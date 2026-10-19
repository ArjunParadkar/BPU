"""E10: continual learning -- the same algorithms on float32 and on BPU tiles.

Five sequential tasks (permuted MNIST, five fixed permutations) on an MLP
784-256-256-10.  Every arm runs on two substrates: float32 (ideal float
weights, the reference) and BPU tiles (sigma_eff 0.04, read noise on every
read, accumulate-and-flush writes at one conductance level, L = 25).

Arms (the algorithm is identical on both substrates):
  naive        plain sequential fine-tuning;
  ewc          elastic weight consolidation (diagonal Fisher after each task,
               penalty lambda/2 sum F (theta - theta*)^2);
  replay       experience replay with a buffer of 1 % of the training set
               (600 samples, equal share per past task), half of every batch;
  fpl          BPU-style consolidation: each synapse is a fast volatile device
               g_f (tau_f = 20 s of chip time) in parallel with a slow
               nonvolatile device g_s; learning writes g_f, which decays; after
               each task a replay phase with learning off transfers g_f into
               g_s only for synapses whose postsynaptic unit is a persistent
               feature (Feature Persistence Lifetime above the median; on
               float32 the two devices are two float tensors);
  fpl_replay   the same with the replay buffer during training;
  joint        all tasks mixed (upper bound, reported for context).

Ablations (BPU, 3 seeds): FPL gate on / off (off = every synapse
consolidates), and device retention tau_ret in {1e3, 1e4, 1e5, inf} s of the
nonvolatile devices at 10 ms of chip time per step, evaluated at the end of
the sequence and after a 1000 s rest.

Criterion (substrate claim): BPU-EWC and BPU-replay within 3 points of
average accuracy of their float32 twins.  Forgetting (mean over the first
four tasks of best-ever minus final accuracy) is reported for every arm,
with writes per device per second against the 6e-3 budget.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from scipy.cluster.hierarchy import linkage

from common import base_parser, save_results, figure_path, Timer
from bpusim.torchtile import TileLinear, TileOptimizer, tile_modules
from bpusim.noise import summarize
from e2_grow_vs_rigl import load_mnist, synthetic_mnist

ARMS = ["naive", "ewc", "replay", "fpl", "fpl_replay"]
SUBSTRATES = ["float32", "bpu"]
BUDGET = 6e-3


class DualTileLinear(nn.Module):
    """Fast volatile tile in parallel with a slow nonvolatile tile: G_eff = g_f + g_s (float32: two tensors)."""

    def __init__(self, i, o, sigma, gen, tau_f_steps, bias=True, write_mode="level"):
        super().__init__()
        self.fast = TileLinear(i, o, bias=bias, sigma=sigma, generator=gen, init_std=0.0, write_mode=write_mode)
        self.slow = TileLinear(i, o, bias=False, sigma=sigma, generator=gen, init_std=0.05, write_mode=write_mode)
        self.slow.W.requires_grad_(False)
        self.decay = float(np.exp(-1.0 / tau_f_steps))
        self.fpl = None

    def forward(self, x):
        return self.fast(x) + self.slow(x)

    @torch.no_grad()
    def step_decay(self):
        f = self.fast
        if f.ideal:
            f.W.mul_(self.decay); return
        f.wp.mul_(self.decay); f.wm.mul_(self.decay); f.pending.mul_(self.decay)
        f.W.copy_(f.scale * f.A * (f.wp - f.wm))

    @torch.no_grad()
    def consolidate(self, gate: torch.Tensor, rate: float):
        """Transfer a fraction `rate` of the fast weight into the slow device for gated output units."""
        delta = self.fast.W * gate[None, :].float() * rate
        realised = self.slow.apply_delta(delta)
        f = self.fast                                     # the fast device gives up what the slow one took
        if f.ideal:
            f.W.sub_(realised); return
        state = realised / f.scale
        f.wp.sub_(torch.clamp(state, min=0)); f.wm.sub_(torch.clamp(-state, min=0))
        f.wp.clamp_(0, 1); f.wm.clamp_(0, 1)
        f.W.copy_(f.scale * f.A * (f.wp - f.wm))

    def retain(self, dt, tau_ret):
        self.slow.retain(dt, tau_ret)


class MLP(nn.Module):
    def __init__(self, sigma, gen, dual=False, tau_f_steps=2000, dims=(784, 256, 256, 10), write_mode="level"):
        super().__init__()
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(DualTileLinear(a, b, sigma, gen, tau_f_steps, write_mode=write_mode) if dual
                          else TileLinear(a, b, sigma=sigma, generator=gen, init_std=0.05, write_mode=write_mode))
        self.layers = nn.ModuleList(layers)
        self.dual = dual

    def forward(self, x, record=False):
        acts = []
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = F.relu(x); acts.append(x)
        return (x, acts) if record else x

    def nonvolatile_tiles(self):
        return [l.slow for l in self.layers] if self.dual else list(self.layers)

    def retain(self, dt, tau_ret):
        for t in self.nonvolatile_tiles():
            t.retain(dt, tau_ret)


def feature_persistence(H: torch.Tensor) -> np.ndarray:
    """Per-unit Feature Persistence Lifetime: single-linkage merge height of the unit's activity vector."""
    Z = H.T.cpu().numpy().astype(float)
    Z = Z - Z.mean(1, keepdims=True)
    nrm = np.linalg.norm(Z, axis=1, keepdims=True); Z = Z / np.maximum(nrm, 1e-9)
    n = Z.shape[0]
    L = linkage(Z, method="single", metric="euclidean")
    death = np.full(n, np.inf)
    members = {i: [i] for i in range(n)}
    for k, (a, b, d, _) in enumerate(L):
        a, b = int(a), int(b)
        for side in (a, b):
            if side < n and np.isinf(death[side]):
                death[side] = d
        members[n + k] = members.pop(a) + members.pop(b)
    death[np.isinf(death)] = L[:, 2].max() if len(L) else 0.0
    death[nrm.ravel() < 1e-6] = 0.0
    return death


class EWC:
    """Diagonal-Fisher elastic weight consolidation over the learnable weights (tile W and biases)."""

    def __init__(self, model, lam):
        self.model, self.lam = model, lam
        self.terms = []

    def params(self):
        return [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]

    def add_task(self, X, y, n_samples=200, gen=None):
        idx = torch.randint(0, len(X), (n_samples,), generator=gen)
        fisher = {n: torch.zeros_like(p) for n, p in self.params()}
        self.model.eval()
        for i in idx.tolist():
            self.model.zero_grad(set_to_none=True)
            loss = F.cross_entropy(self.model(X[i:i + 1]), y[i:i + 1])
            loss.backward()
            for n, p in self.params():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        self.model.train(); self.model.zero_grad(set_to_none=True)
        self.terms.append(({n: f / n_samples for n, f in fisher.items()}, {n: p.detach().clone() for n, p in self.params()}))

    def penalty(self):
        if not self.terms:
            return 0.0
        pen = 0.0
        for fisher, star in self.terms:
            for n, p in self.params():
                pen = pen + (fisher[n] * (p - star[n]) ** 2).sum()
        return 0.5 * self.lam * pen


class ReplayBuffer:
    """1 % of the training set, shared equally by every task seen so far."""

    def __init__(self, capacity, gen):
        self.capacity, self.gen = capacity, gen
        self.tasks = []

    def add(self, X, y):
        self.tasks.append((X, y))
        per = self.capacity // len(self.tasks)
        idx = [torch.randperm(len(Xt), generator=self.gen)[:per].to(Xt.device) for Xt, _ in self.tasks]   # one draw per task for X AND y
        self.X = torch.cat([Xt[i] for (Xt, _), i in zip(self.tasks, idx)])
        self.y = torch.cat([yt[i] for (_, yt), i in zip(self.tasks, idx)])

    def __len__(self):
        return 0 if not self.tasks else len(self.X)

    def sample(self, k):
        idx = torch.randint(0, len(self.X), (k,), generator=self.gen)
        return self.X[idx], self.y[idx]


def accuracy(model, X, y, batch=1000):
    model.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            correct += int((model(X[i:i + batch]).argmax(1) == y[i:i + batch]).sum())
    model.train()
    return correct / len(X)


def train_task(model, opt, X, y, steps, batch, gen, dual=False, ewc=None, buffer=None, tau_ret=math.inf, dt=0.01):
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch,), generator=gen)
        xb, yb = X[idx], y[idx]
        if buffer is not None and len(buffer):
            xr, yr = buffer.sample(batch)
            xb = torch.cat([xb, xr]); yb = torch.cat([yb, yr])
        loss = F.cross_entropy(model(xb), yb)
        if ewc is not None:
            loss = loss + ewc.penalty()
        opt.zero_grad(); loss.backward(); opt.step()
        if dual:
            for l in model.layers:
                l.step_decay()
        if math.isfinite(tau_ret):
            model.retain(dt, tau_ret)


def replay_consolidate(model, X, steps, batch, gen, tau_c_steps, gate_quantile=0.5, tau_ret=math.inf, dt=0.01):
    """Replay with learning off: measure FPL per hidden unit, then consolidate gated synapses."""
    model.eval()
    with torch.no_grad():
        idx = torch.randint(0, len(X), (min(2000, len(X)),), generator=gen)
        _, acts = model(X[idx], record=True)
        gates = []
        for li, H in enumerate(acts):
            fpl = feature_persistence(H)
            if gate_quantile <= 0:
                g = fpl >= 0                                   # gate off: every unit consolidates
            else:
                thr = np.quantile(fpl[fpl > 0], gate_quantile) if np.any(fpl > 0) else 0.0
                g = fpl > thr
            gates.append(torch.tensor(g, device=H.device))
            model.layers[li].fpl = fpl
        gates.append(torch.ones(model.layers[-1].fast.out_features, dtype=torch.bool, device=acts[0].device))
        rate = 1.0 / tau_c_steps
        for _ in range(steps):
            for l, g in zip(model.layers, gates):
                l.consolidate(g, rate)
                l.step_decay()
            if math.isfinite(tau_ret):
                model.retain(dt, tau_ret)
    model.train()
    return [float(g.float().mean()) for g in gates]


def run_arm(arm, substrate, seed, tasks, args, tau_ret=math.inf, gate_quantile=0.5, device="cpu"):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    sigma = None if substrate == "float32" else args.sigma
    dual = arm.startswith("fpl")
    model = MLP(sigma, torch.Generator().manual_seed(seed), dual=dual, tau_f_steps=args.tau_f_steps).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = TileOptimizer(model.modules(), torch.optim.Adam(params, lr=args.lr))
    ewc = EWC(model, args.lam) if arm == "ewc" else None
    buffer = ReplayBuffer(args.buffer, torch.Generator().manual_seed(seed + 500)) if arm.endswith("replay") else None
    acc_matrix = []; gate_fracs = []
    t0 = time.time()
    if arm == "joint":
        Xall = torch.cat([t[0] for t in tasks]); yall = torch.cat([t[1] for t in tasks])
        train_task(model, opt, Xall, yall, args.steps * 5, args.batch, gen, tau_ret=tau_ret, dt=args.step_seconds)
        acc_matrix = [[accuracy(model, t[2], t[3]) for t in tasks]] * 5
    else:
        for ti, (Xa, ya, Xb, yb) in enumerate(tasks):
            train_task(model, opt, Xa, ya, args.steps, args.batch, gen, dual=dual, ewc=ewc, buffer=buffer, tau_ret=tau_ret, dt=args.step_seconds)
            if dual:
                gate_fracs.append(replay_consolidate(model, Xa, args.replay_steps, args.batch, gen, args.tau_c_steps, gate_quantile, tau_ret, args.step_seconds))
            if ewc is not None:
                ewc.add_task(Xa, ya, gen=gen)
            if buffer is not None:
                buffer.add(Xa, ya)
            acc_matrix.append([accuracy(model, t[2], t[3]) for t in tasks])
    A = np.array(acc_matrix)
    final = A[-1]; avg = float(final.mean())
    forgetting = float(np.mean([A[:, j].max() - A[-1, j] for j in range(4)]))
    # after a rest of 1000 s of chip time with no input: fast devices decay fully, nonvolatile ones by exp(-1000 / tau_ret)
    with torch.no_grad():
        if dual:
            for l in model.layers:
                f = l.fast
                if f.ideal:
                    f.W.zero_()
                else:
                    f.wp.zero_(); f.wm.zero_(); f.pending.zero_(); f.W.zero_()
        if math.isfinite(tau_ret) and substrate == "bpu":
            model.retain(1000.0, tau_ret)
    rest_acc = [accuracy(model, t[2], t[3]) for t in tasks]
    rest = {"acc": rest_acc, "avg": float(np.mean(rest_acc)), "forgetting": float(np.mean([A[:, j].max() - rest_acc[j] for j in range(4)]))}
    tiles = tile_modules(model)
    writes = sum(m.n_writes for m in tiles); n_dev = sum(m.n_devices for m in tiles)
    n_steps = args.steps * 5 + (args.replay_steps * 5 if dual else 0)
    wpd = writes / n_dev; wps = wpd / (n_steps * args.step_seconds)
    out = {"arm": arm, "substrate": substrate, "seed": seed, "tau_ret": tau_ret if math.isfinite(tau_ret) else None, "gate_quantile": gate_quantile,
           "acc_matrix": A.tolist(), "final_acc": final.tolist(), "avg_acc": avg, "forgetting": forgetting, "after_rest_1000s": rest,
           "writes_per_device": wpd, "writes_per_device_per_s": wps, "percent_of_budget": 100 * wps / BUDGET,
           "flush_rate": float(np.mean([t.flush_rate for t in tiles])) if substrate == "bpu" else 0.0,
           "gate_fractions": gate_fracs, "seconds": time.time() - t0}
    print(f"  seed {seed} {substrate:8s} {arm:10s} tau_ret={tau_ret:g} gate={gate_quantile}: avg acc {avg:.3f} forgetting {forgetting:.3f} final {np.round(final, 3)} "
          f"| rest {rest['avg']:.3f} | writes/dev/s {wps:.3g} ({100 * wps / BUDGET:.0f}% budget) ({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    ap = base_parser("E10 continual learning (substrate framing)", default_seeds=3)
    ap.add_argument("--steps", type=int, default=600, help="training steps per task")
    ap.add_argument("--replay-steps", type=int, default=200, help="FPL consolidation replay steps")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=0.04)
    ap.add_argument("--lam", type=float, default=100.0, help="EWC lambda")
    ap.add_argument("--buffer", type=int, default=600, help="replay buffer size (1 %% of the 60k training set)")
    ap.add_argument("--tau-f-steps", type=int, default=2000, help="tau_f = 20 s at 10 ms per step")
    ap.add_argument("--tau-c-steps", type=int, default=100)
    ap.add_argument("--step-seconds", type=float, default=0.01)
    ap.add_argument("--tau-ret", type=str, default="1e3,1e4,1e5", help="retention ablation (s); inf is the main run")
    ap.add_argument("--no-ablation", action="store_true")
    ap.add_argument("--only-arms", type=str, default=None, help="run only these arms (comma list) and merge them into the existing results.json, replacing those arms' runs")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    seeds = 10 if args.full else args.seeds
    if args.smoke:
        data = synthetic_mnist(np.random.default_rng(0)); seeds = 1; args.steps, args.replay_steps = 20, 10
    else:
        data = load_mnist()
    dev = args.device
    Xtr = torch.tensor(data["train_x"]).to(dev); ytr = torch.tensor(data["train_y"]).to(dev); Xte = torch.tensor(data["test_x"]).to(dev); yte = torch.tensor(data["test_y"]).to(dev)
    taus = [] if args.no_ablation else [float(x) for x in args.tau_ret.split(",")]
    runs = []
    def tasks_for(seed):
        rng = np.random.default_rng(seed)
        perms = [torch.tensor(rng.permutation(784), device=dev) for _ in range(5)]
        return [(Xtr[:, p], ytr, Xte[:, p], yte) for p in perms]
    # main arms for every seed first, then the ablations, with an incremental save after every run
    plan = [(seed, arm, sub, math.inf, 0.5) for seed in range(seeds) for sub in SUBSTRATES for arm in ARMS + ["joint"]]
    if not args.no_ablation:
        plan += [(seed, arm, sub, math.inf, 0.0) for seed in range(seeds) for sub in SUBSTRATES for arm in ("fpl", "fpl_replay")]
        plan += [(seed, arm, "bpu", tau, 0.5) for seed in range(seeds) for tau in taus for arm in ARMS]
    if args.only_arms:
        keep = args.only_arms.split(",")
        plan = [p for p in plan if p[1] in keep]
        from common import load_results
        runs = [r for r in load_results("e10", args.out)["runs"] if r["arm"] not in keep]
        print(f"merging: {len(runs)} existing runs kept, {len(plan)} runs to redo for {keep}", flush=True)
    tasks_cache = {}
    for i, (seed, arm, sub, tau, gate) in enumerate(plan):
        if seed not in tasks_cache:
            tasks_cache[seed] = tasks_for(seed)
        runs.append(run_arm(arm, sub, seed, tasks_cache[seed], args, tau_ret=tau, gate_quantile=gate, device=dev))
        save_results("e10", {"partial": True, "done": i + 1, "planned": len(plan), "seeds": seeds, "arms": ARMS, "substrates": SUBSTRATES, "runs": runs,
                             "verdict": {"statement": f"PARTIAL: {i + 1}/{len(plan)} runs"}}, args.out)
    # ---- summary
    def sel(arm, substrate, tau=None, gate=0.5):
        return [r for r in runs if r["arm"] == arm and r["substrate"] == substrate and r["tau_ret"] == tau and r["gate_quantile"] == gate]
    def summ(rs):
        return {"avg_acc": summarize([r["avg_acc"] for r in rs]), "forgetting": summarize([r["forgetting"] for r in rs]),
                "avg_acc_after_rest_1000s": summarize([r["after_rest_1000s"]["avg"] for r in rs]),
                "forgetting_after_rest_1000s": summarize([r["after_rest_1000s"]["forgetting"] for r in rs]),
                "writes_per_device_per_s": summarize([r["writes_per_device_per_s"] for r in rs]), "percent_of_budget": summarize([r["percent_of_budget"] for r in rs]),
                "flush_rate": summarize([r["flush_rate"] for r in rs]), "n": len(rs)}
    summary = {s: {a: summ(sel(a, s)) for a in ARMS + ["joint"]} for s in SUBSTRATES}
    twins = {a: {"float32": summary["float32"][a]["avg_acc"]["mean"], "bpu": summary["bpu"][a]["avg_acc"]["mean"],
                 "gap_points": summary["bpu"][a]["avg_acc"]["mean"] - summary["float32"][a]["avg_acc"]["mean"]} for a in ARMS + ["joint"]}
    ok = all(abs(twins[a]["gap_points"]) <= 0.03 for a in ("ewc", "replay"))
    ablation = {"gate_off": {s: {a: summ(sel(a, s, gate=0.0)) for a in ("fpl", "fpl_replay")} for s in SUBSTRATES} if taus or not args.no_ablation else {},
                "tau_ret": {f"{tau:g}": {a: summ(sel(a, "bpu", tau=tau)) for a in ARMS} for tau in taus}}
    out = {"seeds": seeds, "steps": args.steps, "replay_steps": args.replay_steps, "sigma": args.sigma, "lam": args.lam, "buffer": args.buffer,
           "tau_f_steps": args.tau_f_steps, "tau_c_steps": args.tau_c_steps, "step_seconds": args.step_seconds, "device": dev, "smoke": args.smoke,
           "arms": ARMS, "substrates": SUBSTRATES, "runs": runs, "summary": summary, "twins": twins, "ablation": ablation,
           "verdict": {"success": bool(ok), "statement": ("SMOKE RUN. " if args.smoke else "") +
                       f"average accuracy after 5 tasks, float32 / BPU: " + ", ".join(f"{a} {twins[a]['float32']:.3f} / {twins[a]['bpu']:.3f} ({twins[a]['gap_points']:+.3f})" for a in ARMS + ["joint"])
                       + "; forgetting float32 / BPU: " + ", ".join(f"{a} {summary['float32'][a]['forgetting']['mean']:.3f} / {summary['bpu'][a]['forgetting']['mean']:.3f}" for a in ARMS)
                       + f"; BPU writes: " + ", ".join(f"{a} {summary['bpu'][a]['percent_of_budget']['mean']:.0f}% of budget" for a in ARMS)
                       + (". SUCCESS: BPU-EWC and BPU-replay within 3 points of their float32 twins" if ok else
                          f". Criterion not met: EWC gap {twins['ewc']['gap_points']:+.3f}, replay gap {twins['replay']['gap_points']:+.3f} (limit 0.03)")}}
    print("E10 verdict:", out["verdict"]["statement"])
    save_results("e10", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e10_continual"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = res["arms"] + ["joint"]; S = res["summary"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    x = np.arange(len(arms))
    for i, s in enumerate(res["substrates"]):
        ax[0].bar(x + (i - 0.5) * 0.38, [S[s][a]["avg_acc"]["mean"] for a in arms], 0.38, yerr=[S[s][a]["avg_acc"]["sem"] for a in arms], capsize=2, label=s, color="k" if s == "float32" else "C0", alpha=0.85 if s == "float32" else 1)
    ax[0].set_xticks(x); ax[0].set_xticklabels(arms, fontsize=8); ax[0].set_ylim(0, 1); ax[0].set_ylabel("average accuracy after 5 tasks"); ax[0].legend(fontsize=8)
    ax[0].set_title("E10: same algorithm, two substrates", fontsize=9)
    for i, s in enumerate(res["substrates"]):
        ax[1].bar(x[:-1] + (i - 0.5) * 0.38, [S[s][a]["forgetting"]["mean"] for a in res["arms"]], 0.38, yerr=[S[s][a]["forgetting"]["sem"] for a in res["arms"]], capsize=2, label=s, color="k" if s == "float32" else "C0")
    ax[1].set_xticks(x[:-1]); ax[1].set_xticklabels(res["arms"], fontsize=8); ax[1].set_ylabel("forgetting"); ax[1].legend(fontsize=8); ax[1].set_title("forgetting per arm", fontsize=9)
    ab = res.get("ablation", {}).get("tau_ret", {})
    if ab:
        taus = sorted(ab, key=float) + ["inf"]
        for a in res["arms"]:
            ys = [ab[t][a]["avg_acc_after_rest_1000s"]["mean"] for t in taus[:-1]] + [S["bpu"][a]["avg_acc_after_rest_1000s"]["mean"]]
            ax[2].plot(range(len(taus)), ys, marker="o", label=a)
        ax[2].set_xticks(range(len(taus))); ax[2].set_xticklabels(taus); ax[2].set_xlabel("tau_ret (s)"); ax[2].set_ylabel("avg accuracy after a 1000 s rest (BPU)"); ax[2].legend(fontsize=7)
        ax[2].set_title("retention ablation", fontsize=9)
    fig.suptitle(res["verdict"]["statement"][:150], fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e10] figure {path}")


if __name__ == "__main__":
    main()
