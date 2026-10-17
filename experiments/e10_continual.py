"""E10: continual learning without forgetting, with volatile/nonvolatile consolidation.

Five sequential tasks (permuted MNIST, five fixed permutations) on an MLP
whose layers are tiles (784-256-256-10, sigma_eff 0.04).  Arms:

* NAIVE   nonvolatile devices only, plain sequential fine-tuning;
* FROZEN  trained on task 1 only, then frozen (a fixed accelerator);
* JOINT   trained on all five tasks mixed (reference upper bound);
* BPU     every synapse is a fast volatile device g_f (tau_f = 20 s of chip
  time) in parallel with a slow nonvolatile device g_s, G_eff = g_f + g_s.
  Learning writes g_f, which decays.  After each task a replay phase
  presents that task again with learning off; during replay the slow
  device copies the fast one (tau_c) *only for synapses whose
  postsynaptic unit is a persistent feature*, measured by the Feature
  Persistence Lifetime of that unit in the replay activity (the unit's own
  H0 merge distance in a single-linkage filtration of the unit activity
  vectors: distinct, long-lived features consolidate; redundant ones fade).

Reported (3 seeds): the accuracy matrix (after task i, on task j), average
accuracy over the five tasks after the sequence, backward transfer /
forgetting (mean over tasks of best-ever minus final accuracy), the same
numbers after a rest of one tau_f with no input, and writes per device.
Success test from the plan: BPU average accuracy within 5 points of JOINT
and forgetting under 10 points, with NAIVE forgetting more.
"""
from __future__ import annotations

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


class DualTileLinear(nn.Module):
    """Fast volatile tile in parallel with a slow nonvolatile tile: G_eff = g_f + g_s."""

    def __init__(self, i, o, sigma, gen, tau_f_steps, bias=True):
        super().__init__()
        self.fast = TileLinear(i, o, bias=bias, sigma=sigma, generator=gen, init_std=0.0)
        self.slow = TileLinear(i, o, bias=False, sigma=sigma, generator=gen, init_std=0.05)
        self.slow.W.requires_grad_(False)
        self.decay = float(np.exp(-1.0 / tau_f_steps))
        self.fpl = None

    def forward(self, x):
        return self.fast(x) + self.slow(x)

    @torch.no_grad()
    def step_decay(self):
        f = self.fast
        f.wp.mul_(self.decay); f.wm.mul_(self.decay); f.pending.mul_(self.decay)
        f.W.copy_(f.scale * f.A * (f.wp - f.wm))

    @torch.no_grad()
    def consolidate(self, gate: torch.Tensor, rate: float):
        """Transfer a fraction `rate` of the fast weight into the slow device for gated output units."""
        delta = self.fast.W * gate[None, :].float() * rate
        realised = self.slow.apply_delta(delta)
        # the fast device gives up what the slow one took (conserving G_eff)
        f = self.fast
        state = realised / f.scale
        f.wp.sub_(torch.clamp(state, min=0)); f.wm.sub_(torch.clamp(-state, min=0))
        f.wp.clamp_(0, 1); f.wm.clamp_(0, 1)
        f.W.copy_(f.scale * f.A * (f.wp - f.wm))


class MLP(nn.Module):
    def __init__(self, sigma, gen, dual=False, tau_f_steps=2000, dims=(784, 256, 256, 10)):
        super().__init__()
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers.append(DualTileLinear(a, b, sigma, gen, tau_f_steps) if dual else TileLinear(a, b, sigma=sigma, generator=gen, init_std=0.05))
        self.layers = nn.ModuleList(layers)
        self.dual = dual

    def forward(self, x, record=False):
        acts = []
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = F.relu(x); acts.append(x)
        return (x, acts) if record else x


def feature_persistence(H: torch.Tensor) -> np.ndarray:
    """Per-unit Feature Persistence Lifetime: each unit's activity vector over the replay samples is a
    point; single-linkage merge height of the unit (the death of its own H0 bar) is its lifetime."""
    Z = H.T.numpy().astype(float)                       # units x samples
    Z = Z - Z.mean(1, keepdims=True)
    nrm = np.linalg.norm(Z, axis=1, keepdims=True); Z = Z / np.maximum(nrm, 1e-9)
    n = Z.shape[0]
    L = linkage(Z, method="single", metric="euclidean")
    death = np.full(n, np.inf)
    # walk the dendrogram: the first merge involving an original point gives its death time
    members = {i: [i] for i in range(n)}
    for k, (a, b, d, _) in enumerate(L):
        a, b = int(a), int(b)
        for side in (a, b):
            if side < n and np.isinf(death[side]):
                death[side] = d
        # elder rule: the smaller/younger cluster dies; original points already set above
        members[n + k] = members.pop(a) + members.pop(b)
    death[np.isinf(death)] = L[:, 2].max() if len(L) else 0.0
    dead_units = (nrm.ravel() < 1e-6)
    death[dead_units] = 0.0                              # silent units are not features
    return death


def accuracy(model, X, y, batch=1000):
    model.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch):
            correct += int((model(X[i:i + batch]).argmax(1) == y[i:i + batch]).sum())
    model.train()
    return correct / len(X)


def train_task(model, opt, X, y, steps, batch, gen, dual=False):
    for _ in range(steps):
        idx = torch.randint(0, len(X), (batch,), generator=gen)
        loss = F.cross_entropy(model(X[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if dual:
            for l in model.layers:
                l.step_decay()


def replay_consolidate(model, X, y, steps, batch, gen, tau_c_steps, gate_quantile=0.5):
    """Replay with learning off: measure FPL per hidden unit, then consolidate gated synapses."""
    model.eval()
    with torch.no_grad():
        idx = torch.randint(0, len(X), (min(2000, len(X)),), generator=gen)
        _, acts = model(X[idx], record=True)
        gates = []
        for li, H in enumerate(acts):
            fpl = feature_persistence(H)
            thr = np.quantile(fpl[fpl > 0], gate_quantile) if np.any(fpl > 0) else 0.0
            gates.append(torch.tensor(fpl > thr))
            model.layers[li].fpl = fpl
        gates.append(torch.ones(model.layers[-1].fast.out_features, dtype=torch.bool))   # readout consolidates fully
        rate = 1.0 / tau_c_steps
        for _ in range(steps):
            for l, g in zip(model.layers, gates):
                l.consolidate(g, rate)
                l.step_decay()
    model.train()
    return [float(g.float().mean()) for g in gates]


def run_seed(seed, data, args):
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(784) for _ in range(5)]
    Xtr = torch.tensor(data["train_x"]); ytr = torch.tensor(data["train_y"]); Xte = torch.tensor(data["test_x"]); yte = torch.tensor(data["test_y"])
    tasks = [(Xtr[:, p], ytr, Xte[:, p], yte) for p in perms]
    out = {}
    for arm in ("naive", "frozen", "joint", "bpu"):
        dual = arm == "bpu"
        model = MLP(args.sigma, torch.Generator().manual_seed(seed), dual=dual, tau_f_steps=args.tau_f_steps)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = TileOptimizer(model.modules(), torch.optim.Adam(params, lr=args.lr))
        acc_matrix = []; gate_fracs = []
        t0 = time.time()
        if arm == "joint":
            Xall = torch.cat([t[0] for t in tasks]); yall = torch.cat([t[1] for t in tasks])
            train_task(model, opt, Xall, yall, args.steps * 5, args.batch, gen)
            acc_matrix = [[accuracy(model, t[2], t[3]) for t in tasks]] * 5
        else:
            for ti, (Xa, ya, Xb, yb) in enumerate(tasks):
                if not (arm == "frozen" and ti > 0):
                    train_task(model, opt, Xa, ya, args.steps, args.batch, gen, dual=dual)
                    if dual:
                        gate_fracs.append(replay_consolidate(model, Xa, ya, args.replay_steps, args.batch, gen, args.tau_c_steps))
                acc_matrix.append([accuracy(model, t[2], t[3]) for t in tasks])
        A = np.array(acc_matrix)
        final = A[-1]; avg = float(final.mean())
        forgetting = float(np.mean([A[:, j].max() - A[-1, j] for j in range(4)]))
        rest = None
        if dual:   # let the fast devices decay for one tau_f with no input, then re-evaluate
            for _ in range(args.tau_f_steps):
                for l in model.layers:
                    l.step_decay()
            rest_acc = [accuracy(model, t[2], t[3]) for t in tasks]
            rest = {"acc": rest_acc, "avg": float(np.mean(rest_acc)), "forgetting": float(np.mean([A[:, j].max() - rest_acc[j] for j in range(4)]))}
        writes = sum(m.n_writes for m in tile_modules(model)); n_dev = sum(m.n_devices for m in tile_modules(model))
        out[arm] = {"acc_matrix": A.tolist(), "final_acc": final.tolist(), "avg_acc": avg, "forgetting": forgetting, "after_rest": rest,
                    "writes_per_device": writes / n_dev, "gate_fractions": gate_fracs, "seconds": time.time() - t0}
        print(f"  seed {seed} {arm}: avg acc {avg:.3f} forgetting {forgetting:.3f} final {np.round(final, 3)}" + (f" | after rest avg {rest['avg']:.3f} forg {rest['forgetting']:.3f}" if rest else "") + f" ({time.time() - t0:.0f}s)", flush=True)
    return out


def main():
    ap = base_parser("E10 continual learning with consolidation", default_seeds=3)
    ap.add_argument("--steps", type=int, default=600, help="training steps per task")
    ap.add_argument("--replay-steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--sigma", type=float, default=0.04)
    ap.add_argument("--tau-f-steps", type=int, default=2000, help="tau_f = 20 s at 10 ms per step")
    ap.add_argument("--tau-c-steps", type=int, default=100, help="consolidation time constant during replay (steps)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    seeds = 10 if args.full else args.seeds
    if args.smoke:
        data = synthetic_mnist(np.random.default_rng(0)); seeds = 1; args.steps, args.replay_steps = 20, 10
    else:
        data = load_mnist()
    runs = []
    for seed in range(seeds):
        with Timer(f"seed {seed}"):
            runs.append(run_seed(seed, data, args))
    arms = ["naive", "frozen", "joint", "bpu"]
    summary = {a: {"avg_acc": summarize([r[a]["avg_acc"] for r in runs]), "forgetting": summarize([r[a]["forgetting"] for r in runs]),
                   "writes_per_device": summarize([r[a]["writes_per_device"] for r in runs])} for a in arms}
    summary["bpu_after_rest"] = {"avg_acc": summarize([r["bpu"]["after_rest"]["avg"] for r in runs]), "forgetting": summarize([r["bpu"]["after_rest"]["forgetting"] for r in runs])}
    gap = summary["joint"]["avg_acc"]["mean"] - summary["bpu"]["avg_acc"]["mean"]
    ok = gap <= 0.05 and summary["bpu"]["forgetting"]["mean"] < 0.10 and summary["naive"]["forgetting"]["mean"] > summary["bpu"]["forgetting"]["mean"]
    out = {"seeds": seeds, "steps": args.steps, "replay_steps": args.replay_steps, "sigma": args.sigma, "tau_f_steps": args.tau_f_steps,
           "tau_c_steps": args.tau_c_steps, "smoke": args.smoke, "runs": runs, "summary": summary,
           "verdict": {"success": bool(ok), "statement": ("SMOKE RUN. " if args.smoke else "") +
                       f"average accuracy after 5 tasks: joint {summary['joint']['avg_acc']['mean']:.3f}, BPU {summary['bpu']['avg_acc']['mean']:.3f} "
                       f"(after rest {summary['bpu_after_rest']['avg_acc']['mean']:.3f}), naive {summary['naive']['avg_acc']['mean']:.3f}, frozen {summary['frozen']['avg_acc']['mean']:.3f}; "
                       f"forgetting: BPU {summary['bpu']['forgetting']['mean']:.3f}, naive {summary['naive']['forgetting']['mean']:.3f}; "
                       + ("SUCCESS (within 5 points of joint, forgetting < 10 points, naive forgets more)" if ok else
                          f"criterion not met (gap to joint {gap:+.3f}, BPU forgetting {summary['bpu']['forgetting']['mean']:.3f})")}}
    print("E10 verdict:", out["verdict"]["statement"])
    save_results("e10", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e10_continual"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    arms = ["naive", "frozen", "joint", "bpu"]
    for a in arms:
        A = np.mean([r[a]["acc_matrix"] for r in res["runs"]], 0)
        ax[0].plot(range(1, 6), [A[i, 0] for i in range(5)], marker="o", label=a)
    ax[0].set_xlabel("tasks seen"); ax[0].set_ylabel("accuracy on task 1"); ax[0].legend(fontsize=8); ax[0].set_title("E10: retention of task 1", fontsize=10)
    s = res["summary"]
    x = np.arange(len(arms) + 1); labels = arms + ["bpu after rest"]
    av = [s[a]["avg_acc"]["mean"] for a in arms] + [s["bpu_after_rest"]["avg_acc"]["mean"]]
    fg = [s[a]["forgetting"]["mean"] for a in arms] + [s["bpu_after_rest"]["forgetting"]["mean"]]
    ax[1].bar(x - 0.2, av, 0.4, label="average accuracy"); ax[1].bar(x + 0.2, fg, 0.4, label="forgetting")
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=7, rotation=15); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1)
    A = np.mean([r["bpu"]["acc_matrix"] for r in res["runs"]], 0)
    im = ax[2].imshow(A, vmin=0, vmax=1, cmap="viridis"); ax[2].set_xlabel("evaluated task"); ax[2].set_ylabel("after training task"); ax[2].set_title("BPU accuracy matrix", fontsize=10)
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.suptitle(res["verdict"]["statement"][:120], fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e10] figure {path}")


if __name__ == "__main__":
    main()
