"""E9: trainability under drift -- the same algorithm on float32 and on BPU tiles.

The E8 architecture (nanoGPT, 6 layers, d = 192) is trained once on tiny
Shakespeare as an ideal float model, then exposed to a non-stationary
character stream: Shakespeare for ``--seg`` steps, then this repository's own
Python source, then Shakespeare again.  Six deployments see the identical
stream (same batches, same seed):

  frozen       BPU tiles (sigma_eff 0.04), weights LOADed once, no updates
               (a fixed accelerator / FPGA bitstream);
  sgd_float32  the float32 REFERENCE: backprop + AdamW on the ideal model;
  sgd_bpu      the SUBSTRATE CEILING: the same backprop + AdamW, with float
               accumulators (Adam moments, the per-synapse accumulate-and-
               flush accumulator) and BPU devices as the weights -- every read
               is a noisy tile read, every write a whole conductance level
               (L = 25 levels at sigma_eff 0.04), stochastic rounding;
  fa_bpu       feedback alignment: fixed random feedback matrices instead of
               W^T between layers, pulsed through the tiles;
  dfa_bpu      direct feedback alignment (fully local): every tile receives
               the global output error through its own fixed random matrix,
               nothing propagates between layers (the output head uses its own
               error);
  local_bpu    layer-local three-factor rule: every tile has its own digital
               readout to the next-token target and minimises its own
               cross-entropy; the local error at the tile output is the
               (vector) modulator, the presynaptic activity the eligibility,
               nothing propagates between layers.

Criterion (substrate claim): sgd_bpu within 5 % of sgd_float32 on the
post-shift loss (mean loss over the last 200 steps of the Python segment, at
the same step count) AND writes per device per second within the 6e-3
endurance budget (chip time per step is the stated assumption
``--step-seconds``).  Secondary: how close fa / dfa / local come to the
oracle, recovery time after each switch (steps until the 100-step mean loss
closes 90 % of the gap to its segment minimum), writes per device per second
and the percentage of the budget, flush rate.  3 seeds.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common import base_parser, save_results, figure_path, Timer, ROOT, load_results
import json
from bpusim.torchtile import TileLinear, TileOptimizer, tile_modules, DFAContext, write_budget_report
from bpusim.noise import summarize
from e8_nanogpt_tiles import GPT, get_batch, train, evaluate, copy_weights, DATA, model_device

ARMS = ["frozen", "sgd_float32", "sgd_bpu", "fa_bpu", "dfa_bpu", "local_bpu"]
BUDGET = 6e-3


def load_corpora():
    shakes = pathlib.Path(DATA).read_text(encoding="utf-8")
    files = sorted((ROOT / "bpusim").glob("*.py")) + sorted((ROOT / "experiments").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    code = "\n\n".join(f.read_text(encoding="utf-8") for f in files)
    code = (code * (len(shakes) // max(len(code), 1) + 1))[:len(shakes)]   # same length as Shakespeare
    chars = sorted(set(shakes) | set(code))
    stoi = {c: i for i, c in enumerate(chars)}
    enc = lambda t: torch.tensor([stoi[c] for c in t], dtype=torch.long)
    return enc(shakes), enc(code), len(chars)


class LocalLayerLosses:
    """Layer-local three-factor learning for every TileLinear except the head: a digital readout R_l
    (out_features -> vocab) per tile, local loss CE(R_l y_l, next token).  The main forward runs
    without gradients; each tile is then re-read on the captured input and its local loss is
    back-propagated into that tile's weight and readout only."""

    def __init__(self, model: GPT, vocab: int, lr: float, lr_readout: float):
        self.model = model
        self.tiles = [t for t in tile_modules(model) if t is not model.head]
        dev = model_device(model)
        self.readouts = nn.ModuleDict({str(i): nn.Linear(t.out_features, vocab, bias=False).to(dev) for i, t in enumerate(self.tiles)})
        for r in self.readouts.values():
            nn.init.normal_(r.weight, std=0.02)
        self.cache = {}
        for t in self.tiles:
            t.register_forward_hook(self._hook)
        params = [p for t in self.tiles for p in (t.W, t.b) if p is not None] + [model.head.W]
        self.opt = TileOptimizer(model.modules(), torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0))
        self.opt_r = torch.optim.AdamW(self.readouts.parameters(), lr=lr_readout)

    def _hook(self, module, inp, out):
        self.cache[id(module)] = inp[0].detach()

    def step(self, x, y):
        with torch.no_grad():
            logits, loss = self.model(x, y)
        self.opt.zero_grad(); self.opt_r.zero_grad()
        tgt = y.reshape(-1)
        for i, t in enumerate(self.tiles):
            xin = self.cache[id(t)]
            yl = t(xin)                                         # fresh noisy read, gradient to t.W / t.b only
            ll = F.cross_entropy(self.readouts[str(i)](yl).reshape(-1, self.readouts[str(i)].out_features), tgt)
            ll.backward()
        # the head's own error is local to it
        h_in = self.model.ln(self._last_hidden(x))
        lh = F.cross_entropy(self.model.head(h_in).reshape(-1, self.model.head.out_features), tgt)
        lh.backward()
        torch.nn.utils.clip_grad_norm_([p for g in self.opt.opt.param_groups for p in g["params"]], 1.0)
        self.opt.step(); self.opt_r.step()
        return loss

    @torch.no_grad()
    def _last_hidden(self, idx):
        m = self.model
        B, T = idx.shape
        h = m.tok(idx) + m.pos(torch.arange(T, device=idx.device))
        for b in m.blocks:
            h = b(h)
        return h


def recovery_time(curve, switch_idx, window=100, frac=0.9):
    seg = np.asarray(curve[switch_idx:], float)
    if len(seg) < window + 2:
        return None
    ma = np.convolve(seg, np.ones(window) / window, mode="valid")
    start, best = ma[0], ma.min()
    if start - best <= 1e-6:
        return 0
    target = start - frac * (start - best)
    hit = np.nonzero(ma <= target)[0]
    return int(hit[0]) if hit.size else None


LOCALITY = {"sgd_bpu": 0, "fa_bpu": 1, "dfa_bpu": 2, "local_bpu": 3}   # 0 = global backprop ... 3 = layer-local
LOCALITY_LABEL = {0: "backprop (global gradient)", 1: "feedback alignment (global error, random feedback)",
                  2: "direct FA (broadcast error, no inter-layer gradient)", 3: "layer-local three-factor"}


def make_model(arm, vocab, block, sigma, seed, device, theta_flush=1.0, write_mode="nearest_pulse"):
    kw = dict(write_mode=write_mode, theta_flush=theta_flush)
    if arm == "sgd_float32":
        return GPT(vocab, 192, 6, 6, block, None, None, seed=seed).to(device), None
    if arm == "sgd_bpu_pulse":   # diagnostic: the same oracle with the legacy nearest-pulse writes (no level gate)
        return GPT(vocab, 192, 6, 6, block, sigma, None, seed=seed, tile_kw=dict(write_mode="pulse")).to(device), None
    if arm == "dfa_bpu":
        holder = DFAContext()
        return GPT(vocab, 192, 6, 6, block, sigma, None, seed=seed, tile_kw=dict(kw, dfa=holder, dfa_dim=vocab), head_kw=dict(dfa=None, dfa_dim=0)).to(device), holder
    return GPT(vocab, 192, 6, 6, block, sigma, None, fa=(arm == "fa_bpu"), seed=seed, tile_kw=kw).to(device), None


def run_stream(model, arm, segments, block, batch, gen, lr, holder=None, local=None, log_every=200):
    dev = model_device(model)
    losses, writes = [], []
    opt = None
    if arm in ("sgd_float32", "sgd_bpu", "sgd_bpu_pulse", "fa_bpu", "dfa_bpu"):
        opt = TileOptimizer(model.modules(), torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0))
    step = 0; t0 = time.time()
    for seg_i, data in enumerate(segments):
        for _ in range(data["steps"]):
            x, y = get_batch(data["ids"], block, batch, gen, dev)
            if arm == "frozen":
                with torch.no_grad():
                    _, loss = model(x, y)
            elif arm == "local_bpu":
                loss = local.step(x, y)
            else:
                logits, loss = model(x, y)
                opt.zero_grad()
                if holder is not None:   # DFA: broadcast the output error (d CE / d logits, mean over tokens)
                    with torch.no_grad():
                        e = torch.softmax(logits.reshape(-1, logits.shape[-1]), -1)
                        e[torch.arange(e.shape[0], device=dev), y.reshape(-1)] -= 1.0
                        holder.err = e / e.shape[0]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(float(loss))
            writes.append(sum(t.n_writes for t in tile_modules(model)))
            step += 1
            if step % log_every == 0:
                print(f"    {arm} step {step} seg {seg_i} loss {np.mean(losses[-log_every:]):.3f} ({time.time() - t0:.0f}s)", flush=True)
    return losses, writes


def main():
    ap = base_parser("E9 trainability under drift (substrate framing)", default_seeds=3)
    ap.add_argument("--seg", type=int, default=1000, help="steps per stream segment")
    ap.add_argument("--pretrain-steps", type=int, default=1000)
    ap.add_argument("--sigma", type=float, default=0.04)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4, help="AdamW learning rate on the stream (all gradient arms)")
    ap.add_argument("--lr-readout", type=float, default=1e-3, help="local arm: learning rate of the digital per-layer readouts")
    ap.add_argument("--step-seconds", type=float, default=0.01, help="assumed chip time per training step for the write-rate conversion")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--theta-flush", type=str, default="1.0", help="level mode: flush thresholds in conductance levels, comma list; each BPU gradient arm is run at every value")
    ap.add_argument("--write-mode", type=str, default="nearest_pulse", choices=["nearest_pulse", "level", "pulse"],
                    help="write rule of the BPU arms (nearest_pulse is the architecture default since 2026-10-19)")
    ap.add_argument("--merge", type=str, default=None, help="results.json whose runs are reused for arms/thetas not rerun here (e.g. the theta = 1.0 run)")
    ap.add_argument("--resummarize", action="store_true", help="no runs: rebuild summary / verdict / figure from the runs in results.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    arms = args.arms.split(",")
    seeds = args.seeds
    if args.smoke:
        args.seg, args.pretrain_steps, seeds = 30, 30, 1
    shakes, code, vocab = load_corpora()
    n = int(0.9 * len(shakes)); shakes_tr = shakes[:n]
    segments = [{"name": "shakespeare", "ids": shakes_tr, "steps": args.seg}, {"name": "python_source", "ids": code, "steps": args.seg},
                {"name": "shakespeare", "ids": shakes_tr, "steps": args.seg}]
    thetas = [float(x) for x in args.theta_flush.split(",")]
    merged = json.load(open(args.merge))["runs"] if args.merge else None
    runs = []
    if args.resummarize:
        prev = load_results("e9", args.out); runs = prev["runs"]; seeds = prev["seeds"]; thetas = prev.get("thetas", thetas)
        for k in ("seg", "sigma", "pretrain_steps", "batch", "step_seconds", "lr", "write_mode"):
            setattr(args, k, prev.get(k, getattr(args, k)))
        return summarize_and_save(runs, seeds, thetas, arms, segments, args)
    for seed in range(seeds):
        torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
        ideal = GPT(vocab, 192, 6, 6, args.block, None, None, seed=seed).to(args.device)
        with Timer(f"seed {seed}: pretrain ideal on Shakespeare"):
            train(ideal, shakes_tr, args.pretrain_steps, args.block, 24, 1e-3, gen, tiled=False, label="pretrain")
        base_loss = {"shakespeare": evaluate(ideal, shakes[n:], args.block, args.batch, 20, torch.Generator().manual_seed(5)),
                     "python_source": evaluate(ideal, code, args.block, args.batch, 20, torch.Generator().manual_seed(5))}
        print(f"  ideal val loss: {base_loss}", flush=True)
        curves = dict(merged[seed]["curves"]) if merged else {}
        if args.write_mode != "level":
            thetas = [1.0]          # theta_flush only exists in level mode
        jobs = [(a, 1.0) for a in arms if a in ("frozen", "sgd_float32")] + [(a, th) for th in thetas for a in arms if a not in ("frozen", "sgd_float32")]
        for arm, theta in jobs:
            name = arm if (theta == 1.0 or arm in ("frozen", "sgd_float32")) else f"{arm}@{theta:g}"
            if name in curves and merged:
                print(f"  {name}: reused from {args.merge}", flush=True); continue
            model, holder = make_model(arm, vocab, args.block, args.sigma, seed, args.device, theta, args.write_mode)
            model.load_state_dict(ideal.state_dict(), strict=False); copy_weights(ideal, model, verify=True)
            local = LocalLayerLosses(model, vocab, args.lr, args.lr_readout) if arm == "local_bpu" else None
            gen_s = torch.Generator().manual_seed(100 + seed)      # identical stream for every arm of a seed
            with Timer(f"  stream: {arm}"):
                losses, writes = run_stream(model, arm, segments, args.block, args.batch, gen_s, args.lr, holder, local)
            tiles = tile_modules(model)
            n_dev = sum(t.n_devices for t in tiles)
            wpd = writes[-1] / n_dev; total_s = len(losses) * args.step_seconds
            rec = [recovery_time(losses, k * args.seg) for k in (1, 2)]
            seg_end = [float(np.mean(losses[k * args.seg:(k + 1) * args.seg][-200:])) for k in range(3)]
            seg_mean = [float(np.mean(losses[k * args.seg:(k + 1) * args.seg])) for k in range(3)]
            fr = float(np.mean([t.flush_rate for t in tiles])) if arm != "sgd_float32" else 0.0
            curves[name] = {"arm": arm, "theta_flush": theta, "loss": losses[::max(1, len(losses) // 600)], "loss_full_len": len(losses), "writes_cum": writes[::max(1, len(writes) // 600)],
                            "writes_per_device_total": wpd, "writes_per_device_per_s": wpd / total_s, "percent_of_budget": 100.0 * wpd / total_s / BUDGET,
                            "budget": write_budget_report(tiles, seconds=total_s),
                            "flush_rate": fr, "levels": tiles[0].levels if arm != "sgd_float32" else None,
                            "recovery_steps_after_switch": rec, "segment_end_loss": seg_end, "segment_mean_loss": seg_mean}
            print(f"  {name}: segment-end losses {np.round(seg_end, 3)}, recovery {rec}, total writes/dev {wpd:.3g} ({100 * wpd / 1e6:.2g}% endurance), {wpd / total_s:.3g}/dev/s, flush rate {fr:.2g}", flush=True)
            del model
        runs.append({"seed": seed, "ideal_val_loss": base_loss, "curves": curves})
    summarize_and_save(runs, seeds, thetas, arms, segments, args)


def summarize_and_save(runs, seeds, thetas, arms, segments, args):
    # ---- summary over seeds (one entry per arm@theta)
    names = [k for k in runs[0]["curves"] if all(k in r["curves"] for r in runs)]
    summ = {}
    for arm in names:
        summ[arm] = {k: summarize([r["curves"][arm][k] for r in runs]) for k in ("writes_per_device_per_s", "percent_of_budget", "flush_rate", "writes_per_device_total")}
        summ[arm]["percent_of_endurance"] = summarize([100.0 * r["curves"][arm]["writes_per_device_total"] / 1e6 for r in runs])
        summ[arm]["writes_per_tile_per_s"] = summarize([r["curves"][arm].get("budget", {}).get("writes_per_tile_per_s", 0.0) for r in runs])
        summ[arm]["arm"] = runs[0]["curves"][arm].get("arm", arm); summ[arm]["theta_flush"] = runs[0]["curves"][arm].get("theta_flush", 1.0)
        for k in range(3):
            summ[arm][f"segment_end_loss_{k}"] = summarize([r["curves"][arm]["segment_end_loss"][k] for r in runs])
        summ[arm]["recovery_steps_after_switch"] = [[r["curves"][arm]["recovery_steps_after_switch"][i] for r in runs] for i in (0, 1)]
    arms = names
    if "sgd_float32" not in summ or "sgd_bpu" not in summ:   # diagnostic subset run: save the curves and stop
        out = {"seg": args.seg, "sigma": args.sigma, "seeds": seeds, "arms": arms, "step_seconds": args.step_seconds, "runs": runs, "summary": summ, "diagnostic": True,
               "verdict": {"statement": "diagnostic arms only: " + "; ".join(f"{a} post-shift {summ[a]['segment_end_loss_1']['mean']:.3f}, {summ[a]['percent_of_budget']['mean']:.0f}% of budget" for a in arms)}}
        print("E9 (diagnostic):", out["verdict"]["statement"]); save_results("e9", out, args.out); return
    ref = summ["sgd_float32"]["segment_end_loss_1"]["mean"]
    gaps = {arm: 100.0 * (summ[arm]["segment_end_loss_1"]["mean"] / ref - 1) for arm in arms}
    gap_oracle = {arm: 100.0 * (summ[arm]["segment_end_loss_1"]["mean"] / summ["sgd_bpu"]["segment_end_loss_1"]["mean"] - 1) for arm in arms if arm != "sgd_float32"}
    within5 = abs(gaps.get("sgd_bpu", float("inf"))) <= 5.0
    budget_ok = summ["sgd_bpu"]["writes_per_device_per_s"]["mean"] <= BUDGET if "sgd_bpu" in summ else False
    # endurance criterion: within 5 % of float32 at ANY theta with < 1e4 total writes per device
    pareto = [{"name": a, "arm": summ[a]["arm"], "theta_flush": summ[a]["theta_flush"], "post_shift_loss": summ[a]["segment_end_loss_1"]["mean"],
               "gap_pct": gaps[a], "writes_per_device_total": summ[a]["writes_per_device_total"]["mean"]} for a in arms if a not in ("frozen", "sgd_float32")]
    passing = [p for p in pareto if abs(p["gap_pct"]) <= 5.0 and p["writes_per_device_total"] < 1e4]
    success = bool(passing)
    pulse = None
    try:
        pr = load_results("e9_pulse")["summary"]["sgd_bpu_pulse"]
        pulse = {"post_shift_loss": pr["segment_end_loss_1"]["mean"], "writes_per_device_total": pr["writes_per_device_total"]["mean"] if "writes_per_device_total" in pr else pr["writes_per_device_per_s"]["mean"] * args.seg * 3 * args.step_seconds,
                 "gap_pct": 100.0 * (pr["segment_end_loss_1"]["mean"] / ref - 1)}
        pulse["passes"] = bool(abs(pulse["gap_pct"]) <= 5.0 and pulse["writes_per_device_total"] < 1e4)
    except Exception:
        pass
    # loss vs locality: one row per BPU gradient arm (theta = 1 / nearest-pulse), ordered from global backprop to layer-local
    locality = [{"arm": a, "locality": LOCALITY[a], "label": LOCALITY_LABEL[LOCALITY[a]], "post_shift_loss": summ[a]["segment_end_loss_1"]["mean"], "sem": summ[a]["segment_end_loss_1"]["sem"],
                 "gap_pct_vs_float32": gaps[a], "gap_pct_vs_bpu_oracle": gap_oracle.get(a), "writes_per_device_total": summ[a]["writes_per_device_total"]["mean"]}
                for a in ("sgd_bpu", "fa_bpu", "dfa_bpu", "local_bpu") if a in summ]
    out = {"pulse_diagnostic": pulse, "seg": args.seg, "sigma": args.sigma, "pretrain_steps": args.pretrain_steps, "batch": args.batch, "step_seconds": args.step_seconds, "lr": args.lr,
           "write_mode": getattr(args, "write_mode", "level"), "locality": locality,
           "seeds": seeds, "device": args.device, "arms": arms, "segments": [s["name"] for s in segments], "runs": runs, "summary": summ,
           "post_shift_gap_pct_vs_float32": gaps, "post_shift_gap_pct_vs_bpu_oracle": gap_oracle, "budget_writes_per_device_per_s": BUDGET, "smoke": args.smoke,
           "thetas": thetas, "pareto": pareto, "endurance_limit_writes_per_device": 1e4,
           "verdict": {"success": success, "sgd_bpu_within_5pct": bool(within5), "sgd_bpu_within_lifelong_rate": bool(budget_ok), "passing": passing,
                       "statement": ("SMOKE RUN. " if args.smoke else "") +
                       f"post-shift loss (Python segment, last 200 steps, {seeds} seeds): float32 SGD {ref:.3f}; "
                       + "; ".join(f"{p['name']} {p['post_shift_loss']:.3f} ({p['gap_pct']:+.1f}%, {p['writes_per_device_total']:.3g} writes/device = {100 * p['writes_per_device_total'] / 1e6:.2g}% endurance)" for p in pareto)
                       + (". SUCCESS: " + ", ".join(p["name"] for p in passing) + " within 5% of float32 with < 1e4 total writes/device" if success else
                          f". Criterion not met by any {'level-gated' if getattr(args, 'write_mode', 'level') == 'level' else getattr(args, 'write_mode', 'level')} arm: none is within 5% of float32 with < 1e4 total writes per device"
                          + (f" (closest: {min(pareto, key=lambda p: abs(p['gap_pct']))['name']} at {min(pareto, key=lambda p: abs(p['gap_pct']))['gap_pct']:+.1f}%)" if pareto else ""))
                       + (f". Nearest-pulse diagnostic (no level gate): {pulse['post_shift_loss']:.3f} ({pulse['gap_pct']:+.1f}%) at {pulse['writes_per_device_total']:.3g} writes/device = {100 * pulse['writes_per_device_total'] / 1e6:.2g}% endurance"
                          + (": it MEETS both numbers of the criterion, so under the endurance budget the level gate is a cost without a benefit" if pulse.get("passes") else "") if (pulse and getattr(args, "write_mode", "level") == "level") else "")
                       + (". Loss vs locality (write rule " + getattr(args, "write_mode", "level") + "): " + "; ".join(f"{l['arm']} [{l['locality']}] {l['post_shift_loss']:.3f} ({l['gap_pct_vs_float32']:+.1f}%)" for l in locality) if locality else "")}}
    print("E9 verdict:", out["verdict"]["statement"])
    save_results("e9", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e9_trainability"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    seg = res["seg"]; arms = [a for a in res["arms"] if "@" not in a]
    labels = {"frozen": "frozen (BPU, no updates)", "sgd_float32": "SGD float32 (reference)", "sgd_bpu": "SGD on BPU tiles (substrate ceiling)",
              "fa_bpu": "feedback alignment (BPU)", "dfa_bpu": "direct FA, fully local (BPU)", "local_bpu": "layer-local three-factor (BPU)"}
    for arm in arms:
        L = np.mean([r["curves"][arm]["loss"] for r in res["runs"]], 0); x = np.linspace(0, res["runs"][0]["curves"][arm]["loss_full_len"], len(L))
        k = max(1, len(L) // 60); sm = np.convolve(L, np.ones(k) / k, mode="valid")
        ax[0].plot(x[:len(sm)], sm, label=labels.get(arm, arm), ls="--" if arm == "sgd_float32" else "-", color="k" if arm == "sgd_float32" else None)
    for k in (1, 2):
        ax[0].axvline(k * seg, color="k", ls=":", lw=0.8)
    ax[0].set_xlabel("stream step"); ax[0].set_ylabel("loss (nats/char)"); ax[0].legend(fontsize=7)
    ax[0].set_title(f"E9: Shakespeare -> Python -> Shakespeare ({res['seeds']} seeds)", fontsize=9)
    s = res["summary"]
    x = np.arange(len(arms))
    m = [s[a]["segment_end_loss_1"]["mean"] for a in arms]; e = [s[a]["segment_end_loss_1"]["sem"] for a in arms]
    ax[1].bar(x, m, yerr=e, capsize=3, color=["gray" if a == "frozen" else "k" if a == "sgd_float32" else "C0" if a == "sgd_bpu" else "C1" for a in arms])
    ref = s["sgd_float32"]["segment_end_loss_1"]["mean"] if "sgd_float32" in s else None   # the e9_pulse diagnostic run has no float32 arm
    if ref is not None:
        ax[1].axhspan(ref * 0.95, ref * 1.05, color="k", alpha=0.08, label="float32 +-5 %")
    ax[1].set_xticks(x); ax[1].set_xticklabels(arms, rotation=20, fontsize=7); ax[1].set_ylabel("post-shift loss (Python, last 200 steps)"); ax[1].legend(fontsize=7)
    ax[1].set_title("post-shift loss per arm", fontsize=9)
    if res.get("write_mode", "level") != "level" and res.get("locality"):
        L = res["locality"]
        ax[2].errorbar([l["locality"] for l in L], [l["post_shift_loss"] for l in L], [l["sem"] for l in L], marker="o", capsize=3, color="C0", label="BPU tiles")
        if ref is not None:
            ax[2].axhspan(ref * 0.95, ref * 1.05, color="k", alpha=0.08); ax[2].axhline(ref, color="k", ls="--", lw=0.8, label="float32 backprop ± 5 %")
        ax[2].set_xticks([l["locality"] for l in L]); ax[2].set_xticklabels([l["arm"].replace("_bpu", "") for l in L])
        ax[2].set_xlabel("locality: 0 backprop → 3 layer-local"); ax[2].set_ylabel("post-shift loss"); ax[2].legend(fontsize=7)
        ax[2].set_title(f"loss vs locality ({res['write_mode']} writes)", fontsize=9)
        fig.suptitle(res["verdict"]["statement"][:150], fontsize=7)
        fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
        print(f"[e9] figure {path}"); return
    # Pareto: post-shift loss vs total writes per device, one marker per arm, one point per theta
    cols = {"sgd_bpu": "C0", "fa_bpu": "C1", "dfa_bpu": "C2", "local_bpu": "C3", "sgd_bpu_pulse": "C4"}
    for p in res.get("pareto", []):
        ax[2].scatter(max(p["writes_per_device_total"], 1e-3), p["post_shift_loss"], color=cols.get(p["arm"], "k"), s=30 + 40 * p["theta_flush"], label=f"{p['arm']} θ={p['theta_flush']:g}")
    pulse = res.get("pulse_diagnostic")
    if pulse:
        ax[2].scatter(pulse["writes_per_device_total"], pulse["post_shift_loss"], marker="x", color="k", s=50, label="nearest-pulse writes (diagnostic)")
    if ref is not None:
        ax[2].axhspan(ref * 0.95, ref * 1.05, color="k", alpha=0.08); ax[2].axhline(ref, color="k", ls="--", lw=0.8, label="float32 ± 5 %")
    ax[2].axvline(res.get("endurance_limit_writes_per_device", 1e4), color="r", ls=":", label="1e4 writes/device")
    ax[2].set_xscale("log"); ax[2].set_xlabel("total writes per device over the stream"); ax[2].set_ylabel("post-shift loss"); ax[2].legend(fontsize=6)
    ax[2].set_title("Pareto: loss vs writes (marker size = θ_flush)", fontsize=9)
    fig.suptitle(res["verdict"]["statement"][:150], fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e9] figure {path}")


if __name__ == "__main__":
    main()
