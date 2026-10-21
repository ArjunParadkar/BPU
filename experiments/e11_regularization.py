"""E11: read noise and synapse dropout as regularisers (variability programme, 2026-10-19).

The E8 nanoGPT (6 layers, d = 192, 2.7M parameters) is trained FROM SCRATCH on a
small-data slice of tiny Shakespeare (``--train-frac`` of the training text, 10 %
= 100k characters, so the model overfits within the budget and regularisation has
something to do) with

  * read noise sigma_r in {0, 0.02, 0.04, 0.06, 0.1} as the ONLY device noise
    (sigma_d2d = sigma_c2c = 0; the attention cache carries the same read noise),
    set through the MODULATE primitive (``set_noise``), and
  * synapse dropout p in {0, 0.1, 0.3, 0.5} at sigma = 0 (``set_dropout``: every
    read drops each synapse independently with probability p),

on two substrates that run the identical algorithm:
  tiles     BPU devices (nearest-pulse writes, layer-accumulate 8-bit read-out);
  float32   float weights with the same multiplicative read noise / dropout injected.

Evaluation per run: (a) in-distribution validation loss (held-out Shakespeare),
(b) distribution shift: loss on Python source (this repository) that the model
never saw; both read with the training noise ON (as deployed) and OFF (clean
read).  Shift gap = loss_python - loss_val.

Hypotheses (pre-registered): the validation loss has its optimum at sigma_r > 0
(sigma* with a bootstrap CI over seeds); the shift gap narrows with sigma_r
(slope of gap vs sigma with a bootstrap CI and a per-seed sign test).  5 seeds.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from common import base_parser, save_results, figure_path, Timer, load_results
from bpusim.torchtile import tile_modules, write_budget_report
from bpusim.noise import summarize
from e8_nanogpt_tiles import GPT, get_batch, train, evaluate, load_text
from e9_trainability import load_corpora

SIGMAS = [0.0, 0.02, 0.04, 0.06, 0.1, 0.15, 0.2]   # the MODULATE dial (top 0.2); the first sweep stopped at 0.1 and put sigma* on its edge
SIGMAS_BEYOND = [0.3]                              # beyond-dial diagnostic, excluded from the sigma* / slope tests
DROPOUTS = [0.1, 0.3, 0.5]        # p = 0 is the sigma = 0 arm


def set_all_noise(model, sigma, dropout):
    for t in tile_modules(model):
        t.set_noise(sigma); t.set_dropout(dropout)
    for b in model.blocks:
        b.attn_sigma_r = sigma


def arm_list():
    arms = [("sigma", s) for s in SIGMAS + SIGMAS_BEYOND] + [("dropout", p) for p in DROPOUTS]
    return arms


def run_one(kind, value, substrate, seed, vocab, train_ids, val_ids, code_ids, args):
    sigma = value if kind == "sigma" else 0.0
    p_drop = value if kind == "dropout" else 0.0
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    tile_kw = dict(split=(0.0, 0.0, 1.0), write_mode="float" if substrate == "float32" else "nearest_pulse")
    # sigma = 0 on tiles still needs a device model: TileLinear treats sigma=0 as devices without noise
    model = GPT(vocab, 192, 6, 6, args.block, sigma, None, seed=seed, tile_kw=tile_kw).to(args.device)
    set_all_noise(model, sigma, p_drop)
    t0 = time.time()
    losses = train(model, train_ids, args.steps, args.block, args.batch, args.lr, gen, tiled=True, log_every=500, warmup=100, label=f"{substrate} {kind}={value} s{seed}")
    wall = time.time() - t0
    ev = lambda data: evaluate(model, data, args.block, args.batch, args.eval_batches, torch.Generator().manual_seed(99))
    val_noisy, py_noisy = ev(val_ids), ev(code_ids)
    set_all_noise(model, 0.0, 0.0)
    val_clean, py_clean = ev(val_ids), ev(code_ids)
    tiles = tile_modules(model)
    rep = write_budget_report(tiles, seconds=args.steps * args.step_seconds)
    out = {"kind": kind, "value": value, "sigma": sigma, "dropout": p_drop, "substrate": substrate, "seed": seed,
           "train_loss_last100": float(np.mean(losses[-100:])), "train_curve": losses[::max(1, len(losses) // 150)],
           "val_noisy": val_noisy, "val_clean": val_clean, "python_noisy": py_noisy, "python_clean": py_clean,
           "gap_noisy": py_noisy - val_noisy, "gap_clean": py_clean - val_clean,
           "writes_per_device_total": rep.get("writes_per_device_total", 0.0), "n_modulate": rep.get("n_modulate", 0), "wall_s": wall}
    print(f"  {substrate} {kind}={value} seed {seed}: train {out['train_loss_last100']:.3f} val {val_noisy:.3f} (clean {val_clean:.3f}) python {py_noisy:.3f} gap {out['gap_noisy']:.3f} | {wall:.0f}s", flush=True)
    del model
    return out


def bootstrap_argmin(curves, xs, n_boot=2000, seed=0):
    """curves: (seeds, len(xs)); sigma* of the seed-mean curve and its bootstrap distribution over seeds."""
    rng = np.random.default_rng(seed)
    C = np.asarray(curves, float); n = C.shape[0]
    star = xs[int(np.argmin(C.mean(0)))]
    boots = [xs[int(np.argmin(C[rng.integers(0, n, n)].mean(0)))] for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    dist = {str(x): float(np.mean(np.asarray(boots) == x)) for x in xs}
    return {"star": float(star), "ci95": [float(lo), float(hi)], "boot_distribution": dist, "per_seed_argmin": [float(xs[int(np.argmin(c))]) for c in C]}


def bootstrap_slope(curves, xs, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    C = np.asarray(curves, float); n = C.shape[0]; x = np.asarray(xs, float)
    fit = lambda y: float(np.polyfit(x, y, 1)[0])
    slope = fit(C.mean(0))
    boots = [fit(C[rng.integers(0, n, n)].mean(0)) for _ in range(n_boot)]
    per_seed = [fit(c) for c in C]
    return {"slope_per_unit_sigma": slope, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "per_seed_slopes": per_seed, "seeds_negative": int(sum(s < 0 for s in per_seed)),
            "gap_at_zero": float(C.mean(0)[0]), "gap_at_max": float(C.mean(0)[-1])}


def summarize_runs(runs, seeds, args):
    S = {}
    for sub in ("float32", "tiles"):
        S[sub] = {}
        for kind, value in arm_list():
            rs = [r for r in runs if r["substrate"] == sub and r["kind"] == kind and r["value"] == value]
            if not rs:
                continue
            S[sub][f"{kind}={value:g}"] = {k: summarize([r[k] for r in rs]) for k in ("val_noisy", "val_clean", "python_noisy", "python_clean", "gap_noisy", "gap_clean", "train_loss_last100", "writes_per_device_total")}
    hyp = {}
    for sub in ("float32", "tiles"):
        H = {}
        ok = all(all(any(r["substrate"] == sub and r["kind"] == "sigma" and r["value"] == s and r["seed"] == sd for r in runs) for s in SIGMAS) for sd in range(seeds))
        if not ok:
            hyp[sub] = {"complete": False}; continue
        get = lambda key: [[next(r[key] for r in runs if r["substrate"] == sub and r["kind"] == "sigma" and r["value"] == s and r["seed"] == sd) for s in SIGMAS] for sd in range(seeds)]
        for key in ("val_noisy", "val_clean"):
            H[f"sigma_star_{key}"] = bootstrap_argmin(get(key), SIGMAS)
        for key in ("gap_noisy", "gap_clean"):
            H[f"shift_gap_{key}"] = bootstrap_slope(get(key), SIGMAS)
        dp = all(any(r["substrate"] == sub and r["kind"] == "dropout" and r["value"] == p and r["seed"] == sd for r in runs) for p in DROPOUTS for sd in range(seeds))
        if dp:
            P = [0.0] + DROPOUTS
            getp = lambda key: [[next(r[key] for r in runs if r["substrate"] == sub and r["seed"] == sd and ((r["kind"] == "sigma" and r["value"] == 0.0) if p == 0.0 else (r["kind"] == "dropout" and r["value"] == p))) for p in P] for sd in range(seeds)]
            H["p_star_val_clean"] = bootstrap_argmin(getp("val_clean"), P)
            H["p_star_val_noisy"] = bootstrap_argmin(getp("val_noisy"), P)
        H["complete"] = True
        hyp[sub] = H
    # verdict
    parts = []
    for sub in ("tiles", "float32"):
        H = hyp.get(sub, {})
        if not H.get("complete"):
            parts.append(f"{sub}: incomplete"); continue
        a = H["sigma_star_val_clean"]; b = H["sigma_star_val_noisy"]; g = H["shift_gap_gap_clean"]
        parts.append(f"{sub}: sigma* = {a['star']:g} (clean read; 95% CI {a['ci95'][0]:g}..{a['ci95'][1]:g}, per-seed {a['per_seed_argmin']}), {b['star']:g} as deployed (CI {b['ci95'][0]:g}..{b['ci95'][1]:g}); "
                     f"shift gap slope {g['slope_per_unit_sigma']:+.2f} nats per unit sigma (CI {g['ci95'][0]:+.2f}..{g['ci95'][1]:+.2f}), gap {g['gap_at_zero']:.3f} -> {g['gap_at_max']:.3f}, {g['seeds_negative']}/{seeds} seeds narrowing"
                     + (f"; dropout p* = {H['p_star_val_clean']['star']:g} (clean, CI {H['p_star_val_clean']['ci95'][0]:g}..{H['p_star_val_clean']['ci95'][1]:g})" if "p_star_val_clean" in H else "")
                     + "".join(f"; beyond the dial sigma = {sb:g}: val {S[sub][f'sigma={sb:g}']['val_clean']['mean']:.3f} clean / {S[sub][f'sigma={sb:g}']['val_noisy']['mean']:.3f} deployed" for sb in SIGMAS_BEYOND if f"sigma={sb:g}" in S[sub]))
    t = hyp.get("tiles", {})
    met_sigma = bool(t.get("complete") and t["sigma_star_val_clean"]["star"] > 0 and t["sigma_star_val_clean"]["ci95"][0] > 0)
    met_gap = bool(t.get("complete") and t["shift_gap_gap_clean"]["ci95"][1] < 0)
    twin = {}
    if S["tiles"] and S["float32"]:
        for k in S["tiles"]:
            if k in S["float32"]:
                twin[k] = {"val_clean_gap_pct": 100 * (S["tiles"][k]["val_clean"]["mean"] / S["float32"][k]["val_clean"]["mean"] - 1),
                           "val_noisy_gap_pct": 100 * (S["tiles"][k]["val_noisy"]["mean"] / S["float32"][k]["val_noisy"]["mean"] - 1)}
    out = {"sigmas": SIGMAS, "sigmas_beyond": SIGMAS_BEYOND, "dropouts": DROPOUTS, "seeds": seeds, "steps": args.steps, "batch": args.batch, "block": args.block, "lr": args.lr,
           "train_frac": args.train_frac, "eval_batches": args.eval_batches, "device": args.device, "runs": runs, "summary": S, "hypotheses": hyp, "twin_gap": twin,
           "verdict": {"sigma_star_positive_with_ci": met_sigma, "shift_gap_narrows_with_ci": met_gap,
                       "statement": ("SMOKE RUN. " if args.smoke else "") + f"{args.steps} steps on {100 * args.train_frac:.0f}% of Shakespeare, {seeds} seeds. " + " || ".join(parts)
                                    + f". Hypothesis 'optimal val loss at sigma > 0' on tiles: {'SUPPORTED' if met_sigma else 'NOT supported'} (CI excludes 0: {met_sigma}); "
                                    + f"'shift gap narrows with sigma': {'SUPPORTED' if met_gap else 'NOT supported'} (slope CI below 0: {met_gap})."}}
    return out


def main():
    ap = base_parser("E11 read noise / dropout as regularisers", default_seeds=5)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train-frac", type=float, default=0.1, help="fraction of the Shakespeare training split used (small-data regime)")
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--step-seconds", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--substrates", type=str, default="tiles,float32")
    ap.add_argument("--resummarize", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sigmas", type=str, default=None, help="comma list overriding SIGMAS (the tested grid; sigma* and the slope use it)")
    ap.add_argument("--sigmas-beyond", type=str, default=None, help="comma list overriding SIGMAS_BEYOND (diagnostic only); '' for none")
    args = ap.parse_args()
    global SIGMAS, SIGMAS_BEYOND
    if args.sigmas is not None:
        SIGMAS = [float(x) for x in args.sigmas.split(",")]
    if args.sigmas_beyond is not None:
        SIGMAS_BEYOND = [float(x) for x in args.sigmas_beyond.split(",") if x.strip()]
    torch.set_num_threads(args.threads)
    seeds = args.seeds
    if args.smoke:
        args.steps, args.eval_batches, seeds = 40, 2, 1
    if args.resummarize:
        prev = load_results("e11", args.out)
        for k in ("steps", "batch", "block", "lr", "train_frac", "eval_batches"):
            setattr(args, k, prev.get(k, getattr(args, k)))
        out = summarize_runs(prev["runs"], prev["seeds"], args); print("E11 verdict:", out["verdict"]["statement"])
        save_results("e11", out, args.out)
        if not args.no_fig:
            plot(out, figure_path("e11_regularization"))
        return
    shakes, code, vocab = load_corpora()          # shared vocabulary over Shakespeare + Python
    n = int(0.9 * len(shakes)); train_ids = shakes[:int(n * args.train_frac)]; val_ids = shakes[n:]
    print(f"vocab {vocab}, train {len(train_ids)} chars ({100 * args.train_frac:.0f}% of the split), val {len(val_ids)}, python {len(code)}", flush=True)
    runs = []
    try:
        runs = load_results("e11", args.out)["runs"]; print(f"resuming with {len(runs)} runs on disk", flush=True)
    except Exception:
        pass
    done = {(r["substrate"], r["kind"], r["value"], r["seed"]) for r in runs}
    for seed in range(seeds):
        for kind, value in arm_list():
            for sub in args.substrates.split(","):
                if (sub, kind, value, seed) in done:
                    continue
                runs.append(run_one(kind, value, sub, seed, vocab, train_ids, val_ids, code, args))
                partial = summarize_runs(runs, seeds, args); partial["partial"] = True
                save_results("e11", partial, args.out)
    out = summarize_runs(runs, seeds, args)
    print("E11 verdict:", out["verdict"]["statement"])
    save_results("e11", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e11_regularization"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = res["summary"]; sig = res["sigmas"] + res.get("sigmas_beyond", []); top = max(res["sigmas"])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for sub, col in (("float32", "k"), ("tiles", "C0")):
        for key, ls, lab in (("val_clean", "-", "val, clean read"), ("val_noisy", "--", "val, as deployed")):
            m = [S[sub].get(f"sigma={s:g}", {}).get(key, {}).get("mean", np.nan) for s in sig]; e = [S[sub].get(f"sigma={s:g}", {}).get(key, {}).get("sem", 0) for s in sig]
            ax[0].errorbar(sig, m, e, marker="o", ls=ls, color=col, capsize=3, label=f"{sub}: {lab}")
        m = [S[sub].get(f"sigma={s:g}", {}).get("gap_clean", {}).get("mean", np.nan) for s in sig]; e = [S[sub].get(f"sigma={s:g}", {}).get("gap_clean", {}).get("sem", 0) for s in sig]
        ax[1].errorbar(sig, m, e, marker="o", color=col, capsize=3, label=f"{sub}: python − val (clean)")
        P = [0.0] + res["dropouts"]
        m = [S[sub].get("sigma=0" if p == 0 else f"dropout={p:g}", {}).get("val_clean", {}).get("mean", np.nan) for p in P]
        e = [S[sub].get("sigma=0" if p == 0 else f"dropout={p:g}", {}).get("val_clean", {}).get("sem", 0) for p in P]
        ax[2].errorbar(P, m, e, marker="s", color=col, capsize=3, label=f"{sub}: val, clean read")
    for a in ax[:2]:
        a.axvline(top, color="0.6", ls=":", lw=1); a.text(top, a.get_ylim()[1], " dial top", fontsize=7, color="0.4", va="top")
    ax[0].set_xlabel("read noise σ_r (training)"); ax[0].set_ylabel("loss (nats/char)"); ax[0].legend(fontsize=7); ax[0].set_title("E11: validation loss vs training read noise", fontsize=9)
    ax[1].set_xlabel("read noise σ_r"); ax[1].set_ylabel("shift gap (nats/char)"); ax[1].legend(fontsize=7); ax[1].set_title("distribution shift: Python after Shakespeare", fontsize=9)
    ax[2].set_xlabel("synapse dropout p (σ = 0)"); ax[2].set_ylabel("val loss, clean read"); ax[2].legend(fontsize=7); ax[2].set_title("dropout arm", fontsize=9)
    fig.suptitle(res["verdict"]["statement"][:160], fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e11] figure {path}")


if __name__ == "__main__":
    main()
