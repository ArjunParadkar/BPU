"""E11-cal: noise calibration of E11 (round four, 2026-10-23, "learning vs knowing").

Question.  On both data regimes the float32 reference needed about 10x the injected read noise of the
tiles to reach its validation optimum (small data sigma* 1.0 vs 0.1, full data 0.4 vs 0.04).  Is that
10x in sigma an artefact of the axis: does the tile path (8-bit layer-accumulate converter, attention
read-out) already perturb every layer, so that the two optima coincide once each run is placed at the
perturbation its layers actually saw?

Measurement.  E11 saved no checkpoints, so the sigma = 0 arm is retrained with the E11 code, seeds 0-4,
on both regimes (small: 10 % of Shakespeare, 1500 steps; full: 100 %, 5000 steps) and both substrates
(tiles: nearest-pulse writes, layer-accumulate 8-bit read-out; float32 reference); the retrained val loss
is checked against E11's own sigma = 0 run of the same seed.  On every checkpoint, for every sigma in the
E11 grids of that regime and substrate, ``relpert.measure`` gives every read site's local and cumulative
rel_pert(sigma) = ||y_noisy - y_ideal|| / ||y_ideal|| (y_ideal = exact float product of the same stored
weights, no noise, no converter) over 16 batches of 256 tokens (2 x 128), at the setting E11 actually ran
(``as_run``: the MODULATE dial clips the Linear layers to 0.2, the attention cache follows sigma) and at
the nominal unclipped setting.  Robustness: seed 0 is also trained at each substrate's sigma* and
measured the same way (does the calibration of a model trained at sigma* differ from the sigma = 0 one?).

Analysis (``--analyze``, reads only results.json files).  Each E11 run is placed at x = mean_local
rel_pert of its (regime, substrate, sigma) as run, mean over the 5 calibration seeds.  The val-loss (clean
read, E11's primary) and shift-gap (python - val, clean) curves are replotted against x.  Optimum in
rel_pert: bootstrap (2000) over E11 seeds -> sigma*_b, over calibration seeds -> x(sigma*_b); point = x at
the E11 sigma*, CI = 2.5/97.5 percentiles.  Ratio R = x*_float32 / x*_tiles per regime, bootstrap CI from
independent resamples.

PRE-REGISTERED (written 2026-10-23 before any run):
  primary scalar   mean_local rel_pert as run (unweighted mean over the 31 read sites: per block qkv,
                   attention, proj, fc, fc2, plus the head).
  H (lines up)     in BOTH regimes, R's 95 % CI contains 1 AND 0.5 <= R <= 2.  Then the 10x in sigma is
                   an axis artefact: the tile path adds perturbation (rel_pert(0) on tiles, reported with
                   its CI, is the tile's intrinsic perturbation) and/or the differential pair reads noisier
                   than |W| (sqrt(w+^2 + w-^2) > |w+ - w-|).
  NOT lined up     otherwise; R and its CI are reported per regime and the 10x is not an axis artefact
                   of the forward perturbation (write quantisation, the other tile difference, is not
                   measured here and is named as the remaining candidate).
  Either outcome is a result; the verdict states which happened.
Secondary (reported, no threshold): cumulative and logits rel_pert; linear vs attention means; the
sigma*-trained calibration vs the sigma = 0 calibration; the same analysis with the unclipped
float32 curves (results/e11_f32_unclipped, results/e11_full_f32_unclipped) when they exist.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from common import ROOT, save_results, figure_path, load_results
import e11_regularization as e11
from e9_trainability import load_corpora
from relpert import measure, set_noise_as_run, set_noise_unclipped, batches_from

REGIMES = {"small": dict(train_frac=0.1, steps=1500), "full": dict(train_frac=1.0, steps=5000)}
# E11 results directories per (regime, substrate); later entries win on overlapping sigma
CURVES = {("small", "tiles"): ["results/e11"], ("small", "float32"): ["results/e11", "results/e11_f32_bracket"],
          ("full", "tiles"): ["results/e11_full_tiles"], ("full", "float32"): ["results/e11_full_f32"]}
CURVES_UNCLIPPED = {("small", "float32"): ["results/e11_f32_unclipped"], ("full", "float32"): ["results/e11_full_f32_unclipped"]}
OUT = "results/e11cal"


def load_json(rel):
    p = ROOT / rel / "results.json"
    return json.load(open(p)) if p.exists() else None


def curve_runs(dirs, substrate):
    """sigma -> seed -> run (sigma arms only, later dirs override)."""
    C = {}
    for d in dirs:
        r = load_json(d)
        if r is None:
            continue
        for x in r["runs"]:
            if x["substrate"] == substrate and x["kind"] == "sigma":
                C.setdefault(float(x["value"]), {})[int(x["seed"])] = x
    return C


def grid(regime, substrate):
    s = set()
    for d in CURVES[(regime, substrate)] + CURVES_UNCLIPPED.get((regime, substrate), []):
        s |= set(curve_runs([d], substrate).keys())
    if not s:   # smoke / missing results: the E11 dial grid
        s = set(e11.SIGMAS)
    return sorted(s)


# ------------------------------------------------------------------ measurement
def measure_part(regime, substrate, seed, train_sigma, args, data):
    key = f"{regime}_{substrate}_s{seed}_train{train_sigma:g}"
    part = ROOT / OUT / "parts" / f"{key}.json"
    if part.exists() and not args.smoke:
        print(f"  {key}: on disk", flush=True); return
    shakes, code, vocab = data
    R = REGIMES[regime]
    a = argparse.Namespace(steps=R["steps"], batch=32, block=128, lr=1e-3, train_frac=R["train_frac"], eval_batches=40, step_seconds=0.01,
                           device=args.device, smoke=args.smoke, legacy_dial_clip=True)   # retrain exactly as E11 ran
    if args.smoke:
        a.steps, a.eval_batches = 20, 2
    n = int(0.9 * len(shakes)); train_ids = shakes[:int(n * a.train_frac)]; val_ids = shakes[n:]
    t0 = time.time()
    run, model = e11.run_one("sigma", train_sigma, substrate, seed, vocab, train_ids, val_ids, code, a, keep_model=True)
    batches = batches_from(val_ids, 2 if args.smoke else args.meas_batches, args.tokens_per_batch, 128, args.device)
    sig = grid(regime, substrate)
    rows = {"as_run": [], "unclipped": []}
    for s in sig:
        rows["as_run"].append(measure(model, batches, s, set_noise_as_run, cumulative=True))
        rows["unclipped"].append(measure(model, batches, s, set_noise_unclipped, cumulative=True))
        print(f"    {key} sigma={s:g}: mean_local as run {rows['as_run'][-1]['mean_local']:.4f} unclipped {rows['unclipped'][-1]['mean_local']:.4f}"
              f" logits {rows['as_run'][-1]['logits_cumulative']:.4f}", flush=True)
    ref = curve_runs(CURVES[(regime, substrate)][:1], substrate).get(float(train_sigma), {}).get(seed)
    rec = {"regime": regime, "substrate": substrate, "seed": seed, "train_sigma": train_sigma, "sigmas": sig, "rows": rows,
           "retrained_val_clean": run["val_clean"], "e11_val_clean_same_seed": ref["val_clean"] if ref else None,
           "train_run": {k: v for k, v in run.items() if k != "train_curve"}, "wall_s": time.time() - t0}
    part.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(part, "w"), indent=1)
    print(f"  {key}: done in {time.time() - t0:.0f}s (retrained val {run['val_clean']:.3f}, E11 {ref['val_clean'] if ref else float('nan'):.3f})", flush=True)
    del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None


# ------------------------------------------------------------------ analysis
def load_parts():
    d = ROOT / OUT / "parts"
    return [json.load(open(p)) for p in sorted(d.glob("*.json"))] if d.exists() else []


def calib_table(parts, regime, substrate, setting="as_run", train_sigma=0.0, key="mean_local"):
    """sigma -> array over calibration seeds."""
    T = {}
    for p in parts:
        if p["regime"] == regime and p["substrate"] == substrate and p["train_sigma"] == train_sigma:
            for row in p["rows"][setting]:
                T.setdefault(float(row["sigma"]), []).append(row[key])
    return {s: np.asarray(v, float) for s, v in T.items()}


def optimum_in_relpert(curves, cal, n_boot=2000, seed=0):
    """curves: sigma -> seed -> run.  Returns sigma*, x* (rel_pert at sigma*), bootstrap CI of x*."""
    sig = [s for s in sorted(curves) if s in cal]
    seeds = sorted(set.intersection(*[set(curves[s]) for s in sig]))
    V = np.asarray([[curves[s][sd]["val_clean"] for s in sig] for sd in seeds])
    rng = np.random.default_rng(seed)
    i_star = int(np.argmin(V.mean(0)))
    x_mean = {s: float(cal[s].mean()) for s in sig}
    boots = []
    for _ in range(n_boot):
        vb = V[rng.integers(0, len(seeds), len(seeds))].mean(0); sb = sig[int(np.argmin(vb))]
        c = cal[sb]; boots.append(float(c[rng.integers(0, len(c), len(c))].mean()))
    return {"sigma_star": sig[i_star], "x_star": x_mean[sig[i_star]], "x_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "boots": boots, "n_curve_seeds": len(seeds), "sigmas": sig, "x_of_sigma": x_mean,
            "val_mean": V.mean(0).tolist(), "gap_mean": np.asarray([[curves[s][sd]["gap_clean"] for s in sig] for sd in seeds]).mean(0).tolist()}


def analyze(args):
    parts = load_parts()
    res = {"calibration": {}, "optima": {}, "ratio": {}, "reproduction": [], "robustness": {}, "unclipped_float32": {}}
    for p in parts:
        if p["train_sigma"] == 0.0:
            res["reproduction"].append({"regime": p["regime"], "substrate": p["substrate"], "seed": p["seed"], "retrained": p["retrained_val_clean"], "e11": p["e11_val_clean_same_seed"]})
    lined = {}
    for regime in REGIMES:
        for sub in ("tiles", "float32"):
            for setting in ("as_run", "unclipped"):
                for key in ("mean_local", "mean_local_linear", "mean_local_attn", "logits_cumulative", "mean_cumulative"):
                    cal = calib_table(parts, regime, sub, setting, 0.0, key)
                    if cal:
                        res["calibration"].setdefault(f"{regime}/{sub}/{setting}", {})[key] = {f"{s:g}": {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": len(v)} for s, v in sorted(cal.items())}
            cal = calib_table(parts, regime, sub)
            curves = curve_runs(CURVES[(regime, sub)], sub)
            if cal and curves:
                o = optimum_in_relpert(curves, cal); res["optima"][f"{regime}/{sub}"] = {k: v for k, v in o.items() if k != "boots"}; lined[(regime, sub)] = o
                if 0.0 in cal:
                    v = cal[0.0]; rng = np.random.default_rng(1)
                    b = [float(v[rng.integers(0, len(v), len(v))].mean()) for _ in range(2000)]
                    res["optima"][f"{regime}/{sub}"]["x_at_sigma0"] = {"mean": float(v.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}
            # robustness: calibration of the model trained at sigma*
            for p in parts:
                if p["regime"] == regime and p["substrate"] == sub and p["train_sigma"] != 0.0:
                    c0 = calib_table([q for q in parts if q["seed"] == p["seed"]], regime, sub)
                    res["robustness"][f"{regime}/{sub}/train{p['train_sigma']:g}"] = {f"{r['sigma']:g}": {"trained_at_star": r["mean_local"], "trained_at_0": float(c0[float(r['sigma'])][0]) if float(r['sigma']) in c0 else None} for r in p["rows"]["as_run"]}
            uc = CURVES_UNCLIPPED.get((regime, sub))
            if uc and curve_runs(uc, sub):
                calu = calib_table(parts, regime, sub, "unclipped")
                cu = curve_runs(CURVES[(regime, sub)], sub)
                for s, v in curve_runs(uc, sub).items():
                    if s > 0.2:
                        cu[s] = v
                    else:
                        cu.setdefault(s, v)
                # below the dial top as-run == unclipped, above it the unclipped runs replace the clipped ones
                o = optimum_in_relpert({s: v for s, v in cu.items() if s in calu}, calu)
                res["unclipped_float32"][regime] = {k: v for k, v in o.items() if k != "boots"}
                lined[(regime, "float32_unclipped")] = o
    rng = np.random.default_rng(2)
    verdict_parts = []; ok_all = True; complete = True
    for regime in REGIMES:
        if (regime, "tiles") not in lined or (regime, "float32") not in lined:
            complete = False; continue
        t, f = lined[(regime, "tiles")], lined[(regime, "float32")]
        R = f["x_star"] / t["x_star"] if t["x_star"] > 0 else float("inf")
        rb = np.asarray(f["boots"]) / np.maximum(np.asarray(t["boots"])[rng.permutation(len(t["boots"]))], 1e-12)
        ci = [float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))]
        ok = bool(ci[0] <= 1.0 <= ci[1] and 0.5 <= R <= 2.0); ok_all &= ok
        res["ratio"][regime] = {"R": R, "ci95": ci, "lines_up": ok, "sigma_ratio": f["sigma_star"] / t["sigma_star"] if t["sigma_star"] > 0 else None}
        x0 = res["optima"][f"{regime}/tiles"].get("x_at_sigma0", {})
        verdict_parts.append(f"{regime} data: tiles sigma* {t['sigma_star']:g} -> rel_pert {t['x_star']:.4f} (CI {t['x_ci95'][0]:.4f}..{t['x_ci95'][1]:.4f}); "
                             f"float32 sigma* {f['sigma_star']:g} -> {f['x_star']:.4f} (CI {f['x_ci95'][0]:.4f}..{f['x_ci95'][1]:.4f}); R = {R:.2f} (CI {ci[0]:.2f}..{ci[1]:.2f}) vs {res['ratio'][regime]['sigma_ratio'] or float('nan'):.1f}x in sigma; "
                             f"tile intrinsic rel_pert at sigma 0 = {x0.get('mean', float('nan')):.4f} (CI {x0.get('ci95', [float('nan')] * 2)[0]:.4f}..{x0.get('ci95', [float('nan')] * 2)[1]:.4f})"
                             + ("" if (regime, "float32_unclipped") not in lined else f"; with the unclipped float32 rerun sigma* {lined[(regime, 'float32_unclipped')]['sigma_star']:g} -> {lined[(regime, 'float32_unclipped')]['x_star']:.4f}"))
    hyp = "LINES UP (supported)" if (complete and ok_all) else ("does NOT line up" if complete else "incomplete")
    res["verdict"] = {"lines_up": bool(complete and ok_all), "complete": complete,
                      "statement": ("SMOKE RUN. " if args.smoke else "") + f"Optima in rel_pert space (mean local rel_pert as run, 5 calibration seeds): {hyp}. " + " || ".join(verdict_parts)
                                   + ". Note: E11 set every Linear through the MODULATE dial, which clips at 0.2 on the float32 reference too; every arm above 0.2 ran its Linear layers at 0.2 and only the attention cache at sigma (the as-run calibration measures exactly that)."}
    res["preregistered"] = "H lines up: in both regimes R = x*_f32 / x*_tiles has a 95% CI containing 1 and 0.5 <= R <= 2 (primary scalar mean_local rel_pert, as run)."
    print("E11-cal verdict:", res["verdict"]["statement"])
    save_results("e11cal", res, OUT)
    if not args.no_fig:
        plot(res, figure_path("e11cal_relpert"))
    return res


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {("small", "tiles"): "C0", ("full", "tiles"): "C2", ("small", "float32"): "k", ("full", "float32"): "C3"}
    for (regime, sub), col in cols.items():
        c = res["calibration"].get(f"{regime}/{sub}/as_run", {}).get("mean_local")
        if c:
            s = [float(k) for k in c]; m = [c[k]["mean"] for k in c]; e = [c[k]["sd"] for k in c]
            ax[0].errorbar(s, m, e, marker="o", color=col, capsize=2, label=f"{sub}, {regime} data (as run)")
        cu = res["calibration"].get(f"{regime}/{sub}/unclipped", {}).get("mean_local")
        if cu and sub == "float32":
            ax[0].plot([float(k) for k in cu], [cu[k]["mean"] for k in cu], ":", color=col, lw=1, label=f"{sub}, {regime}, unclipped")
        o = res["optima"].get(f"{regime}/{sub}")
        if o:
            xs = list(o["x_of_sigma"].values())
            ax[1].plot(xs, o["val_mean"], "o-", color=col, label=f"{sub}, {regime}")
            ax[1].axvspan(o["x_ci95"][0], o["x_ci95"][1], color=col, alpha=0.12)
            ax[2].plot(xs, o["gap_mean"], "o-", color=col, label=f"{sub}, {regime}")
    for a in ax:
        a.set_xscale("symlog", linthresh=1e-3)
    ax[0].set_xlabel("injected read noise σ"); ax[0].set_ylabel("mean local rel_pert"); ax[0].set_title("calibration: layer perturbation vs σ", fontsize=9); ax[0].legend(fontsize=6)
    ax[1].set_xlabel("measured rel_pert (as run)"); ax[1].set_ylabel("E11 val loss, clean read"); ax[1].set_title("val loss vs rel_pert (band = CI of the optimum)", fontsize=9); ax[1].legend(fontsize=6)
    ax[2].set_xlabel("measured rel_pert (as run)"); ax[2].set_ylabel("shift gap python − val (clean)"); ax[2].set_title("shift gap vs rel_pert", fontsize=9); ax[2].legend(fontsize=6)
    fig.suptitle(res["verdict"]["statement"][:170], fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e11cal] figure {path}")


def main():
    ap = argparse.ArgumentParser(description="E11-cal noise calibration")
    ap.add_argument("--regimes", default="small,full")
    ap.add_argument("--substrates", default="tiles,float32")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--meas-batches", type=int, default=16)
    ap.add_argument("--tokens-per-batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-star", action="store_true", help="skip the sigma*-trained robustness checkpoints")
    ap.add_argument("--analyze", action="store_true", help="only (re)build results.json + figure from the parts on disk")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    global OUT
    if args.smoke:
        OUT = "results/hpcc_smoke/e11cal"; args.seeds = 1; args.regimes = "small"; args.no_star = True
    if not args.analyze:
        data = load_corpora()
        star = {("small", "tiles"): 0.1, ("small", "float32"): 1.0, ("full", "tiles"): 0.04, ("full", "float32"): 0.4}   # the E11 sigma* (as reported)
        for regime in args.regimes.split(","):
            for sub in args.substrates.split(","):
                for seed in range(args.seeds):
                    measure_part(regime, sub, seed, 0.0, args, data)
                if not args.no_star:
                    measure_part(regime, sub, 0, star[(regime, sub)], args, data)
    analyze(args)


if __name__ == "__main__":
    main()
