"""E13: stochastic resonance of a LIF row driven through a tile (variability programme).

A row of N = 64 leaky integrate-and-fire neurons (tau_m = 20 ms, threshold 1, dt = 1 ms, the
``neurons.LIF`` model) is driven by a SUBTHRESHOLD periodic current: a sinusoid at f = 5 Hz
whose peak is 0.8x the threshold current, I(t) = 0.4 + 0.4 sin(2 pi f t) (0.8 at the crest),
delivered through one memristor per neuron on a 1 x 64 tile (weight 0.8, input x(t) = I(t)/0.8).
The tile's read noise sigma_r (the MODULATE dial) is swept {0, 0.005, ..., 0.2} (41 levels):
every read of the device multiplies the current by (1 + sigma_r * N(0, 1)), fresh each ms.
Reference: the same LIF row with the same multiplicative Gaussian noise injected on the current
(FloatTile).  The tiles carry the device model (d2d 0.03, c2c 0.03, write-verified programming
of the 0.8 weight); nothing else differs.

Metric: SNR of the output spike train at the drive frequency = power of the population spike
train at f divided by the mean power in the 2-Hz band around f (the drive bin excluded), in dB,
from a 20 s record (100 cycles); per-neuron SNR is reported as well.  10 seeds.

Hypothesis: the SNR vs sigma_r is a U-curve (inverted) with its peak in [0.02, 0.06].  Verdict
rule: an interior maximum that beats both ends of the sweep by >= 3 dB with a paired sign test
over seeds (p < 0.05); sigma* with a bootstrap CI over seeds.

Physics check: the LIF membrane low-passes the per-ms read noise by sqrt(dt / 2 tau_m) ~ 0.16,
so the membrane noise at the crest is sigma_r x 0.8 x 0.16 (0.025 at sigma_r = 0.2) against a
gap of 0.2 to threshold.  The main arm may therefore show no resonance in the dial's range; a
diagnostic grid over the drive amplitude {0.8, 0.9, 0.95} x tau_m {20, 5, 2} ms maps where the
peak falls into the band, which is the physical anchor of the "noise band" claim.
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from common import base_parser, save_results, figure_path, Timer, sign_test
from bpusim.device import DeviceParams
from bpusim.tile import Tile, FloatTile
from bpusim.neurons import LIF
from bpusim.noise import summarize

SIGMAS = [round(0.005 * k, 3) for k in range(0, 41)]
BEYOND_DIAL = [0.3, 0.5, 0.7, 1.0]        # diagnostic levels above the dial's range (0.2), to locate the descending branch
N = 64
F_DRIVE = 5.0
T_REC = 20.0
DT = 1e-3


def snr_db(train, f0, dt, band=2.0, exclude=0.15):
    """Power at f0 over the mean power in [f0 - band, f0 + band] excluding +-exclude Hz around f0."""
    train = np.asarray(train, float)
    if train.sum() == 0:
        return float("nan")
    x = train - train.mean()
    P = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(len(x), dt)
    i0 = int(np.argmin(np.abs(fr - f0)))
    peak = P[max(i0 - 1, 0):i0 + 2].max()
    bg = P[(np.abs(fr - f0) <= band) & (np.abs(fr - f0) > exclude)]
    return float(10 * np.log10(peak / max(bg.mean(), 1e-30)))


def run_one(job):
    sigma, seed, substrate, amp, tau_m = job
    rng = np.random.default_rng([seed, int(sigma * 1e4), substrate == "tiles", int(amp * 100), int(tau_m * 1e4)])
    if substrate == "tiles":
        tile = Tile(1, N, rng, DeviceParams(sigma_r=0.02))     # d2d 0.03, c2c 0.03 defaults; programmed at the nominal read noise
        tile.program(np.full((1, N), 0.8), tol=0.005)
    else:
        tile = FloatTile(1, N, rng); tile.program(np.full((1, N), 0.8))
    if sigma <= tile.MODULATE_RANGE[1]:
        tile.set_noise(sigma)
    else:                                                      # beyond the dial: a diagnostic, not a setting the chip offers
        if substrate == "tiles":
            tile.p = tile.p.replace(sigma_r=sigma); tile.gp.p = tile.p; tile.gm.p = tile.p
        else:
            tile.sigma_r = sigma
    lif = LIF(N, rng, tau_m=tau_m, dt=DT, theta=1.0)
    n = int(T_REC / DT); t = np.arange(n) * DT
    I0 = 0.5 * amp; I1 = 0.5 * amp
    x = (I0 + I1 * np.sin(2 * np.pi * F_DRIVE * t)) / 0.8          # device weight 0.8 -> current I(t) at the crest = amp
    spikes = np.zeros((n, N), bool)
    for k in range(n):
        I = tile.read(np.array([x[k]]))
        spikes[k] = lif.step(I)
    pop = spikes.sum(1)
    per = [snr_db(spikes[:, j], F_DRIVE, DT) for j in range(N)]
    per = [v for v in per if np.isfinite(v)]
    return {"sigma": sigma, "seed": seed, "substrate": substrate, "amp": amp, "tau_m": tau_m, "snr_pop_db": snr_db(pop, F_DRIVE, DT),
            "snr_neuron_db": float(np.mean(per)) if per else float("nan"), "frac_spiking": float((spikes.sum(0) > 0).mean()),
            "rate_hz": float(spikes.sum() / N / T_REC), "n_reads": tile.n_reads}


def verdict_for(curve_by_sigma, sigmas, seeds):
    """curve_by_sigma: {sigma: [per-seed SNR]} (NaN = no spikes -> floor at the minimum finite value - 3 dB)."""
    M = np.array([[v if np.isfinite(v) else np.nan for v in curve_by_sigma[s]] for s in sigmas], float)
    floor = np.nanmin(M) - 3.0 if np.isfinite(M).any() else 0.0
    M = np.where(np.isfinite(M), M, floor)
    mean = M.mean(1)
    i = int(np.argmax(mean)); s_star = sigmas[i]
    p0, w0 = sign_test(-M[i], -M[0]); p1, w1 = sign_test(-M[i], -M[-1])      # SNR at the peak larger than at the ends
    beats = bool(0 < i < len(sigmas) - 1 and mean[i] - mean[0] >= 3 and mean[i] - mean[-1] >= 3 and p0 < 0.05 and p1 < 0.05)
    rng = np.random.default_rng(0); boots = [sigmas[int(np.argmax(M[:, rng.integers(0, seeds, seeds)].mean(1)))] for _ in range(2000)]
    return {"sigma_star": s_star, "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "snr_at_star": float(mean[i]),
            "snr_at_zero": float(mean[0]), "snr_at_max": float(mean[-1]), "p_vs_zero": p0, "p_vs_max": p1, "u_curve": beats,
            "peak_in_band": bool(beats and 0.02 <= s_star <= 0.06), "mean_curve": mean.tolist()}


def main():
    ap = base_parser("E13 stochastic resonance through a tile", default_seeds=10)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--amps", type=str, default="0.8,0.9,0.95", help="drive crest as a fraction of threshold (0.8 = the headline arm)")
    ap.add_argument("--taus", type=str, default="0.02,0.005,0.002", help="membrane time constants (20 ms = the headline arm)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = args.seeds; sigmas = SIGMAS + BEYOND_DIAL
    amps = [float(a) for a in args.amps.split(",")]; taus = [float(t) for t in args.taus.split(",")]
    if args.smoke:
        seeds = 2; sigmas = [0.0, 0.05, 0.1, 0.2, 0.5]; amps = [0.8, 0.95]; taus = [0.02, 0.002]
    # headline: amp 0.8, tau 20 ms, both substrates; diagnostic grid: tiles only, the other (amp, tau) cells
    cells = [(a, tau, sub) for a in amps for tau in taus for sub in (("tiles", "float32") if (a == amps[0] and tau == taus[0]) else ("tiles",))]
    jobs = [(s, seed, sub, a, tau) for (a, tau, sub) in cells for s in sigmas for seed in range(seeds)]
    with Timer(f"E13 {len(jobs)} runs"):
        with mp.Pool(args.procs) as pool:
            runs = pool.map(run_one, jobs, chunksize=4)
    out = {"sigmas": sigmas, "seeds": seeds, "N": N, "f_drive": F_DRIVE, "T": T_REC, "cells": {}, "verdicts": {}}
    for (a, tau, sub) in cells:
        key = f"amp{a:g}_tau{tau * 1e3:g}ms_{sub}"
        rs = [r for r in runs if r["substrate"] == sub and r["amp"] == a and r["tau_m"] == tau]
        sweep = {str(s): {"snr_pop_db": summarize([r["snr_pop_db"] for r in rs if r["sigma"] == s]), "snr_neuron_db": summarize([r["snr_neuron_db"] for r in rs if r["sigma"] == s]),
                          "rate_hz": summarize([r["rate_hz"] for r in rs if r["sigma"] == s]), "frac_spiking": float(np.mean([r["frac_spiking"] for r in rs if r["sigma"] == s]))} for s in sigmas}
        out["cells"][key] = sweep
        dial = [s for s in sigmas if s <= 0.2]
        v = verdict_for({s: [r["snr_pop_db"] for r in rs if r["sigma"] == s] for s in dial}, dial, seeds)
        v_all = verdict_for({s: [r["snr_pop_db"] for r in rs if r["sigma"] == s] for s in sigmas}, sigmas, seeds)
        v["beyond_dial"] = {"sigma_star": v_all["sigma_star"], "ci95": v_all["ci95"], "u_curve": v_all["u_curve"], "snr_at_star": v_all["snr_at_star"], "snr_at_max": v_all["snr_at_max"]}
        out["verdicts"][key] = v
        print(f"  {key}: dial sigma* {v['sigma_star']} (CI {v['ci95']}), SNR {v['snr_at_zero']:.1f} -> {v['snr_at_star']:.1f} -> {v['snr_at_max']:.1f} dB, U-curve {v['u_curve']}, in band {v['peak_in_band']}; "
              f"with the beyond-dial levels: sigma* {v['beyond_dial']['sigma_star']} U-curve {v['beyond_dial']['u_curve']}; rate at sigma* {sweep[str(v['sigma_star'])]['rate_hz']['mean']:.2f} Hz", flush=True)
    head = f"amp{amps[0]:g}_tau{taus[0] * 1e3:g}ms_tiles"; ref = f"amp{amps[0]:g}_tau{taus[0] * 1e3:g}ms_float32"
    hv, rv = out["verdicts"][head], out["verdicts"].get(ref)
    in_band = [k for k, v in out["verdicts"].items() if v["peak_in_band"]]
    u_cells = [k for k, v in out["verdicts"].items() if v["u_curve"]]
    out["verdict"] = {"headline_u_curve": hv["u_curve"], "headline_sigma_star": hv["sigma_star"], "headline_peak_in_band": hv["peak_in_band"],
                      "cells_with_u_curve": u_cells, "cells_with_peak_in_band": in_band,
                      "statement": ("SMOKE RUN. " if args.smoke else "") + f"headline (drive {amps[0]:g} x threshold, tau_m {taus[0] * 1e3:g} ms, {seeds} seeds): tiles sigma* = {hv['sigma_star']} (CI {hv['ci95'][0]}..{hv['ci95'][1]}), SNR {hv['snr_at_zero']:.1f} dB at 0 -> {hv['snr_at_star']:.1f} at sigma* -> {hv['snr_at_max']:.1f} at 0.2; "
                                   + (f"float32 reference sigma* = {rv['sigma_star']}, SNR {rv['snr_at_zero']:.1f} -> {rv['snr_at_star']:.1f} -> {rv['snr_at_max']:.1f}; " if rv else "")
                                   + ("U-curve CONFIRMED" if hv["u_curve"] else "no U-curve in the dial's range") + (", peak in [0.02, 0.06]" if hv["peak_in_band"] else ", peak NOT in [0.02, 0.06]")
                                   + f". Diagnostic grid within the dial: U-curve in {len(u_cells)}/{len(out['verdicts'])} cells ({', '.join(u_cells) or 'none'}); peak in band in {len(in_band)} ({', '.join(in_band) or 'none'})"
                                   + ". With the beyond-dial levels (0.3-1.0, diagnostic): " + "; ".join(f"{k}: sigma* {v['beyond_dial']['sigma_star']}{' U' if v['beyond_dial']['u_curve'] else ''}" for k, v in out["verdicts"].items()) + "."}
    print("E13 verdict:", out["verdict"]["statement"])
    save_results("e13", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e13_stochastic_resonance"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = np.array(res["sigmas"], float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for key, sweep in res["cells"].items():
        m = np.array([sweep[str(s)]["snr_pop_db"]["mean"] for s in res["sigmas"]]); e = np.array([sweep[str(s)]["snr_pop_db"]["sem"] for s in res["sigmas"]])
        head = key.startswith(f"amp{res['cells'] and list(res['cells'])[0].split('_')[0][3:]}")
        a = ax[0] if key.endswith("float32") or key == list(res["cells"])[0] else ax[1]
        a.errorbar(S, m, e, marker=".", capsize=2, label=key, ls="--" if key.endswith("float32") else "-", color="k" if key.endswith("float32") else None)
    for a in ax:
        a.axvspan(0.02, 0.06, color="C2", alpha=0.12, label="hypothesised band"); a.axvline(0.2, color="r", ls=":", lw=0.8, label="dial limit"); a.set_xscale("symlog", linthresh=0.02); a.set_xlabel("read noise σ_r"); a.set_ylabel("SNR at 5 Hz (dB)"); a.legend(fontsize=6)
    ax[0].set_title("E13 headline: 0.8× threshold, τ_m 20 ms, tiles vs float32", fontsize=9); ax[1].set_title("diagnostic grid (tiles): drive × τ_m", fontsize=9)
    fig.suptitle(res["verdict"]["statement"][:150], fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e13] figure {path}")


if __name__ == "__main__":
    main()
