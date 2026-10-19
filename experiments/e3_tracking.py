"""E3 (moving cue): tracking error of a ring attractor versus dynamic noise.

The static-drift protocol (e3_drift.py: path integration of an OU velocity
with no cue) is replaced by a *moving cue*: a weak external bump of input
(amplitude ``--cue-amp`` x the tonic drive, width 1 rad) rotates at constant
angular velocity omega in {0.1, 0.5, 1.0} rad/s and the ring's own bump has to
follow it.  Tracking error = mean |phi_bump - phi_cue| over T seconds.

Substrate framing: the SAME ring (J0 = -0.5, J1 = 3, N = 128, relu, 16
physical 32x32 tiles per matrix) runs as
  (a) float32 reference: ideal weights, no tiles, no mismatch, dynamic noise
      as additive neuron noise only (there are no tile reads to perturb);
  (b) BPU tiles with fabrication mismatch sigma_d2d in {0.02, 0.05, 0.10}
      (plus a mismatch-free tile control), dynamic noise half multiplicative
      read noise, half additive neuron noise.
Dynamic noise sweep {0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08}
(0 added as the baseline of the sign test), homeostasis off (tau_H = 0),
10 seeds paired by mismatch draw and starting angle.  The pinning landscape
at zero noise (Delta U) is measured for every mismatch level as in e3_drift.
The U test is the one used before: an interior minimum that beats sigma = 0
in a paired sign test (p < 0.05) with the error rising again at the top of
the sweep.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import pathlib

import numpy as np

from common import base_parser, save_results, figure_path, Timer, sign_test
from bpusim.attractor import RingAttractor, circ_dist
from bpusim.device import DeviceParams
from bpusim.noise import summarize
from e3_drift import landscape_one, J0, J1

LEVELS = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08]
OMEGAS = [0.1, 0.5, 1.0]


def cue_input(theta, phi, amp, width=1.0):
    d = circ_dist(theta, phi)
    return amp * np.exp(-d ** 2 / (2 * width ** 2))


_PROGRAMMED = {}   # (N, mismatch, seed) -> programmed device states; one hardware per seed, shared across noise levels


def build(N, mismatch, sigma_dyn, seed, substrate):
    rng_hw = np.random.default_rng([seed, 1, N, int(mismatch * 1e4)])
    if substrate == "float32":
        ring = RingAttractor(N, J0, J1, rng=rng_hw, use_tiles=False, sigma_neuron=sigma_dyn)
    else:
        # the fabricated hardware (d2d gains) and its write-verified programming are a property of the
        # seed, not of the dynamic-noise level: program once (verify at the device's nominal read noise)
        # and reuse the device states for every sigma_dyn.  Write-verify of 2 x 16k devices was 75 % of a
        # run's time and, with sigma_r tied to sigma_dyn, gave every noise level a different hardware.
        key = (N, mismatch, seed)
        if key not in _PROGRAMMED:
            p0 = DeviceParams(sigma_d2d=mismatch, sigma_c2c=0.0, sigma_r=0.02)
            r0 = RingAttractor(N, J0, J1, rng=np.random.default_rng([seed, 1, N, int(mismatch * 1e4)]), params=p0,
                               sigma_mismatch=mismatch, tau_H=0.0)
            _PROGRAMMED[key] = {t: (getattr(r0, t).gp.w.copy(), getattr(r0, t).gm.w.copy(), getattr(r0, t).gp.d2d_gain.copy(), getattr(r0, t).gm.d2d_gain.copy()) for t in ("tile_W", "tile_Wp")}
        params = DeviceParams(sigma_d2d=mismatch, sigma_c2c=0.0, sigma_r=0.5 * sigma_dyn)
        ring = RingAttractor.__new__(RingAttractor)
        _init_ring_from_cache(ring, N, params, rng_hw, sigma_dyn, _PROGRAMMED[key])
    ring.set_dynamic_rng(np.random.default_rng([seed, 3, int(round(sigma_dyn * 1e4))]))
    return ring


def _init_ring_from_cache(ring, N, params, rng, sigma_dyn, cache):
    """RingAttractor.__init__ without the programming step: tiles are created and their device states
    and d2d gains copied from the cached programmed hardware."""
    from bpusim.neurons import RateNeurons
    from bpusim.tile import Tile
    ring.rng = rng
    ring.N, ring.J0, ring.J1, ring.I = int(N), float(J0), float(J1), 1.0
    ring.tau, ring.dt, ring.gamma = 10e-3, 1e-3, 1.0
    ring.tau_H, ring.tau_rate = 0.0, 20.0
    ring.h = np.ones(ring.N); ring.r_bar = None
    ring.theta = 2 * np.pi * np.arange(ring.N) / ring.N
    d = ring.theta[None, :] - ring.theta[:, None]
    ring.W_ideal = J0 + J1 * np.cos(d); ring.Wp_ideal = -J1 * np.sin(d)
    ring.scale = abs(J0) + abs(J1)
    ring.use_tiles = True; ring.p = params
    ring.neurons = RateNeurons(ring.N, rng, tau=ring.tau, dt=ring.dt, f="relu", sigma=0.5 * sigma_dyn)
    for name in ("tile_W", "tile_Wp"):
        t = Tile(ring.N, ring.N, rng, params)
        wp, wm, gp_gain, gm_gain = cache[name]
        t.gp.w[:] = wp; t.gm.w[:] = wm; t.gp.d2d_gain[:] = gp_gain; t.gm.d2d_gain[:] = gm_gain
        setattr(ring, name, t)
    ring.reset()


def run_one(args):
    N, mismatch, sigma_dyn, seed, omega, amp, T, substrate = args
    dt = 1e-3
    ring = build(N, mismatch, sigma_dyn, seed, substrate)
    phi0 = np.random.default_rng([seed, 2]).uniform(-np.pi, np.pi)
    ring.reset(angle=phi0)
    for _ in range(500):
        ring.step(0.0, cue_input(ring.theta, phi0, amp))
    n = int(T / dt)
    err = np.empty(n)
    for t in range(n):
        phi = phi0 + omega * (t + 1) * dt
        ring.step(0.0, cue_input(ring.theta, phi, amp))
        err[t] = circ_dist(ring.decode(), phi)
    r = ring.r
    return {"N": N, "mismatch": mismatch, "sigma": sigma_dyn, "seed": seed, "omega": omega, "amp": amp, "substrate": substrate,
            "track_err": float(np.mean(np.abs(err))), "lag_mean": float(np.mean(err)), "err_rms": float(np.sqrt(np.mean(err ** 2))),
            "bump_alive": bool((r.max() - r.min()) / max(r.mean(), 1e-9) > 0.5), "err_trace": err[::max(1, n // 60)].tolist()}


def verdict_for(sweep, levels, seeds):
    means = [sweep[str(s)]["track_err"]["mean"] for s in levels]
    i_min = int(np.argmin(means))
    r0 = [r["track_err"] for r in sweep[str(levels[0])]["runs"]]; rb = [r["track_err"] for r in sweep[str(levels[i_min])]["runs"]]
    p, wins = sign_test(rb, r0)
    u = bool(0 < i_min < len(levels) - 1 and means[-1] > means[i_min] and p < 0.05)
    return {"sigma_star_empirical": levels[i_min], "err_at_zero": means[0], "err_at_min": means[i_min], "err_at_max": means[-1],
            "sign_test_p": p, "seeds_better_than_zero": wins, "u_shaped": u,
            "statement": (f"U-shaped: minimum at sigma={levels[i_min]} beats sigma=0 in {wins}/{seeds} seeds (p={p:.3g}) and error rises again" if u else
                          "NOT U-shaped: " + ("minimum at zero noise" if i_min == 0 else
                                              f"interior minimum at sigma={levels[i_min]} not significant ({wins}/{seeds} seeds, p={p:.3g})" if p >= 0.05 else "no rise at high noise"))}


def main():
    ap = base_parser("E3 moving-cue tracking vs dynamic noise", default_seeds=10)
    ap.add_argument("--T", type=float, default=15.0, help="seconds of tracking per run (30 s in the first attempt; 15 s fits the laptop)")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--mismatch", type=str, default="0.02,0.05,0.10")
    ap.add_argument("--omegas", type=str, default=",".join(map(str, OMEGAS)))
    ap.add_argument("--cue-amp", type=str, default="0.1,0.05", help="cue amplitudes (fraction of the tonic drive); the first is the headline arm")
    ap.add_argument("--procs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--finalize", action="store_true", help="no simulation: build the verdict, comparison and figure from the (partial) results.json on disk")
    args = ap.parse_args()
    if args.finalize:
        from common import load_results
        out = load_results("e3_tracking", args.out)
        out["cue_amps"] = sorted({float(k.split("_")[0][3:]) for k in out["sweeps"]}, reverse=True) or out["cue_amps"]
        out.pop("partial", None); out["finalized_from_partial"] = True
        finish(out, args); return
    seeds = 50 if args.full else args.seeds
    N = args.N; mismatches = [float(x) for x in args.mismatch.split(",")]
    omegas = [float(x) for x in args.omegas.split(",")]; amps = [float(x) for x in args.cue_amp.split(",")]
    out = {"levels": LEVELS, "seeds": seeds, "N": N, "mismatch": mismatches, "omegas": omegas, "cue_amps": amps, "J0": J0, "J1": J1,
           "T": args.T, "tau_H": 0.0, "sweeps": {}, "landscape": {}, "verdicts": {}}
    pool = mp.Pool(args.procs)
    with Timer("pinning landscapes"):
        for mismatch in mismatches + [0.0]:
            L = pool.map(landscape_one, [(N, mismatch, s, 0.0) for s in range(min(seeds, 10))])
            out["landscape"][f"mismatch{mismatch}"] = {"delta_U": summarize([l["delta_U"] for l in L]), "rms_drift": summarize([l["rms_drift"] for l in L]), "example": L[0]}
            print(f"  mismatch {mismatch}: Delta U = {out['landscape'][f'mismatch{mismatch}']['delta_U']['mean']:.3g}", flush=True)
    arms = [("float32", 0.0)] + [("bpu", m) for m in mismatches + [0.0]]
    for amp in amps:
        for omega in omegas:
            for substrate, mismatch in arms:
                key = f"amp{amp:g}_omega{omega:g}_{substrate}" + (f"_mismatch{mismatch}" if substrate == "bpu" else "")
                with Timer(f"  {key} ({len(LEVELS)} x {seeds})"):
                    jobs = [(N, mismatch, s, seed, omega, amp, args.T, substrate) for s in LEVELS for seed in range(seeds)]
                    res = pool.map(run_one, jobs)
                sweep = {}
                for s in LEVELS:
                    rs = [r for r in res if r["sigma"] == s]
                    sweep[str(s)] = {"track_err": summarize([r["track_err"] for r in rs]), "lag_mean": summarize([r["lag_mean"] for r in rs]),
                                     "bump_alive_frac": float(np.mean([r["bump_alive"] for r in rs])),
                                     "runs": [{k: r[k] for k in ("seed", "track_err", "lag_mean", "err_rms", "bump_alive")} for r in rs],
                                     "err_trace_seed0": rs[0]["err_trace"]}
                out["sweeps"][key] = sweep
                v = verdict_for(sweep, LEVELS, seeds); out["verdicts"][key] = v
                print("    err: " + " ".join(f"{s}:{sweep[str(s)]['track_err']['mean']:.3f}" for s in LEVELS) + f" | {v['statement']}", flush=True)
                partial = dict(out); partial["partial"] = True
                partial["verdict"] = {"statement": "PARTIAL"}
                save_results("e3_tracking", partial, args.out)
    pool.close(); pool.join()
    finish(out, args)


def finish(out, args):
    LEVELS = out["levels"]; amps = out["cue_amps"]; omegas = out["omegas"]; mismatches = out["mismatch"]
    main_keys = [k for k in out["verdicts"] if "_bpu_mismatch" in k and not k.endswith("mismatch0.0")]
    n_u = sum(out["verdicts"][k]["u_shaped"] for k in main_keys)
    # substrate comparison at the headline amplitude: BPU tracking error vs the float32 reference at every (omega, sigma)
    comp = {}
    amp0 = amps[0]
    for omega in omegas:
        if f"amp{amp0:g}_omega{omega:g}_float32" not in out["sweeps"]:
            continue
        ref = out["sweeps"][f"amp{amp0:g}_omega{omega:g}_float32"]
        for mismatch in mismatches + [0.0]:
            if f"amp{amp0:g}_omega{omega:g}_bpu_mismatch{mismatch}" not in out["sweeps"]:
                continue
            b = out["sweeps"][f"amp{amp0:g}_omega{omega:g}_bpu_mismatch{mismatch}"]
            comp[f"omega{omega:g}_mismatch{mismatch}"] = {str(s): {"float32": ref[str(s)]["track_err"]["mean"], "bpu": b[str(s)]["track_err"]["mean"],
                                                                  "ratio": b[str(s)]["track_err"]["mean"] / max(ref[str(s)]["track_err"]["mean"], 1e-9)} for s in LEVELS}
    out["substrate_comparison"] = comp
    out["verdict"] = {"n_u_shaped": n_u, "n_conditions": len(main_keys),
                      "statement": f"{n_u}/{len(main_keys)} mismatched BPU conditions (cue amp x omega x sigma_d2d) show a significant U in tracking error vs dynamic noise; " +
                                   "; ".join(f"{k}: {out['verdicts'][k]['statement']}" for k in main_keys)}
    print("E3 verdict:", out["verdict"]["statement"])
    save_results("e3_tracking", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e3_tracking"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = np.array(res["levels"], float)
    omegas = res["omegas"]; amp0 = res["cue_amps"][0]
    fig, axes = plt.subplots(1, len(omegas) + 1, figsize=(4.6 * (len(omegas) + 1), 4))
    for ax, omega in zip(axes, omegas):
        for key, sweep in res["sweeps"].items():
            if not key.startswith(f"amp{amp0:g}_omega{omega:g}_"):
                continue
            m = np.array([sweep[str(s)]["track_err"]["mean"] for s in res["levels"]]); e = np.array([sweep[str(s)]["track_err"]["sem"] for s in res["levels"]])
            lab = key.split("_", 2)[2].replace("_mismatch", " mm ")
            ax.errorbar(L, m, e, marker="o", capsize=3, ls="--" if lab == "float32" else "-", color="k" if lab == "float32" else None, label=lab)
        ax.set_xscale("symlog", linthresh=0.001); ax.set_xlabel("dynamic noise $\\sigma_{dyn}$"); ax.set_ylabel("tracking error |$\\phi_{bump}-\\phi_{cue}$| (rad)")
        ax.set_title(f"$\\omega$ = {omega} rad/s, cue {amp0}, N = {res['N']}, {res['seeds']} seeds", fontsize=9); ax.legend(fontsize=7)
    ax = axes[-1]
    for key, Ld in res["landscape"].items():
        ex = Ld["example"]; ax.plot(ex["phi"], ex["U"], label=f"{key} ΔU={Ld['delta_U']['mean']:.3g}", lw=1)
    ax.set_xlabel("bump position"); ax.set_ylabel("pinning potential U"); ax.set_title("pinning landscape, zero noise", fontsize=9); ax.legend(fontsize=6)
    fig.suptitle(res["verdict"]["statement"][:120], fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e3_tracking] figure {path}")


if __name__ == "__main__":
    main()
