"""E3: path-integration error of a ring attractor versus dynamic noise (noise optimum test).

Rings of N in {128, 256} neurons (16 and 64 physical 32x32 tiles routed as one
matrix each) with fixed device mismatch sigma_d2d in {0.02, 0.05, 0.10}
(programmed weights W (1 + sigma zeta)), homeostatic scaling ON
(tau_H = 5 s, read-side gain), dynamic noise swept over nine levels
(half multiplicative read noise, half additive neuron noise), T = 30 s of
Ornstein-Uhlenbeck velocity per run, 20 seeds paired by mismatch draw and
trajectory.  Before each sweep the pinning landscape at zero dynamic noise
is measured (bump drift velocity vs position, integrated to U(phi); barrier
estimate Delta U).  The curve is fitted with E(D) = c1 exp(DeltaU / D) + 2 D T
and the fitted D* is reported with a bootstrap CI over seeds.  A no-mismatch
control is included.  The script states explicitly whether the curve is
U-shaped (interior minimum, significantly below sigma = 0, rising again).
``--quick`` reproduces the original 32-neuron, 3-second setting.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os

import numpy as np

from common import base_parser, save_results, figure_path, Timer, sign_test
from bpusim.attractor import RingAttractor, circ_dist
from bpusim.device import DeviceParams
from bpusim.noise import summarize

LEVELS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
J0, J1 = -0.5, 3.0


def ou_velocity(rng, n, dt, tau_v=0.3, sigma_v=1.5):
    v = np.empty(n); x = 0.0
    a = np.exp(-dt / tau_v); s = sigma_v * np.sqrt(1 - a * a)
    for i in range(n):
        x = a * x + s * rng.standard_normal(); v[i] = x
    return v


def build_ring(N, mismatch, sigma_dyn, seed, tau_H, dt=1e-3):
    rng_hw = np.random.default_rng([seed, 1, N, int(mismatch * 1e4)])
    params = DeviceParams(sigma_d2d=mismatch, sigma_c2c=0.0, sigma_r=0.5 * sigma_dyn)
    ring = RingAttractor(N, J0, J1, rng=rng_hw, params=params, sigma_mismatch=mismatch,
                         sigma_neuron=0.5 * sigma_dyn, dt=dt, tau_H=tau_H)
    ring.set_dynamic_rng(np.random.default_rng([seed, 3, int(round(sigma_dyn * 1e4))]))
    return ring


def run_one(args):
    N, mismatch, sigma_dyn, seed, c_gain, T, tau_H = args
    dt = 1e-3
    ring = build_ring(N, mismatch, sigma_dyn, seed, tau_H, dt)
    rng_traj = np.random.default_rng([seed, 2])
    phi0 = rng_traj.uniform(-np.pi, np.pi)
    ring.reset(angle=phi0); ring.settle(500)
    a0 = ring.decode()
    n = int(T / dt)
    v_true = ou_velocity(rng_traj, n, dt)
    angles = ring.run(v_true / c_gain, record=False)
    true = a0 + np.cumsum(v_true) * dt
    err = circ_dist(angles, true)
    r = ring.r
    return {"N": N, "mismatch": mismatch, "sigma": sigma_dyn, "seed": seed, "rmse": float(np.sqrt(np.mean(err ** 2))),
            "final_abs_err": float(abs(err[-1])), "bump_alive": bool((r.max() - r.min()) / max(r.mean(), 1e-9) > 0.5),
            "err_trace": err[::max(1, n // 60)].tolist(), "h_range": [float(ring.h.min()), float(ring.h.max())]}


def landscape_one(args):
    N, mismatch, seed, tau_H = args
    ring = build_ring(N, mismatch, 0.0, seed, tau_H)
    # let homeostasis settle on a free-running bump before probing
    ring.reset(angle=0.0); ring.settle(3000)
    L = ring.pinning_landscape(n_pos=48, settle=300, probe=200)
    return {"N": N, "mismatch": mismatch, "seed": seed, **L}


def fit_curve(levels, means, T):
    """Fit E(D) = c1 exp(dU/D) + 2 D T with D = sigma^2 (diffusion ~ noise variance); returns D*, params."""
    from scipy.optimize import curve_fit
    D = np.asarray(levels, float) ** 2
    E = np.asarray(means, float)
    m = D > 0
    def f(D, c1, dU, k):
        return c1 * np.exp(dU / D) + 2 * k * D * T
    try:
        p, _ = curve_fit(f, D[m], E[m], p0=[E[m].min() * 0.5, D[m].min(), 1.0], bounds=([0, 0, 0], [np.inf, np.inf, np.inf]), maxfev=20000)
        c1, dU, k = p
        Dgrid = np.linspace(D[m].min(), D.max(), 400)
        Dstar = float(Dgrid[np.argmin(f(Dgrid, *p))])
        return {"c1": float(c1), "delta_U_fit": float(dU), "k": float(k), "D_star": Dstar, "sigma_star": float(np.sqrt(Dstar)),
                "interior": bool(D[m].min() < Dstar < D.max())}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def main():
    ap = base_parser("E3 ring drift vs dynamic noise", default_seeds=20)
    ap.add_argument("--T", type=float, default=30.0)
    ap.add_argument("--N", type=str, default="128,256")
    ap.add_argument("--mismatch", type=str, default="0.02,0.05,0.10")
    ap.add_argument("--tau-H", type=str, default="5,0", help="homeostatic time constants to run as arms (0 = off)")
    ap.add_argument("--procs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--quick", action="store_true", help="original setting: N=32, T=3 s, mismatch 0.05, 3 seeds")
    args = ap.parse_args()
    seeds = 50 if args.full else args.seeds
    Ns = [int(x) for x in args.N.split(",")]; mismatches = [float(x) for x in args.mismatch.split(",")]
    tau_Hs = [float(x) for x in args.tau_H.split(",")]
    if args.quick:
        Ns, mismatches, seeds, args.T = [32], [0.05], 3, 3.0
    out = {"levels": LEVELS, "seeds": seeds, "N": Ns, "mismatch": mismatches, "J0": J0, "J1": J1, "T": args.T, "tau_H": tau_Hs,
           "velocity_gain": {}, "sweeps": {}, "landscape": {}, "fits": {}, "verdicts": {}}
    pool = mp.Pool(args.procs)
    for N in Ns:
        c = RingAttractor(N, J0, J1, use_tiles=False).calibrate_velocity_gain()
        out["velocity_gain"][str(N)] = c
        print(f"N={N}: velocity gain c = {c:.2f}; tiles per matrix = {(N // 32) ** 2}")
        for tau_H in tau_Hs:
          for mismatch in mismatches + [0.0]:
            key = f"N{N}_mismatch{mismatch}_tauH{tau_H:g}"
            with Timer(f"  {key} landscape"):
                L = pool.map(landscape_one, [(N, mismatch, s, tau_H) for s in range(min(seeds, 10))])
            out["landscape"][key] = {"delta_U": summarize([l["delta_U"] for l in L]), "rms_drift": summarize([l["rms_drift"] for l in L]),
                                     "example": L[0]}
            print(f"    Delta U = {out['landscape'][key]['delta_U']['mean']:.3g} +- {out['landscape'][key]['delta_U']['sem']:.2g} (rad^2/s units of U), rms drift {out['landscape'][key]['rms_drift']['mean']:.3g} rad/s")
            with Timer(f"  {key} sweep ({len(LEVELS)} x {seeds})"):
                jobs = [(N, mismatch, s, seed, c, args.T, tau_H) for s in LEVELS for seed in range(seeds)]
                res = pool.map(run_one, jobs)
            sweep = {}
            for s in LEVELS:
                rs = [r for r in res if r["sigma"] == s]
                sweep[str(s)] = {"rmse": summarize([r["rmse"] for r in rs]), "bump_alive_frac": float(np.mean([r["bump_alive"] for r in rs])),
                                 "runs": [{k: r[k] for k in ("seed", "rmse", "final_abs_err", "bump_alive", "h_range")} for r in rs],
                                 "err_trace_seed0": rs[0]["err_trace"]}
            out["sweeps"][key] = sweep
            means = [sweep[str(s)]["rmse"]["mean"] for s in LEVELS]
            print("    RMSE: " + " ".join(f"{s}:{m:.3f}" for s, m in zip(LEVELS, means)))
            # verdict + fit + bootstrap
            i_min = int(np.argmin(means))
            r0 = [r["rmse"] for r in sweep[str(LEVELS[0])]["runs"]]; rb = [r["rmse"] for r in sweep[str(LEVELS[i_min])]["runs"]]
            p, wins = sign_test(rb, r0)
            u = bool(i_min > 0 and i_min < len(LEVELS) - 1 and means[-1] > means[i_min] and p < 0.05)
            fit = fit_curve(LEVELS, means, args.T)
            rng = np.random.default_rng(0); boot = []
            per_seed = np.array([[sweep[str(s)]["runs"][k]["rmse"] for s in LEVELS] for k in range(seeds)])
            for _ in range(300):
                idx = rng.integers(0, seeds, seeds); f = fit_curve(LEVELS, per_seed[idx].mean(0), args.T)
                if "sigma_star" in f:
                    boot.append(f["sigma_star"])
            fit["sigma_star_bootstrap_CI95"] = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if boot else None
            fit["bootstrap_frac_interior"] = float(np.mean([LEVELS[1] <= b <= LEVELS[-1] for b in boot])) if boot else None
            out["fits"][key] = fit
            out["verdicts"][key] = {"sigma_star_empirical": LEVELS[i_min], "rmse_at_zero": means[0], "rmse_at_min": means[i_min],
                                    "rmse_at_max": means[-1], "sign_test_p": p, "seeds_better_than_zero": wins, "u_shaped": u,
                                    "statement": (f"U-shaped: minimum at sigma={LEVELS[i_min]} beats sigma=0 in {wins}/{seeds} seeds (p={p:.3g}) and error rises again"
                                                  if u else f"NOT U-shaped: " + ("minimum at zero noise" if i_min == 0 else f"interior minimum at sigma={LEVELS[i_min]} not significant ({wins}/{seeds} seeds, p={p:.3g})" if p >= 0.05 else "no rise at high noise"))
                                    + f"; fitted sigma* = {fit.get('sigma_star', float('nan')):.3g} (bootstrap 95% CI {fit.get('sigma_star_bootstrap_CI95')})"}
            print("    " + out["verdicts"][key]["statement"], flush=True)
    pool.close(); pool.join()
    main_keys = [k for k in out["verdicts"] if "mismatch0.0_" not in k]
    n_u = sum(out["verdicts"][k]["u_shaped"] for k in main_keys)
    out["verdict"] = {"n_u_shaped": n_u, "n_conditions": len(main_keys),
                      "statement": f"{n_u}/{len(main_keys)} mismatched conditions (N x sigma_d2d) show a significant U; " +
                                   "; ".join(f"{k}: {out['verdicts'][k]['statement']}" for k in main_keys)}
    print("E3 verdict:", out["verdict"]["statement"])
    save_results("e3", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e3_drift"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = np.array(res["levels"], float)
    Ns = res["N"]
    fig, axes = plt.subplots(1, len(Ns) + 1, figsize=(5 * (len(Ns) + 1), 4))
    for ax, N in zip(axes, Ns):
        for key, sweep in res["sweeps"].items():
            if not key.startswith(f"N{N}_"):
                continue
            m = np.array([sweep[str(s)]["rmse"]["mean"] for s in res["levels"]]); e = np.array([sweep[str(s)]["rmse"]["sem"] for s in res["levels"]])
            ax.errorbar(L, m, e, marker="o", capsize=3, ls="-" if key.endswith("tauH5") else "--",
                        label=key.split("_", 1)[1].replace("mismatch", "mm ").replace("_tauH", " tauH="))
        ax.set_xscale("symlog", linthresh=0.005); ax.set_xlabel("dynamic noise $\\sigma_{dyn}$"); ax.set_ylabel("path-integration RMSE (rad)")
        ax.set_title(f"E3: N = {N}, T = {res['T']} s, {res['seeds']} seeds", fontsize=10); ax.legend(fontsize=8)
    ax = axes[-1]
    for key, Ld in res["landscape"].items():
        ex = Ld["example"]
        ax.plot(ex["phi"], ex["U"], label=f"{key} (seed 0)", lw=1)
    ax.set_xlabel("bump position $\\phi$"); ax.set_ylabel("pinning potential U (arb.)"); ax.set_title("pinning landscape at zero dynamic noise", fontsize=10); ax.legend(fontsize=6)
    fig.suptitle(res["verdict"]["statement"][:110], fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e3] figure {path}")


if __name__ == "__main__":
    main()
