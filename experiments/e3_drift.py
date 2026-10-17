"""E3: path-integration error of a 32-neuron ring on a 32x32 tile versus dynamic noise.

Fixed device mismatch sigma_d2d = 0.05 (programmed weights W (1 + 0.05 zeta)).
Dynamic noise sigma_dyn is swept and split half into multiplicative tile read
noise and half into additive neuron noise.  Velocity trajectories are
Ornstein-Uhlenbeck random walks; the velocity command is scaled by the gain
calibrated on the ideal network so that the bump should move at v rad/s.
Reported: circular RMSE between the decoded bump angle and the integrated
velocity, per noise level, over seeds.  A control sweep without mismatch is
included.  The script states explicitly whether the curve is U-shaped.
"""
from __future__ import annotations

import numpy as np

from common import base_parser, save_results, figure_path, Timer, sign_test
from bpusim.attractor import RingAttractor, circ_dist
from bpusim.device import DeviceParams
from bpusim.noise import summarize

LEVELS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2]
N = 32
J0, J1 = -0.5, 3.0


def ou_velocity(rng, n, dt, tau_v=0.3, sigma_v=1.5):
    v = np.empty(n); x = 0.0
    a = np.exp(-dt / tau_v); s = sigma_v * np.sqrt(1 - a * a)
    for i in range(n):
        x = a * x + s * rng.standard_normal(); v[i] = x
    return v


def run_one(sigma_dyn: float, seed: int, mismatch: float, c_gain: float, T: float = 3.0, dt: float = 1e-3):
    # three independent streams so that one seed pairs the same fabricated mismatch and the
    # same velocity trajectory across every dynamic-noise level
    rng_hw = np.random.default_rng([seed, 1])
    rng_traj = np.random.default_rng([seed, 2])
    rng_dyn = np.random.default_rng([seed, 3, int(round(sigma_dyn * 1e4))])
    params = DeviceParams(sigma_d2d=mismatch, sigma_c2c=0.0, sigma_r=0.5 * sigma_dyn)
    ring = RingAttractor(N, J0, J1, rng=rng_hw, params=params, sigma_mismatch=mismatch,
                         sigma_neuron=0.5 * sigma_dyn, dt=dt)
    ring.set_dynamic_rng(rng_dyn)
    phi0 = rng_traj.uniform(-np.pi, np.pi)
    ring.reset(angle=phi0); ring.settle(300)
    a0 = ring.decode()
    n = int(T / dt)
    v_true = ou_velocity(rng_traj, n, dt)
    angles = ring.run(v_true / c_gain, record=False)
    true = a0 + np.cumsum(v_true) * dt
    err = circ_dist(angles, true)
    r = ring.r
    return {"rmse": float(np.sqrt(np.mean(err ** 2))), "final_abs_err": float(abs(err[-1])),
            "bump_alive": bool((r.max() - r.min()) / max(r.mean(), 1e-9) > 0.5),
            "err_trace": err[::50].tolist()}


def main():
    ap = base_parser("E3 ring drift vs dynamic noise", default_seeds=3)
    ap.add_argument("--T", type=float, default=3.0)
    args = ap.parse_args()
    seeds = 10 if args.full else args.seeds
    ref = RingAttractor(N, J0, J1, use_tiles=False)
    c = ref.calibrate_velocity_gain()
    print(f"velocity gain c = {c:.2f} rad/s per unit command (gamma = 1)")
    out = {"levels": LEVELS, "seeds": seeds, "N": N, "J0": J0, "J1": J1, "T": args.T, "velocity_gain": c,
           "sweeps": {}}
    for label, mismatch in (("mismatch_0.05", 0.05), ("mismatch_0.00", 0.0)):
        sweep = {}
        with Timer(label):
            for s in LEVELS:
                runs = [run_one(s, seed, mismatch, c, args.T) for seed in range(seeds)]
                sweep[str(s)] = {"runs": runs, "rmse": summarize([r["rmse"] for r in runs]),
                                 "bump_alive_frac": float(np.mean([r["bump_alive"] for r in runs]))}
                print(f"  {label} sigma_dyn={s:<6} RMSE={sweep[str(s)]['rmse']['mean']:.3f} +- {sweep[str(s)]['rmse']['sem']:.3f} rad")
        out["sweeps"][label] = sweep
    # ----- verdict on the main sweep
    main_sweep = out["sweeps"]["mismatch_0.05"]
    means = np.array([main_sweep[str(s)]["rmse"]["mean"] for s in LEVELS])
    i_min = int(np.nanargmin(means))
    r0 = [r["rmse"] for r in main_sweep[str(LEVELS[0])]["runs"]]
    rb = [r["rmse"] for r in main_sweep[str(LEVELS[i_min])]["runs"]]
    p, wins = sign_test(rb, r0)
    rises_after = bool(i_min < len(LEVELS) - 1 and means[-1] > means[i_min])
    u_shaped = bool(i_min > 0 and rises_after and p < 0.05)
    out["verdict"] = {
        "sigma_star": LEVELS[i_min], "rmse_at_zero": float(means[0]), "rmse_at_min": float(means[i_min]),
        "rmse_at_max_noise": float(means[-1]), "sign_test_p_min_vs_zero": p, "seeds_better_than_zero": wins,
        "u_shaped": u_shaped,
        "statement": ("U-shaped: an interior noise level beats zero noise (sign test p=%.3g) and error rises again at high noise." % p)
        if u_shaped else
        ("NOT U-shaped: " + ("minimum is at zero noise" if i_min == 0 else
                             ("interior minimum not significant (p=%.3g)" % p if not p < 0.05 else
                              "error does not rise again at high noise")) + "; the curve is reported as measured."),
    }
    print("E3 verdict:", out["verdict"]["statement"])
    save_results("e3", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e3_drift"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L = np.array(res["levels"], float)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for label, sweep in res["sweeps"].items():
        m = np.array([sweep[str(s)]["rmse"]["mean"] for s in res["levels"]])
        e = np.array([sweep[str(s)]["rmse"]["sem"] for s in res["levels"]])
        ax[0].errorbar(L, m, e, marker="o", capsize=3, label=label.replace("_", " = "))
    ax[0].set_xlabel("dynamic noise $\\sigma_{dyn}$ (read + neuron)"); ax[0].set_ylabel("path-integration RMSE (rad)")
    ax[0].set_xscale("symlog", linthresh=0.005); ax[0].legend(); ax[0].set_title("E3: ring drift vs dynamic noise")
    sweep = res["sweeps"]["mismatch_0.05"]
    for s in res["levels"][::2]:
        tr = np.array(sweep[str(s)]["runs"][0]["err_trace"])
        ax[1].plot(np.arange(len(tr)) * 50 * 1e-3, np.abs(tr), label=f"$\\sigma$={s}")
    ax[1].set_xlabel("time (s)"); ax[1].set_ylabel("|angle error| (rad), seed 0"); ax[1].legend(fontsize=7)
    ax[1].set_title(res["verdict"]["statement"][:60], fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e3] figure {path}")


if __name__ == "__main__":
    main()
