"""E4: self-organisation of a ring attractor from a sparse random tile.

32 neurons with preferred angles on a 32x32 tile start at 10% random
connectivity with small weights.  A cue (Gaussian bump of input) sweeps around the
ring at constant speed, reversing every lap (uniform coverage; a random-walk
cue is available with --cue walk) while STDP (symmetric window) writes the tile
through the pulse model, growth adds synapses where <x_i xi_j> is large and
prunes weak ones, and Oja-style column normalisation bounds the excitation.
A fixed global inhibition is not learned.  After development the cue is
removed; the network is kicked to random angles and the free-running
activity is recorded.  Persistent homology certifies beta1 = 1 against the
circular-shift null.  An untrained control with the same initial tile is
certified the same way, and WhiteBox metrics (SV, TC, FPL) are tracked
across development checkpoints.
"""
from __future__ import annotations

import numpy as np

from common import base_parser, save_results, figure_path, Timer
from bpusim.attractor import SelfOrganizingSheet, SelfOrgConfig, ring_dist, circ_dist, sweep_1d, telegraph_walk_1d
from bpusim.topology import certify
from bpusim.metrics import whitebox_track

N = 32


def cue_walk(rng, n_steps, dt=1e-3, speed=2.0):
    """Random-walk cue angle: Ornstein-Uhlenbeck velocity integrated on the circle."""
    v = 0.0; a = 0.0; out = np.empty(n_steps)
    k = np.exp(-dt / 0.3); s = speed * np.sqrt(1 - k * k)
    for i in range(n_steps):
        v = k * v + s * rng.standard_normal(); a += v * dt; out[i] = a
    return out


def population_angle(r, theta):
    return float(np.arctan2(np.sum(r * np.sin(theta)), np.sum(r * np.cos(theta))))


def profile(W, theta):
    """Mean weight as a function of the angular difference (binned)."""
    d = circ_dist(theta[None, :], theta[:, None])
    off = ~np.eye(len(theta), dtype=bool)          # no self-connections: leave the diagonal out
    bins = np.linspace(-np.pi, np.pi, 17)
    idx = np.digitize(d[off], bins) - 1
    w = W[off]
    prof = np.array([w[idx == b].mean() if np.any(idx == b) else 0 for b in range(16)])
    return (0.5 * (bins[1:] + bins[:-1])).tolist(), prof.tolist()


def run_seed(seed: int, n_dev: int, n_checkpoints: int, n_null: int, cue_kind: str = "sweep", write_budget: float = 0.0,
             write_mode: str = "threshold", theta_flush: float = 1.0):
    rng = np.random.default_rng(seed)
    theta = 2 * np.pi * np.arange(N) / N
    cfg = SelfOrgConfig(write_budget=write_budget, write_mode=write_mode, theta_flush=theta_flush)
    sheet = SelfOrganizingSheet(theta, ring_dist, rng, cfg)
    W0 = sheet.tile.W.copy(); n_active0 = sheet.tile.n_active
    starts = rng.uniform(-np.pi, np.pi, 40)
    # control: certify the untrained tile first (same object, before learning)
    X_ctrl, persisted_ctrl = sheet.sample_manifold(starts)
    cert_ctrl = certify(X_ctrl, maxdim=1, n_points=300, n_null=n_null, rng=rng)
    cue = (sweep_1d if cue_kind == "sweep" else telegraph_walk_1d)(rng, n_dev)
    clouds, ckpt_steps, ckpt_alive = [], [], []
    chunk = n_dev // n_checkpoints
    for k in range(n_checkpoints):
        for t in range(k * chunk, (k + 1) * chunk):
            sheet.step(cue[t], learn=True)
        X, persisted = sheet.sample_manifold(starts, n_keep=25)
        clouds.append(X); ckpt_steps.append((k + 1) * chunk); ckpt_alive.append(float(persisted.mean()))
    sheet.finish_development()
    X, persisted = sheet.sample_manifold(starts)
    cert = certify(X, maxdim=1, n_points=300, n_null=n_null, rng=rng)
    # where do the released bumps end up?  (coverage of the ring by the free-running states)
    ends = np.array([population_angle(X[i * 50:(i + 1) * 50].mean(0), theta) for i in range(len(starts))])
    occupancy = np.histogram(ends, bins=8, range=(-np.pi, np.pi))[0]
    drift = np.abs(circ_dist(ends, starts))
    wb = whitebox_track(clouds, [chunk * 1e-3] * (len(clouds) - 1), maxdim=1, dim=1, n_points=150, rng=rng)
    ang, prof0 = profile(W0, theta)
    _, prof1 = profile(sheet.tile.W, theta)
    return {
        "seed": seed, "certified_ring": cert["is_ring"], "betti1": cert["betti"][1], "betti0_extra": cert["betti"][0],
        "writes_per_device_per_s": sheet.tile.writes_per_device(seconds=n_dev * 1e-3),
        "write_budget_per_device_per_s": 6e-3,
        "percent_of_write_budget": 100.0 * sheet.tile.writes_per_device(seconds=n_dev * 1e-3) / 6e-3,
        "flush_rate": sheet.tile.flush_rate, "levels": sheet.tile.levels, "write_mode": write_mode, "theta_flush": theta_flush,
        "budget": sheet.tile.write_budget_report(seconds=n_dev * 1e-3),
        "top_H1_lifetimes": cert["top_lifetimes"][1], "H1_threshold": cert["threshold"][1],
        "control_certified_ring": cert_ctrl["is_ring"], "control_betti1": cert_ctrl["betti"][1], "control_betti0_extra": cert_ctrl["betti"][0],
        "control_top_H1": cert_ctrl["top_lifetimes"][1], "control_H1_threshold": cert_ctrl["threshold"][1],
        "bump_persists_frac": float(persisted.mean()), "control_bump_persists_frac": float(persisted_ctrl.mean()),
        "ring_occupancy_bins_of_8": int(np.count_nonzero(occupancy)), "occupancy": occupancy.tolist(),
        "median_drift_from_cue_rad": float(np.median(drift)),
        "n_active_final": sheet.tile.n_active, "n_active_initial": n_active0,
        "n_grown": sheet.growth.n_grown, "n_pruned": sheet.growth.n_pruned,
        "weight_profile": {"angle": ang, "initial": prof0, "final": prof1},
        "checkpoints": {"step": ckpt_steps, "bump_persists_frac": ckpt_alive, "SV": wb["SV"], "TC": wb["TC"], "FPL": wb["FPL"]},
        "diagram_H1": cert["diagrams"][1], "control_diagram_H1": cert_ctrl["diagrams"][1],
        "n_active_trace": sheet.log["n_active"][::200],
    }


def _job(a):
    return run_seed(*a)


def main():
    ap = base_parser("E4 ring self-organisation", default_seeds=3)
    ap.add_argument("--steps", type=int, default=40000, help="development steps (dt = 1 ms)")
    ap.add_argument("--write-budget", type=float, default=0.0, help="endurance cap, write events per device per second (6e-3 = the technical document's budget)")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--cue", type=str, default="sweep", choices=["sweep", "walk"])
    ap.add_argument("--write-mode", type=str, default="nearest_pulse", choices=["nearest_pulse", "threshold", "level"],
                    help="nearest_pulse (default since 2026-10-19): 16-bit accumulator flushed every step as whole pulses; "
                         "threshold: the original E4 rule; level: accumulate-and-flush at theta_flush conductance levels")
    ap.add_argument("--theta-flush", type=float, default=1.0, help="level mode: flush threshold in conductance levels (0.25, 0.5, 1.0)")
    ap.add_argument("--checkpoints", type=int, default=5)
    ap.add_argument("--null", type=int, default=20)
    args = ap.parse_args()
    seeds = 10 if args.full else args.seeds
    runs = []
    if args.procs > 1:
        import multiprocessing as mp
        with mp.Pool(args.procs) as pool:
            runs = pool.map(_job, [(seed, args.steps, args.checkpoints, args.null, args.cue, args.write_budget, args.write_mode, args.theta_flush) for seed in range(seeds)])
    else:
        for seed in range(seeds):
            with Timer(f"seed {seed}"):
                runs.append(run_seed(seed, args.steps, args.checkpoints, args.null, args.cue, args.write_budget, args.write_mode, args.theta_flush))
    for r in runs:
        seed = r["seed"]
        print(f"  seed {seed}: certified={r['certified_ring']} betti1={r['betti1']} extra betti0={r['betti0_extra']} writes/dev/s={r['writes_per_device_per_s']:.3g} top H1={np.round(r['top_H1_lifetimes'][:3], 3)} thr={r['H1_threshold']:.3f}"
              f" | control certified={r['control_certified_ring']} betti1={r['control_betti1']} | persists {r['bump_persists_frac']:.2f} (ctrl {r['control_bump_persists_frac']:.2f})"
              f" | occupancy {r['ring_occupancy_bins_of_8']}/8 bins, drift {r['median_drift_from_cue_rad']:.2f} rad"
              f" | synapses {r['n_active_initial']}->{r['n_active_final']}")
    n_cert = sum(r["certified_ring"] for r in runs); n_ctrl = sum(r["control_certified_ring"] for r in runs)
    occ = [r["ring_occupancy_bins_of_8"] for r in runs]
    wpd = float(np.mean([r['writes_per_device_per_s'] for r in runs]))
    out = {"seeds": seeds, "dev_steps": args.steps, "cue": args.cue, "write_budget": args.write_budget, "write_mode": args.write_mode, "theta_flush": args.theta_flush,
           "writes_per_device_total": float(np.mean([r["budget"]["writes_per_device_total"] for r in runs])),
           "percent_of_endurance": float(np.mean([r["budget"]["percent_of_endurance"] for r in runs])),
           "writes_per_tile_per_s": float(np.mean([r["budget"]["writes_per_tile_per_s"] for r in runs])),
           "percent_of_tile_bandwidth": float(np.mean([r["budget"]["percent_of_tile_bandwidth"] for r in runs])),
           "development_time_s": args.steps * 1e-3, "runs": runs,
           "writes_per_device_per_s": wpd, "percent_of_write_budget": 100.0 * wpd / 6e-3, "flush_rate": float(np.mean([r["flush_rate"] for r in runs])),
           "verdict": {"certified_fraction": n_cert / seeds, "control_certified_fraction": n_ctrl / seeds,
                       "mean_occupancy_bins_of_8": float(np.mean(occ)),
                       "statement": f"{n_cert}/{seeds} trained rings certified beta1 = 1 against the circular-shift null "
                                    f"({n_ctrl}/{seeds} untrained controls certified); released bumps occupy "
                                    f"{np.mean(occ):.1f}/8 bins of the ring on average (8 = a fully explored ring); "
                                    f"development {args.steps * 1e-3:.0f} s at {wpd:.3g} writes/device/s = {100 * wpd / 6e-3:.0f}% of the 6e-3 budget"
                                    + (f" (quota {args.write_budget:g})" if args.write_budget else " (no quota)")
                                    + (f", accumulate-and-flush at theta_flush = {args.theta_flush:g} levels (L = {runs[0]['levels']}, flush rate {np.mean([r['flush_rate'] for r in runs]):.2g})" if args.write_mode == "level"
                                       else f", nearest-pulse writes (16-bit accumulator flushed every step, flush rate {np.mean([r['flush_rate'] for r in runs]):.2g})" if args.write_mode == "nearest_pulse" else ", threshold writes")
                                    + f"; total {np.mean([r['budget']['writes_per_device_total'] for r in runs]):.3g} writes/device = {np.mean([r['budget']['percent_of_endurance'] for r in runs]):.2g}% of the 1e6 endurance, {np.mean([r['budget']['writes_per_tile_per_s'] for r in runs]):.3g} writes/tile/s = {np.mean([r['budget']['percent_of_tile_bandwidth'] for r in runs]):.2g}% of the 1e4 driver bandwidth."}}
    print("E4 verdict:", out["verdict"]["statement"])
    save_results("e4", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e4_ring_selforg"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = res["runs"][0]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    ax[0].plot(r["weight_profile"]["angle"], r["weight_profile"]["initial"], label="initial")
    ax[0].plot(r["weight_profile"]["angle"], r["weight_profile"]["final"], label="after development")
    ax[0].set_xlabel("$\\theta_{post}-\\theta_{pre}$"); ax[0].set_ylabel("mean W"); ax[0].legend(); ax[0].set_title("learned kernel (seed 0)")
    for key, lab in (("diagram_H1", "trained"), ("control_diagram_H1", "control")):
        D = np.array(r[key]).reshape(-1, 2)
        if len(D):
            ax[1].scatter(D[:, 0], D[:, 1], s=12, label=lab)
    lim = max([np.max(np.array(r[k]).reshape(-1, 2)) if len(r[k]) else 1 for k in ("diagram_H1", "control_diagram_H1")] + [1e-3])
    ax[1].plot([0, lim], [0, lim], "k--", lw=0.5)
    ax[1].plot([0, lim], [r["H1_threshold"], lim + r["H1_threshold"]], "r:", lw=0.8, label="null 99%")
    ax[1].set_xlabel("birth"); ax[1].set_ylabel("death"); ax[1].legend(fontsize=7); ax[1].set_title(f"H1 barcode, certified={r['certified_ring']}")
    ck = r["checkpoints"]
    ax[2].plot(ck["step"], ck["TC"], "o-", label="TC (H1)"); ax[2].set_xlabel("development step"); ax[2].set_ylabel("total persistence")
    ax2 = ax[2].twinx(); ax2.plot(ck["step"][1:], ck["SV"], "s--", color="C1", label="SV"); ax2.set_ylabel("SV (W$_2$/s)")
    ax[2].set_title("WhiteBox metrics"); ax[2].legend(loc="upper left", fontsize=7); ax2.legend(loc="upper right", fontsize=7)
    ax[3].plot(np.arange(len(r["n_active_trace"])) * 200, r["n_active_trace"]); ax[3].set_xlabel("development step"); ax[3].set_ylabel("active synapses")
    ax[3].set_title(res["verdict"]["statement"][:50], fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e4] figure {path}")


if __name__ == "__main__":
    main()
