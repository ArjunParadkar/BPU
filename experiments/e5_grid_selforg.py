"""E5: self-organisation of grid (torus) modules on 5x6 and 16x16 sheets.

Same developmental protocol as E4 on a periodic 2D sheet: sparse random
tile, exploration cue sweeping the torus along an irrational heading, STDP + growth + normalisation
with a fixed global inhibition and a developmental (sub-threshold) gain.
After development the cue is removed, the sheet is kicked to 160 random
positions (5 free-running samples kept per kick, so the torus is covered
as a point sample rather than as a few clusters) and the activity is certified against the
circular-shift null: a torus requires exactly two significant H1 bars and
one H2 bar.  The programmed difference-of-Gaussians module (GridModule) is
certified alongside as a positive reference, and the untrained tile as a
negative control.
"""
from __future__ import annotations

import numpy as np

from common import base_parser, save_results, figure_path, Timer
from bpusim.attractor import (SelfOrganizingSheet, SelfOrgConfig, torus_dist_factory, sweep_2d,
                              GridModule)
from bpusim.topology import certify


def sheet_coords(nx, ny):
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return np.column_stack([ix.ravel(), iy.ravel()]).astype(float)


def random_positions(rng, nx, ny, n):
    return np.column_stack([rng.uniform(0, nx, n), rng.uniform(0, ny, n)])


def certify_sheet(X, rng, n_null, n_points, n_points_h2=None):
    c = certify(X, maxdim=2, n_points=n_points, n_null=n_null, rng=rng, n_points_h2=n_points_h2)
    return {"verdict": c["verdict"], "is_torus": c["is_torus"], "betti": c["betti"], "threshold": c["threshold"],
            "top_H1": c["top_lifetimes"][1], "top_H2": c["top_lifetimes"][2], "diagram_H1": c["diagrams"][1],
            "diagram_H2": c["diagrams"][2]}


def grid_kernel_for(nx, ny):
    """Burak-Fiete style zero-centre DoG scaled to the sheet: s_e = max(1, nx/16), s_i = 2 s_e, gain set
    so the kernel's Fourier peak is 3.7 (the 16x16 default's instability margin)."""
    from bpusim.attractor import torus_offsets, dog_kernel
    s_e = max(1.0, nx / 16.0); s_i = 2 * s_e
    dx, dy = torus_offsets(nx, ny)
    peak = np.fft.fft2(dog_kernel(dx, dy, 1, s_e, 1, s_i)[0].reshape(nx, ny)).real.max()
    return s_e, s_i, 3.7 / peak


def reference_cloud(nx, ny, rng, n_steps=30000, every=30, speed=0.4, use_tiles=True, turn_every=300):
    """Programmed DoG grid module path-integrating a uniform-coverage velocity: constant speed,
    a random turn every ``turn_every`` ms, 30 s of trajectory sampled every 30 ms.  A short
    trajectory (3 s) left the torus covered by a few line segments, whose loops the certifier
    counted; and straight sweeps make every neuron's trace periodic, which a circular shift does
    not destroy (the null kept the loops).  Dense coverage with frequent turns fixes both."""
    s_e, s_i, gain = grid_kernel_for(nx, ny)
    g = GridModule(nx, ny, s_e=s_e, s_i=s_i, gain=gain, rng=rng, use_tiles=use_tiles)
    g.settle(600)
    heading = np.arctan(np.sqrt(2.0)); out = []
    for t in range(n_steps):
        if t % turn_every == 0:
            heading += rng.uniform(-1.5, 1.5)
        r = g.step(vx=speed * np.cos(heading), vy=speed * np.sin(heading))
        if t % every == 0:
            out.append(r.copy())
    return np.array(out)


def run_reference_only(nx, ny, seed, n_null, n_points, n_points_h2):
    rng = np.random.default_rng([seed, nx, ny, 5])
    ref = certify_sheet(reference_cloud(nx, ny, rng), rng, n_null, n_points, n_points_h2)
    return {"seed": seed, "sheet": [nx, ny], "reference_programmed": ref, "trained": None, "control": None}


def _run_sheet_job(a):
    return run_sheet(*a)


def run_sheet(nx, ny, seed, n_dev, n_null, n_points, cfg_kw, n_starts=160, n_points_h2=None):
    rng = np.random.default_rng([seed, nx, ny])
    if nx * ny > 64:            # large sheets: a 256x256 tile costs ~1 ms per step, so sample more sparingly
        n_starts = 100; n_dev = min(n_dev, 12000)
    coords = sheet_coords(nx, ny)
    dist = torus_dist_factory(nx, ny)
    cfg = SelfOrgConfig(cue_width=1.0, **cfg_kw)
    sheet = SelfOrganizingSheet(coords, dist, rng, cfg)
    starts = random_positions(rng, nx, ny, n_starts)
    n_free = 200 if nx * ny <= 64 else 120
    X_ctrl, pers_ctrl = sheet.sample_manifold(starts, n_cue=80, n_free=n_free, n_keep=5)
    ctrl = certify_sheet(X_ctrl, rng, n_null, n_points, n_points_h2)
    cue = sweep_2d(rng, n_dev, nx, ny, speed=3.0)
    for t in range(n_dev):
        sheet.step(cue[t], learn=True)
    sheet.finish_development()
    X, pers = sheet.sample_manifold(starts, n_cue=80, n_free=n_free, n_keep=5)
    trained = certify_sheet(X, rng, n_null, n_points, n_points_h2)
    ref = certify_sheet(reference_cloud(nx, ny, rng), rng, n_null, n_points, n_points_h2)
    return {"seed": seed, "sheet": [nx, ny], "trained": trained, "control": ctrl, "reference_programmed": ref,
            "bump_persists_frac": float(pers.mean()), "control_bump_persists_frac": float(pers_ctrl.mean()),
            "n_active_final": sheet.tile.n_active, "n_grown": sheet.growth.n_grown, "n_pruned": sheet.growth.n_pruned,
            "mean_sheet_after": sheet.neurons.r.reshape(nx, ny).tolist()}


def main():
    ap = base_parser("E5 grid self-organisation", default_seeds=3)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--null", type=int, default=8)
    ap.add_argument("--points", type=int, default=250)
    ap.add_argument("--sheets", type=str, default="5x6,16x16")
    ap.add_argument("--points-h2", type=int, default=None, help="separate (smaller) sample for H2")
    ap.add_argument("--reference-only", action="store_true", help="certify only the programmed module (no self-organisation)")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--write-mode", type=str, default="nearest_pulse", choices=["nearest_pulse", "threshold", "level"])
    args = ap.parse_args()
    seeds = 10 if args.full else args.seeds
    out = {"seeds": seeds, "dev_steps": args.steps, "write_mode": args.write_mode, "sheets": {}}
    statements = []
    for spec in args.sheets.split(","):
        nx, ny = map(int, spec.split("x"))
        runs = []
        if args.reference_only:
            if args.procs > 1:
                import multiprocessing as mp
                with mp.Pool(args.procs) as pool:
                    runs = pool.starmap(run_reference_only, [(nx, ny, seed, args.null, args.points, args.points_h2) for seed in range(seeds)])
            else:
                for seed in range(seeds):
                    with Timer(f"{spec} seed {seed} (reference)"):
                        runs.append(run_reference_only(nx, ny, seed, args.null, args.points, args.points_h2))
            for r in runs:
                c = r["reference_programmed"]
                print(f"  {spec} seed {r['seed']}: programmed reference {c['verdict']} betti={c['betti']} H1={np.round(c['top_H1'][:3], 2)} H2={np.round(c['top_H2'][:2], 2)} thr={ {k: round(v, 2) for k, v in c['threshold'].items()} }")
        else:
            cfg_kw = {"write_mode": args.write_mode}
            if args.procs > 1:
                import multiprocessing as mp
                with mp.Pool(min(args.procs, seeds)) as pool:
                    runs = pool.map(_run_sheet_job, [(nx, ny, seed, args.steps, args.null, args.points, cfg_kw, 160, args.points_h2) for seed in range(seeds)])
            else:
                for seed in range(seeds):
                    with Timer(f"{spec} seed {seed}"):
                        runs.append(run_sheet(nx, ny, seed, args.steps, args.null, args.points, cfg_kw, n_points_h2=args.points_h2))
            for r in runs:
                seed = r["seed"]
                print(f"  {spec} seed {seed}: trained {r['trained']['verdict']} betti={r['trained']['betti']} | control {r['control']['verdict']} {r['control']['betti']}"
                      f" | programmed reference {r['reference_programmed']['verdict']} {r['reference_programmed']['betti']} | persists {r['bump_persists_frac']:.2f}")
        n_t = sum(bool(r["trained"] and r["trained"]["is_torus"]) for r in runs); n_c = sum(bool(r["control"] and r["control"]["is_torus"]) for r in runs)
        n_r = sum(r["reference_programmed"]["is_torus"] for r in runs)
        out["sheets"][spec] = {"runs": runs, "trained_torus_fraction": n_t / seeds, "control_torus_fraction": n_c / seeds,
                               "reference_torus_fraction": n_r / seeds}
        statements.append(f"{spec}: programmed DoG reference {n_r}/{seeds}" + ("" if args.reference_only else f"; {n_t}/{seeds} self-organised sheets certified (1,2,1); untrained control {n_c}/{seeds}"))
    out["verdict"] = {"statement": "; ".join(statements)}
    print("E5 verdict:", out["verdict"]["statement"])
    save_results("e5", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e5_grid_selforg"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sheets = list(res["sheets"].keys())
    fig, axes = plt.subplots(len(sheets), 3, figsize=(13, 4 * len(sheets)), squeeze=False)
    for row, spec in enumerate(sheets):
        r = res["sheets"][spec]["runs"][0]
        for col, (key, title) in enumerate((("trained", "self-organised"), ("reference_programmed", "programmed DoG"), ("control", "untrained control"))):
            ax = axes[row, col]
            c = r[key]
            if c is None:
                ax.set_title(f"{spec} {title}: not run", fontsize=9); ax.axis("off"); continue
            for dim, mk in ((1, "o"), (2, "^")):
                D = np.array(c[f"diagram_H{dim}"]).reshape(-1, 2)
                if len(D):
                    ax.scatter(D[:, 0], D[:, 1], s=14, marker=mk, label=f"H{dim}")
                thr = c["threshold"][str(dim)] if str(dim) in c["threshold"] else c["threshold"][dim]
                ax.plot([0, 3], [thr, 3 + thr], ":", lw=0.8)
            lim = max([np.max(np.array(c[f"diagram_H{d}"]).reshape(-1, 2)) if len(c[f"diagram_H{d}"]) else 0.1 for d in (1, 2)])
            ax.plot([0, lim], [0, lim], "k--", lw=0.5); ax.set_xlim(0, lim * 1.05); ax.set_ylim(0, lim * 1.05)
            ax.set_title(f"{spec} {title}: {c['verdict']} betti={c['betti']}", fontsize=9); ax.set_xlabel("birth"); ax.set_ylabel("death"); ax.legend(fontsize=7)
    fig.suptitle(res["verdict"]["statement"], fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e5] figure {path}")


if __name__ == "__main__":
    main()
