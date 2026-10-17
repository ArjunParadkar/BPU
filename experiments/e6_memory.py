"""E6: scaffold memory and Hopfield attention under device noise.

32-bit content patterns are paired with 32-bit index patterns and stored on
32x32 tiles.  Recall uses the loop cue -> index -> attractor cleanup ->
content with corrupted cues: by default 20% of the cue bits are flipped
(``--flip``); ``--erase`` erases a fraction instead or in addition (the
first version of this experiment erased 50%).  Three
readers are compared: the pseudo-inverse scaffold (write-verify programmed),
the delta-rule scaffold (learned through the noisy tile), and Hopfield
attention over the content patterns.  Accuracy (fraction of pairs recalled
with every bit correct) is reported versus the number of stored pairs and
versus device noise sigma (split equally over d2d, c2c and read noise).
"""
from __future__ import annotations

import numpy as np

from common import base_parser, save_results, figure_path, Timer
from bpusim.memory import ScaffoldMemory, HopfieldAttention
from bpusim.noise import equal_split, summarize

COUNTS = [4, 8, 12, 16, 20, 24, 28, 32]
SIGMAS = [0.0, 0.02, 0.04, 0.08, 0.12, 0.2]
N = 32
BETA = 0.5
ERASE = 0.5


def corrupt(rng, s, erase=ERASE, flip=0.0):
    c = s.copy()
    c[rng.random(len(s)) < erase] = 0.0
    if flip > 0:
        m = rng.random(len(s)) < flip
        c[m] = -c[m]
    return c


def run_cell(K: int, sigma: float, seed: int, n_cues: int = 2, flip: float = 0.0, delta_epochs: int = 20, erase: float = ERASE):
    rng = np.random.default_rng([seed, K, int(sigma * 1e4)])
    params = equal_split(sigma).device_params()
    P = rng.choice([-1.0, 1.0], (N, K)); S = rng.choice([-1.0, 1.0], (N, K))
    out = {}
    mem = ScaffoldMemory(N, N, rng, params); mem.store(P, S, method="pinv")
    memd = ScaffoldMemory(N, N, rng, params); memd.store(P, S, method="delta", eta=0.02, epochs=delta_epochs)
    att = HopfieldAttention(S, BETA, rng, params)
    acc = {"pinv_loop": [], "pinv_direct": [], "delta_loop": [], "attention": []}
    bits = {k: [] for k in acc}
    for mu in range(K):
        for _ in range(n_cues):
            cue = corrupt(rng, S[:, mu], erase=erase, flip=flip)
            r = mem.recall(cue)["content"]; acc["pinv_loop"].append(np.array_equal(r, S[:, mu])); bits["pinv_loop"].append(np.mean(r == S[:, mu]))
            r = np.sign(mem.W_ps.read(np.sign(mem.W_sp.read(cue)))); r[r == 0] = 1
            acc["pinv_direct"].append(np.array_equal(r, S[:, mu])); bits["pinv_direct"].append(np.mean(r == S[:, mu]))
            r = memd.recall(cue)["content"]; acc["delta_loop"].append(np.array_equal(r, S[:, mu])); bits["delta_loop"].append(np.mean(r == S[:, mu]))
            r = np.sign(att.retrieve(cue)["xi"]); r[r == 0] = 1
            acc["attention"].append(np.array_equal(r, S[:, mu])); bits["attention"].append(np.mean(r == S[:, mu]))
    for k in acc:
        out[k] = {"acc": float(np.mean(acc[k])), "bit_acc": float(np.mean(bits[k]))}
    return out


def main():
    ap = base_parser("E6 memory and attention under device noise", default_seeds=3)
    ap.add_argument("--flip", type=float, default=0.2, help="fraction of cue bits flipped (default 0.2)")
    ap.add_argument("--erase", type=float, default=0.0, help="fraction of cue entries erased (set to 0)")
    args = ap.parse_args()
    seeds = 10 if args.full else args.seeds
    methods = ["pinv_loop", "pinv_direct", "delta_loop", "attention"]
    grid = {}
    with Timer("E6 grid"):
        for sigma in SIGMAS:
            for K in COUNTS:
                cells = [run_cell(K, sigma, seed, flip=args.flip, erase=args.erase) for seed in range(seeds)]
                grid[f"{sigma}|{K}"] = {m: {"acc": summarize([c[m]["acc"] for c in cells]),
                                           "bit_acc": summarize([c[m]["bit_acc"] for c in cells])} for m in methods}
            line = " ".join(f"K={K}:{grid[f'{sigma}|{K}']['pinv_loop']['acc']['mean']:.2f}/{grid[f'{sigma}|{K}']['attention']['acc']['mean']:.2f}" for K in COUNTS)
            print(f"  sigma={sigma:<5} (pinv loop / attention) {line}")
    # capacity at >= 90% accuracy per sigma
    capacity = {}
    for m in methods:
        capacity[m] = {}
        for sigma in SIGMAS:
            ok = [K for K in COUNTS if grid[f"{sigma}|{K}"][m]["acc"]["mean"] >= 0.9]
            capacity[m][str(sigma)] = max(ok) if ok else 0
    out = {"counts": COUNTS, "sigmas": SIGMAS, "seeds": seeds, "N": N, "beta": BETA, "erase_fraction": args.erase,
           "flip_fraction": args.flip, "methods": methods, "grid": grid, "capacity_at_90pct": capacity,
           "verdict": {"statement": "Capacity (largest K with >= 90% exact recall) per sigma: " +
                       "; ".join(f"{m}: " + ", ".join(f"s={s}->{capacity[m][str(s)]}" for s in SIGMAS) for m in methods)}}
    print("E6:", out["verdict"]["statement"])
    save_results("e6", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e6_memory"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = res["methods"]
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 3.6), sharey=True)
    for ax, m in zip(axes, methods):
        for sigma in res["sigmas"]:
            y = [res["grid"][f"{sigma}|{K}"][m]["acc"]["mean"] for K in res["counts"]]
            e = [res["grid"][f"{sigma}|{K}"][m]["acc"]["sem"] for K in res["counts"]]
            ax.errorbar(res["counts"], y, e, marker="o", ms=3, capsize=2, label=f"$\\sigma$={sigma}")
        ax.set_title(m.replace("_", " ")); ax.set_xlabel("stored pairs K"); ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel(f"exact recall accuracy ({100 * res['flip_fraction']:.0f}% flipped, {100 * res['erase_fraction']:.0f}% erased)"); axes[-1].legend(fontsize=7)
    fig.suptitle("E6: recall on 32x32 tiles vs pattern count and device noise", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e6] figure {path}")


if __name__ == "__main__":
    main()
