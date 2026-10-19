"""E14: read noise as a sampler — retrieval from ambiguous cues (variability programme).

Modern-Hopfield / attention retrieval  xi' = X softmax(beta X^T xi)  from a store of P = 8
random +-1 patterns of dimension N = 64 held on a tile (``memory.HopfieldAttention``: the score
read and the value read both go through the noisy tile).  The cue is an AMBIGUOUS blend of two
stored patterns, c = alpha x_a + (1 - alpha) x_b, alpha in {0.5, 0.55, 0.6}; the retrieved
pattern of a trial is the stored pattern with the largest overlap with the output.

The "correct posterior" is the Boltzmann posterior of the energy at the retrieval temperature,
p(mu | c) = softmax_mu(beta x_mu . c), which is what the softmax computes from noise-free scores:
at alpha = 0.5 it is (0.5, 0.5) on the two blended patterns; alpha = 0.55 / 0.6 make it
asymmetric (~0.78 / 0.93 on x_a at beta = 0.2, N = 64).  Over 400 trials with fresh read noise
the empirical retrieval frequencies p_hat(mu) are compared with the posterior by
KL(p || p_hat) (Laplace-smoothed) as a function of the read-noise dial sigma_r in
{0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2}.  Reference: the same retrieval
on float32 weights with the same multiplicative read noise injected (FloatTile).  5 seeds
(stored patterns).

Hypotheses: at sigma = 0 retrieval is deterministic (one pattern every time, KL large); the KL
has its minimum at sigma > 0 (sigma* with a bootstrap CI over seeds); the same sigma* serves
every alpha (a sampler at the right temperature matches asymmetric posteriors too).
"""
from __future__ import annotations

import multiprocessing as mp

import numpy as np

from common import base_parser, save_results, figure_path, Timer
from bpusim.device import DeviceParams
from bpusim.memory import HopfieldAttention
from bpusim.noise import summarize

SIGMAS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1, 0.15, 0.2]
BEYOND_DIAL = [0.3, 0.5, 0.7, 1.0]        # diagnostic levels above the dial's range, to locate the rise (spurious retrievals)
ALPHAS = [0.5, 0.55, 0.6]
N, P, BETA = 64, 8, 0.2
TRIALS = 400


def posterior(X, c, beta):
    s = beta * (X.T @ c); s = s - s.max(); p = np.exp(s); return p / p.sum()


def kl(p, q_counts):
    q = (q_counts + 1.0) / (q_counts.sum() + len(q_counts))        # Laplace smoothing
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / q[m])))


def run_one(job):
    sigma, alpha, seed, substrate = job
    rng = np.random.default_rng([seed, 14])
    X = rng.choice([-1.0, 1.0], (N, P))
    params = DeviceParams(sigma_r=sigma)                             # d2d 0.03 / c2c 0.03 defaults on tiles; dial = read noise
    mem = HopfieldAttention(X, BETA, np.random.default_rng([seed, 14, int(sigma * 1e4), substrate == "tiles"]), params, float_ref=(substrate == "float32"))
    t = mem.store.tile
    if sigma <= t.MODULATE_RANGE[1]:
        t.set_noise(sigma)
    elif substrate == "tiles":                                  # beyond the dial: diagnostic only
        t.p = t.p.replace(sigma_r=sigma); t.gp.p = t.p; t.gm.p = t.p
    else:
        t.sigma_r = sigma
    out = {"sigma": sigma, "alpha": alpha, "seed": seed, "substrate": substrate, "pairs": []}
    kls = []; p_a_err = []; n_other = 0; distinct = []
    pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    for a, b in pairs:
        c = alpha * X[:, a] + (1 - alpha) * X[:, b]
        post = posterior(X, c, BETA)
        counts = np.zeros(P)
        for _ in range(TRIALS):
            xi = mem.retrieve(c)["xi"]
            counts[int(np.argmax(X.T @ xi))] += 1
        kls.append(kl(post, counts)); p_a_err.append(float(counts[a] / TRIALS - post[a])); n_other += counts.sum() - counts[a] - counts[b]
        distinct.append(int((counts > 0).sum()))
        out["pairs"].append({"a": a, "b": b, "posterior_ab": [float(post[a]), float(post[b])], "freq_ab": [float(counts[a] / TRIALS), float(counts[b] / TRIALS)], "kl": kls[-1]})
    out.update({"kl": float(np.mean(kls)), "p_a_error": float(np.mean(p_a_err)), "frac_other": float(n_other / (TRIALS * len(pairs))), "distinct_patterns": float(np.mean(distinct))})
    return out


def main():
    ap = base_parser("E14 read noise as a sampler (ambiguous-cue retrieval)", default_seeds=5)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = args.seeds; sigmas = SIGMAS + BEYOND_DIAL; alphas = ALPHAS
    if args.smoke:
        seeds = 1; sigmas = [0.0, 0.04, 0.2, 0.5]; alphas = [0.5]
    jobs = [(s, a, seed, sub) for sub in ("float32", "tiles") for a in alphas for s in sigmas for seed in range(seeds)]
    with Timer(f"E14 {len(jobs)} runs"):
        with mp.Pool(args.procs) as pool:
            runs = pool.map(run_one, jobs)
    out = {"sigmas": sigmas, "alphas": alphas, "seeds": seeds, "N": N, "P": P, "beta": BETA, "trials": TRIALS, "sweeps": {}, "verdicts": {}}
    rng = np.random.default_rng(0)
    for sub in ("tiles", "float32"):
        for a in alphas:
            key = f"alpha{a:g}_{sub}"
            rs = [r for r in runs if r["substrate"] == sub and r["alpha"] == a]
            sweep = {str(s): {"kl": summarize([r["kl"] for r in rs if r["sigma"] == s]), "p_a_error": summarize([r["p_a_error"] for r in rs if r["sigma"] == s]),
                              "frac_other": summarize([r["frac_other"] for r in rs if r["sigma"] == s]), "distinct_patterns": summarize([r["distinct_patterns"] for r in rs if r["sigma"] == s])} for s in sigmas}
            out["sweeps"][key] = sweep
            M = np.array([[next(r["kl"] for r in rs if r["sigma"] == s and r["seed"] == sd) for sd in range(seeds)] for s in sigmas])
            mean = M.mean(1); dial = [j for j, s in enumerate(sigmas) if s <= 0.2]
            i = int(dial[int(np.argmin(mean[dial]))]); i_all = int(np.argmin(mean))
            boots = [sigmas[dial[int(np.argmin(M[dial][:, rng.integers(0, seeds, seeds)].mean(1)))]] for _ in range(2000)]
            out["verdicts"][key] = {"sigma_star": sigmas[i], "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))], "kl_at_zero": float(mean[0]), "kl_min": float(mean[i]),
                                    "kl_at_dial_max": float(mean[dial[-1]]), "kl_at_max": float(mean[-1]), "sigma_star_beyond_dial": sigmas[i_all], "kl_min_beyond_dial": float(mean[i_all]),
                                    "rises_beyond_dial": bool(mean[-1] > mean[i_all] + 0.05),
                                    "deterministic_at_zero": bool(sweep["0.0"]["distinct_patterns"]["mean"] <= 1.0 + 1e-9),
                                    "min_at_positive_sigma": bool(i > 0 and np.percentile(boots, 2.5) > 0), "mean_curve": mean.tolist()}
            v = out["verdicts"][key]
            print(f"  {key}: KL {v['kl_at_zero']:.2f} at 0 -> {v['kl_min']:.3f} at sigma* {v['sigma_star']} (CI {v['ci95']}) -> {v['kl_at_dial_max']:.3f} at 0.2 -> {v['kl_at_max']:.2f} at {sigmas[-1]}; deterministic at 0: {v['deterministic_at_zero']}; beyond dial sigma* {v['sigma_star_beyond_dial']}", flush=True)
    tv = {a: out["verdicts"][f"alpha{a:g}_tiles"] for a in alphas}
    stars = [tv[a]["sigma_star"] for a in alphas]
    out["verdict"] = {"kl_min_at_positive_sigma_all_alphas": all(tv[a]["min_at_positive_sigma"] for a in alphas), "deterministic_at_zero": all(tv[a]["deterministic_at_zero"] for a in alphas),
                      "sigma_star_by_alpha_tiles": {str(a): tv[a]["sigma_star"] for a in alphas}, "same_sigma_star": bool(max(stars) <= 2 * max(min(stars), 1e-9)),
                      "statement": ("SMOKE RUN. " if args.smoke else "") + f"{seeds} seeds, {TRIALS} trials per cue, beta {BETA}: " + "; ".join(f"alpha {a:g}: tiles KL {tv[a]['kl_at_zero']:.2f} -> {tv[a]['kl_min']:.3f} at sigma* {tv[a]['sigma_star']} (CI {tv[a]['ci95'][0]}..{tv[a]['ci95'][1]}), float32 sigma* {out['verdicts'][f'alpha{a:g}_float32']['sigma_star']} (KL min {out['verdicts'][f'alpha{a:g}_float32']['kl_min']:.3f})" for a in alphas)
                                   + f". sigma = 0 deterministic: {all(tv[a]['deterministic_at_zero'] for a in alphas)}; KL minimum at sigma > 0 with CI excluding 0 for every alpha: {all(tv[a]['min_at_positive_sigma'] for a in alphas)}; one sigma* serves every alpha (within 2x): {bool(max(stars) <= 2 * max(min(stars), 1e-9))}"
                                   + ". Beyond the dial (0.3-1.0, diagnostic): " + "; ".join(f"alpha {a:g} sigma* {tv[a]['sigma_star_beyond_dial']}, KL {tv[a]['kl_min_beyond_dial']:.3f} -> {tv[a]['kl_at_max']:.2f} at 1.0{' (rises)' if tv[a]['rises_beyond_dial'] else ''}" for a in alphas) + "."}
    print("E14 verdict:", out["verdict"]["statement"])
    save_results("e14", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e14_sampling"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = np.array(res["sigmas"], float)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, col in zip(res["alphas"], ("C0", "C1", "C2")):
        for sub, ls in (("tiles", "-"), ("float32", "--")):
            sw = res["sweeps"][f"alpha{a:g}_{sub}"]
            m = [sw[str(s)]["kl"]["mean"] for s in res["sigmas"]]; e = [sw[str(s)]["kl"]["sem"] for s in res["sigmas"]]
            ax[0].errorbar(S, m, e, marker=".", ls=ls, color=col, capsize=2, label=f"α={a:g} {sub}")
            m = [sw[str(s)]["p_a_error"]["mean"] for s in res["sigmas"]]
            ax[1].plot(S, m, marker=".", ls=ls, color=col, label=f"α={a:g} {sub}")
            m = [sw[str(s)]["frac_other"]["mean"] for s in res["sigmas"]]
            ax[2].plot(S, m, marker=".", ls=ls, color=col, label=f"α={a:g} {sub}")
    for a_ in ax:
        a_.axvline(0.2, color="r", ls=":", lw=0.8)
    ax[0].set_xscale("symlog", linthresh=0.005); ax[0].set_xlabel("read noise σ_r"); ax[0].set_ylabel("KL(posterior ‖ retrieval frequencies)"); ax[0].legend(fontsize=6); ax[0].set_title("E14: sampling quality vs read noise", fontsize=9)
    ax[1].set_xscale("symlog", linthresh=0.005); ax[1].axhline(0, color="k", lw=0.5); ax[1].set_xlabel("σ_r"); ax[1].set_ylabel("p̂(a) − p(a | cue)"); ax[1].legend(fontsize=6); ax[1].set_title("calibration of the dominant pattern", fontsize=9)
    ax[2].set_xscale("symlog", linthresh=0.005); ax[2].set_xlabel("σ_r"); ax[2].set_ylabel("fraction retrieving a third pattern"); ax[2].legend(fontsize=6); ax[2].set_title("spurious retrievals", fontsize=9)
    fig.suptitle(res["verdict"]["statement"][:150], fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e14] figure {path}")


if __name__ == "__main__":
    main()
