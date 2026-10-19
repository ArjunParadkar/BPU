"""E12: the read-noise dial as the exploration noise of node-perturbation RL (variability programme).

A 2-layer policy network (input -> 32 hidden units with a tonic drive, clipped linear ->
K actions; both layers on tiles) learns two tasks with the L3 three-factor rule (node
perturbation):

  bandit     contextual bandit of features: context x in R^8 (Gaussian), K = 4 arms, reward 1
             if the chosen arm is argmax(M^T x) for a hidden random map M, else 0.
             Criterion: running accuracy (200 decisions) >= 0.7 (softmax REINFORCE reaches 0.91).
  grid       4x4 grid navigation: one-hot state, 4 actions, fixed goal, +1 at the goal minus
             0.02 per step, episodes of at most 16 steps.
             Criterion: mean steps-to-goal over 50 episodes <= 1.5x the optimal mean.

Exploration comes ONLY from the tiles' read noise (the MODULATE primitive, ``set_noise``): the
policy is greedy on a noisy read.  Every context (bandit) or start state (grid) is presented
TWICE, "two renditions", A and B, with independent read noise; the per-unit noise sample of a
read is its deviation from the noise-free read (the periphery's estimate of the mean; in
hardware an averaged read), the eligibility of a synapse is sum_t x_pre(t) xi_post(t) over
the rendition, and the rule is  dW = eta (R_A - R_B) (e_A - e_B)  — a two-sided node
perturbation: when the two renditions choose the same actions the update is exactly zero, and
the reward difference is its own baseline.  (A single-rendition rule with a running-mean or
linear-critic baseline was tried first and failed on the bandit: a greedy policy's chosen arm
has a positively biased noise sample, the baseline mismatch turns that into a drift, and the
accuracy ended at or below chance at every sigma up to 1.0; measured, see DECISIONS.md.)
Presynaptic normalisation: each eligibility is divided by |x_pre|^2 (a layer-wide activity
signal), so the step in the post-synaptic activity is eta (R_A - R_B) xi.

Noise-dial schedule arms: fixed sigma in {0.01, 0.04, 0.1} (+ 0.02 as an extra point for the
exponent fit), annealed 0.1 -> 0.01 (geometric over the budget), reward-gated: sigma = 0.1
while the reward is not rising (short-minus-long running mean <= 0), 0.01 while it rises; every
change goes through set_noise and is logged as a MODULATE event.  Reference: the same algorithm
on float32 weights with the same multiplicative Gaussian read noise injected (FloatTile).
Physics to keep in mind: multiplicative read noise flips a greedy decision only when the margin
is within ~sigma of the scores, so the fraction of renditions that differ (logged as
``flip_rate``) is the actual exploration rate; it falls as the learned margins grow.

Hypotheses: reward-gated beats the fixed arms (fewer trials to criterion, paired over seeds);
learning speed (1 / trials-to-criterion) scales as sigma^2 at small sigma (log-log exponent with
a bootstrap CI).  5 seeds per cell; trial budget --trials (one trial = one pair of renditions).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import zlib

import numpy as np

from common import base_parser, save_results, figure_path, Timer, sign_test
from bpusim.device import DeviceParams
from bpusim.tile import Tile, FloatTile
from bpusim.noise import summarize

FIXED = [0.01, 0.02, 0.04, 0.1]
ARMS = ["fixed0.01", "fixed0.02", "fixed0.04", "fixed0.1", "annealed", "gated"]
SIG_HI, SIG_LO = 0.1, 0.01
N_H = 32
TONIC = 0.25
BANDIT_CRIT = 0.7


class Policy:
    """x -> h = clip(W1^T x + tonic, 0, 2) -> z = W2^T h, both layers on a substrate; greedy on one noisy read."""

    def __init__(self, d_in, K, rng, substrate, sigma, eta, params=None):
        self.rng = rng; self.eta = eta
        mk = (lambda i, o: FloatTile(i, o, rng)) if substrate == "float32" else (lambda i, o: Tile(i, o, rng, params))
        self.W1 = mk(d_in, N_H); self.W2 = mk(N_H, K)
        self.W1.program(rng.normal(0, 0.3, (d_in, N_H)) / np.sqrt(d_in))
        self.W2.program(rng.normal(0, 0.3, (N_H, K)) / np.sqrt(N_H))
        self.set_noise(sigma)
        self.e1 = np.zeros((d_in, N_H)); self.e2 = np.zeros((N_H, K))
        self.n_dec = 0; self.n_flip = 0

    def set_noise(self, sigma):
        self.sigma = self.W1.set_noise(sigma); self.W2.set_noise(sigma)

    def clean(self, x):
        h = np.clip(x @ self.W1.W + TONIC, 0, 2); return h, h @ self.W2.W

    def act(self, x):
        """One noisy read decides; the noise sample per unit is the deviation from the noise-free read."""
        h = np.clip(self.W1.read(x) + TONIC, 0, 2); z = self.W2.read(h)
        h0, z0 = self.clean(x)
        a = int(np.argmax(z)); self.n_dec += 1; self.n_flip += int(a != int(np.argmax(z0)))
        self.e1 += np.outer(x, h - h0) / max(float(x @ x), 1.0)
        self.e2 += np.outer(h0, z - z0) / max(float(h0 @ h0), 1.0)
        return a

    def take(self):
        e = (self.e1.copy(), self.e2.copy()); self.e1[:] = 0.0; self.e2[:] = 0.0; return e

    def learn_pair(self, dR, eA, eB):
        if dR == 0.0:
            return
        self.W1.accumulate(self.eta * dR * (eA[0] - eB[0])); self.W2.accumulate(self.eta * dR * (eA[1] - eB[1]))

    @property
    def n_writes(self):
        return self.W1.n_writes + self.W2.n_writes

    @property
    def n_devices(self):
        return 2 * (self.W1.W.size + self.W2.W.size)


class Bandit:
    def __init__(self, rng, d=8, K=4):
        self.rng, self.d, self.K = rng, d, K
        self.M = rng.normal(0, 1, (d, K))
        self.optimal_mean = 1.0

    def new_context(self):
        self.x = self.rng.normal(0, 1, self.d); self.best = int(np.argmax(self.M.T @ self.x))

    def rendition(self, pol):
        a = pol.act(self.x)
        return float(a == self.best), 1


class Grid:
    def __init__(self, rng, n=4, max_steps=16):
        self.rng, self.n, self.max_steps = rng, n, max_steps
        self.goal = (n - 1, n - 1)
        self.K = 4
        d = [abs(i - self.goal[0]) + abs(j - self.goal[1]) for i in range(n) for j in range(n) if (i, j) != self.goal]
        self.optimal_mean = float(np.mean(d))

    def new_context(self):
        n = self.n
        while True:
            self.start = (int(self.rng.integers(n)), int(self.rng.integers(n)))
            if self.start != self.goal:
                break

    def rendition(self, pol):
        n = self.n; s = self.start
        for t in range(1, self.max_steps + 1):
            x = np.zeros(n * n); x[s[0] * n + s[1]] = 1.0
            a = pol.act(x)
            di, dj = ((1, 0), (-1, 0), (0, 1), (0, -1))[a]
            s = (min(max(s[0] + di, 0), n - 1), min(max(s[1] + dj, 0), n - 1))
            if s == self.goal:
                return 1.0 - 0.02 * t, t
        return 0.0, self.max_steps


def schedule(arm, k, n_trials, trend):
    if arm.startswith("fixed"):
        return float(arm[5:])
    if arm == "annealed":
        return SIG_HI * (SIG_LO / SIG_HI) ** (k / max(n_trials - 1, 1))
    return SIG_HI if trend <= 0.0 else SIG_LO           # gated: explore while the reward is not rising


def run_one(job):
    task, arm, substrate, seed, n_trials, eta = job
    rng = np.random.default_rng([seed, zlib.crc32(task.encode()) % 1000, zlib.crc32(arm.encode()) % 1000, substrate == "tiles"])
    env = Bandit(rng) if task == "bandit" else Grid(rng)
    d_in = env.d if task == "bandit" else env.n * env.n
    params = DeviceParams(sigma_r=0.0)           # d2d 0.03, c2c 0.03 (device defaults); read noise = the dial
    pol = Policy(d_in, env.K, rng, substrate, schedule(arm, 0, n_trials, 0.0), eta, params)
    r_short = None; r_long = None
    rewards = np.empty(n_trials); steps = np.empty(n_trials); sig = np.empty(n_trials); flips = np.empty(n_trials)
    crit_at = None
    win_r, win_s = 100, 25                           # windows in pairs: 200 decisions / 50 episodes
    for k in range(n_trials):
        trend = 0.0 if r_short is None else r_short - r_long
        s_new = schedule(arm, k, n_trials, trend)
        if s_new != pol.sigma:
            pol.set_noise(s_new)
        sig[k] = pol.sigma
        env.new_context()
        f0 = pol.n_flip
        rA, nA = env.rendition(pol); eA = pol.take()
        rB, nB = env.rendition(pol); eB = pol.take()
        pol.learn_pair(rA - rB, eA, eB)
        r = 0.5 * (rA + rB); rewards[k] = r; steps[k] = 0.5 * (nA + nB); flips[k] = (pol.n_flip - f0) / max(nA + nB, 1)
        if r_short is None:
            r_short = r_long = r
        r_short += (r - r_short) / 20.0; r_long += (r - r_long) / 200.0
        if crit_at is None:
            if task == "bandit" and k + 1 >= win_r and rewards[k + 1 - win_r:k + 1].mean() >= BANDIT_CRIT:
                crit_at = 2 * (k + 1)
            if task == "grid" and k + 1 >= win_s and steps[k + 1 - win_s:k + 1].mean() <= 1.5 * env.optimal_mean:
                crit_at = 2 * (k + 1)
    n_mod = pol.W1.n_modulate
    final = float(rewards[-win_r:].mean()) if task == "bandit" else float(steps[-win_s:].mean())
    q = max(1, n_trials // 100)
    return {"task": task, "arm": arm, "substrate": substrate, "seed": seed, "trials_to_criterion": crit_at, "reached": crit_at is not None,
            "final_metric": final, "optimal_mean_steps": env.optimal_mean if task == "grid" else None, "flip_rate": float(flips.mean()),
            "reward_curve": rewards[:n_trials // q * q].reshape(-1, q).mean(1).tolist(), "steps_curve": steps[:n_trials // q * q].reshape(-1, q).mean(1).tolist(),
            "flip_curve": flips[:n_trials // q * q].reshape(-1, q).mean(1).tolist(),
            "sigma_curve": sig[::q].tolist(), "n_modulate": n_mod, "writes_per_device_total": pol.n_writes / pol.n_devices}


def fit_exponent(points):
    """points: list of (sigma, speed); log-log slope."""
    P = [(s, v) for s, v in points if v > 0]
    if len(P) < 2:
        return None
    x = np.log([p[0] for p in P]); y = np.log([p[1] for p in P])
    return float(np.polyfit(x, y, 1)[0])


def main():
    ap = base_parser("E12 noise dial as exploration (node perturbation)", default_seeds=5)
    ap.add_argument("--trials", type=int, default=10000, help="pairs of renditions per run (20000 decisions on the bandit)")
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--tasks", type=str, default="bandit,grid")
    ap.add_argument("--arms", type=str, default=",".join(ARMS))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    seeds = args.seeds
    if args.smoke:
        args.trials, seeds = 600, 1
    tasks = args.tasks.split(","); arms = args.arms.split(",")
    jobs = [(t, a, s, seed, args.trials, args.eta) for t in tasks for a in arms for s in ("float32", "tiles") for seed in range(seeds)]
    with Timer(f"E12 {len(jobs)} runs"):
        if args.procs > 1:
            with mp.Pool(args.procs) as pool:
                runs = pool.map(run_one, jobs)
        else:
            runs = [run_one(j) for j in jobs]
    budget = 2 * args.trials
    S = {}; H = {}
    for t in tasks:
        S[t] = {}
        for s in ("float32", "tiles"):
            S[t][s] = {}
            for a in arms:
                rs = [r for r in runs if r["task"] == t and r["substrate"] == s and r["arm"] == a]
                ttc = [r["trials_to_criterion"] if r["reached"] else budget for r in rs]
                S[t][s][a] = {"trials_to_criterion": summarize(ttc), "median_trials": float(np.median(ttc)), "reached_frac": float(np.mean([r["reached"] for r in rs])),
                              "final_metric": summarize([r["final_metric"] for r in rs]), "n_modulate": summarize([r["n_modulate"] for r in rs]), "flip_rate": summarize([r["flip_rate"] for r in rs]),
                              "writes_per_device_total": summarize([r["writes_per_device_total"] for r in rs]), "per_seed": ttc}
            # hypothesis 1: gated beats the best fixed arm (paired over seeds, sign test on trials-to-criterion)
            fixed = [a for a in arms if a.startswith("fixed")]
            if "gated" in arms and fixed:
                best_fixed = min(fixed, key=lambda a: S[t][s][a]["median_trials"])
                g = S[t][s]["gated"]["per_seed"]; f = S[t][s][best_fixed]["per_seed"]
                p, wins = sign_test(g, f)
                H[f"{t}|{s}|gated_vs_best_fixed"] = {"best_fixed": best_fixed, "gated_median": float(np.median(g)), "fixed_median": float(np.median(f)),
                                                      "seeds_gated_faster": wins, "sign_test_p": p, "gated_wins": bool(np.median(g) < np.median(f) and p < 0.05)}
            # hypothesis 2: speed ~ sigma^2 at small sigma (sigma <= 0.04), exponent with bootstrap over seeds
            small = [a for a in fixed if float(a[5:]) <= 0.04]
            if len(small) >= 2:
                per_seed = []
                for seed in range(seeds):
                    pts = [(float(a[5:]), 1.0 / S[t][s][a]["per_seed"][seed]) for a in small]
                    per_seed.append(fit_exponent(pts))
                mean_pts = [(float(a[5:]), 1.0 / np.mean(S[t][s][a]["per_seed"])) for a in small]
                rng = np.random.default_rng(0); boots = []
                for _ in range(2000):
                    idx = rng.integers(0, seeds, seeds)
                    boots.append(fit_exponent([(float(a[5:]), 1.0 / np.mean([S[t][s][a]["per_seed"][i] for i in idx])) for a in small]))
                boots = [b for b in boots if b is not None]
                H[f"{t}|{s}|speed_exponent"] = {"sigmas": [float(a[5:]) for a in small], "exponent": fit_exponent(mean_pts), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
                                                "per_seed": per_seed, "consistent_with_2": bool(boots and np.percentile(boots, 2.5) <= 2.0 <= np.percentile(boots, 97.5))}
    parts = []
    for t in tasks:
        for s in ("tiles", "float32"):
            line = f"{t}/{s}: " + ", ".join(f"{a} {S[t][s][a]['median_trials']:.0f} ({S[t][s][a]['reached_frac']:.0%} reached, final {S[t][s][a]['final_metric']['mean']:.2f}, flips {S[t][s][a]['flip_rate']['mean']:.3f})" for a in arms)
            h1 = H.get(f"{t}|{s}|gated_vs_best_fixed"); h2 = H.get(f"{t}|{s}|speed_exponent")
            if h1:
                line += f"; gated vs {h1['best_fixed']}: {h1['gated_median']:.0f} vs {h1['fixed_median']:.0f}, {h1['seeds_gated_faster']}/{seeds} seeds faster, p={h1['sign_test_p']:.2g}" + (" GATED WINS" if h1["gated_wins"] else "")
            if h2 and h2["exponent"] is not None:
                line += f"; speed ∝ σ^{h2['exponent']:.2f}" + (f" (CI {h2['ci95'][0]:.2f}..{h2['ci95'][1]:.2f}{', consistent with 2' if h2['consistent_with_2'] else ''})" if h2["ci95"] else "")
            parts.append(line)
    gated_wins = [H[k]["gated_wins"] for k in H if k.endswith("gated_vs_best_fixed") and "|tiles|" in k]
    out = {"tasks": tasks, "arms": arms, "seeds": seeds, "trials": args.trials, "eta": args.eta, "sigma_hi_lo": [SIG_HI, SIG_LO], "runs": runs, "summary": S, "hypotheses": H,
           "verdict": {"gated_beats_fixed_on_tiles": bool(gated_wins and all(gated_wins)), "n_tasks_gated_wins_tiles": int(sum(gated_wins)),
                       "statement": ("SMOKE RUN. " if args.smoke else "") + f"trials to criterion (median over {seeds} seeds, budget {args.trials}): " + " || ".join(parts)}}
    print("E12 verdict:", out["verdict"]["statement"])
    save_results("e12", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e12_exploration"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = res["tasks"]; arms = res["arms"]; S = res["summary"]
    fig, axes = plt.subplots(1, 2 * len(tasks), figsize=(5.2 * 2 * len(tasks), 4))
    for i, t in enumerate(tasks):
        ax = axes[2 * i]
        x = np.arange(len(arms)); w = 0.38
        for j, (s, col) in enumerate((("float32", "k"), ("tiles", "C0"))):
            m = [S[t][s][a]["median_trials"] for a in arms]
            ax.bar(x + (j - 0.5) * w, m, w, color=col, alpha=0.8, label=s)
        ax.set_xticks(x); ax.set_xticklabels(arms, rotation=25, fontsize=7); ax.set_ylabel("trials to criterion (median)"); ax.set_yscale("log"); ax.legend(fontsize=7)
        ax.set_title(f"{t}: steps to criterion per noise schedule", fontsize=9)
        ax = axes[2 * i + 1]
        for a, ls in (("fixed0.04", "-"), ("gated", "--"), ("annealed", ":")):
            if a not in arms:
                continue
            for s, col in (("float32", "k"), ("tiles", "C0")):
                rs = [r for r in res["runs"] if r["task"] == t and r["substrate"] == s and r["arm"] == a]
                key = "reward_curve" if t == "bandit" else "steps_curve"
                c = np.mean([r[key] for r in rs], 0); xx = np.linspace(0, 2 * res["trials"], len(c))
                ax.plot(xx, c, ls=ls, color=col, label=f"{s} {a}")
        ax.set_xlabel("decisions / episodes"); ax.set_ylabel("reward" if t == "bandit" else "steps to goal"); ax.legend(fontsize=6); ax.set_title(f"{t}: learning curves", fontsize=9)
    fig.suptitle(res["verdict"]["statement"][:150], fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e12] figure {path}")


if __name__ == "__main__":
    main()
