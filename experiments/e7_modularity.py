"""E7: self-assembly of a module graph from eight generic tiles on two tasks.

Eight generic modules (32 rate neurons on a 32x32 recurrent tile each) start
unconnected.  Module 0 receives navigation input, module 1 receives recall
cues, and a readout reads modules 6 and 7.  Two tasks are interleaved:

* NAV    : input (cos th, sin th, cos v, sin v) -> population code of th + v,
* RECALL : a 16-bit pattern with half of its entries erased -> the clean pattern
           (eight stored patterns).

Weights learn with layer-local losses: every module has its own readout to
the task target and minimises its own loss with gradients that never leave
the module; the global readout uses the output error.  Node perturbation is
kept for structural decisions only (the route statistic).  Routes
between modules are created where ||<m xi_l z_k^T>|| exceeds theta_e within
an edge budget, and pruned when they stay weak.  We log Newman modularity Q
of the routing graph over training, the mutual information between every
module's activity and the task id, and *compositional transfer*: a new task
(bearing from the current heading to the location whose cue is shown,
phi_mu - th, composing RECALL and NAV) is evaluated zero-shot and after a
short fine-tune, for the grown graph and for two fixed layered baselines:
one with the same synapse budget trained for the same number of trials
("fixed"), and one trained for as many trials as match the grown graph's
total multiply-accumulates ("fixed_compute").
"""
from __future__ import annotations

import numpy as np

from common import base_parser, save_results, figure_path, Timer, sign_test
from bpusim.modules import ModuleGraph, GraphConfig, layered_routing
from bpusim.device import DeviceParams
from bpusim.attractor import circ_dist
from bpusim.noise import summarize

K, N_MOD, OUT = 8, 32, 16
PHI = 2 * np.pi * np.arange(OUT) / OUT
N_PATTERNS = 8
ERASE = 0.5


def popcode(angle):
    return np.cos(angle - PHI)


def decode(y):
    return float(np.arctan2(np.sum(y * np.sin(PHI)), np.sum(y * np.cos(PHI))))


class Tasks:
    def __init__(self, rng):
        self.rng = rng
        self.patterns = rng.choice([-1.0, 1.0], (N_PATTERNS, OUT))
        self.loc_angle = 2 * np.pi * np.arange(N_PATTERNS) / N_PATTERNS

    def sample(self, task):
        rng = self.rng
        if task == 0:   # NAV
            th = rng.uniform(-np.pi, np.pi); v = rng.uniform(-np.pi / 2, np.pi / 2)
            return {0: np.array([np.cos(th), np.sin(th), np.cos(v), np.sin(v)]), 1: np.zeros(OUT)}, popcode(th + v), {"angle": th + v}
        mu = int(rng.integers(N_PATTERNS))
        cue = self.patterns[mu].copy(); cue[rng.random(OUT) < ERASE] = 0.0
        if task == 1:   # RECALL
            return {0: np.zeros(4), 1: cue}, self.patterns[mu].copy(), {"mu": mu}
        th = rng.uniform(-np.pi, np.pi)   # COMPOSITE: bearing to the recalled location
        return {0: np.array([np.cos(th), np.sin(th), 0.0, 0.0]), 1: cue}, popcode(self.loc_angle[mu] - th), {"angle": self.loc_angle[mu] - th}

    @staticmethod
    def score(task, y, target, info):
        if task == 1:
            bits = float(np.mean(np.sign(y) == np.sign(target)))
            return {"bit_acc": bits, "correct": bits >= 14 / 16}
        err = abs(circ_dist(decode(y), info["angle"]))
        return {"abs_err": float(err), "correct": err < np.pi / 8}


def make_graph(kind, rng, cfg):
    if kind == "grown":
        return ModuleGraph(cfg, {0: 4, 1: OUT}, OUT, [0, 1], [6, 7], rng, DeviceParams())
    R = layered_routing(K, [0, 1], [2, 3, 4, 5], [6, 7])
    cfg_f = GraphConfig(**{**cfg.__dict__, "B_density": cfg.edge_budget * cfg.B_density / R.sum()})
    return ModuleGraph(cfg_f, {0: 4, 1: OUT}, OUT, [0, 1], [6, 7], rng, DeviceParams(), R_fixed=R)


def train(graph, tasks, task_ids, n_trials, structure, log_every=100):
    losses = {t: [] for t in set(task_ids)}
    for i in range(n_trials):
        t = task_ids[i % len(task_ids)]
        inp, target, info = tasks.sample(t)
        fwd = graph.forward(inp)
        loss = float(np.sum((fwd["y"] - target) ** 2))
        graph.learn(fwd, loss, task=t, structure=structure, target=target)
        losses[t].append(loss)
        if (i + 1) % log_every == 0:
            graph.log(np.mean([losses[t][-log_every // len(task_ids):][-1] for t in losses]))
    return {t: float(np.mean(v[-200:])) for t, v in losses.items()}


def evaluate(graph, tasks, task, n=200, record=False):
    scores, Z = [], []
    for _ in range(n):
        inp, target, info = tasks.sample(task)
        fwd = graph.forward(inp, record=record)
        scores.append(Tasks.score(task, fwd["y"], target, info))
        if record:
            Z.append(np.array(fwd["z_mean"]))   # (K, n)
    out = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
    out["n"] = n
    return (out, np.array(Z)) if record else out


def run_seed(seed, n_base, n_ft, cfg_kw):
    rng = np.random.default_rng([seed, 7])
    tasks = Tasks(np.random.default_rng([seed, 77]))
    res = {}
    grown_macs = None
    for kind in ("grown", "fixed", "fixed_compute"):
        tasks.rng = np.random.default_rng([seed, 77, 1])
        cfg = GraphConfig(n_per_module=N_MOD, n_modules=K, out_dim=OUT, **cfg_kw)
        g = make_graph("grown" if kind == "grown" else "fixed", np.random.default_rng([seed, 8]), cfg)
        nb, nf = n_base, n_ft
        if kind == "fixed_compute":
            # matched compute: the fixed graph gets as many trials as match the grown graph's MACs
            ratio = grown_macs / (g.macs_per_trial() * (n_base + n_ft))
            nb, nf = max(100, int(n_base * ratio)), max(50, int(n_ft * ratio))
        with Timer(f"  {kind}: base training ({nb} trials)"):
            base_loss = train(g, tasks, [0, 1], nb, structure=(kind == "grown"))
        ev_nav, Z_nav = evaluate(g, tasks, 0, record=True)
        ev_rec, Z_rec = evaluate(g, tasks, 1, record=True)
        Z = np.concatenate([Z_nav, Z_rec], 0); task_ids = np.array([0] * len(Z_nav) + [1] * len(Z_rec))
        mi = g.module_task_information([Z[:, k, :] for k in range(K)], task_ids)
        Q, labels = g.modularity()
        zero_shot = evaluate(g, tasks, 2)
        with Timer(f"  {kind}: composite fine-tune ({nf} trials)"):
            ft_loss = train(g, tasks, [2], nf, structure=(kind == "grown"))
        after = evaluate(g, tasks, 2)
        if kind == "grown":
            grown_macs = g.macs_total
        res[kind] = {"base_loss": base_loss, "nav": ev_nav, "recall": ev_rec, "Q": Q, "communities": labels,
                     "edges": int(g.n_edges), "routes": [[int(a), int(b)] for a, b in zip(*np.nonzero(g.R))],
                     "n_synapses": g.n_synapses, "mi_bits": [m["mi_bits"] for m in mi], "mi_shuffle": [m["mi_shuffle"] for m in mi],
                     "composite_zero_shot": zero_shot, "composite_after_finetune": after, "composite_ft_loss": ft_loss,
                     "trials": [nb, nf], "macs_total": int(g.macs_total), "history": g.history}
        print(f"  seed {seed} {kind}: edges={g.n_edges} syn={g.n_synapses} Q={Q:.3f} nav acc={ev_nav['correct']:.2f} (err {ev_nav['abs_err']:.2f}) "
              f"recall bits={ev_rec['bit_acc']:.2f} | composite zero-shot {zero_shot['correct']:.2f} -> after ft {after['correct']:.2f} (err {after['abs_err']:.2f}) | MI max {max(res[kind]['mi_bits']):.2f} bits")
    res["transfer_gap"] = res["grown"]["composite_after_finetune"]["correct"] - res["fixed"]["composite_after_finetune"]["correct"]
    res["transfer_gap_matched_compute"] = res["grown"]["composite_after_finetune"]["correct"] - res["fixed_compute"]["composite_after_finetune"]["correct"]
    return res


def main():
    ap = base_parser("E7 modular self-assembly", default_seeds=3)
    ap.add_argument("--base-trials", type=int, default=9000)
    ap.add_argument("--ft-trials", type=int, default=3000)
    ap.add_argument("--edge-budget", type=int, default=12)
    ap.add_argument("--theta-e", type=float, default=0.02)
    ap.add_argument("--rule", type=str, default="local", choices=["local", "np"])
    ap.add_argument("--eta-local", type=float, default=0.01)
    ap.add_argument("--procs", type=int, default=1)
    args = ap.parse_args()
    seeds = 10 if args.full else args.seeds
    cfg_kw = {"edge_budget": args.edge_budget, "theta_e": args.theta_e, "rule": args.rule, "eta_local": args.eta_local}
    if args.procs > 1:
        import multiprocessing as mp
        with mp.Pool(args.procs) as pool:
            runs = pool.starmap(run_seed, [(seed, args.base_trials, args.ft_trials, cfg_kw) for seed in range(seeds)])
    else:
        runs = []
        for seed in range(seeds):
            with Timer(f"seed {seed}"):
                runs.append(run_seed(seed, args.base_trials, args.ft_trials, cfg_kw))
    Q_final = [r["grown"]["Q"] for r in runs]
    Q_hist = np.array([r["grown"]["history"]["Q"] for r in runs])
    gap = [r["transfer_gap"] for r in runs]; gap_c = [r["transfer_gap_matched_compute"] for r in runs]
    p_gap, wins = sign_test([r["fixed"]["composite_after_finetune"]["correct"] for r in runs],
                            [r["grown"]["composite_after_finetune"]["correct"] for r in runs])
    mi_grown = np.array([r["grown"]["mi_bits"] for r in runs]); mi_sh = np.array([r["grown"]["mi_shuffle"] for r in runs])
    specialised = float(np.mean((mi_grown - mi_sh) > 0.1))
    kinds = ("grown", "fixed", "fixed_compute")
    out = {"seeds": seeds, "base_trials": args.base_trials, "ft_trials": args.ft_trials, "rule": args.rule, "n_per_module": N_MOD, "runs": runs,
           "summary": {"Q_final_grown": summarize(Q_final), "Q_over_time_mean": Q_hist.mean(0).tolist() if Q_hist.size else [],
                       "Q_rises": bool(Q_hist.size and Q_hist.mean(0)[-1] > Q_hist.mean(0)[0] + 0.05),
                       "nav_acc": {k: summarize([r[k]["nav"]["correct"] for r in runs]) for k in kinds},
                       "recall_bit_acc": {k: summarize([r[k]["recall"]["bit_acc"] for r in runs]) for k in kinds},
                       "composite_after_ft": {k: summarize([r[k]["composite_after_finetune"]["correct"] for r in runs]) for k in kinds},
                       "composite_zero_shot": {k: summarize([r[k]["composite_zero_shot"]["correct"] for r in runs]) for k in kinds},
                       "transfer_gap": summarize(gap), "transfer_gap_matched_compute": summarize(gap_c), "transfer_sign_test_p": p_gap,
                       "trials_fixed_compute": [r["fixed_compute"]["trials"] for r in runs],
                       "fraction_modules_task_informative": specialised,
                       "mi_bits_per_module_mean": mi_grown.mean(0).tolist(), "mi_shuffle_mean": mi_sh.mean(0).tolist()}}
    s = out["summary"]
    out["verdict"] = {"statement": (
        f"Q(final, grown) = {s['Q_final_grown']['mean']:.2f} +- {s['Q_final_grown']['sem']:.2f} "
        f"({'rises' if s['Q_rises'] else 'does not rise'} over training); "
        f"{100 * specialised:.0f}% of modules carry > 0.1 bit about the task above shuffle; "
        f"compositional transfer gap (grown - fixed) = {s['transfer_gap']['mean']:+.2f} at matched synapses (sign test p = {p_gap:.2g}), "
        f"{s['transfer_gap_matched_compute']['mean']:+.2f} at matched compute; "
        f"base-task accuracy: nav {s['nav_acc']['grown']['mean']:.2f} / {s['nav_acc']['fixed']['mean']:.2f}, "
        f"recall bits {s['recall_bit_acc']['grown']['mean']:.2f} / {s['recall_bit_acc']['fixed']['mean']:.2f} (grown / fixed)")}
    print("E7 verdict:", out["verdict"]["statement"])
    save_results("e7", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e7_modularity"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
    for r in res["runs"]:
        h = r["grown"]["history"]
        ax[0].plot(h["trial"], h["Q"], color="C0", alpha=0.6)
        ax[0].plot(h["trial"], np.array(h["edges"]) / max(1, max(h["edges"])), color="C1", alpha=0.3)
    ax[0].set_xlabel("trial"); ax[0].set_ylabel("modularity Q (blue), edges/max (orange)"); ax[0].set_title("E7: Q over training (grown graphs)")
    r = res["runs"][0]
    G = nx.DiGraph(); G.add_nodes_from(range(K)); G.add_edges_from([tuple(e) for e in r["grown"]["routes"]])
    pos = nx.circular_layout(G)
    cols = [f"C{c}" for c in r["grown"]["communities"]]
    nx.draw_networkx(G, pos, ax=ax[1], node_color=cols, with_labels=True, arrows=True, node_size=500)
    ax[1].set_title(f"grown routing, seed 0 (Q={r['grown']['Q']:.2f}); in: 0,1  out: 6,7", fontsize=9); ax[1].axis("off")
    mi = np.array(res["summary"]["mi_bits_per_module_mean"]); sh = np.array(res["summary"]["mi_shuffle_mean"])
    ax[2].bar(np.arange(K) - 0.2, mi, 0.4, label="I(Z_k; task)"); ax[2].bar(np.arange(K) + 0.2, sh, 0.4, label="shuffle")
    ax[2].set_xlabel("module"); ax[2].set_ylabel("bits"); ax[2].legend(fontsize=8); ax[2].set_title("task information per module (grown)")
    s = res["summary"]
    labels = ["nav", "recall bits", "composite 0-shot", "composite after ft"]
    gv = [s["nav_acc"]["grown"]["mean"], s["recall_bit_acc"]["grown"]["mean"], s["composite_zero_shot"]["grown"]["mean"], s["composite_after_ft"]["grown"]["mean"]]
    fv = [s["nav_acc"]["fixed"]["mean"], s["recall_bit_acc"]["fixed"]["mean"], s["composite_zero_shot"]["fixed"]["mean"], s["composite_after_ft"]["fixed"]["mean"]]
    x = np.arange(4); ax[3].bar(x - 0.2, gv, 0.4, label="grown"); ax[3].bar(x + 0.2, fv, 0.4, label="fixed layered")
    ax[3].set_xticks(x); ax[3].set_xticklabels(labels, fontsize=7, rotation=15); ax[3].set_ylim(0, 1); ax[3].legend(fontsize=8)
    ax[3].set_title(f"transfer gap {s['transfer_gap']['mean']:+.2f}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e7] figure {path}")


if __name__ == "__main__":
    main()
