"""L8: modules, routing and the self-assembling module graph.

A :class:`Module` is a group of rate neurons with its own recurrent tile and
mask (a generic tile; attractor or memory types reuse the same interface
through ``F``).  A :class:`ModuleGraph` holds K modules, a boolean routing
matrix R (R[k, l] = module k projects to module l) and, for every active
route, a projection tile B_kl.  External inputs enter designated input
modules through fixed random tiles; a readout tile reads designated output
modules.

All synapses learn with the same local three-factor rule (node perturbation,
Lesson 4.2): every neuron carries its own noise xi, the eligibility of a
synapse is sum_t x_pre(t) xi_post(t) over a trial, a single broadcast
modulator m = -(L - L_bar) gates the write, dW = eta m e, and writes below
theta_w are held in a surprise gate.  The same estimate one level up
decides structure: a route k -> l is created when
||< m xi_l z_k^T >||_F exceeds theta_e (within an edge budget), seeded along
the sign of that correlation; routes whose tile norm stays small are
pruned.  Modularity Q and per-module mutual information with the task id
are computed by :mod:`bpusim.metrics`.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .device import DeviceParams
from .tile import Tile
from .plasticity import SurpriseGate
from .metrics import best_modularity, task_information


@dataclass
class GraphConfig:
    n_per_module: int = 16
    n_modules: int = 8
    dt_over_tau: float = 0.5        # relaxation step per iteration
    n_relax: int = 8                # iterations per trial
    sigma_xi: float = 0.1           # neuron noise (the node-perturbation probe)
    explore_one: bool = True        # per trial, only one randomly chosen module carries the probe noise
                                    # (a global gating signal; raises the estimate's SNR by ~sqrt(K))
    r_max: float = 2.0
    I_tonic: float = 0.25           # background drive so every module has baseline activity to probe
    eta: float = 1e-4               # three-factor learning rate; the applied rate is eta / sigma_xi**2
                                    # because the node-perturbation estimate <m x xi> = sigma_xi**2 * gradient
                                    # (node perturbation has SNR ~ 1/sqrt(#perturbed units) per trial; at
                                    # 3e-3 the hidden updates randomised the features faster than the readout
                                    # could track them and accuracy fell to chance; see DECISIONS.md)
    theta_w: float = 0.02           # write threshold (~ half a pulse from w = 0; smaller requests round to
                                    # no pulse and would be lost without the residual-keeping gate)
    eta_readout: float = 0.03       # readout uses the delta rule with the output error as the (vector)
                                    # modulator: dW_out = eta_readout z (target - y)^T, a local three-factor
                                    # rule because the error is available at the output neurons
    theta_e: float = 0.02           # route creation threshold on ||<m xi_l z_k^T>||_F
    edge_budget: int = 12
    tau_C: float = 200.0            # running average length (trials) for the route statistic
    tau_L: float = 50.0             # running average length for the loss baseline
    B_density: float = 0.5          # synapse density of a new projection tile
    seed_scale: float = 0.5         # |initial weight| of a seeded route (~1/sqrt(n * density) keeps the
                                    # layer gain near one so activity reaches the output modules)
    prune_norm: float = 0.02        # prune a route whose mean |W| stays below this ...
    prune_patience: int = 300       # ... for this many trials
    recurrent_density: float = 0.3
    recurrent_scale: float = 0.2
    rule: str = "local"             # "local": every module minimises its own readout loss (layer-local
                                    # gradient inside the module, no gradient between modules); node
                                    # perturbation is used for structural decisions only.  "np": the
                                    # original node-perturbation weight learning.
    eta_local: float = 0.001        # layer-local learning rate (0.01 saturated 32-neuron modules; 0.001 gives nav 0.97 on the fixed graph)
    out_dim: int = 16


def _scale_for(n: int) -> float:
    """Weight scales are set for 16-neuron modules; larger modules sum more inputs, so scale by
    sqrt(16 / n) to keep the layer gain (and the fraction of saturated units) the same.  Without
    this, 32-neuron modules saturated at r_max and learned nothing (nav accuracy 0.18 vs 0.94)."""
    return float(np.sqrt(16.0 / n))


class Module:
    def __init__(self, idx: int, n: int, rng: np.random.Generator, params: DeviceParams, cfg: GraphConfig):
        self.idx, self.n, self.rng, self.cfg = idx, n, rng, cfg
        mask = rng.random((n, n)) < cfg.recurrent_density
        np.fill_diagonal(mask, False)
        self.W = Tile(n, n, rng, params, mask=mask)
        sc = cfg.recurrent_scale * _scale_for(n)
        self.W.program(np.where(mask, rng.uniform(-sc, sc, (n, n)), 0.0))
        self.gate = SurpriseGate((n, n), cfg.theta_w)
        self.z = np.zeros(n)
        self.xi = np.zeros(n)
        self.u = np.zeros(n)
        # local readout (layer-local loss): y_k = R_k^T z_k
        self.R = Tile(n, cfg.out_dim, rng, params)
        self.R.program(rng.uniform(-0.1, 0.1, (n, cfg.out_dim)))
        self.R_gate = SurpriseGate((n, cfg.out_dim), 0.2 * cfg.theta_w)

    def reset(self):
        self.z[:] = 0.0; self.xi[:] = 0.0


class ModuleGraph:
    def __init__(self, cfg: GraphConfig, in_dims: dict, out_dim: int, input_modules: list, output_modules: list,
                 rng: np.random.Generator, params: DeviceParams | None = None, R_fixed: np.ndarray | None = None):
        """in_dims: {module index: external input dimension}; R_fixed: a frozen routing matrix (baseline)."""
        self.cfg, self.rng = cfg, rng
        self.params = params or DeviceParams()
        cfg.out_dim = int(out_dim)          # local readouts share the task's output dimension
        K, n = cfg.n_modules, cfg.n_per_module
        self.K, self.n = K, n
        self.modules = [Module(k, n, rng, self.params, cfg) for k in range(K)]
        self.input_modules = list(input_modules); self.output_modules = list(output_modules)
        self.U = {}; self.U_gate = {}
        for k, d in in_dims.items():
            t = Tile(d, n, rng, self.params)
            t.program(rng.uniform(-1, 1, (d, n)) * 0.8)   # input fan-in d is task-fixed, not n
            self.U[k] = t; self.U_gate[k] = SurpriseGate((d, n), cfg.theta_w)
        self.out_dim = out_dim
        self.readout = Tile(n * len(output_modules), out_dim, rng, self.params)
        self.readout.program(rng.uniform(-0.1, 0.1, (n * len(output_modules), out_dim)))
        self.readout_gate = SurpriseGate(self.readout.shape, 0.2 * cfg.theta_w)
        self.R = np.zeros((K, K), bool)
        self.B = {}
        self.B_gate = {}
        self.C = np.zeros((K, K))          # route statistic ||<m xi_l z_k^T>||
        self.Cmat = np.zeros((K, K, n, n))  # the running correlation matrices themselves
        self.low_count = np.zeros((K, K), int)
        self.fixed = R_fixed is not None
        if self.fixed:
            for k, l in zip(*np.nonzero(R_fixed)):
                self._create_route(int(k), int(l), seed=None)
        self.L_bar = {}
        self.history = {"Q": [], "edges": [], "trial": [], "loss": []}
        self.trial = 0
        self.macs_total = 0

    def macs_per_trial(self) -> int:
        """Multiply-accumulates of one trial: recurrent + route + input reads per relaxation step, readout."""
        n = self.n
        per_step = sum(m.W.n_active for m in self.modules) + sum(t.n_active for t in self.B.values()) + sum(t.n_active for t in self.U.values())
        return int(self.cfg.n_relax * per_step + self.readout.n_active)

    # ----------------------------------------------------------------- routes
    def _create_route(self, k, l, seed=None):
        n = self.n
        mask = self.rng.random((n, n)) < self.cfg.B_density
        t = Tile(n, n, self.rng, self.params, mask=mask)
        sc = self.cfg.seed_scale * _scale_for(n)
        if seed is None:
            W0 = self.rng.uniform(-sc, sc, (n, n))
        else:
            W0 = sc * np.sign(seed) * (np.abs(seed) / max(np.abs(seed).max(), 1e-12))
        t.program(np.where(mask, W0, 0.0))
        self.B[(k, l)] = t
        self.B_gate[(k, l)] = SurpriseGate((n, n), self.cfg.theta_w)
        self.R[k, l] = True
        self.low_count[k, l] = 0

    def _remove_route(self, k, l):
        del self.B[(k, l)]; del self.B_gate[(k, l)]
        self.R[k, l] = False

    @property
    def n_edges(self):
        return int(self.R.sum())

    @property
    def n_synapses(self):
        s = sum(m.W.n_active for m in self.modules) + sum(t.n_active for t in self.B.values())
        return int(s)

    # ---------------------------------------------------------------- forward
    def forward(self, inputs: dict, record: bool = False):
        """Relax the graph on one trial.  inputs: {module index: vector}.

        Returns output y, and the per-synapse eligibilities needed for learning.
        """
        cfg = self.cfg
        for m in self.modules:
            m.reset()
        n = self.n
        elig_W = [np.zeros((n, n)) for _ in self.modules]
        elig_B = {kl: np.zeros((n, n)) for kl in self.B}
        elig_U = {k: np.zeros(self.U[k].shape) for k in self.U}
        xi_sum = [np.zeros(n) for _ in self.modules]
        z_sum = [np.zeros(n) for _ in self.modules]
        trace = [] if record else None
        a = cfg.dt_over_tau
        probe = int(self.rng.integers(self.K)) if cfg.explore_one else None
        for it in range(cfg.n_relax):
            z_prev = [m.z.copy() for m in self.modules]
            for l, m in enumerate(self.modules):
                u = m.W.read(z_prev[l])
                if l in self.U and l in inputs:
                    u = u + self.U[l].read(inputs[l])
                for k in range(self.K):
                    if self.R[k, l]:
                        u = u + self.B[(k, l)].read(z_prev[k])
                xi = cfg.sigma_xi * self.rng.standard_normal(n) if (probe is None or probe == l) else np.zeros(n)
                target = np.clip(u + cfg.I_tonic, 0.0, cfg.r_max)
                m.z = (1 - a) * m.z + a * (target + xi)
                m.xi = xi
                m.u = u + cfg.I_tonic
            # eligibilities: pre activity x post noise (node perturbation)
            for l, m in enumerate(self.modules):
                elig_W[l] += np.outer(z_prev[l], m.xi)
                if l in self.U and l in inputs:
                    elig_U[l] += np.outer(inputs[l], m.xi)
                for k in range(self.K):
                    if self.R[k, l]:
                        elig_B[(k, l)] += np.outer(z_prev[k], m.xi)
                xi_sum[l] += m.xi; z_sum[l] += m.z
            if record:
                trace.append(np.concatenate([m.z for m in self.modules]))
        z_out = np.concatenate([self.modules[k].z for k in self.output_modules])
        y = self.readout.read(z_out)
        self.macs_total += self.macs_per_trial()
        return {"y": y, "elig_W": elig_W, "elig_B": elig_B, "elig_U": elig_U, "z_out": z_out, "probe": probe,
                "z_prev": z_prev, "inputs": inputs,
                "xi_mean": [x / cfg.n_relax for x in xi_sum], "z_mean": [z / cfg.n_relax for z in z_sum],
                "trace": np.array(trace) if record else None}

    # ------------------------------------------------------------------ learn
    def learn(self, fwd: dict, loss: float, task: int, structure: bool = True, target=None) -> dict:
        cfg = self.cfg
        # modulator: the loss relative to its running mean, z-scored by its running deviation and
        # clipped, so the write size does not scale with the raw loss (a broadcast "surprise" signal)
        Lb, Ls = self.L_bar.get(task, (loss, abs(loss) + 1e-3))
        m_mod = float(np.clip(-(loss - Lb) / (Ls + 1e-6), -3.0, 3.0))
        Lb_new = Lb + (loss - Lb) / cfg.tau_L
        Ls_new = Ls + (abs(loss - Lb) - Ls) / cfg.tau_L
        self.L_bar[task] = (Lb_new, Ls_new)
        writes = 0
        if cfg.rule == "local" and target is not None:
            writes += self._learn_local(fwd, np.asarray(target, float))
        else:
            eta = cfg.eta / cfg.sigma_xi ** 2
            for l, mod in enumerate(self.modules):
                writes += gated_write(mod.W, mod.gate, eta * m_mod * fwd["elig_W"][l])
            for kl, t in self.B.items():
                writes += gated_write(t, self.B_gate[kl], eta * m_mod * fwd["elig_B"][kl])
            for k, t in self.U.items():
                if k in fwd["elig_U"]:
                    writes += gated_write(t, self.U_gate[k], eta * m_mod * fwd["elig_U"][k])
        if target is not None:
            err = np.asarray(target, float) - fwd["y"]
            writes += gated_write(self.readout, self.readout_gate, cfg.eta_readout * np.outer(fwd["z_out"], err))
        # route statistic for every ordered pair (k -> l), k != l.  With single-module
        # exploration the estimate for target l is conditioned on the trials where l carried
        # the probe (otherwise xi_l = 0 and the average would be diluted by 1/K).
        a = 1.0 / cfg.tau_C
        probe = fwd.get("probe")
        for l in range(self.K):
            if probe is not None and l != probe:
                continue
            xl = fwd["xi_mean"][l]
            for k in range(self.K):
                if k == l:
                    continue
                est = m_mod * np.outer(fwd["z_mean"][k], xl)
                self.Cmat[k, l] += a * (est - self.Cmat[k, l])
        self.C = np.linalg.norm(self.Cmat, axis=(2, 3))
        grown = pruned = 0
        if structure and not self.fixed:
            grown, pruned = self._structure_step()
        self.trial += 1
        return {"m": m_mod, "writes": writes, "grown": grown, "pruned": pruned}

    def _learn_local(self, fwd, target) -> int:
        """Layer-local losses: module k minimises ||R_k^T z_k - target||^2 with gradients that stay inside
        the module (its readout, its recurrent tile, its incoming routes and input tile); nothing
        propagates between modules.  One-step gradient at the final relaxation state."""
        cfg = self.cfg; writes = 0; n = self.n
        for k, mod in enumerate(self.modules):
            y_k = mod.R.read(mod.z)
            err = y_k - target                                   # d L_k / d y_k (up to 2)
            writes += gated_write(mod.R, mod.R_gate, -cfg.eta_local * np.outer(mod.z, err))
            active = ((mod.u > 0) & (mod.u < cfg.r_max)).astype(float)   # f'(u) of the clipped relu
            delta = (mod.R.W @ err) * active * cfg.dt_over_tau       # d L_k / d u_k
            if not np.any(delta):
                continue
            writes += gated_write(mod.W, mod.gate, -cfg.eta_local * np.outer(fwd["z_prev"][k], delta))
            for l in range(self.K):
                if self.R[l, k]:
                    writes += gated_write(self.B[(l, k)], self.B_gate[(l, k)], -cfg.eta_local * np.outer(fwd["z_prev"][l], delta))
            if k in self.U and k in fwd["inputs"]:
                writes += gated_write(self.U[k], self.U_gate[k], -cfg.eta_local * np.outer(fwd["inputs"][k], delta))
        return writes

    def _structure_step(self):
        cfg = self.cfg
        grown = pruned = 0
        # prune routes that stay weak
        for (k, l), t in list(self.B.items()):
            if np.mean(np.abs(t.W[t.A])) < cfg.prune_norm if t.n_active else True:
                self.low_count[k, l] += 1
            else:
                self.low_count[k, l] = 0
            if self.low_count[k, l] >= cfg.prune_patience:
                self._remove_route(k, l); pruned += 1
        # connect the strongest candidate above threshold, one per trial, within budget
        if self.n_edges < cfg.edge_budget:
            cand = np.where(~self.R & ~np.eye(self.K, dtype=bool), self.C, -np.inf)
            k, l = np.unravel_index(int(np.argmax(cand)), cand.shape)
            if cand[k, l] > cfg.theta_e:
                self._create_route(int(k), int(l), seed=self.Cmat[k, l]); grown += 1
                self.Cmat[k, l] = 0.0
        return grown, pruned

    # ---------------------------------------------------------------- metrics
    def modularity(self):
        Q, labels = best_modularity(self.R.astype(float))
        return float(Q), labels.tolist()

    def module_task_information(self, Z_by_module: list, task_ids: np.ndarray, n_clusters: int = 6):
        return [task_information(Z, task_ids, n_clusters=n_clusters) for Z in Z_by_module]

    def log(self, loss: float):
        Q, _ = self.modularity()
        self.history["Q"].append(Q); self.history["edges"].append(self.n_edges)
        self.history["trial"].append(self.trial); self.history["loss"].append(float(loss))


def gated_write(tile: Tile, gate: SurpriseGate, dW: np.ndarray) -> int:
    """Accumulate dW in the surprise gate, write what crosses the threshold, and put back whatever
    the pulsed tile could not realise (changes below half a pulse), so small persistent updates are
    eventually written instead of being silently dropped."""
    out, k = gate(dW)
    if k:
        realised = tile.update(out)
        gate.pending += out - realised
    return k


def layered_routing(K: int, inputs: list, hidden: list, outputs: list) -> np.ndarray:
    """A fixed feed-forward routing matrix: inputs -> hidden -> outputs, all-to-all between layers."""
    R = np.zeros((K, K), bool)
    for i in inputs:
        for h in hidden:
            R[i, h] = True
    for h in hidden:
        for o in outputs:
            R[h, o] = True
    return R
