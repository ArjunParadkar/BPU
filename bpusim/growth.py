"""L3: noise-powered structural plasticity on a tile.

Grow rule (Lesson 4.2).  Each output's fluctuation about its running mean,
``xi_j = y_j - <y_j>``, correlated with the broadcast modulator ``m`` and the
input ``x_i`` estimates the loss gradient of a synapse that may not exist::

    C_ij = < m * x_i * xi_j >           (exponential running average)
    A_ij : 0 -> 1   if |C_ij| > theta_grow   (within the synapse budget)

New synapses are seeded along ``sign(C_ij)`` with a small weight.

Prune rule (Lesson 4.3).  An active synapse whose |W_ij| stays below
``theta_prune`` for ``T_prune`` consecutive steps is released; dead devices
are pruned immediately.

``spawn`` copies a tile into a fresh one with programming noise
(neurogenesis + symmetry breaking, Lesson 4.4).
"""
from __future__ import annotations

import numpy as np

from .tile import Tile


class GrowthRule:
    def __init__(self, tile: Tile, theta_grow: float = 1e-3, theta_prune: float = 0.02,
                 T_prune: int = 50, budget: int | None = None, tau_C: float = 50.0,
                 tau_mean: float = 20.0, seed_w: float = 0.05, max_grow_per_call: int | None = None):
        self.tile = tile
        self.theta_grow, self.theta_prune, self.T_prune = theta_grow, theta_prune, T_prune
        self.budget = int(budget) if budget is not None else tile.A.size
        self.a_C = 1.0 / tau_C
        self.a_mean = 1.0 / tau_mean
        self.seed_w = seed_w
        self.max_grow = max_grow_per_call
        self.C = np.zeros(tile.shape)
        self.y_mean = np.zeros(tile.n_out)
        self.below = np.zeros(tile.shape, int)
        self.n_grown = 0
        self.n_pruned = 0
        self.history = []

    # ------------------------------------------------------------ statistics
    def observe(self, m: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Update the running correlation C with one (m, x, y) sample; returns xi."""
        y = np.asarray(y, float); x = np.asarray(x, float)
        xi = y - self.y_mean
        self.y_mean += self.a_mean * (y - self.y_mean)
        self.C += self.a_C * (m * np.outer(x, xi) - self.C)
        return xi

    def observe_batch(self, m: np.ndarray, X: np.ndarray, Y: np.ndarray) -> None:
        """Batched version: m (B,), X (B, n_in), Y (B, n_out)."""
        Xi = Y - self.y_mean[None, :]
        self.y_mean += self.a_mean * (Y.mean(0) - self.y_mean)
        est = (X * m[:, None]).T @ Xi / len(m)
        self.C += self.a_C * (est - self.C)

    # ----------------------------------------------------------------- grow
    def grow(self) -> int:
        """Activate up to the budget the inactive synapses with |C| > theta_grow, largest first."""
        t = self.tile
        free = self.budget - t.n_active
        if free <= 0:
            return 0
        cand = (~t.A) & (~t.dead) & (np.abs(self.C) > self.theta_grow)
        n_cand = int(cand.sum())
        if n_cand == 0:
            return 0
        k = min(free, n_cand, self.max_grow or n_cand)
        scores = np.where(cand, np.abs(self.C), -np.inf).ravel()
        idx = np.argpartition(-scores, k - 1)[:k]
        rows, cols = np.unravel_index(idx, t.shape)
        sgn = np.sign(self.C[rows, cols])
        t.A[rows, cols] = True
        # seed along sign(C): positive -> G+ device, negative -> G- device
        tp = t.gp.w.copy(); tm = t.gm.w.copy()
        tp[rows, cols] = np.where(sgn > 0, self.seed_w, 0.0)
        tm[rows, cols] = np.where(sgn < 0, self.seed_w, 0.0)
        t.gp.program_target(tp, tol=0.01)
        t.gm.program_target(tm, tol=0.01)
        self.below[rows, cols] = 0
        self.n_grown += k
        return k

    # ---------------------------------------------------------------- prune
    def prune(self) -> int:
        """Prune synapses weak for T_prune consecutive steps and dead devices immediately."""
        t = self.tile
        weak = t.A & (np.abs(t.W) < self.theta_prune)
        self.below = np.where(weak, self.below + 1, 0)
        kill = (self.below >= self.T_prune) | (t.A & t.dead)
        n = int(kill.sum())
        if n:
            t.A[kill] = False
            self.below[kill] = 0
            # release the devices: reset both to the bottom so a regrown synapse starts clean
            t.reset_pair(kill, 0.0, 0.0)
            self.C[kill] = 0.0
            self.n_pruned += n
        return n

    def step(self, m, x, y, do_structure: bool = True) -> dict:
        """Convenience: observe then prune+grow.  Returns counts."""
        self.observe(m, x, y)
        out = {"grown": 0, "pruned": 0}
        if do_structure:
            out["pruned"] = self.prune()
            out["grown"] = self.grow()
        self.history.append(self.tile.n_active)
        return out


def spawn(parent: Tile, rng: np.random.Generator, sigma_copy: float = 0.05, tol: float = 0.02) -> Tile:
    """Neurogenesis: a new tile programmed to a noisy copy of the parent's weights and mask.

    The child has its own fresh device-to-device mismatch, and the copy itself
    is perturbed by ``sigma_copy`` relative noise, so two children of one
    parent are never identical (symmetry breaking, Lesson 4.4).
    """
    child = Tile(parent.n_in, parent.n_out, rng, parent.p, mask=parent.A.copy())
    W = parent.W * (1.0 + sigma_copy * rng.standard_normal(parent.shape))
    child.program(np.clip(W, -1, 1), tol=tol)
    return child
