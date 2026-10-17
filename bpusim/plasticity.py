"""L1: local plasticity rules.

* STDP window  ``A+ exp(-dt/tau+)`` for pre-before-post (dt > 0) and
  ``-A- exp(dt/tau-)`` for post-before-pre (dt < 0), with an online
  trace-based implementation (:class:`STDP`).
* Eligibility trace  ``tau_e de/dt = -e + tau_e * event``.
* Three-factor update  ``dW = eta * m * e`` gated by a write threshold
  ``theta_w`` (surprise gating: a write happens only when |dW| >= theta_w).
* Homeostatic scaling  ``tau_H dG/dt = G (r* - r) / r*`` per postsynaptic neuron.
* Oja's rule / explicit column normalisation.
"""
from __future__ import annotations

import numpy as np


def stdp_window(dt, A_plus=0.01, A_minus=0.0105, tau_plus=20e-3, tau_minus=20e-3):
    """STDP weight change for a spike-time difference dt = t_post - t_pre (array ok)."""
    dt = np.asarray(dt, float)
    return np.where(dt > 0, A_plus * np.exp(-dt / tau_plus), -A_minus * np.exp(dt / tau_minus)) * (dt != 0)


class STDP:
    """Online pair-based STDP with exponential pre/post traces.

    Each call to :meth:`step` with boolean pre/post spike vectors returns the
    weight change matrix ``dW`` of shape ``(n_in, n_out)``.  Summing these
    over time reproduces :func:`stdp_window` for every spike pair (all-to-all
    pairing).
    """

    def __init__(self, n_in, n_out, A_plus=0.01, A_minus=0.0105, tau_plus=20e-3, tau_minus=20e-3,
                 dt=1e-3):
        self.n_in, self.n_out = n_in, n_out
        self.A_plus, self.A_minus = A_plus, A_minus
        self.tau_plus, self.tau_minus, self.dt = tau_plus, tau_minus, dt
        self.x_pre = np.zeros(n_in)    # pre trace
        self.x_post = np.zeros(n_out)  # post trace

    def reset(self):
        self.x_pre[:] = 0
        self.x_post[:] = 0

    def step(self, pre_spikes, post_spikes) -> np.ndarray:
        pre = np.asarray(pre_spikes, float)
        post = np.asarray(post_spikes, float)
        # decay traces, then add this step's spikes
        self.x_pre *= np.exp(-self.dt / self.tau_plus)
        self.x_post *= np.exp(-self.dt / self.tau_minus)
        # pre spike now: depress by the existing post trace (post happened before pre)
        dW = -self.A_minus * np.outer(pre, self.x_post)
        # post spike now: potentiate by the existing pre trace (pre happened before post)
        dW += self.A_plus * np.outer(self.x_pre, post)
        self.x_pre += pre
        self.x_post += post
        return dW


class EligibilityTrace:
    """tau_e de/dt = -e + tau_e * event  (Euler, step dt).

    With this scaling a unit event lasting one step leaves ``e = dt`` which
    then decays with ``tau_e``; an event of rate 1 per unit time saturates at
    ``e = tau_e``.
    """

    def __init__(self, shape, tau_e=1.0, dt=1e-3):
        self.e = np.zeros(shape)
        self.tau_e, self.dt = tau_e, dt

    def reset(self):
        self.e[:] = 0

    def step(self, event) -> np.ndarray:
        self.e += self.dt * (-self.e / self.tau_e + np.asarray(event))
        return self.e


def three_factor(eta: float, m, e, theta_w: float = 0.0):
    """dW = eta * m * e, zeroed where |dW| < theta_w.  ``m`` may be a scalar or per-column vector.

    Returns (dW, n_writes).
    """
    dW = eta * np.asarray(m) * np.asarray(e)
    if theta_w > 0:
        dW = np.where(np.abs(dW) >= theta_w, dW, 0.0)
    return dW, int(np.count_nonzero(dW))


class SurpriseGate:
    """Accumulates sub-threshold three-factor updates so small but persistent
    changes are eventually written (write-budget friendly)."""

    def __init__(self, shape, theta_w):
        self.pending = np.zeros(shape)
        self.theta_w = theta_w

    def __call__(self, dW):
        self.pending += dW
        write = np.abs(self.pending) >= self.theta_w
        out = np.where(write, self.pending, 0.0)
        self.pending[write] = 0.0
        return out, int(write.sum())


def homeostatic_scaling(G, r, r_star, tau_H, dt):
    """Multiplicative scaling of each postsynaptic column: tau_H dG/dt = G (r* - r)/r*.

    G : (n_in, n_out) conductances or weights; r : (n_out,) rates.
    Returns dG (same shape as G).
    """
    G = np.asarray(G, float)
    factor = (dt / tau_H) * (r_star - np.asarray(r, float)) / r_star
    return G * factor[None, :]


def oja_update(x, y, W, eta):
    """Oja's rule for a linear layer y = W^T x:  dW = eta * (x y^T - y^2 * W)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    return eta * (np.outer(x, y) - (y ** 2)[None, :] * W)


def normalize_columns(W, target_norm=1.0, eps=1e-12):
    """Return W with every column rescaled to L2 norm ``target_norm`` (explicit Oja limit)."""
    W = np.asarray(W, float)
    n = np.linalg.norm(W, axis=0, keepdims=True)
    return W * (target_norm / np.maximum(n, eps))
