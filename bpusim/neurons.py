"""Neuron models: leaky integrate-and-fire and rate units, both with optional additive noise.

LIF:   tau_m dv/dt = -(v - v_rest) + R I ;  spike when v >= theta ; v <- v_reset
Rate:  tau   dr/dt = -r + f(u)

Both are integrated with forward Euler at step ``dt``.  Additive noise of
standard deviation ``sigma`` (per unit time, scaled by sqrt(dt)) is added to
the state each step.
"""
from __future__ import annotations

import numpy as np

ACTIVATIONS = {
    "relu": lambda u: np.maximum(u, 0.0),
    "tanh": np.tanh,
    "sigmoid": lambda u: 1.0 / (1.0 + np.exp(-u)),
    "linear": lambda u: u,
    "satlin": lambda u: np.clip(u, 0.0, 1.0),
}


def get_activation(f):
    if callable(f):
        return f
    return ACTIVATIONS[f]


class LIF:
    """Leaky integrate-and-fire population of ``n`` neurons."""

    def __init__(self, n: int, rng: np.random.Generator, tau_m: float = 20e-3, v_rest: float = 0.0,
                 R: float = 1.0, theta: float = 1.0, v_reset: float = 0.0, dt: float = 1e-3,
                 sigma: float = 0.0, t_ref: float = 0.0):
        self.n, self.rng = int(n), rng
        self.tau_m, self.v_rest, self.R, self.theta, self.v_reset = tau_m, v_rest, R, theta, v_reset
        self.dt, self.sigma, self.t_ref = dt, sigma, t_ref
        self.v = np.full(self.n, v_rest, float)
        self.refrac = np.zeros(self.n)
        self.t = 0.0

    def reset(self):
        self.v[:] = self.v_rest
        self.refrac[:] = 0.0
        self.t = 0.0

    def step(self, I: np.ndarray) -> np.ndarray:
        """Advance one dt with input current ``I`` (shape (n,)); returns boolean spike vector."""
        dv = (-(self.v - self.v_rest) + self.R * np.asarray(I)) * (self.dt / self.tau_m)
        if self.sigma > 0:
            dv = dv + self.sigma * np.sqrt(self.dt) * self.rng.standard_normal(self.n)
        free = self.refrac <= 0
        self.v = np.where(free, self.v + dv, self.v)
        spikes = self.v >= self.theta
        self.v[spikes] = self.v_reset
        self.refrac = np.where(spikes, self.t_ref, self.refrac - self.dt)
        self.t += self.dt
        return spikes

    @property
    def I_threshold(self) -> float:
        """Current below which the neuron never fires: theta / R (for v_rest = 0)."""
        return (self.theta - self.v_rest) / self.R


class RateNeurons:
    """Rate population: tau dr/dt = -r + f(u) (+ noise)."""

    def __init__(self, n: int, rng: np.random.Generator, tau: float = 10e-3, dt: float = 1e-3,
                 f="relu", sigma: float = 0.0, r_init: float = 0.0):
        self.n, self.rng = int(n), rng
        self.tau, self.dt, self.sigma = tau, dt, sigma
        self.f = get_activation(f)
        self.r = np.full(self.n, r_init, float)

    def reset(self, r=0.0):
        self.r = np.broadcast_to(np.asarray(r, float), (self.n,)).copy()

    def step(self, u: np.ndarray) -> np.ndarray:
        """Advance one dt with total input ``u``; returns the new rate vector."""
        dr = (-self.r + self.f(np.asarray(u))) * (self.dt / self.tau)
        if self.sigma > 0:
            dr = dr + self.sigma * np.sqrt(self.dt / self.tau) * self.rng.standard_normal(self.n)
        self.r = self.r + dr
        return self.r
