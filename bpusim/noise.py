"""Noise configuration and sweep helpers.

The BPU separates *static* noise (device-to-device mismatch, fixed after
fabrication) from *dynamic* noise (cycle-to-cycle write noise, read noise and
neuron noise, fresh every step).  E3 fixes the static part and sweeps the
dynamic part; E1/E6 sweep a single ``sigma`` that is split equally over the
three device sources so that ``sigma_eff = sigma``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .device import DeviceParams


@dataclass
class NoiseConfig:
    sigma_d2d: float = 0.03
    sigma_c2c: float = 0.03
    sigma_r: float = 0.02
    sigma_neuron: float = 0.0

    @property
    def sigma_eff(self) -> float:
        return math.sqrt(self.sigma_d2d ** 2 + self.sigma_c2c ** 2 + self.sigma_r ** 2)

    def device_params(self, base: DeviceParams | None = None) -> DeviceParams:
        base = base or DeviceParams()
        return base.replace(sigma_d2d=self.sigma_d2d, sigma_c2c=self.sigma_c2c, sigma_r=self.sigma_r)


def equal_split(sigma: float) -> NoiseConfig:
    """Split a total effective device noise equally over d2d, c2c and read."""
    s = sigma / math.sqrt(3.0)
    return NoiseConfig(s, s, s, 0.0)


def dynamic_only(sigma_dyn: float, sigma_d2d: float = 0.05, neuron_fraction: float = 0.5) -> NoiseConfig:
    """Fixed mismatch, dynamic noise shared between read noise and neuron noise.

    ``sigma_dyn`` is the total dynamic level; a fraction goes to additive
    neuron noise (in rate units) and the rest to multiplicative read noise.
    """
    return NoiseConfig(sigma_d2d=sigma_d2d, sigma_c2c=0.0,
                       sigma_r=sigma_dyn * (1 - neuron_fraction),
                       sigma_neuron=sigma_dyn * neuron_fraction)


def sweep(levels, fn, seeds, **kw):
    """Run fn(sigma, seed, **kw) for every level and seed; returns {sigma: [results]}."""
    out = {}
    for s in levels:
        out[float(s)] = [fn(float(s), int(seed), **kw) for seed in seeds]
    return out


def summarize(values):
    """mean, std, sem, n of a list of floats (NaNs ignored)."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "n": 0}
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "sem": float(v.std(ddof=1) / math.sqrt(v.size)) if v.size > 1 else 0.0, "n": int(v.size)}
