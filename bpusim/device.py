"""L0: the stochastic MoS2 memristor synapse.

Every synapse is a device with an internal state ``w`` in [0, 1] that maps
linearly onto a conductance::

    G = G_min + (G_max - G_min) * w

Pulses move ``w`` with the saturating steps of the paper (Lesson 1)::

    potentiation  dw+ =  alpha_p * exp(-beta_p * w)
    depression    dw- = -alpha_d * exp(-beta_d * (1 - w))

Three noise sources, each a relative (multiplicative) Gaussian:

* device-to-device ``sigma_d2d`` -- a fixed gain per device that multiplies
  every step size (fabrication mismatch),
* cycle-to-cycle ``sigma_c2c`` -- a fresh factor on every written step,
* read ``sigma_r`` -- a fresh factor on every read of ``G``.

``sigma_eff = sqrt(sigma_d2d**2 + sigma_c2c**2 + sigma_r**2)``.

Retention relaxes ``G`` toward ``G_inf`` with time constant ``tau_ret``
(``inf`` = non-volatile).  A boolean ``dead`` mask marks fabrication
failures: a dead device is stuck at ``G_min`` and ignores pulses.

``program_target`` is write-verify: read, compare with the target within a
tolerance, pulse in the right direction, repeat.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np


@dataclass
class DeviceParams:
    """Physical parameters of a MoS2 synapse (units: siemens for G)."""
    G_min: float = 1e-6          # 1 uS
    G_max: float = 1e-4          # 100 uS
    alpha_p: float = 0.05
    beta_p: float = 3.0
    alpha_d: float = 0.05
    beta_d: float = 3.0
    sigma_d2d: float = 0.03
    sigma_c2c: float = 0.03
    sigma_r: float = 0.02
    tau_ret: float = math.inf    # seconds; inf = non-volatile
    w_inf: float = 0.0           # relaxation target as a state (G_inf = G(w_inf))
    dead_frac: float = 0.0       # fraction of devices dead at fabrication

    @property
    def sigma_eff(self) -> float:
        return math.sqrt(self.sigma_d2d ** 2 + self.sigma_c2c ** 2 + self.sigma_r ** 2)

    @property
    def G_range(self) -> float:
        return self.G_max - self.G_min

    @property
    def levels(self) -> int:
        """Number of distinguishable conductance levels L: two levels are distinguishable when they
        differ by one effective noise standard deviation at full scale, so L = round(1 / sigma_eff),
        clipped to [2, 256] (8 bits is the most the read-out ever resolves).  One level is
        (G_max - G_min) / L in conductance, 1 / L in state units."""
        s = self.sigma_eff
        return 256 if s <= 0 else int(max(2, min(256, round(1.0 / s))))

    def replace(self, **kw) -> "DeviceParams":
        d = asdict(self)
        d.update(kw)
        return DeviceParams(**d)


class DeviceArray:
    """An array of independent memristors with shared parameters.

    Parameters
    ----------
    shape : tuple
        Array shape (any rank).
    rng : numpy.random.Generator
        Source of all randomness (d2d gains, dead mask, c2c and read noise).
    params : DeviceParams
    w_init : float | ndarray
        Initial state (clipped to [0, 1]).
    """

    def __init__(self, shape, rng: np.random.Generator, params: DeviceParams | None = None,
                 w_init=0.0):
        self.shape = tuple(shape)
        self.rng = rng
        self.p = params or DeviceParams()
        self.w = np.clip(np.broadcast_to(np.asarray(w_init, float), self.shape).copy(), 0.0, 1.0)
        # device-to-device mismatch: fixed per device, multiplies the step size
        self.d2d_gain = 1.0 + self.p.sigma_d2d * rng.standard_normal(self.shape)
        self.d2d_gain = np.clip(self.d2d_gain, 0.05, None)  # a device never steps backwards
        self.dead = rng.random(self.shape) < self.p.dead_frac
        self.w[self.dead] = 0.0
        self.n_pulses = 0  # write-energy accounting

    # ------------------------------------------------------------------ state
    @property
    def G(self) -> np.ndarray:
        """Noise-free conductance (what a perfect read would give)."""
        return self.p.G_min + self.p.G_range * self.w

    def w_from_G(self, G) -> np.ndarray:
        return (np.asarray(G) - self.p.G_min) / self.p.G_range

    def read(self) -> np.ndarray:
        """Conductance with fresh multiplicative read noise."""
        G = self.G
        if self.p.sigma_r > 0:
            G = G * (1.0 + self.p.sigma_r * self.rng.standard_normal(self.shape))
        return G

    # ----------------------------------------------------------------- pulses
    def _step_sizes(self):
        p = self.p
        up = p.alpha_p * np.exp(-p.beta_p * self.w)
        down = p.alpha_d * np.exp(-p.beta_d * (1.0 - self.w))
        return up, down

    def pulse(self, n) -> None:
        """Apply ``n`` pulses per device (positive = potentiate, negative = depress).

        Pulses are applied one at a time so the saturating step law holds
        exactly; each step carries its own cycle-to-cycle noise.
        """
        n = np.broadcast_to(np.asarray(n), self.shape).astype(int).copy()
        n[self.dead] = 0
        remaining = n.copy()
        self.n_pulses += int(np.abs(n).sum())
        while True:
            active = remaining != 0
            if not active.any():
                break
            up, down = self._step_sizes()
            step = np.where(remaining > 0, up, -down) * self.d2d_gain
            if self.p.sigma_c2c > 0:
                step = step * (1.0 + self.p.sigma_c2c * self.rng.standard_normal(self.shape))
            self.w = np.where(active, self.w + step, self.w)
            remaining -= np.sign(remaining)
        np.clip(self.w, 0.0, 1.0, out=self.w)
        self.w[self.dead] = 0.0

    def pulses_for(self, w_target) -> np.ndarray:
        """Nominal (noise-free) signed pulse count to move from ``w`` to ``w_target``.

        From dw/dn = alpha_p e^{-beta_p w}:  n = (e^{beta_p w1} - e^{beta_p w0}) / (alpha_p beta_p).
        Depression is the mirror image in u = 1 - w.
        """
        p = self.p
        w0 = self.w
        w1 = np.clip(np.broadcast_to(np.asarray(w_target, float), self.shape), 0.0, 1.0)
        n_up = (np.exp(p.beta_p * w1) - np.exp(p.beta_p * w0)) / (p.alpha_p * p.beta_p)
        n_down = (np.exp(p.beta_d * (1 - w1)) - np.exp(p.beta_d * (1 - w0))) / (p.alpha_d * p.beta_d)
        n = np.where(w1 >= w0, n_up, -n_down)
        return np.rint(n).astype(int)

    # -------------------------------------------------------------- retention
    def retain(self, dt: float) -> None:
        """Relax w toward w_inf with time constant tau_ret over an interval dt."""
        if not math.isfinite(self.p.tau_ret):
            return
        a = 1.0 - math.exp(-dt / self.p.tau_ret)
        self.w += a * (self.p.w_inf - self.w)
        self.w[self.dead] = 0.0

    # ----------------------------------------------------------- write-verify
    def program_target(self, w_target, tol: float = 0.02, max_iter: int = 100,
                       verify_reads: int = 8, damping: float = 0.7) -> int:
        """Write-verify loop: read, compare, pulse until |w_read - w_target| <= tol.

        Each iteration applies the nominal pulse count for the measured gap
        (at least one pulse), so programming converges in a few iterations
        even from far away.  ``verify_reads`` reads are averaged per check
        (the verify circuit's own read noise is what limits precision: with
        sigma_r = 0.02 a single read of a device near w = 1 is uncertain by
        0.02 in w, so the default averages eight reads) and the pulse count is
        damped by ``damping`` to stop measurement noise from driving a
        ping-pong between the coarse depression step near w = 1 and the
        coarse potentiation step near w = 0.  Returns the number of verify
        iterations used.
        """
        w_target = np.clip(np.broadcast_to(np.asarray(w_target, float), self.shape), 0.0, 1.0)
        it = 0
        for it in range(1, max_iter + 1):
            G = np.mean([self.read() for _ in range(verify_reads)], axis=0)
            w_meas = self.w_from_G(G)
            err = w_target - w_meas
            # a device cannot be placed closer than half of one pulse step; near w = 0 the
            # potentiation step (alpha_p) can exceed tol, so the tolerance floors there
            up, down = self._step_sizes()
            resolution = 0.5 * np.maximum(up, down) * self.d2d_gain
            todo = (np.abs(err) > np.maximum(tol, resolution)) & ~self.dead
            if not todo.any():
                return it - 1
            # nominal pulses for (a damped fraction of) the measured gap, at least one pulse
            n = self.pulses_for(np.clip(self.w + damping * err, 0, 1))
            n = np.where(n == 0, np.sign(err).astype(int), n)
            n = np.where(todo, n, 0)
            self.pulse(n)
        return it

    def __repr__(self):
        return f"DeviceArray(shape={self.shape}, mean_w={self.w.mean():.3f}, dead={int(self.dead.sum())})"
