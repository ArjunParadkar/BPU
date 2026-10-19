"""L2: a crossbar tile of differential memristor pairs.

A tile holds an ``n_in x n_out`` array of synapses.  Each synapse is a pair
of devices ``(G+, G-)`` and a boolean mask ``A``::

    W = A * (G+ - G-) / G_scale          (G_scale = G_max - G_min, so |W| <= 1)

``read(x)`` performs the analog matrix-vector product ``W_read^T x`` where
``W_read`` is built from one fresh noisy read of both device arrays.
``update(dW)`` converts a desired weight change into signed pulse counts on
the appropriate device of each pair and applies them with write noise.
``accumulate(dW)`` is the write-budget path: changes are summed in a digital
accumulator and a device is pulsed only when a whole conductance level is
pending (stochastic rounding of the remainder), so writes are counted only on
pulses; ``flush_rate`` and ``writes_per_device`` report the result.
"""
from __future__ import annotations

import numpy as np

from .device import DeviceArray, DeviceParams


class Tile:
    """Differential-pair crossbar.

    Parameters
    ----------
    n_in, n_out : int
    rng : numpy.random.Generator
    params : DeviceParams
    mask : bool ndarray (n_in, n_out) | None
        Initial connectivity; default all-on.
    w_init : float
        Initial state of both devices of each pair (W starts at 0).
    """

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator,
                 params: DeviceParams | None = None, mask=None, w_init: float = 0.0, tile_size: int = 32,
                 levels: int | None = None, acc_rng: np.random.Generator | None = None):
        """A matrix larger than ``tile_size`` x ``tile_size`` is routed as several physical tiles
        (``n_tiles``); the device model is identical, the difference on hardware is the extra
        A/D conversions and routed events at tile boundaries, which ``conversions_per_read``
        and ``events_per_read`` account for."""
        self.n_in, self.n_out = int(n_in), int(n_out)
        self.tile_size = int(tile_size)
        self.rng = rng
        self.p = params or DeviceParams()
        shape = (self.n_in, self.n_out)
        self.gp = DeviceArray(shape, rng, self.p, w_init)
        self.gm = DeviceArray(shape, rng, self.p, w_init)
        self.A = np.ones(shape, bool) if mask is None else np.asarray(mask, bool).copy()
        self.G_scale = self.p.G_range
        self.n_writes = 0
        # eta(V_G): the memtransistor gate scales plasticity without touching the stored state;
        # 0 is the hardware kill switch (claim 6), 1 is nominal
        self.plasticity_gain = 1.0
        # accumulate-and-flush write budget: a digital per-synapse accumulator of requested changes
        # (weight units); a device is pulsed only when the accumulator holds at least one conductance
        # level (1 / L in state units, L distinguishable levels from sigma_eff), see ``accumulate``
        self.levels = int(levels) if levels else self.p.levels
        self.acc = np.zeros(shape)
        self.n_acc_calls = 0          # accumulate() calls
        self.n_acc_writes = 0         # synapse write events produced by accumulate()
        self.acc_rng = acc_rng or rng  # stochastic rounding draws

    # ----------------------------------------------------------------- views
    @property
    def shape(self):
        return (self.n_in, self.n_out)

    @property
    def W(self) -> np.ndarray:
        """Noise-free effective weight (masked)."""
        return self.A * (self.gp.G - self.gm.G) / self.G_scale

    @property
    def dead(self) -> np.ndarray:
        """A synapse is dead if either device of its pair is dead."""
        return self.gp.dead | self.gm.dead

    @property
    def n_active(self) -> int:
        return int(self.A.sum())

    @property
    def density(self) -> float:
        return self.n_active / self.A.size

    def W_read(self) -> np.ndarray:
        """Weight matrix built from one fresh noisy read of both arrays."""
        return self.A * (self.gp.read() - self.gm.read()) / self.G_scale

    # ----------------------------------------------------------------- reads
    def read(self, x: np.ndarray) -> np.ndarray:
        """y = W_read^T x  for x of shape (n_in,) or (batch, n_in)."""
        return np.asarray(x) @ self.W_read()

    def read_T(self, y: np.ndarray) -> np.ndarray:
        """x = W_read y: drive the crossbar from the output side (used by attention)."""
        return np.asarray(y) @ self.W_read().T

    # --------------------------------------------------------------- writes
    def update(self, dW: np.ndarray, max_pulses: int | None = None) -> np.ndarray:
        """Apply a weight change through pulse counts on the differential pair.

        Positive dW potentiates G+ (and if G+ is saturated, depresses G-);
        negative dW does the mirror image.  Only masked-on synapses are
        written.  Returns the *realised* weight change (W_after - W_before),
        so a caller accumulating sub-pulse changes can keep the residual.
        ``self.n_writes`` counts synapses that received at least one pulse.
        """
        dW = np.asarray(dW, float) * self.A * self.plasticity_gain
        W_before = self.W
        if not np.any(dW):
            return np.zeros(self.shape)
        wp, wm = self.gp.w, self.gm.w
        # desired change in state units (W = wp - wm for G_scale = G_range)
        pos = np.clip(dW, 0, None)
        neg = np.clip(-dW, 0, None)
        # route each positive change: first into G+ headroom, remainder out of G-
        head_p = 1.0 - wp
        d_wp_up = np.minimum(pos, head_p)
        d_wm_down = np.minimum(pos - d_wp_up, wm)
        head_m = 1.0 - wm
        d_wm_up = np.minimum(neg, head_m)
        d_wp_down = np.minimum(neg - d_wm_up, wp)
        tgt_p = wp + d_wp_up - d_wp_down
        tgt_m = wm + d_wm_up - d_wm_down
        # nearest whole pulse count; a change smaller than half a pulse is not written
        # (callers that need to accumulate sub-pulse changes use plasticity.SurpriseGate)
        n_p = self.gp.pulses_for(tgt_p)
        n_m = self.gm.pulses_for(tgt_m)
        if max_pulses is not None:
            n_p = np.clip(n_p, -max_pulses, max_pulses)
            n_m = np.clip(n_m, -max_pulses, max_pulses)
        self.gp.pulse(n_p)
        self.gm.pulse(n_m)
        self.n_writes += int(((n_p != 0) | (n_m != 0)).sum())
        return self.W - W_before

    @property
    def level(self) -> float:
        """One conductance level in weight units (= state units for a differential pair): 1 / L."""
        return 1.0 / self.levels

    def accumulate(self, dW: np.ndarray, write_budget: float | None = None, dt: float | None = None,
                   theta_flush: float = 1.0) -> np.ndarray:
        """Accumulate-and-flush write (the write-budget mechanism).

        ``acc[i, j] += dW[i, j]`` in a digital accumulator; a synapse is considered for a write
        only when ``|acc| >= theta_flush`` conductance levels (one level = 1 / L; theta_flush in
        {0.25, 0.5, 1.0}).  The flushed amount is a whole number of levels: the integer part of
        ``|acc| / level`` plus the remainder rounded stochastically (probability = remainder), so
        the expected flushed amount equals the accumulator and, below one level, the device is
        pulsed with probability |acc| / level; the flushed amount (not the realised one) is
        subtracted, since the digital side does not read the device back.  Writes are counted
        only when a device is pulsed.  ``write_budget`` (write events per device per second,
        with ``dt`` the call interval) optionally caps the number of flushes per call to the
        largest accumulators, the quota used by E4.  Returns the realised weight change.
        """
        dW = np.asarray(dW, float) * self.A * self.plasticity_gain
        self.acc += dW
        self.n_acc_calls += 1
        lvl = self.level
        q = np.abs(self.acc) / lvl
        gate = (q >= theta_flush) & self.A
        if write_budget and dt:
            self.write_credit = getattr(self, "write_credit", 0.0) + write_budget * 2 * self.A.size * dt
            n_allow = int(self.write_credit)
            if n_allow < gate.sum():
                order = np.argsort(-np.where(gate, q, 0.0).ravel())[:n_allow]
                sel = np.zeros(self.A.size, bool); sel[order] = True
                gate &= sel.reshape(self.shape)
        if not gate.any():
            return np.zeros(self.shape)
        n_int = np.floor(q)
        frac = q - n_int
        n_lv = n_int + (self.acc_rng.random(self.shape) < frac)
        flushed = np.where(gate & (n_lv > 0), np.sign(self.acc) * n_lv * lvl, 0.0)
        before = self.n_writes
        realised = self.update(flushed)
        self.acc -= flushed
        self.n_acc_writes += self.n_writes - before
        if write_budget and dt:
            self.write_credit -= int(gate.sum())
        return realised

    @property
    def flush_rate(self) -> float:
        """Fraction of accumulate() opportunities (calls x active synapses) that produced a write."""
        d = self.n_acc_calls * max(self.n_active, 1)
        return self.n_acc_writes / d if d else 0.0

    @property
    def n_pulses(self) -> int:
        return self.gp.n_pulses + self.gm.n_pulses

    @property
    def n_tiles(self) -> int:
        import math
        return math.ceil(self.n_in / self.tile_size) * math.ceil(self.n_out / self.tile_size)

    @property
    def conversions_per_read(self) -> int:
        import math
        return self.n_out * math.ceil(self.n_in / self.tile_size)

    @property
    def events_per_read(self) -> int:
        return self.n_tiles

    ENDURANCE = 1e6              # write cycles per device over its life (technical document)
    TILE_BANDWIDTH = 1e4         # write driver limit: write events per physical tile per second
    LIFELONG_RATE = 6e-3         # the earlier per-second budget (1e6 cycles over 5 years), kept for reference

    def writes_per_device(self, seconds: float | None = None) -> float:
        """Average write events per device over the run (per second if ``seconds`` is given)."""
        w = self.n_writes / (2.0 * self.A.size)
        return w / seconds if seconds else w

    def write_budget_report(self, seconds: float | None = None) -> dict:
        """The three write-budget numbers: endurance (total writes per device against 1e6), the write
        driver bandwidth (write events per physical 32x32 tile per second against 1e4) and the old
        lifelong rate (writes per device per second against 6e-3)."""
        wpd = self.writes_per_device()
        rep = {"writes_per_device_total": wpd, "percent_of_endurance": 100.0 * wpd / self.ENDURANCE,
               "flush_rate": self.flush_rate, "levels": self.levels}
        if seconds:
            per_tile_s = self.n_writes / self.n_tiles / seconds
            rep.update({"writes_per_tile_per_s": per_tile_s, "percent_of_tile_bandwidth": 100.0 * per_tile_s / self.TILE_BANDWIDTH,
                        "writes_per_device_per_s": wpd / seconds, "percent_of_lifelong_rate": 100.0 * wpd / seconds / self.LIFELONG_RATE})
        return rep

    def program(self, W_target: np.ndarray, tol: float = 0.02, max_iter: int = 400) -> int:
        """Write-verify the whole tile to W_target in [-1, 1] (positive part on G+, negative on G-)."""
        W_target = np.clip(np.asarray(W_target, float), -1.0, 1.0)
        it1 = self.gp.program_target(np.clip(W_target, 0, 1), tol, max_iter)
        it2 = self.gm.program_target(np.clip(-W_target, 0, 1), tol, max_iter)
        self.acc[:] = 0.0            # a whole-tile rewrite supersedes whatever was pending
        return max(it1, it2)

    def reset_pair(self, idx, w_plus: float = 0.0, w_minus: float = 0.0) -> None:
        """Set the devices of the synapses ``idx`` (boolean mask or index tuple) by write-verify."""
        tp = self.gp.w.copy(); tm = self.gm.w.copy()
        tp[idx] = w_plus; tm[idx] = w_minus
        self.gp.program_target(tp, tol=0.02)
        self.gm.program_target(tm, tol=0.02)

    def retain(self, dt: float) -> None:
        self.gp.retain(dt)
        self.gm.retain(dt)

    def set_mask(self, mask) -> None:
        self.A = np.asarray(mask, bool).copy()
        self.acc[~self.A] = 0.0

    def __repr__(self):
        return f"Tile({self.n_in}x{self.n_out}, active={self.n_active}, |W|max={np.abs(self.W).max():.3f})"
