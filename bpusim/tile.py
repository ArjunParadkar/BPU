"""L2: a crossbar tile of differential memristor pairs.

A tile holds an ``n_in x n_out`` array of synapses.  Each synapse is a pair
of devices ``(G+, G-)`` and a boolean mask ``A``::

    W = A * (G+ - G-) / G_scale          (G_scale = G_max - G_min, so |W| <= 1)

``read(x)`` performs the analog matrix-vector product ``W_read^T x`` where
``W_read`` is built from one fresh noisy read of both device arrays.
``update(dW)`` converts a desired weight change into signed pulse counts on
the appropriate device of each pair and applies them with write noise.
``accumulate(dW)`` is the learning-rule write path.  Default write rule
``"nearest_pulse"`` (2026-10-19): requested changes are summed in a 16-bit
digital accumulator that is flushed EVERY step as the nearest whole number of
pulses; the nominal (noise-free) effect of those pulses is subtracted, the
sub-pulse remainder stays.  The earlier ``"level"`` rule (pulse only when
theta_flush conductance levels are pending, stochastic rounding) is kept as an
option; under an endurance budget it was a cost without a benefit (E9, E4).
Two more primitives (2026-10-19): ``set_noise(sigma_r)`` is the per-tile
read-noise dial (the MODULATE primitive, range 0.005-0.2) and
``set_dropout(p)`` makes every read drop each synapse with probability p.
Both are logged in ``primitive_log``.
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

    WRITE_RULES = ("nearest_pulse", "level")
    ACC_BITS = 16                       # digital accumulator resolution (signed, full scale = one weight unit)
    MODULATE_RANGE = (0.005, 0.2)       # read-noise dial range (set_noise)

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator,
                 params: DeviceParams | None = None, mask=None, w_init: float = 0.0, tile_size: int = 32,
                 levels: int | None = None, acc_rng: np.random.Generator | None = None,
                 write_rule: str = "nearest_pulse"):
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
        assert write_rule in self.WRITE_RULES, write_rule
        self.write_rule = write_rule
        self.acc_step = 2.0 ** -(self.ACC_BITS - 1)   # 16-bit signed accumulator: 1 weight unit = 2^15 counts
        # per-tile primitives (2026-10-19): read-noise dial and per-read synapse dropout, both logged
        self.dropout = 0.0
        self.n_reads = 0
        self.primitive_log: list[dict] = []
        self.n_modulate = 0

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
        """Weight matrix built from one fresh noisy read of both arrays (plus the per-read dropout mask)."""
        self.n_reads += 1
        W = self.A * (self.gp.read() - self.gm.read()) / self.G_scale
        if self.dropout > 0:
            W = W * (self.rng.random(self.shape) >= self.dropout)
        return W

    # ------------------------------------------------------------ primitives
    def set_noise(self, sigma_r: float) -> float:
        """MODULATE primitive: set this tile's read-noise level (relative, per read).  The dial range is
        ``MODULATE_RANGE``; values outside it are clipped and the clip is logged.  Returns the value set."""
        lo, hi = self.MODULATE_RANGE
        v = float(sigma_r); clipped = v < lo or v > hi
        if v != 0.0:                       # 0 (dial off) is allowed for the reference arms
            v = float(min(max(v, lo), hi))
        self.p = self.p.replace(sigma_r=v)
        self.gp.p = self.p; self.gm.p = self.p
        self.n_modulate += 1
        self.primitive_log.append({"primitive": "MODULATE", "read_index": self.n_reads, "sigma_r": v, "clipped": clipped})
        return v

    def set_dropout(self, p: float) -> None:
        """DROPOUT primitive: from now on every read drops each synapse independently with probability p."""
        assert 0.0 <= p < 1.0, p
        self.dropout = float(p)
        self.primitive_log.append({"primitive": "DROPOUT", "read_index": self.n_reads, "p": self.dropout})

    # ----------------------------------------------------------------- reads
    def read(self, x: np.ndarray) -> np.ndarray:
        """y = W_read^T x  for x of shape (n_in,) or (batch, n_in)."""
        return np.asarray(x) @ self.W_read()

    def read_T(self, y: np.ndarray) -> np.ndarray:
        """x = W_read y: drive the crossbar from the output side (used by attention)."""
        return np.asarray(y) @ self.W_read().T

    # --------------------------------------------------------------- writes
    def update(self, dW: np.ndarray, max_pulses: int | None = None, return_nominal: bool = False, apply_gain: bool = True):
        """Apply a weight change through pulse counts on the differential pair.

        Positive dW potentiates G+ (and if G+ is saturated, depresses G-);
        negative dW does the mirror image.  Only masked-on synapses are
        written.  Returns the *realised* weight change (W_after - W_before),
        so a caller accumulating sub-pulse changes can keep the residual;
        with ``return_nominal`` also the *nominal* change (the noise-free,
        gain-free effect of the pulse counts, which is all the digital side
        knows without a read-back).
        ``self.n_writes`` counts synapses that received at least one pulse.
        """
        dW = np.asarray(dW, float) * self.A * (self.plasticity_gain if apply_gain else 1.0)
        W_before = self.W
        if not np.any(dW):
            z = np.zeros(self.shape)
            return (z, z.copy()) if return_nominal else z
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
        nominal = None
        if return_nominal:
            nominal = self.A * ((self.gp.nominal_state_after(n_p) - wp) - (self.gm.nominal_state_after(n_m) - wm))
        self.gp.pulse(n_p)
        self.gm.pulse(n_m)
        self.n_writes += int(((n_p != 0) | (n_m != 0)).sum())
        realised = self.W - W_before
        return (realised, nominal) if return_nominal else realised

    @property
    def level(self) -> float:
        """One conductance level in weight units (= state units for a differential pair): 1 / L."""
        return 1.0 / self.levels

    def _quantize_acc(self) -> None:
        """Hold the accumulator at 16-bit signed fixed point (full scale = one weight unit)."""
        np.clip(np.rint(self.acc / self.acc_step) * self.acc_step, -1.0 + self.acc_step, 1.0 - self.acc_step, out=self.acc)

    def accumulate(self, dW: np.ndarray, write_budget: float | None = None, dt: float | None = None,
                   theta_flush: float | None = None, rule: str | None = None) -> np.ndarray:
        """Learning-rule write through the digital accumulator; ``rule`` defaults to ``self.write_rule``.

        **nearest_pulse** (default): ``acc += dW`` (16-bit), then the whole accumulator is requested
        every step and realised as the nearest whole number of pulses per device; the nominal effect
        of those pulses is subtracted (the digital side has no read-back), so the sub-pulse remainder
        and the quantisation error carry over.  No threshold, no stochastic rounding.

        **level** (the 2026-10-18 write budget, kept as an option): accumulate-and-flush at
        ``theta_flush`` conductance levels (default 1.0) with stochastic rounding, as below.

        ``write_budget`` (write events per device per second, with ``dt`` the call interval)
        optionally caps the number of flushes per call to the largest accumulators (E4's quota).
        Returns the realised weight change.
        """
        rule = rule or self.write_rule
        if rule == "nearest_pulse":
            return self._accumulate_nearest_pulse(dW, write_budget, dt)
        return self._accumulate_level(dW, write_budget, dt, 1.0 if theta_flush is None else theta_flush)

    def _accumulate_nearest_pulse(self, dW, write_budget, dt):
        dW = np.asarray(dW, float) * self.A * self.plasticity_gain
        self.acc += dW
        self._quantize_acc()
        self.n_acc_calls += 1
        req = self.acc.copy()
        if write_budget and dt:
            # quota: only the n_allow largest pending changes may be written this call
            self.write_credit = getattr(self, "write_credit", 0.0) + write_budget * 2 * self.A.size * dt
            n_allow = int(self.write_credit)
            cand = np.abs(req) > 0
            if n_allow < cand.sum():
                order = np.argsort(-np.abs(req).ravel())[:n_allow]
                sel = np.zeros(self.A.size, bool); sel[order] = True
                req = np.where(sel.reshape(self.shape), req, 0.0)
        before = self.n_writes
        realised, nominal = self.update(req, return_nominal=True, apply_gain=False)
        self.acc -= nominal
        self._quantize_acc()
        self.n_acc_writes += self.n_writes - before
        if write_budget and dt:
            self.write_credit -= int((nominal != 0).sum())
        return realised

    def _accumulate_level(self, dW: np.ndarray, write_budget: float | None = None, dt: float | None = None,
                          theta_flush: float = 1.0) -> np.ndarray:
        """Accumulate-and-flush write (the 2026-10-18 write-budget mechanism).

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
        realised = self.update(flushed, apply_gain=False)
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
               "flush_rate": self.flush_rate, "levels": self.levels, "write_rule": self.write_rule,
               "n_modulate": self.n_modulate, "dropout": self.dropout, "n_reads": self.n_reads}
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


class FloatTile:
    """The float32 REFERENCE substrate with the Tile interface (2026-10-19).

    Weights are float32 with ideal (unquantised, noise-free) updates; every read injects the same
    multiplicative Gaussian read noise ``sigma_r`` and the same per-read dropout as a Tile would, so
    the variability experiments (E12-E14) run the identical algorithm on both substrates and the only
    difference is the device physics (d2d / c2c, pulse quantisation, write-verify).  No writes are
    counted (there are no devices)."""

    MODULATE_RANGE = Tile.MODULATE_RANGE

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator, sigma_r: float = 0.0, w_init: float = 0.0):
        self.n_in, self.n_out = int(n_in), int(n_out)
        self.rng = rng
        self.W = np.full((self.n_in, self.n_out), float(w_init))
        self.A = np.ones(self.W.shape, bool)
        self.sigma_r = float(sigma_r)
        self.dropout = 0.0
        self.n_reads = 0; self.n_writes = 0; self.n_modulate = 0
        self.primitive_log: list[dict] = []
        self.write_rule = "float"

    @property
    def shape(self):
        return self.W.shape

    def W_read(self) -> np.ndarray:
        self.n_reads += 1
        W = self.W
        if self.sigma_r > 0:
            W = W * (1.0 + self.sigma_r * self.rng.standard_normal(self.shape))
        if self.dropout > 0:
            W = W * (self.rng.random(self.shape) >= self.dropout)
        return W

    def read(self, x):
        return np.asarray(x) @ self.W_read()

    def read_T(self, y):
        return np.asarray(y) @ self.W_read().T

    def update(self, dW, **kw):
        before = self.W.copy()
        self.W = np.clip(self.W + np.asarray(dW, float) * self.A, -1.0, 1.0)
        return self.W - before

    accumulate = update

    def program(self, W_target, **kw) -> int:
        self.W = np.clip(np.asarray(W_target, float), -1.0, 1.0).copy(); return 0

    def set_noise(self, sigma_r: float) -> float:
        """The float32 reference has no dial: the value is set as given (2026-10-23; before, it was clipped to
        MODULATE_RANGE like a device tile; E13 / E14 set beyond-dial values directly and were not affected)."""
        v = float(sigma_r); clipped = False
        self.sigma_r = v; self.n_modulate += 1
        self.primitive_log.append({"primitive": "MODULATE", "read_index": self.n_reads, "sigma_r": v, "clipped": clipped})
        return v

    def set_dropout(self, p: float) -> None:
        assert 0.0 <= p < 1.0, p
        self.dropout = float(p)
        self.primitive_log.append({"primitive": "DROPOUT", "read_index": self.n_reads, "p": self.dropout})

    def writes_per_device(self, seconds=None) -> float:
        return 0.0

    def write_budget_report(self, seconds=None) -> dict:
        return {"writes_per_device_total": 0.0, "percent_of_endurance": 0.0, "write_rule": "float", "n_modulate": self.n_modulate}
