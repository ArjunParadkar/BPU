"""L2: a crossbar tile of differential memristor pairs.

A tile holds an ``n_in x n_out`` array of synapses.  Each synapse is a pair
of devices ``(G+, G-)`` and a boolean mask ``A``::

    W = A * (G+ - G-) / G_scale          (G_scale = G_max - G_min, so |W| <= 1)

``read(x)`` performs the analog matrix-vector product ``W_read^T x`` where
``W_read`` is built from one fresh noisy read of both device arrays.
``update(dW)`` converts a desired weight change into signed pulse counts on
the appropriate device of each pair and applies them with write noise.
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
                 params: DeviceParams | None = None, mask=None, w_init: float = 0.0):
        self.n_in, self.n_out = int(n_in), int(n_out)
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
    def n_pulses(self) -> int:
        return self.gp.n_pulses + self.gm.n_pulses

    def writes_per_device(self, seconds: float | None = None) -> float:
        """Average write events per device (per second if ``seconds`` is given).  The
        technical document's endurance budget is ~6e-3 writes per device per second
        (1e6 cycles over 5 years)."""
        w = self.n_writes / (2.0 * self.A.size)
        return w / seconds if seconds else w

    def program(self, W_target: np.ndarray, tol: float = 0.02, max_iter: int = 400) -> int:
        """Write-verify the whole tile to W_target in [-1, 1] (positive part on G+, negative on G-)."""
        W_target = np.clip(np.asarray(W_target, float), -1.0, 1.0)
        it1 = self.gp.program_target(np.clip(W_target, 0, 1), tol, max_iter)
        it2 = self.gm.program_target(np.clip(-W_target, 0, 1), tol, max_iter)
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

    def __repr__(self):
        return f"Tile({self.n_in}x{self.n_out}, active={self.n_active}, |W|max={np.abs(self.W).max():.3f})"
