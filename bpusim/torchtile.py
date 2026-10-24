"""PyTorch mirror of the tile model for model-scale experiments (E8-E10, E2).

:class:`TileLinear` executes a linear layer through differential memristor
pairs with the same semantics as :class:`bpusim.tile.Tile`:

* state ``w+``, ``w-`` in [0, 1] per synapse, effective weight
  ``W = scale * A * (w+ - w-)``, mask ``A``;
* fixed device-to-device gain per device (``sigma_d2d``), fresh
  multiplicative read noise per device on every forward (``sigma_r``),
  fresh multiplicative write noise per pulse (``sigma_c2c``);
* soft-bound updates: a requested change is converted to a whole number of
  pulses of size ``alpha exp(-beta w)`` (potentiation) or
  ``alpha exp(-beta (1 - w))`` (depression) on the matching device of the
  pair.  Default write rule ``"nearest_pulse"`` (2026-10-19): a 16-bit
  per-synapse accumulator is flushed every step as the nearest whole pulses
  and the NOMINAL effect of those pulses is subtracted (no read-back), so the
  sub-pulse remainder carries over.  ``"level"`` (accumulate-and-flush at
  theta_flush conductance levels, stochastic rounding) and ``"pulse"`` (the
  legacy rule that subtracted the realised change) are kept as options;
  ``"float"`` keeps float32 weights with ideal updates and injects the same
  read noise / dropout on every read: the reference arm of E11-E14;
* primitives: ``set_noise(sigma_r)`` is the per-tile read-noise dial (the
  MODULATE primitive, range 0.005-0.2, logged) and ``set_dropout(p)`` drops
  every synapse independently with probability p on every read;
* read-out: by default every device tile converts its outputs through a
  layer-accumulate 8-bit converter (``ReadoutSpec.default()``, FoM 10 fJ,
  range tracked automatically); E8b overrides it per cell;
* optional feedback alignment: the backward pass carries the input
  gradient through a fixed random matrix instead of ``W^T``; optional direct
  feedback alignment (:class:`DFAContext`): every layer receives the global
  output error through its own fixed random matrix and nothing travels
  between layers.

An :class:`OpCounter` tallies MACs, tile reads, A/D conversions, pulses and
routed events so that E8b can project energy from the actual run.

``ideal=True`` (or constructing with ``sigma=None``) makes the layer a plain
float linear layer with the same interface, used for baselines.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class OpCounter:
    macs: int = 0
    reads: int = 0            # tile read operations (one matrix-vector product each)
    conversions: int = 0      # A/D conversions at tile outputs (one per 32-row tile partial sum)
    conversions_out: int = 0  # A/D conversions if partial sums are accumulated in analog across tiles (one per layer output)
    pulses: int = 0           # device write pulses
    writes: int = 0           # synapses that received >= 1 pulse (write events)
    events: int = 0           # routed vectors between tiles / layers
    cache_writes: int = 0     # volatile attention-cache device writes
    modulates: int = 0        # MODULATE primitive events (read-noise dial changes)
    enabled: bool = True
    log: dict = field(default_factory=dict)

    def reset(self):
        self.macs = self.reads = self.conversions = self.conversions_out = self.pulses = self.writes = self.events = self.cache_writes = self.modulates = 0

    def snapshot(self):
        return {"macs": self.macs, "reads": self.reads, "conversions": self.conversions, "conversions_out": self.conversions_out, "pulses": self.pulses,
                "writes": self.writes, "events": self.events, "cache_writes": self.cache_writes, "modulates": self.modulates}


class _FALinear(torch.autograd.Function):
    """y = x W + b with a fixed random feedback matrix B in the backward pass (feedback alignment)."""

    @staticmethod
    def forward(ctx, x, W_read, b, B):
        ctx.save_for_backward(x, B)
        y = x @ W_read
        if b is not None:
            y = y + b
        ctx.has_bias = b is not None
        return y

    @staticmethod
    def backward(ctx, gy):
        x, B = ctx.saved_tensors
        gx = gy @ B.T                                   # feedback through B, not W^T
        gW = x.reshape(-1, x.shape[-1]).T @ gy.reshape(-1, gy.shape[-1])
        gb = gy.reshape(-1, gy.shape[-1]).sum(0) if ctx.has_bias else None
        return gx, gW, gb, None


class ReadoutSpec:
    """How tile outputs leave the analog domain (E8b read-out modes).

    mode  "tile":  one b-bit ADC per 32-row tile partial sum, partial sums added digitally;
          "layer": partial sums accumulated in analog across the tiles of a layer, one b-bit ADC
                   per layer output;
          "lif":   a LIF neuron per layer output, no ADC: the output current is integrated and the
                   spike count (at most 2^b - 1 per token, sign carried by a complementary pair)
                   encodes |y|; spikes are counted in ``spikes`` for the energy model.
    Ranges are calibrated on one batch (``calibrating=True`` records the max |value| seen per
    read site); afterwards values outside the range clip, as a real converter would.
    """

    DEFAULT = ("layer", 8, 10.0)    # architecture default (2026-10-19): layer-accumulate, 8-bit, FoM_W 10 fJ

    def __init__(self, mode: str, bits: int, calib_pct: float = 100.0, auto: bool = False, fom_fJ: float = 10.0):
        """calib_pct: the converter range is this percentile of |value| on the calibration batch
        (100 = the maximum; 99.9 clips the rare outliers, the usual choice for a low-bit converter).
        auto: no calibration pass; the range of each read site follows the largest |value| seen so far
        (an automatic-gain converter, the default for training runs).  fom_fJ: Walden figure of merit,
        E_ADC = fom x 2^bits (recorded for the energy model)."""
        assert mode in ("tile", "layer", "lif")
        self.mode, self.bits, self.calib_pct = mode, int(bits), float(calib_pct)
        self.auto, self.fom_fJ = bool(auto), float(fom_fJ)
        self.calibrating = False
        self.ranges: dict = {}
        self.spikes = 0
        self.spike_sites = 0        # output values that went through the LIF readout
        self.clipped = 0            # values that hit the converter rail

    @classmethod
    def default(cls):
        m, b, f = cls.DEFAULT
        return cls(m, b, auto=True, fom_fJ=f)

    @property
    def E_ADC_J(self) -> float:
        return self.fom_fJ * 1e-15 * 2 ** self.bits

    def reset_counts(self):
        self.spikes = 0; self.spike_sites = 0; self.clipped = 0

    def convert(self, v: torch.Tensor, site: str) -> torch.Tensor:
        """Quantise ``v`` through this read-out; the gradient passes straight through (the converter is in
        the forward path of training runs since the layer-accumulate 8-bit read-out became the default)."""
        q = self._convert(v.detach(), site)
        return v + (q - v.detach()) if v.requires_grad else q

    def _convert(self, v: torch.Tensor, site: str) -> torch.Tensor:
        if self.auto:
            m = v.detach().abs().max()
            r = self.ranges.get(site)
            self.ranges[site] = m if r is None else torch.maximum(r, m)
            R = torch.clamp(self.ranges[site], min=1e-9)
            if self.mode == "lif":
                n_max = 2 ** self.bits - 1; step = R / n_max
                n = torch.clamp(torch.floor(v.abs() / step), max=n_max)
                self.spikes += int(n.sum()); self.spike_sites += n.numel()
                return torch.sign(v) * n * step
            q_max = 2 ** (self.bits - 1) - 1; step = R / (q_max + 1)
            return torch.clamp(torch.round(v / step), -q_max - 1, q_max) * step
        if self.calibrating:
            a = v.detach().abs().flatten()
            if self.calib_pct >= 100.0:
                m = float(a.max())
            else:
                if a.numel() > 400000:
                    a = a[torch.randint(0, a.numel(), (400000,), device=a.device)]
                m = float(torch.quantile(a.float(), self.calib_pct / 100.0))
            self.ranges[site] = max(self.ranges.get(site, 0.0), m)
            return v
        R = max(self.ranges.get(site, 0.0), 1e-9)
        if self.mode == "lif":
            n_max = 2 ** self.bits - 1
            step = R / n_max
            n = torch.floor(v.abs() / step)                   # integrate-and-fire: whole thresholds crossed
            self.clipped += int((n > n_max).sum())
            n = torch.clamp(n, max=n_max)
            self.spikes += int(n.sum()); self.spike_sites += n.numel()
            return torch.sign(v) * n * step
        q_max = 2 ** (self.bits - 1) - 1
        step = R / (q_max + 1)
        q = torch.round(v / step)
        self.clipped += int(((q > q_max) | (q < -q_max - 1)).sum())
        return torch.clamp(q, -q_max - 1, q_max) * step


class DFAContext:
    """Holds the global output error for direct feedback alignment: set ``err`` (n_vectors x out_dim)
    before calling backward; every DFA layer projects it through its own fixed random matrix."""

    def __init__(self):
        self.err = None


class _DFALinear(torch.autograd.Function):
    """y = x W + b; backward uses the GLOBAL error projected by a fixed random matrix B (out_dim x
    out_features) instead of the incoming gradient, and sends a zero gradient upstream (fully local:
    no gradient travels between layers, the error is broadcast to every layer)."""

    @staticmethod
    def forward(ctx, x, W_read, b, B, holder):
        ctx.save_for_backward(x, B)
        ctx.holder = holder
        ctx.has_bias = b is not None
        y = x @ W_read
        return y + b if b is not None else y

    @staticmethod
    def backward(ctx, gy):
        x, B = ctx.saved_tensors
        e = ctx.holder.err                                   # (n_vec, out_dim) global error
        g = e @ B                                            # (n_vec, out_features) local error signal
        gW = x.reshape(-1, x.shape[-1]).T @ g
        gb = g.sum(0) if ctx.has_bias else None
        return torch.zeros_like(x), gW, gb, None, None


def F_pad(x, pad):
    return torch.nn.functional.pad(x, (0, pad)) if pad else x


def F_pad_rows(W, pad):
    return torch.nn.functional.pad(W, (0, 0, 0, pad)) if pad else W


class TileLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, sigma: float | None = 0.0,
                 split=(1.0, 1.0, 1.0), alpha: float = 0.02, beta: float = 3.0, scale: float = 0.5,
                 density: float = 1.0, feedback_alignment: bool = False, counter: OpCounter | None = None,
                 tile: int = 32, generator: torch.Generator | None = None, init_std: float = 0.02,
                 write_mode: str = "nearest_pulse", levels: int | None = None, dfa: DFAContext | None = None, dfa_dim: int = 0,
                 theta_flush: float = 1.0, readout="default"):
        """sigma: total effective device noise (None = ideal float layer).  ``split`` gives the relative
        weights of (d2d, c2c, read) so that sigma_eff = sigma; default equal split.

        write_mode: "nearest_pulse" (default since 2026-10-19): the 16-bit accumulator is flushed every
        step as the nearest whole pulses and the nominal effect is subtracted; "pulse" (legacy, E8): the
        same rounding but the realised (noisy) change is subtracted and the accumulator is float; "level"
        (the 2026-10-18 write budget): accumulate-and-flush, a synapse is written only when its
        accumulator holds theta_flush conductance levels (1 / L in state units, L = round(1 / sigma_eff)
        unless ``levels`` is given), stochastic rounding, flushed amount subtracted; "float": float32
        weights with ideal updates and the layer's read noise / dropout injected on every read (the
        reference arm of the variability experiments).
        readout: "default" = ReadoutSpec.default() (layer-accumulate 8-bit, auto range) on device tiles,
        None = no read-out emulation, or a ReadoutSpec."""
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.ideal = sigma is None
        assert write_mode in ("nearest_pulse", "pulse", "level", "float"), write_mode
        self.float_weights = write_mode == "float"
        self.alpha, self.beta, self.scale = alpha, beta, scale
        self.tile = tile
        self.counter = counter
        self.fa = feedback_alignment
        self.write_mode = write_mode
        self.theta_flush = float(theta_flush)       # level mode: flush when |acc| >= theta_flush levels
        self.acc_step = scale * 2.0 ** -15          # nearest_pulse mode: 16-bit signed accumulator, full scale = scale
        self.dropout = 0.0                          # per-read synapse failure probability (set_dropout)
        self.primitive_log: list = []
        self.readout: ReadoutSpec | None = (ReadoutSpec.default() if readout == "default" else readout) if not (self.ideal or self.float_weights) else None
        self.site = f"tile{id(self)}"
        self.levels = int(levels) if levels else (256 if (sigma is None or sigma <= 0) else int(max(2, min(256, round(1.0 / float(sigma))))))
        self.n_acc_calls = 0
        self.n_pulses = 0
        self.n_writes = 0
        g = generator
        W0 = torch.randn(in_features, out_features, generator=g) * init_std
        self.W = nn.Parameter(W0)
        self.b = nn.Parameter(torch.zeros(out_features)) if bias else None
        mask = (torch.rand(in_features, out_features, generator=g) < density) if density < 1.0 else torch.ones(in_features, out_features, dtype=torch.bool)
        self.register_buffer("A", mask)
        if feedback_alignment:
            self.register_buffer("B_fa", torch.randn(in_features, out_features, generator=g) / math.sqrt(out_features))
        self.dfa = dfa
        if dfa is not None:
            self.register_buffer("B_dfa", torch.randn(dfa_dim, out_features, generator=g) / math.sqrt(dfa_dim))
        if self.float_weights:
            s = torch.tensor(split, dtype=torch.float32); s = s / s.norm() * float(sigma or 0.0)
            self.sigma_d2d, self.sigma_c2c, self.sigma_r = 0.0, 0.0, float(s[2])
        elif not self.ideal:
            s = torch.tensor(split, dtype=torch.float32); s = s / s.norm() * float(sigma)
            self.sigma_d2d, self.sigma_c2c, self.sigma_r = (float(v) for v in s)
            self.register_buffer("gp", torch.clamp(torch.randn(in_features, out_features, generator=g) * self.sigma_d2d + 1.0, 0.05))
            self.register_buffer("gm", torch.clamp(torch.randn(in_features, out_features, generator=g) * self.sigma_d2d + 1.0, 0.05))
            self.register_buffer("wp", torch.zeros(in_features, out_features))
            self.register_buffer("wm", torch.zeros(in_features, out_features))
            self.register_buffer("pending", torch.zeros(in_features, out_features))
            self.n_pulses = 0
            self.n_writes = 0
            self.load(W0, verify=False)
        with torch.no_grad():
            self.W.mul_(self.A)

    # ------------------------------------------------------------------ state
    @property
    def n_tiles(self) -> int:
        return math.ceil(self.in_features / self.tile) * math.ceil(self.out_features / self.tile)

    @property
    def n_devices(self) -> int:
        return 2 * self.in_features * self.out_features

    @torch.no_grad()
    def load(self, W_target: torch.Tensor, verify: bool = True, tol: float = 0.01):
        """Program the pair state to W_target (write-verify emulated as programming to within tol
        with one pulse-resolution error draw); sets self.W to the realised weight."""
        if self.ideal or self.float_weights:
            self.W.copy_(W_target * self.A); return
        t = torch.clamp(W_target / self.scale, -1.0, 1.0)
        wp = torch.clamp(t, 0, 1); wm = torch.clamp(-t, 0, 1)
        if verify:
            # residual error of write-verify: uniform within the tolerance floor max(tol, half a pulse)
            res_p = 0.5 * self.alpha * torch.exp(-self.beta * wp) * self.gp
            res_m = 0.5 * self.alpha * torch.exp(-self.beta * wm) * self.gm
            wp = wp + (torch.rand_like(wp) * 2 - 1) * torch.maximum(res_p, torch.full_like(res_p, tol)) * (wp > 0)
            wm = wm + (torch.rand_like(wm) * 2 - 1) * torch.maximum(res_m, torch.full_like(res_m, tol)) * (wm > 0)
        self.wp.copy_(torch.clamp(wp, 0, 1)); self.wm.copy_(torch.clamp(wm, 0, 1))
        self.pending.zero_()
        self.W.copy_(self.scale * self.A * (self.wp - self.wm))

    # ---------------------------------------------------------------- forward
    def read_weight(self) -> torch.Tensor:
        if self.ideal:
            return self.W * self.A
        W = self.W * self.A
        if self.sigma_r > 0:
            if self.float_weights:
                # the reference arm: the same multiplicative Gaussian read noise on float32 weights
                W = W + (self.sigma_r * W.abs() * torch.randn_like(W)).detach()
            else:
                # independent read noise on each device of the pair: (w+ e1 - w- e2) has variance w+^2 + w-^2,
                # so one Gaussian draw scaled by sqrt(w+^2 + w-^2) is the same distribution at half the cost
                W = W + (self.scale * self.A * self.sigma_r * torch.sqrt(self.wp ** 2 + self.wm ** 2) * torch.randn_like(self.wp)).detach()
        if self.dropout > 0:
            W = W * (torch.rand_like(W) >= self.dropout)
        return W

    # ------------------------------------------------------------ primitives
    MODULATE_RANGE = (0.005, 0.2)

    def set_noise(self, sigma_r: float) -> float:
        """MODULATE primitive: set this layer's read-noise level (relative, per read); range 0.005-0.2,
        0 = dial off; values outside the range are clipped and the clip is logged."""
        assert not self.ideal, "an ideal layer has no read noise to modulate"
        lo, hi = self.MODULATE_RANGE
        v = float(sigma_r); clipped = v != 0.0 and (v < lo or v > hi)
        if self.float_weights:
            # 2026-10-23 fix: the float32 REFERENCE has no dial, so its injected noise is not clipped (until this
            # date it was, and every E11 float32 arm above 0.2 ran its Linear layers at 0.2; DECISIONS.md)
            clipped = False
        elif v != 0.0:
            v = float(min(max(v, lo), hi))
        self.sigma_r = v
        self.primitive_log.append({"primitive": "MODULATE", "call": self.n_acc_calls, "sigma_r": v, "clipped": clipped})
        if self.counter is not None and self.counter.enabled:
            self.counter.modulates += 1
        return v

    def set_dropout(self, p: float) -> None:
        """DROPOUT primitive: every read drops each synapse independently with probability p."""
        assert 0.0 <= p < 1.0, p
        self.dropout = float(p)
        self.primitive_log.append({"primitive": "DROPOUT", "call": self.n_acc_calls, "p": self.dropout})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_read = self.read_weight()
        if self.counter is not None and self.counter.enabled:
            n_vec = x.numel() // x.shape[-1]
            self.counter.macs += n_vec * int(self.A.sum()) if self.A.dtype == torch.bool else n_vec * self.in_features * self.out_features
            self.counter.reads += n_vec * self.n_tiles
            self.counter.conversions += n_vec * self.out_features * math.ceil(self.in_features / self.tile)
            self.counter.conversions_out += n_vec * self.out_features
            self.counter.events += n_vec
        if self.fa:
            return _FALinear.apply(x, W_read, self.b, self.B_fa)
        if self.dfa is not None:
            return _DFALinear.apply(x, W_read, self.b, self.B_dfa, self.dfa)
        ro = self.readout
        if ro is not None and not self.ideal:
            if ro.mode == "tile":
                # partial sums per 32-row block of inputs, each converted on its own, then added digitally
                n_in = self.in_features; t = self.tile; nch = math.ceil(n_in / t)
                pad = nch * t - n_in
                xp = F_pad(x, pad); Wp = F_pad_rows(W_read, pad)
                xs = xp.reshape(-1, nch, t); Ws = Wp.reshape(nch, t, self.out_features)
                partial = torch.einsum("nci,cio->nco", xs, Ws)
                y = ro.convert(partial, self.site).sum(1).reshape(*x.shape[:-1], self.out_features)
            else:
                y = ro.convert(x @ W_read, self.site)
        else:
            y = x @ W_read
        return y + self.b if self.b is not None else y

    # ----------------------------------------------------------------- writes
    @torch.no_grad()
    def apply_delta(self, dW: torch.Tensor, theta_w: float = 0.0) -> torch.Tensor:
        """Write a requested weight change through quantised, noisy, soft-bounded pulses.

        Returns the realised change.  Sub-pulse residuals stay in ``pending``.
        """
        if self.ideal or self.float_weights:
            self.n_acc_calls += 1
            self.W.add_(dW * self.A); return dW * self.A
        self.pending.add_(dW * self.A)
        if self.write_mode == "nearest_pulse":    # 16-bit signed accumulator, full scale = scale
            torch.clamp(torch.round(self.pending / self.acc_step) * self.acc_step, -self.scale + self.acc_step, self.scale - self.acc_step, out=self.pending)
        self.n_acc_calls += 1
        if self.write_mode == "level":
            lvl = self.scale / self.levels                     # one conductance level in weight units
            q = self.pending.abs() / lvl
            gate = q >= self.theta_flush
            if not bool(gate.any()):
                return torch.zeros_like(dW)
            n_int = torch.floor(q)
            n_lv = n_int + (torch.rand_like(q) < (q - n_int)).to(q.dtype)
            flushed = torch.where(gate & (n_lv > 0), torch.sign(self.pending) * n_lv * lvl, torch.zeros_like(q))
            req = flushed / self.scale
        else:
            flushed = None
            req = self.pending / self.scale                    # requested change in state units
            if theta_w > 0:
                req = torch.where(req.abs() >= theta_w / self.scale, req, torch.zeros_like(req))
        pos = torch.clamp(req, min=0); neg = torch.clamp(-req, min=0)
        # route positive requests into w+ headroom, the remainder out of w- ; mirror for negative
        d_wp_up = torch.minimum(pos, 1 - self.wp); d_wm_down = torch.minimum(pos - d_wp_up, self.wm)
        d_wm_up = torch.minimum(neg, 1 - self.wm); d_wp_down = torch.minimum(neg - d_wm_up, self.wp)
        d_wp = d_wp_up - d_wp_down; d_wm = d_wm_up - d_wm_down
        step_p = torch.where(d_wp >= 0, self.alpha * torch.exp(-self.beta * self.wp), self.alpha * torch.exp(-self.beta * (1 - self.wp)))
        step_m = torch.where(d_wm >= 0, self.alpha * torch.exp(-self.beta * self.wm), self.alpha * torch.exp(-self.beta * (1 - self.wm)))
        n_p = torch.round(d_wp / step_p); n_m = torch.round(d_wm / step_m)
        noise_p = 1 + self.sigma_c2c * torch.randn_like(self.wp) if self.sigma_c2c > 0 else 1.0
        noise_m = 1 + self.sigma_c2c * torch.randn_like(self.wm) if self.sigma_c2c > 0 else 1.0
        real_p = n_p * step_p * self.gp * noise_p
        real_m = n_m * step_m * self.gm * noise_m
        wp_new = torch.clamp(self.wp + real_p, 0, 1); wm_new = torch.clamp(self.wm + real_m, 0, 1)
        realised_state = (wp_new - self.wp) - (wm_new - self.wm)
        if self.write_mode == "nearest_pulse":
            # the digital side knows only the nominal effect of the pulses it sent (no read-back)
            nominal = self.scale * self.A * ((torch.clamp(self.wp + n_p * step_p, 0, 1) - self.wp) - (torch.clamp(self.wm + n_m * step_m, 0, 1) - self.wm))
        self.wp.copy_(wp_new); self.wm.copy_(wm_new)
        realised = self.scale * self.A * realised_state
        if self.write_mode == "nearest_pulse":
            self.pending.sub_(nominal)
            torch.clamp(torch.round(self.pending / self.acc_step) * self.acc_step, -self.scale + self.acc_step, self.scale - self.acc_step, out=self.pending)
        else:
            self.pending.sub_(flushed if flushed is not None else realised)
        self.W.copy_(self.scale * self.A * (self.wp - self.wm))
        pulses = int(n_p.abs().sum() + n_m.abs().sum()); writes = int(((n_p != 0) | (n_m != 0)).sum())
        self.n_pulses += pulses; self.n_writes += writes
        if self.counter is not None and self.counter.enabled:
            self.counter.pulses += pulses; self.counter.writes += writes
        return realised

    def writes_per_device(self, seconds: float | None = None) -> float:
        if self.ideal or self.float_weights:
            return 0.0
        w = self.n_writes / self.n_devices
        return w / seconds if seconds else w

    @torch.no_grad()
    def retain(self, dt: float, tau_ret: float):
        """Retention: every device relaxes toward G_min (w = 0) with time constant tau_ret over dt seconds."""
        if self.ideal or self.float_weights or not math.isfinite(tau_ret) or tau_ret <= 0:
            return
        f = math.exp(-dt / tau_ret)
        self.wp.mul_(f); self.wm.mul_(f)
        self.W.copy_(self.scale * self.A * (self.wp - self.wm))

    @property
    def flush_rate(self) -> float:
        """Fraction of write opportunities (apply_delta calls x synapses) that pulsed a device."""
        if self.ideal or self.float_weights or self.n_acc_calls == 0:
            return 0.0
        return self.n_writes / (self.n_acc_calls * self.in_features * self.out_features)

    ENDURANCE = 1e6; TILE_BANDWIDTH = 1e4; LIFELONG_RATE = 6e-3


def write_budget_report(tiles, seconds: float | None = None) -> dict:
    """Aggregate write-budget report over TileLinear layers: endurance (total writes per device vs 1e6),
    write-driver bandwidth (write events per physical tile per second vs 1e4), and the old lifelong
    rate (writes per device per second vs 6e-3)."""
    tiles = [t for t in tiles if isinstance(t, TileLinear) and not t.ideal and not t.float_weights]
    if not tiles:
        return {"writes_per_device_total": 0.0, "percent_of_endurance": 0.0, "flush_rate": 0.0}
    n_w = sum(t.n_writes for t in tiles); n_dev = sum(t.n_devices for t in tiles); n_tiles = sum(t.n_tiles for t in tiles)
    wpd = n_w / n_dev
    rep = {"writes_per_device_total": wpd, "percent_of_endurance": 100.0 * wpd / TileLinear.ENDURANCE,
           "flush_rate": float(sum(t.flush_rate for t in tiles) / len(tiles)), "levels": tiles[0].levels, "theta_flush": tiles[0].theta_flush,
           "write_mode": tiles[0].write_mode, "n_modulate": int(sum(sum(1 for e in t.primitive_log if e["primitive"] == "MODULATE") for t in tiles)),
           "max_writes_per_device_any_tile": max(t.n_writes / t.n_devices for t in tiles)}
    if seconds:
        per_tile_s = n_w / n_tiles / seconds
        rep.update({"writes_per_tile_per_s": per_tile_s, "percent_of_tile_bandwidth": 100.0 * per_tile_s / TileLinear.TILE_BANDWIDTH,
                    "max_writes_per_tile_per_s": max(t.n_writes / t.n_tiles for t in tiles) / seconds,
                    "writes_per_device_per_s": wpd / seconds, "percent_of_lifelong_rate": 100.0 * wpd / seconds / TileLinear.LIFELONG_RATE})
    return rep


class TileOptimizer:
    """Wraps a torch optimizer: the optimizer proposes a float step, the tiles realise it with pulses."""

    def __init__(self, modules, opt: torch.optim.Optimizer, theta_w: float = 0.0):
        self.tiles = [m for m in modules if isinstance(m, TileLinear)]
        self.opt = opt
        self.theta_w = theta_w

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        before = [t.W.detach().clone() for t in self.tiles]
        self.opt.step()
        for t, W0 in zip(self.tiles, before):
            if t.ideal or t.float_weights:
                t.W.mul_(t.A); t.n_acc_calls += 1; continue
            dW = t.W - W0
            t.W.copy_(W0)
            t.apply_delta(dW, self.theta_w)


def tile_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, TileLinear)]


def tile_attention(q, k, v, sigma_w: float, sigma_r: float, counter: OpCounter | None = None, causal: bool = True,
                   tile: int = 32, readout: ReadoutSpec | None = None, site: str = "attn"):
    """Hopfield-form attention through volatile crossbars.

    Keys and values are *written* to the cache (multiplicative write noise sigma_w), the scores
    q K^T are one tile read and the output softmax(.) V a second tile read, each with fresh read
    noise sigma_r on the stored devices.  Shapes: (B, H, T, d).
    """
    B, H, T, d = q.shape
    if sigma_w > 0:
        k = k * (1 + sigma_w * torch.randn_like(k)); v = v * (1 + sigma_w * torch.randn_like(v))
    k_read = k * (1 + sigma_r * torch.randn_like(k)) if sigma_r > 0 else k
    scores = (q @ k_read.transpose(-2, -1)) / math.sqrt(d)
    if readout is not None:          # the score read leaves the key crossbar through the read-out (head dim = one tile)
        scores = readout.convert(scores, site + ":scores")
    if causal:
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
    att = torch.softmax(scores, dim=-1)
    v_read = v * (1 + sigma_r * torch.randn_like(v)) if sigma_r > 0 else v
    out = att @ v_read
    if readout is not None:
        if readout.mode == "tile":   # the value read spans ceil(T / 32) key tiles: one conversion per block
            nb = math.ceil(T / tile); pad = nb * tile - T
            att_p = torch.nn.functional.pad(att, (0, pad)); v_p = torch.nn.functional.pad(v_read, (0, 0, 0, pad))
            partial = torch.einsum("bhtcj,bhcjd->bhtcd", att_p.reshape(B, H, T, nb, tile), v_p.reshape(B, H, nb, tile, d))
            out = readout.convert(partial, site + ":values").sum(3)
        else:
            out = readout.convert(out, site + ":values")
    if counter is not None and counter.enabled:
        n_pairs = T * (T + 1) // 2 if causal else T * T
        counter.macs += 2 * B * H * n_pairs * d
        counter.reads += 2 * B * H * T * math.ceil(T / tile) * math.ceil(d / tile)
        counter.conversions += B * H * (n_pairs + T * d) * math.ceil(d / tile)
        counter.conversions_out += B * H * (n_pairs + T * d)
        counter.cache_writes += 2 * B * H * T * d
        counter.events += 2 * B * H * T
    return out
