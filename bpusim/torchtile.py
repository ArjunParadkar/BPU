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
  pair, sub-pulse residuals are kept in a pending accumulator (the surprise
  gate), so writes are sparse and quantised;
* optional feedback alignment: the backward pass carries the input
  gradient through a fixed random matrix instead of ``W^T``.

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
    enabled: bool = True
    log: dict = field(default_factory=dict)

    def reset(self):
        self.macs = self.reads = self.conversions = self.conversions_out = self.pulses = self.writes = self.events = self.cache_writes = 0

    def snapshot(self):
        return {"macs": self.macs, "reads": self.reads, "conversions": self.conversions, "conversions_out": self.conversions_out, "pulses": self.pulses,
                "writes": self.writes, "events": self.events, "cache_writes": self.cache_writes}


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


class TileLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, sigma: float | None = 0.0,
                 split=(1.0, 1.0, 1.0), alpha: float = 0.02, beta: float = 3.0, scale: float = 0.5,
                 density: float = 1.0, feedback_alignment: bool = False, counter: OpCounter | None = None,
                 tile: int = 32, generator: torch.Generator | None = None, init_std: float = 0.02):
        """sigma: total effective device noise (None = ideal float layer).  ``split`` gives the relative
        weights of (d2d, c2c, read) so that sigma_eff = sigma; default equal split."""
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.ideal = sigma is None
        self.alpha, self.beta, self.scale = alpha, beta, scale
        self.tile = tile
        self.counter = counter
        self.fa = feedback_alignment
        g = generator
        W0 = torch.randn(in_features, out_features, generator=g) * init_std
        self.W = nn.Parameter(W0)
        self.b = nn.Parameter(torch.zeros(out_features)) if bias else None
        mask = (torch.rand(in_features, out_features, generator=g) < density) if density < 1.0 else torch.ones(in_features, out_features, dtype=torch.bool)
        self.register_buffer("A", mask)
        if feedback_alignment:
            self.register_buffer("B_fa", torch.randn(in_features, out_features, generator=g) / math.sqrt(out_features))
        if not self.ideal:
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
        if self.ideal:
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
        if self.ideal or self.sigma_r == 0:
            return self.W * self.A
        # independent read noise on each device of the pair: (w+ e1 - w- e2) has variance w+^2 + w-^2,
        # so one Gaussian draw scaled by sqrt(w+^2 + w-^2) is the same distribution at half the cost
        noise = self.scale * self.A * self.sigma_r * torch.sqrt(self.wp ** 2 + self.wm ** 2) * torch.randn_like(self.wp)
        return self.W * self.A + noise.detach()

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
        y = x @ W_read
        return y + self.b if self.b is not None else y

    # ----------------------------------------------------------------- writes
    @torch.no_grad()
    def apply_delta(self, dW: torch.Tensor, theta_w: float = 0.0) -> torch.Tensor:
        """Write a requested weight change through quantised, noisy, soft-bounded pulses.

        Returns the realised change.  Sub-pulse residuals stay in ``pending``.
        """
        if self.ideal:
            self.W.add_(dW * self.A); return dW * self.A
        self.pending.add_(dW * self.A)
        req = self.pending / self.scale                        # requested change in state units
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
        self.wp.copy_(wp_new); self.wm.copy_(wm_new)
        realised = self.scale * self.A * realised_state
        self.pending.sub_(realised)
        self.W.copy_(self.scale * self.A * (self.wp - self.wm))
        pulses = int(n_p.abs().sum() + n_m.abs().sum()); writes = int(((n_p != 0) | (n_m != 0)).sum())
        self.n_pulses += pulses; self.n_writes += writes
        if self.counter is not None and self.counter.enabled:
            self.counter.pulses += pulses; self.counter.writes += writes
        return realised

    def writes_per_device(self, seconds: float | None = None) -> float:
        if self.ideal:
            return 0.0
        w = self.n_writes / self.n_devices
        return w / seconds if seconds else w


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
            if t.ideal:
                t.W.mul_(t.A); continue
            dW = t.W - W0
            t.W.copy_(W0)
            t.apply_delta(dW, self.theta_w)


def tile_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, TileLinear)]


def tile_attention(q, k, v, sigma_w: float, sigma_r: float, counter: OpCounter | None = None, causal: bool = True,
                   tile: int = 32):
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
    if causal:
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1)
        scores = scores.masked_fill(mask, float("-inf"))
    att = torch.softmax(scores, dim=-1)
    v_read = v * (1 + sigma_r * torch.randn_like(v)) if sigma_r > 0 else v
    out = att @ v_read
    if counter is not None and counter.enabled:
        n_pairs = T * (T + 1) // 2 if causal else T * T
        counter.macs += 2 * B * H * n_pairs * d
        counter.reads += 2 * B * H * T * math.ceil(T / tile) * math.ceil(d / tile)
        counter.conversions += B * H * (n_pairs + T * d) * math.ceil(d / tile)
        counter.conversions_out += B * H * (n_pairs + T * d)
        counter.cache_writes += 2 * B * H * T * d
        counter.events += 2 * B * H * T
    return out
