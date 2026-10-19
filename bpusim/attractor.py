"""L4: continuous attractors on memristor tiles.

:class:`RingAttractor`
    N rate neurons at angles theta_i with W(d) = J0 + J1 cos(d) programmed on
    a tile, plus a second tile holding W'(d) = -J1 sin(d) for velocity shift::

        tau dr/dt = -r + f(u),   u = (scale/N) [W r + gamma v W' r] + I

    A bump forms iff J1 > 2 (J0 < 1 keeps the uniform mode stable).  The bump
    angle is decoded by the population vector.  Mismatch is modelled as
    programmed weights W (1 + sigma_d2d zeta_ij) where zeta is the tile's own
    fabricated device-to-device deviation; dynamic noise is read noise on
    every tile read plus additive neuron noise.

:class:`GridModule`
    A 2D periodic sheet with a difference-of-Gaussians kernel on three tiles
    (W, dW/dx, dW/dy); 2D velocity shift; phase = lattice position decoded
    from the Fourier peak.

:class:`SelfOrganizingSheet`
    Generic tile + rate neurons + STDP + growth with a fixed global
    inhibition, driven by a moving external cue; used by E4 (ring) and E5
    (grid) to grow an attractor from a sparse random start.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .device import DeviceParams
from .tile import Tile
from .neurons import RateNeurons, get_activation
from .plasticity import STDP, normalize_columns
from .growth import GrowthRule


def circ_dist(a, b):
    """Signed circular difference a - b wrapped to (-pi, pi]."""
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2 * np.pi) - np.pi


def population_vector(r, theta):
    """Decode an angle from rates r on preferred angles theta."""
    r = np.asarray(r, float)
    return float(np.arctan2(np.sum(r * np.sin(theta)), np.sum(r * np.cos(theta))))


def program_with_mismatch(tile: Tile, W: np.ndarray, sigma_d2d: float, tol: float = 0.01):
    """Program W (|W| <= 1) with the paper's mismatch model W (1 + sigma zeta_ij).

    zeta is derived from the tile's fabricated d2d gains of the device that
    carries the weight (G+ for positive entries, G- for negative), so the
    mismatch is a fixed property of the hardware, not a fresh draw.  If the
    tile's own sigma_d2d is zero the mismatch is zero.
    """
    gain = np.where(W >= 0, tile.gp.d2d_gain, tile.gm.d2d_gain)
    if tile.p.sigma_d2d > 0:
        zeta = (gain - 1.0) / tile.p.sigma_d2d
    else:
        zeta = np.zeros_like(W)
    W_prog = np.clip(W * (1.0 + sigma_d2d * zeta), -1.0, 1.0)
    tile.program(W_prog, tol=tol)
    return W_prog


class RingAttractor:
    def __init__(self, N: int = 32, J0: float = -0.5, J1: float = 3.0, I: float = 1.0,
                 tau: float = 10e-3, dt: float = 1e-3, gamma: float = 1.0, f="relu",
                 rng: np.random.Generator | None = None, params: DeviceParams | None = None,
                 sigma_mismatch: float | None = None, sigma_neuron: float = 0.0,
                 use_tiles: bool = True, tau_H: float = 0.0, tau_rate: float = 20.0):
        """tau_H > 0 enables homeostatic scaling (Lesson 5.7): each neuron's input gain h_j follows
        tau_H dh_j/dt = h_j (r* - r_bar_j)/r*, with r* the population-mean running rate, applied as
        a read-side gain (one analog parameter per neuron, no device rewrites).  ``tau_rate`` must be
        long compared with the time the bump takes to visit the whole ring (20 s here): with a 1 s
        average the gain chased the moving bump and destroyed path integration (RMSE 1.25 rad on a
        mismatch-free ring that integrates to 0.05 rad without homeostasis)."""
        self.rng = rng or np.random.default_rng(0)
        self.N, self.J0, self.J1, self.I = int(N), float(J0), float(J1), float(I)
        self.tau, self.dt, self.gamma = tau, dt, gamma
        self.tau_H, self.tau_rate = tau_H, tau_rate
        self.h = np.ones(self.N)
        self.r_bar = None
        self.theta = 2 * np.pi * np.arange(self.N) / self.N
        d = self.theta[None, :] - self.theta[:, None]            # theta_post - theta_pre
        self.W_ideal = J0 + J1 * np.cos(d)
        self.Wp_ideal = -J1 * np.sin(d)
        self.scale = abs(J0) + abs(J1)
        self.use_tiles = use_tiles
        self.p = params or DeviceParams()
        self.neurons = RateNeurons(self.N, self.rng, tau=tau, dt=dt, f=f, sigma=sigma_neuron)
        if use_tiles:
            self.tile_W = Tile(self.N, self.N, self.rng, self.p)
            self.tile_Wp = Tile(self.N, self.N, self.rng, self.p)
            sm = self.p.sigma_d2d if sigma_mismatch is None else sigma_mismatch
            program_with_mismatch(self.tile_W, self.W_ideal / self.scale, sm)
            program_with_mismatch(self.tile_Wp, self.Wp_ideal / self.scale, sm)
        self.reset()

    def set_dynamic_rng(self, rng: np.random.Generator):
        """Use ``rng`` for all *dynamic* randomness (read noise, neuron noise) from now on,
        leaving the fabricated mismatch (already drawn) untouched.  Lets a sweep pair
        the same hardware and trajectory across noise levels."""
        self.rng = rng
        self.neurons.rng = rng
        if self.use_tiles:
            for t in (self.tile_W, self.tile_Wp):
                t.rng = rng; t.gp.rng = rng; t.gm.rng = rng
        return self

    # ------------------------------------------------------------------ state
    def reset(self, angle: float | None = None, amp: float = 0.1):
        """Start from small random activity, or a weak bump at ``angle``."""
        if angle is None:
            r0 = amp * self.rng.random(self.N)
        else:
            r0 = amp * (1 + np.cos(self.theta - angle))
        self.neurons.reset(r0)

    @property
    def r(self):
        return self.neurons.r

    def recurrent_input(self, r, v: float = 0.0):
        if self.use_tiles:
            u = self.tile_W.read(r)
            if v != 0.0:
                u = u + self.gamma * v * self.tile_Wp.read(r)
            u = u * self.scale
        else:
            u = r @ self.W_ideal
            if v != 0.0:
                u = u + self.gamma * v * (r @ self.Wp_ideal)
        return u / self.N

    def step(self, v: float = 0.0, I_ext=None):
        u = self.recurrent_input(self.neurons.r, v)
        if self.tau_H > 0:
            u = u * self.h
        u = u + self.I
        if I_ext is not None:
            u = u + I_ext
        r = self.neurons.step(u)
        if self.tau_H > 0:
            a = self.dt / self.tau_rate
            # start the running mean uniform at the population mean: seeding it with the instantaneous
            # bump would make homeostasis push the bump away from wherever it happened to start
            self.r_bar = np.full(self.N, float(r.mean())) if self.r_bar is None else self.r_bar + a * (r - self.r_bar)
            r_star = max(float(self.r_bar.mean()), 1e-9)
            self.h += (self.dt / self.tau_H) * self.h * (r_star - self.r_bar) / r_star
            np.clip(self.h, 0.5, 2.0, out=self.h)
        return r

    def decode(self) -> float:
        return population_vector(self.neurons.r, self.theta)

    def pinning_landscape(self, n_pos: int = 64, settle: int = 300, probe: int = 200) -> dict:
        """Drift velocity of the free bump vs start position at the current (dynamic) noise.

        The bump is placed at n_pos angles, settled, and its drift over ``probe`` steps measured.
        Integrating the drift velocity around the ring gives the potential U(phi) up to a constant
        (phi_dot = -dU/dphi); the barrier height estimate is max U - min U.
        """
        phis = np.linspace(-np.pi, np.pi, n_pos, endpoint=False)
        vel = np.empty(n_pos)
        for i, phi in enumerate(phis):
            self.reset(angle=phi); self.settle(settle)
            a0 = self.decode(); self.settle(probe); a1 = self.decode()
            vel[i] = circ_dist(a1, a0) / (probe * self.dt)
        dphi = 2 * np.pi / n_pos
        U = -np.cumsum(vel) * dphi
        U -= U.mean()
        return {"phi": phis.tolist(), "drift_velocity": vel.tolist(), "U": U.tolist(),
                "delta_U": float(U.max() - U.min()), "rms_drift": float(np.sqrt(np.mean(vel ** 2)))}

    def bump_modulation(self) -> float:
        """(max - min) / mean of the rate profile; ~0 for the uniform state."""
        r = self.neurons.r
        return float((r.max() - r.min()) / max(r.mean(), 1e-12))

    def settle(self, n_steps: int = 300, v: float = 0.0):
        for _ in range(n_steps):
            self.step(v)
        return self.neurons.r.copy()

    def run(self, velocity: np.ndarray, record: bool = True):
        """Integrate a velocity trace; returns decoded angles (and rates if record)."""
        angles = np.empty(len(velocity))
        rates = np.empty((len(velocity), self.N)) if record else None
        for t, v in enumerate(velocity):
            r = self.step(float(v))
            angles[t] = population_vector(r, self.theta)
            if record:
                rates[t] = r
        return (angles, rates) if record else angles

    def calibrate_velocity_gain(self, v: float = 1.0, T: float = 0.5) -> float:
        """Measure d(angle)/dt per unit v on the *ideal* network (no tiles, no noise)."""
        ideal = RingAttractor(self.N, self.J0, self.J1, self.I, self.tau, self.dt, self.gamma,
                              rng=np.random.default_rng(0), use_tiles=False)
        ideal.reset(0.0)
        ideal.settle(500)
        a0 = ideal.decode()
        n = int(T / self.dt)
        ang = ideal.run(np.full(n, v), record=False)
        total = np.sum(circ_dist(ang[1:], ang[:-1])) + circ_dist(ang[0], a0)
        return float(total / (v * T))


# ---------------------------------------------------------------------------
# Grid module
# ---------------------------------------------------------------------------
def torus_offsets(nx, ny):
    """Signed minimal offsets (dx, dy) between every pair of sites on an nx x ny torus."""
    ix, iy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    X = ix.ravel(); Y = iy.ravel()
    dx = (X[None, :] - X[:, None] + nx // 2) % nx - nx // 2   # post - pre
    dy = (Y[None, :] - Y[:, None] + ny // 2) % ny - ny // 2
    return dx.astype(float), dy.astype(float)


def dog_kernel(dx, dy, a=1.0, s_e=1.0, b=0.5, s_i=2.0):
    d2 = dx ** 2 + dy ** 2
    return a * np.exp(-d2 / (2 * s_e ** 2)) - b * np.exp(-d2 / (2 * s_i ** 2))


def dog_gradient(dx, dy, a=1.0, s_e=1.0, b=0.5, s_i=2.0):
    """dW/d(dx), dW/d(dy) of the difference of Gaussians."""
    d2 = dx ** 2 + dy ** 2
    ge = a * np.exp(-d2 / (2 * s_e ** 2)) / s_e ** 2
    gi = b * np.exp(-d2 / (2 * s_i ** 2)) / s_i ** 2
    common = -(ge - gi)
    return common * dx, common * dy


class GridModule:
    """2D periodic sheet with DoG recurrence and 2D velocity shift on three tiles.

    Default kernel: zero-centre difference of Gaussians (a = b = 1, s_i = 2 s_e),
    i.e. a purely inhibitory surround with uniform excitatory drive I.  With
    gain 2 the Fourier transform of the kernel peaks at 3.7 > 1 at a nonzero
    wavenumber while the uniform mode has W(0) << 1, so a hexagonal pattern
    forms on 16x16 and a single bump on 5x6, and the relu activity stays bounded
    (a positive-centre DoG with relu neurons and no saturation runs away).
    """

    def __init__(self, nx: int = 16, ny: int = 16, a: float = 1.0, s_e: float = 1.0,
                 b: float = 1.0, s_i: float = 2.0, gain: float = 2.0, I: float = 1.0,
                 tau: float = 10e-3, dt: float = 1e-3, gamma: float = 1.0, f="relu",
                 rng=None, params: DeviceParams | None = None, sigma_mismatch: float | None = None,
                 sigma_neuron: float = 0.0, use_tiles: bool = True):
        self.rng = rng or np.random.default_rng(0)
        self.nx, self.ny = int(nx), int(ny)
        self.N = self.nx * self.ny
        self.tau, self.dt, self.gamma, self.I = tau, dt, gamma, I
        dx, dy = torus_offsets(self.nx, self.ny)
        self.dx, self.dy = dx, dy
        self.W_ideal = gain * dog_kernel(dx, dy, a, s_e, b, s_i)
        gx, gy = dog_gradient(dx, dy, a, s_e, b, s_i)
        self.Wx_ideal, self.Wy_ideal = gain * gx, gain * gy
        self.scale = max(np.abs(self.W_ideal).max(), np.abs(self.Wx_ideal).max(), np.abs(self.Wy_ideal).max())
        self.p = params or DeviceParams()
        self.use_tiles = use_tiles
        self.neurons = RateNeurons(self.N, self.rng, tau=tau, dt=dt, f=f, sigma=sigma_neuron)
        if use_tiles:
            sm = self.p.sigma_d2d if sigma_mismatch is None else sigma_mismatch
            self.tile_W = Tile(self.N, self.N, self.rng, self.p)
            self.tile_Wx = Tile(self.N, self.N, self.rng, self.p)
            self.tile_Wy = Tile(self.N, self.N, self.rng, self.p)
            program_with_mismatch(self.tile_W, self.W_ideal / self.scale, sm)
            program_with_mismatch(self.tile_Wx, self.Wx_ideal / self.scale, sm)
            program_with_mismatch(self.tile_Wy, self.Wy_ideal / self.scale, sm)
        self.reset()

    def reset(self, amp: float = 0.1):
        self.neurons.reset(amp * self.rng.random(self.N))

    @property
    def r(self):
        return self.neurons.r

    def sheet(self):
        return self.neurons.r.reshape(self.nx, self.ny)

    def recurrent_input(self, r, vx=0.0, vy=0.0):
        if self.use_tiles:
            u = self.tile_W.read(r)
            if vx:
                u = u + self.gamma * vx * self.tile_Wx.read(r)
            if vy:
                u = u + self.gamma * vy * self.tile_Wy.read(r)
            u = u * self.scale
        else:
            u = r @ self.W_ideal
            if vx:
                u = u + self.gamma * vx * (r @ self.Wx_ideal)
            if vy:
                u = u + self.gamma * vy * (r @ self.Wy_ideal)
        return u

    def step(self, vx=0.0, vy=0.0, I_ext=None):
        u = self.recurrent_input(self.neurons.r, vx, vy) + self.I
        if I_ext is not None:
            u = u + I_ext
        return self.neurons.step(u)

    def settle(self, n_steps=300):
        for _ in range(n_steps):
            self.step()
        return self.neurons.r.copy()

    def lattice_wavevectors(self, n: int = 2):
        """The ``n`` dominant non-collinear integer wavevectors of the current sheet (upper half-plane)."""
        S = self.sheet() - self.sheet().mean()
        F = np.abs(np.fft.fft2(S))
        F[0, 0] = 0
        order = np.argsort(F.ravel())[::-1]
        found = []
        for flat in order:
            kx, ky = np.unravel_index(flat, F.shape)
            kx = kx if kx <= self.nx // 2 else kx - self.nx
            ky = ky if ky <= self.ny // 2 else ky - self.ny
            if (ky < 0) or (ky == 0 and kx <= 0):
                continue  # keep one of each +-k pair
            if any(kx * q[1] - ky * q[0] == 0 for q in found):
                continue  # collinear with one already kept
            found.append((int(kx), int(ky)))
            if len(found) == n:
                break
        return found

    def decode_phase(self, ks=None):
        """Lattice phase in [0,1)^2: the phases of the Fourier coefficients at the two lattice wavevectors.

        For a single bump the two wavevectors are (1,0) and (0,1) and the phase is the
        bump position in units of the sheet; for a multi-bump lattice it is the lattice
        offset modulo one lattice period.
        """
        ks = ks or self.lattice_wavevectors(2)
        S = self.sheet() - self.sheet().mean()
        F = np.fft.fft2(S)
        out = []
        for (kx, ky) in ks:
            out.append((-np.angle(F[kx % self.nx, ky % self.ny]) / (2 * np.pi)) % 1.0)
        return np.array(out)

    def shift_between(self, S0, S1):
        """Integer (dx, dy) circular shift that best maps S0 onto S1 (cross-correlation peak)."""
        C = np.real(np.fft.ifft2(np.fft.fft2(S1) * np.conj(np.fft.fft2(S0))))
        dx, dy = np.unravel_index(np.argmax(C), C.shape)
        dx = dx if dx <= self.nx // 2 else dx - self.nx
        dy = dy if dy <= self.ny // 2 else dy - self.ny
        return int(dx), int(dy)

    def centroid(self):
        """Circular centroid of activity on the torus, in units of sites."""
        S = self.sheet()
        ax = 2 * np.pi * np.arange(self.nx) / self.nx
        ay = 2 * np.pi * np.arange(self.ny) / self.ny
        px = np.arctan2((S.sum(1) * np.sin(ax)).sum(), (S.sum(1) * np.cos(ax)).sum()) % (2 * np.pi)
        py = np.arctan2((S.sum(0) * np.sin(ay)).sum(), (S.sum(0) * np.cos(ay)).sum()) % (2 * np.pi)
        return np.array([px * self.nx / (2 * np.pi), py * self.ny / (2 * np.pi)])


# ---------------------------------------------------------------------------
# Self-organisation: sparse random tile + STDP + growth, driven by a moving cue
# ---------------------------------------------------------------------------
@dataclass
class SelfOrgConfig:
    density0: float = 0.1          # initial random connectivity
    budget_density: float = 0.75   # synapse budget as a fraction of N*N (a 32-ring needs ~0.47 for the
                                   # near-neighbour kernel alone; the rest holds weak far synapses)
    w0: float = 0.02               # initial random weight magnitude
    J_inh: float = 2.5             # fixed global inhibition (not learned)
    I_tonic: float = 1.0           # uniform background drive (the I of the ring equations)
    gain: float = 12.0             # recurrent gain at test (a unit-norm cos+ kernel sustains a bump from gain ~ 11)
    gain_dev: float = 3.0          # recurrent gain during development: below the bump threshold, so the
                                   # cue, not a self-sustained bump, drives the activity that STDP sees
    tau_H: float = 0.0             # homeostatic time constant (s); 0 disables.  With a uniform sweep cue
                                   # homeostasis is unnecessary, and with a random-walk cue it imprinted the
                                   # cue's coverage statistics on the weights (see DECISIONS.md), so E4/E5 run
                                   # without it; set e.g. 50.0 to enable
    tau_rate: float = 10.0         # running-mean rate time constant (s)
    write_mode: str = "threshold"  # "threshold": write when |pending| >= theta_w, keep the sub-pulse residual
                                   # (the original E4); "level": the tile's accumulate-and-flush write budget
                                   # (a synapse is pulsed only when one conductance level 1/L is pending,
                                   # stochastic rounding of the remainder; theta_w is then unused)
    write_budget: float = 0.0      # max write events per device per second during development (0 = no
                                   # cap).  With a cap, pending STDP changes are flushed in order of size
                                   # only as the budget allows (a per-tile write quota), so learning takes
                                   # proportionally longer in simulated time
    stdp_mode: str = "spike"       # "spike": Bernoulli spikes through the STDP traces; "rate": the
                                   # expected STDP change (A+ tau+ - A- tau-) rho_i rho_j dt, no sampling noise
    norm_rows: bool = True         # also normalise presynaptic rows (axonal resource constraint); without
                                   # it row sums vary ~4x around the ring and pin the bump
    I_cue: float = 1.5             # cue amplitude
    cue_width: float = 1.0         # cue width (radians on a ring, sites on a sheet); sets the kernel width
    rate_dt: float = 1e-3
    tau: float = 10e-3
    A_plus: float = 0.01
    A_minus: float = 0.005           # A+ > A-: co-active pairs potentiate on balance (a balanced
                                     # window gives zero net change for symmetric co-activity)
    tau_plus: float = 20e-3
    tau_minus: float = 20e-3
    spike_rate_scale: float = 100.0  # Hz at r = 1
    eta: float = 0.02              # STDP -> tile write scaling
    theta_w: float = 0.01          # write threshold
    norm_every: int = 5000         # steps between normalisations.  Every normalisation rewrites the
                                   # whole tile through write-verify and each rewrite adds a pulse-resolution
                                   # error (~0.02 in w near w = 0); 200 rewrites accumulate ~0.3, 8 rewrites ~0.06
    col_norm: float = 1.0          # target column L2 norm (in tile units, |W|<=1)
    theta_grow: float = 2e-4
    theta_prune: float = 0.03
    T_prune: int = 20
    struct_every: int = 100        # steps between grow/prune
    sigma_neuron: float = 0.02
    r_max: float = 2.0             # rate saturation (bounds the Hebbian positive feedback)
    max_grow_per_step: int | None = None   # synapses grown per structural step (default max(8, N // 4))


class SelfOrganizingSheet:
    """Grow a continuous attractor from a sparse random tile under a moving cue.

    ``coords`` are neuron coordinates (angles for a ring, (x, y) site positions
    for a torus); ``cue_fn(t)`` returns the cue position; ``dist_fn(coords, pos)``
    returns the distance of each neuron from the cue.
    """

    def __init__(self, coords, dist_fn, rng, cfg: SelfOrgConfig | None = None,
                 params: DeviceParams | None = None):
        self.cfg = cfg or SelfOrgConfig()
        self.rng = rng
        self.coords = coords
        self.dist_fn = dist_fn
        self.N = len(coords)
        c = self.cfg
        self.p = params or DeviceParams()
        mask = rng.random((self.N, self.N)) < c.density0
        np.fill_diagonal(mask, False)
        self.tile = Tile(self.N, self.N, rng, self.p, mask=mask)
        W0 = np.where(mask, rng.uniform(0, c.w0, (self.N, self.N)), 0.0)
        self.tile.program(W0, tol=0.01)
        self.neurons = RateNeurons(self.N, rng, tau=c.tau, dt=c.rate_dt,
                                   f=lambda u: np.clip(u, 0.0, c.r_max), sigma=c.sigma_neuron)
        self.stdp = STDP(self.N, self.N, c.A_plus, c.A_minus, c.tau_plus, c.tau_minus, dt=c.rate_dt)
        self.growth = GrowthRule(self.tile, theta_grow=c.theta_grow, theta_prune=c.theta_prune,
                                 T_prune=c.T_prune, budget=int(c.budget_density * self.N * self.N),
                                 tau_C=200, tau_mean=50, seed_w=0.05,
                                 max_grow_per_call=c.max_grow_per_step or max(8, self.N // 4))
        self.pending = np.zeros((self.N, self.N))
        self.r_bar = np.full(self.N, np.nan)   # running mean rate per neuron
        self.h = np.ones(self.N)                # homeostatic scale of each neuron's incoming column
        self.gain = self.cfg.gain_dev           # developmental gain; finish_development() raises it
        self.write_credit = 0.0                 # accumulated write allowance under a write budget
        self.t = 0
        self.log = {"n_active": [], "step": []}

    def cue(self, pos):
        d = self.dist_fn(self.coords, pos)
        return self.cfg.I_cue * np.exp(-d ** 2 / (2 * self.cfg.cue_width ** 2))

    def input(self, r, I_ext=None):
        c = self.cfg
        u = self.gain * self.tile.read(r) / self.N - c.J_inh * r.mean() + c.I_tonic
        if I_ext is not None:
            u = u + I_ext
        return u

    def step(self, cue_pos=None, learn: bool = True):
        c = self.cfg
        I_ext = self.cue(cue_pos) if cue_pos is not None else None
        r = self.neurons.step(self.input(self.neurons.r, I_ext))
        if learn:
            a = c.rate_dt / c.tau_rate
            self.r_bar = r.copy() if np.isnan(self.r_bar).any() else self.r_bar + a * (r - self.r_bar)
            if c.stdp_mode == "spike":
                p_spike = np.clip(r * c.spike_rate_scale * c.rate_dt, 0, 1)
                spikes = self.rng.random(self.N) < p_spike
                dW = self.stdp.step(spikes, spikes) * c.eta
            else:
                rho = r * c.spike_rate_scale
                dW = (c.A_plus * c.tau_plus - c.A_minus * c.tau_minus) * np.outer(rho, rho) * c.rate_dt * c.eta
            np.fill_diagonal(dW, 0.0)
            if c.write_mode == "level":
                self.tile.accumulate(dW, write_budget=c.write_budget or None, dt=c.rate_dt)
                write = np.zeros_like(self.tile.A)
            else:
                self.pending += dW
            write = (np.abs(self.pending) >= c.theta_w) & self.tile.A if c.write_mode != "level" else write
            if c.write_budget > 0 and c.write_mode != "level":
                # endurance cap: so many write events per step for the whole tile (2 devices per synapse)
                self.write_credit += c.write_budget * 2 * self.N * self.N * c.rate_dt
                n_allow = int(self.write_credit)
                if n_allow < write.sum():
                    order = np.argsort(-np.abs(np.where(write, self.pending, 0.0)).ravel())[:n_allow]
                    sel = np.zeros(self.N * self.N, bool); sel[order] = True
                    write &= sel.reshape(self.N, self.N)
            if write.any():
                realised = self.tile.update(np.where(write, self.pending, 0.0))
                self.pending -= realised          # keep the sub-pulse residual
                if c.write_budget > 0:
                    self.write_credit -= int(write.sum())
            self.growth.observe(1.0, r, r)
            if (self.t + 1) % c.struct_every == 0:
                self.growth.prune()
                self.growth.grow()
                self.pending[~self.tile.A] = 0.0
                self.tile.acc[~self.tile.A] = 0.0
            if (self.t + 1) % c.norm_every == 0:
                self._normalize()
            self.log["n_active"].append(self.tile.n_active)
            self.log["step"].append(self.t)
        self.t += 1
        return r

    def _normalize(self):
        """Oja-style column normalisation with homeostatic scaling.

        Every post-synaptic column is rescaled to L2 norm ``col_norm * h_j`` (excitatory,
        >= 0).  ``h_j`` follows the paper's homeostatic rule tau_H dG/dt = G (r* - r_j)/r*
        with r* = the population-mean rate, applied once per normalisation interval, so
        neurons whose mean rate runs above the population lose input and those below gain
        it; this flattens the mismatch landscape (Lesson 5.7).
        """
        c = self.cfg
        if c.tau_H > 0 and not np.isnan(self.r_bar).any():
            r_star = max(self.r_bar.mean(), 1e-6)
            self.h *= 1.0 + (c.norm_every * c.rate_dt / c.tau_H) * (r_star - self.r_bar) / r_star
            self.h = np.clip(self.h, 0.5, 2.0)
        W = np.clip(self.tile.W, 0, None)
        if c.norm_rows:
            rn = np.linalg.norm(W, axis=1, keepdims=True)
            W = W * (rn.mean() / np.maximum(rn, 1e-9))
        Wn = normalize_columns(W, 1.0) * (self.cfg.col_norm * self.h)[None, :]
        Wn = np.where(self.tile.A, Wn, 0.0)
        self.tile.program(Wn, tol=0.01)

    def develop(self, cue_fn, n_steps: int):
        for k in range(n_steps):
            self.step(cue_fn(k), learn=True)

    def finish_development(self):
        """Final normalisation, then switch from the developmental gain to the operating gain."""
        self._normalize()
        self.gain = self.cfg.gain

    def free_run(self, n_steps: int, I_ext=None):
        out = np.empty((n_steps, self.N))
        for k in range(n_steps):
            out[k] = self.step(None, learn=False) if I_ext is None else self.neurons.step(self.input(self.neurons.r, I_ext))
        return out

    def sample_manifold(self, positions, n_cue: int = 100, n_free: int = 300, n_keep: int = 50):
        """For each start position: cue briefly, release, record the end of the free run.

        Returns (rates (n_pos * n_keep, N), persisted (n_pos,) bool) where
        persisted means the activity stayed structured after the cue was removed.
        """
        clouds, persisted = [], []
        gain_saved, self.gain = self.gain, self.cfg.gain   # always evaluate at the operating gain
        for pos in positions:
            self.neurons.reset(0.05 * self.rng.random(self.N))
            for _ in range(n_cue):
                self.step(pos, learn=False)
            traj = np.empty((n_free, self.N))
            for k in range(n_free):
                traj[k] = self.step(None, learn=False)
            tail = traj[-n_keep:]
            clouds.append(tail)
            r = tail.mean(0)
            persisted.append(bool(r.mean() > 1e-3 and (r.max() - r.min()) / max(r.mean(), 1e-9) > 0.5))
        self.gain = gain_saved
        return np.concatenate(clouds, 0), np.array(persisted)


def ring_dist(theta, pos):
    return np.abs(circ_dist(theta, pos))


def telegraph_walk_1d(rng, n_steps, dt=1e-3, speed=3.0, flip_rate=0.5, jitter=1.0):
    """Exploration cue on a ring: constant speed with Poisson direction reversals plus OU jitter.

    Covers the ring uniformly (many laps) instead of diffusing slowly like a
    pure random walk.
    """
    a = 0.0; sgn = 1.0; j = 0.0
    k = np.exp(-dt / 0.3); s = jitter * np.sqrt(1 - k * k)
    out = np.empty(n_steps)
    for i in range(n_steps):
        if rng.random() < flip_rate * dt:
            sgn = -sgn
        j = k * j + s * rng.standard_normal()
        a += (sgn * speed + j) * dt
        out[i] = a
    return out


def sweep_1d(rng, n_steps, dt=1e-3, speed=3.0, laps_per_leg=1.0, jitter=0.3):
    """Systematic exploration cue on a ring: constant speed, direction reversed every
    ``laps_per_leg`` laps, plus small OU jitter.  Coverage is uniform by construction."""
    a = rng.uniform(-np.pi, np.pi); sgn = 1.0; j = 0.0
    k = np.exp(-dt / 0.3); s = jitter * np.sqrt(1 - k * k)
    leg = 2 * np.pi * laps_per_leg / speed
    out = np.empty(n_steps)
    for i in range(n_steps):
        if i > 0 and (i * dt) % leg < dt:
            sgn = -sgn
        j = k * j + s * rng.standard_normal()
        a += (sgn * speed + j) * dt
        out[i] = a
    return out


def sweep_2d(rng, n_steps, nx, ny, dt=1e-3, speed=3.0, leg=10.0):
    """Systematic exploration cue on a torus: straight motion with an irrational heading
    (a Lissajous-like path that covers the torus uniformly); the heading is reflected
    every ``leg`` seconds so both directions along each axis are visited."""
    pos = np.array([rng.uniform(0, nx), rng.uniform(0, ny)])
    heading = np.arctan(np.sqrt(2.0)) + rng.uniform(-0.1, 0.1)
    out = np.empty((n_steps, 2))
    for i in range(n_steps):
        if i > 0 and (i * dt) % leg < dt:
            heading += np.pi + rng.uniform(-0.2, 0.2)
        pos += speed * dt * np.array([np.cos(heading), np.sin(heading)])
        pos[0] %= nx; pos[1] %= ny
        out[i] = pos
    return out


def telegraph_walk_2d(rng, n_steps, nx, ny, dt=1e-3, speed=3.0, turn_rate=1.0, turn_sigma=1.0):
    """Exploration cue on a torus: constant speed, heading changed by Poisson random turns."""
    pos = np.array([rng.uniform(0, nx), rng.uniform(0, ny)])
    heading = rng.uniform(0, 2 * np.pi)
    out = np.empty((n_steps, 2))
    for i in range(n_steps):
        if rng.random() < turn_rate * dt:
            heading += turn_sigma * rng.standard_normal()
        pos += speed * dt * np.array([np.cos(heading), np.sin(heading)])
        pos[0] %= nx; pos[1] %= ny
        out[i] = pos
    return out


def torus_dist_factory(nx, ny):
    def dist(coords, pos):
        dx = (coords[:, 0] - pos[0] + nx / 2) % nx - nx / 2
        dy = (coords[:, 1] - pos[1] + ny / 2) % ny - ny / 2
        return np.sqrt(dx ** 2 + dy ** 2)
    return dist
