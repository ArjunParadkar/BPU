"""Bit-accurate fixed-point reference for the BPU FPGA tile (rung ii of the validation ladder).

This module IS the contract.  Every number in it is an integer and every operation is one a
digital circuit performs (add, multiply, shift, compare, table look-up); there is no floating
point anywhere in the datapath.  The FPGA (``fpga/hdl``) and the PS firmware (``fpga/ps``) must
reproduce it bit for bit; ``docs/reference_contract.md`` is the prose version and
``tests/test_fixed.py`` ties it to the float model in ``bpusim.tile`` / ``bpusim.device``.

Why this file exists.  ``bpusim.tile.Tile`` is a float64 model: it has no DAC code width, no
conductance Q-format, no accumulator width and no ADC; its noise comes from numpy's PCG64 +
ziggurat ``standard_normal`` which no sane piece of RTL reproduces.  The only integer
quantisation bpusim had before this file was the 16-bit write accumulator (``Tile.ACC_BITS``)
and the 8-bit round-half-even read-out in ``torchtile.ReadoutSpec`` (mode ``"tile"``).  So the
fixed-point formats are DEFINED here (chosen to represent the float model with the resolution
the paper assumes: 256 conductance levels, 8-bit read-out) and the float model is the thing
this reference approximates, not the other way round.  ``tests/test_fixed.py`` measures the
approximation error against ``Tile`` and asserts it stays inside the analytic quantisation
bound; that test is the only place where the two models meet.

Two noise modes, both bit-exact (see the contract, section 5):

* ``"stream"`` (mode A): the caller supplies the quantised noise samples ``eta_q`` (Q1.14, one
  per device per read).  ``eta_stream_from_numpy`` turns a numpy generator's Gaussian draws
  into that stream, i.e. the float simulator's own noise, quantised.  The FPGA DMAs the same
  stream in.  This is the mode the bit-exact gate uses.
* ``"prng"`` (mode B): 64 xoshiro128** generators (one per (column, device)) seeded by
  splitmix64 from a 64-bit seed, Gaussian by a 1024-entry inverse-CDF table in Q3.13, scaled
  by sigma_q (Q2.14).  Implemented identically here and in ``fpga/hdl/eta_prng.sv``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

# ----------------------------------------------------------------------------- formats (contract §2)
ROWS = 32                 # word lines (activations)
COLS = 32                 # bit lines (outputs)
ACT_BITS = 8              # DAC resolution per pass: int8 two's complement code, x = code / 128 (Q1.7)
ACT_PASSES_MAX = 2        # a 16-bit activation (Q1.15) is converted as two 8-bit passes, see shift_add
W_BITS = 8                # stored state per device: w_q in [0, 255], w = w_q / 255 (256 levels)
W_LEVELS = (1 << W_BITS) - 1
GMIN_Q = 655              # conductance code of w = 0  (G_min = 1 uS)
GSTEP = 254               # conductance code per state level; G_q = GMIN_Q + GSTEP * w_q, max 65425 (16 bits)
GMAX_Q = GMIN_Q + GSTEP * W_LEVELS
G_LSB_S = (100e-6 - 1e-6) / (GSTEP * W_LEVELS)   # siemens per conductance code (1.5285e-9 S)
ETA_FRAC = 14             # eta_q is Q1.14: eta = eta_q / 16384
ETA_ONE = 1 << ETA_FRAC
ETA_MAX = ETA_ONE - 1     # |eta_q| <= 16383, so the gain (ETA_ONE + eta_q) is in [1, 32767]
GHAT_BITS = 17            # G_hat_q = (G_q * (ETA_ONE + eta_q) + 2^13) >> 14   (max 130846 < 2^17), exact, no saturation
CURRENT_BITS = 32         # per-column I+ / I- : |d| <= 255, G_hat < 2^17, 32 rows  ->  < 2^30, signed 32
ACC_BITS = 40             # Y_j = sum_pass I_j << (8 * pass): < 2^39 for 2 passes, signed 40
ADC_BITS = 8              # SAR ADC: y = sat8( rne( Y / 2^S ) ), S in [0, 31]
ADC_MAX = (1 << (ADC_BITS - 1)) - 1
ADC_MIN = -(1 << (ADC_BITS - 1))
SIGMA_FRAC = 14           # sigma_q is Q2.14 (sigma = sigma_q / 16384), prng mode only
ZLUT_BITS = 10            # Gaussian inverse-CDF table: 1024 entries, index = top 10 bits of the prng word
ZLUT_FRAC = 13            # table entries are Q3.13 (|z| <= 3.29 fits a signed 16-bit word)
VERIFY_CODE = 64          # one-hot activation code used by write-verify reads: G_hat = I / 64 exactly (shift)

# write path (device model, PS side, contract §6)
WS_FRAC = 15              # internal device state w_s is Q1.15 in [0, 32768]; the read side sees w_q = (w_s + 64) >> 7
WS_ONE = 1 << WS_FRAC
GAIN_FRAC = 14            # d2d gain g_q Q2.14 (1.0 = 16384), floored at 0.05 -> 819
GAIN_ONE = 1 << GAIN_FRAC
GAIN_MIN = 819
LUT_BITS = 10             # step / cumulative-pulse tables indexed by the top 10 bits of w_s
TOL_Q = 655               # program_target tol 0.02 in Q1.15
DAMP_Q = 11469            # damping 0.7 in Q2.14
VERIFY_READS = 8          # reads averaged per verify (sum >> 3)
MAX_ITER = 100


def rne(x):
    """Round half to even (numpy.rint == torch.round) -> int64.  Float -> int boundary only."""
    return np.rint(np.asarray(x, dtype=np.float64)).astype(np.int64)


def sat(x, lo, hi):
    return np.clip(np.asarray(x, dtype=np.int64), lo, hi)


def rne_shift(x, s: int):
    """Round-half-to-even arithmetic right shift of a signed integer by s bits (the SAR ADC rounding).

    r = floor(x / 2^s); the remainder decides: > half -> r + 1, == half -> round to even, < half -> r."""
    x = np.asarray(x, dtype=np.int64)
    if s == 0:
        return x.copy()
    r = x >> s                                 # floor
    rem = x - (r << s)                         # 0 <= rem < 2^s
    half = np.int64(1) << (s - 1)
    up = (rem > half) | ((rem == half) & ((r & 1) == 1))
    return r + up.astype(np.int64)


# ----------------------------------------------------------------------------- conversions float -> code
def act_to_code(x, passes: int = 1):
    """Activation in [-1, 1) -> integer code: int8 (Q1.7) for one pass, int16 (Q1.15) for two."""
    frac = 7 if passes == 1 else 15
    hi = (1 << frac) - 1
    return sat(rne(np.asarray(x, float) * (1 << frac)), -hi - 1, hi)


def w_to_code(w):
    """Device state in [0, 1] -> w_q in [0, 255] (round half even)."""
    return sat(rne(np.clip(np.asarray(w, float), 0.0, 1.0) * W_LEVELS), 0, W_LEVELS)


def eta_to_code(eta):
    """Relative noise sample -> eta_q (Q1.14, clipped to +-16383)."""
    return sat(rne(np.asarray(eta, float) * ETA_ONE), -ETA_MAX, ETA_MAX)


def sigma_to_code(sigma: float) -> int:
    return int(sat(rne(sigma * (1 << SIGMA_FRAC)), 0, (1 << 16) - 1))


def eta_stream_from_numpy(rng: np.random.Generator, sigma_r: float, n_reads: int = 1):
    """Mode A: draw the float simulator's own read noise and quantise it into the FPGA stream order.

    bpusim draws ``gp.read()`` (32x32 row-major) then ``gm.read()``; the stream interleaves them as
    index = ((row * 32 + col) * 2 + device), device 0 = G+, 1 = G-, one block of 2048 per read."""
    out = np.empty((n_reads, ROWS, COLS, 2), np.int64)
    for k in range(n_reads):
        ep = eta_to_code(sigma_r * rng.standard_normal((ROWS, COLS)))
        em = eta_to_code(sigma_r * rng.standard_normal((ROWS, COLS)))
        out[k, :, :, 0] = ep
        out[k, :, :, 1] = em
    return out.reshape(n_reads, ROWS * COLS * 2)


# ----------------------------------------------------------------------------- the PRNG (contract §5B)
M32 = (1 << 32) - 1
M64 = (1 << 64) - 1


def splitmix64(state: int):
    """One splitmix64 step: returns (new_state, output)."""
    state = (state + 0x9E3779B97F4A7C15) & M64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return state, z ^ (z >> 31)


def rotl32(x: int, k: int) -> int:
    return ((x << k) | (x >> (32 - k))) & M32


class Xoshiro128ss:
    """xoshiro128** (Blackman & Vigna), 32-bit words; the FPGA instance is identical."""

    def __init__(self, s0: int, s1: int, s2: int, s3: int):
        self.s = [s0 & M32, s1 & M32, s2 & M32, s3 & M32]

    def next(self) -> int:
        s = self.s
        result = (rotl32((s[1] * 5) & M32, 7) * 9) & M32
        t = (s[1] << 9) & M32
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = rotl32(s[3], 11)
        return result


def prng_seed_words(seed64: int, instance: int):
    """Seed of generator ``instance`` (0..63, = 2 * col + device): four successive splitmix64 outputs
    (low 32 bits each) of the state seed64 + instance * 0x9E3779B97F4A7C15 * 4 (so instances never overlap)."""
    st = (seed64 + (instance * 4 * 0x9E3779B97F4A7C15)) & M64
    words = []
    for _ in range(4):
        st, z = splitmix64(st)
        words.append(z & M32)
    if all(w == 0 for w in words):              # xoshiro must not be seeded with all zeros
        words[0] = 1
    return words


def gaussian_lut():
    """Inverse-CDF table: z_q[k] = rne( Phi^-1((k + 0.5) / 1024) * 2^13 ), k = 0..1023 (Q3.13, |z| <= 3.29)."""
    from scipy.special import ndtri
    k = np.arange(1 << ZLUT_BITS)
    return rne(ndtri((k + 0.5) / (1 << ZLUT_BITS)) * (1 << ZLUT_FRAC))


_ZLUT = None


def zlut():
    global _ZLUT
    if _ZLUT is None:
        _ZLUT = gaussian_lut()
    return _ZLUT


class EtaPRNG:
    """Mode B noise source: 64 generators, one per (column, device), each advanced once per row per read."""

    def __init__(self, seed64: int, sigma_q: int):
        self.gens = [Xoshiro128ss(*prng_seed_words(seed64, m)) for m in range(2 * COLS)]
        self.sigma_q = int(sigma_q)
        self.lut = zlut()

    def eta(self, col: int, device: int) -> int:
        word = self.gens[2 * col + device].next()
        z = int(self.lut[word >> (32 - ZLUT_BITS)])
        e = (self.sigma_q * z + (1 << (ZLUT_FRAC - 1))) >> ZLUT_FRAC    # Q2.14 * Q3.13 -> Q.27; floor((. + 2^12) / 2^13) == >>> in SV
        return max(-ETA_MAX, min(ETA_MAX, e))

    def read_block(self):
        """The 2048 samples of one read in stream order (row, col, device)."""
        out = np.empty(ROWS * COLS * 2, np.int64)
        for r in range(ROWS):
            for c in range(COLS):
                out[(r * COLS + c) * 2 + 0] = self.eta(c, 0)
                out[(r * COLS + c) * 2 + 1] = self.eta(c, 1)
        return out


# ----------------------------------------------------------------------------- the read datapath (contract §3-4)
@dataclass
class TileReadResult:
    y_adc: np.ndarray        # (32,) int8 codes
    Y: np.ndarray            # (32,) pre-ADC accumulators (signed 40-bit)
    I_plus: np.ndarray       # (passes, 32) column currents of the G+ array per pass (signed 32)
    I_minus: np.ndarray      # (passes, 32)
    clipped: int             # number of columns that hit an ADC rail


def conductance_q(w_q):
    return GMIN_Q + GSTEP * np.asarray(w_q, np.int64)


def noisy_conductance_q(g_q, eta_q):
    """G_hat_q = floor((G_q * (2^14 + eta_q) + 2^13) / 2^14); exact (17 bits), no saturation."""
    return (np.asarray(g_q, np.int64) * (ETA_ONE + np.asarray(eta_q, np.int64)) + (1 << (ETA_FRAC - 1))) >> ETA_FRAC


def dac_pass_codes(x_code, passes: int):
    """wl_dac: the row drive per pass.  passes = 1: d = x (int8).  passes = 2: pass 0 = low byte unsigned
    [0, 255], pass 1 = high byte signed [-128, 127], so x = 256 * d1 + d0 exactly."""
    x = np.asarray(x_code, np.int64)
    if passes == 1:
        return [x]
    return [x & 0xFF, x >> 8]


def tile_read(w_plus_q, w_minus_q, mask, x_code, eta_q, adc_shift: int, passes: int = 1) -> TileReadResult:
    """One tile read, bit-exact.

    w_plus_q, w_minus_q : (32, 32) uint8 state codes (row = input, col = output)
    mask                : (32, 32) bool connectivity A
    x_code              : (32,) int8 (passes = 1) or int16 (passes = 2) activation codes
    eta_q               : (2048,) int16 noise stream in order ((row * 32 + col) * 2 + device)
    adc_shift           : S, the ADC range exponent (step = 2^S current codes)
    """
    wp = np.asarray(w_plus_q, np.int64); wm = np.asarray(w_minus_q, np.int64)
    A = np.asarray(mask, bool)
    eta = np.asarray(eta_q, np.int64).reshape(ROWS, COLS, 2)
    gp_hat = noisy_conductance_q(conductance_q(wp), eta[:, :, 0])
    gm_hat = noisy_conductance_q(conductance_q(wm), eta[:, :, 1])
    gp_hat = np.where(A, gp_hat, 0); gm_hat = np.where(A, gm_hat, 0)
    Ip = np.zeros((passes, COLS), np.int64); Im = np.zeros((passes, COLS), np.int64)
    Y = np.zeros(COLS, np.int64)
    for p, d in enumerate(dac_pass_codes(x_code, passes)):
        Ip[p] = (d[:, None] * gp_hat).sum(axis=0)          # xbar_tile + col_integrate: exact row-serial sum
        Im[p] = (d[:, None] * gm_hat).sum(axis=0)
        Y += (Ip[p] - Im[p]) << (ACT_BITS * p)               # differential readout + shift_add
    q = rne_shift(Y, int(adc_shift))                         # sar_adc
    y = sat(q, ADC_MIN, ADC_MAX)
    clipped = int(((q < ADC_MIN) | (q > ADC_MAX)).sum())
    return TileReadResult(y.astype(np.int8), Y, Ip, Im, clipped)


def verify_read(w_q, mask_row, eta_row_q, row: int):
    """write-verify read of one word line: drive row ``row`` with code VERIFY_CODE, all other rows 0, one
    pass; returns the 32 column currents of ONE device array (int64) -- G_hat_q * 64 for every masked column.
    ``eta_row_q`` are the 32 noise samples of that array's row."""
    g = noisy_conductance_q(conductance_q(np.asarray(w_q, np.int64)[row]), np.asarray(eta_row_q, np.int64))
    g = np.where(np.asarray(mask_row, bool), g, 0)
    return g * VERIFY_CODE


# ----------------------------------------------------------------------------- the write path (contract §6)
def _lut_index(w_s):
    return np.minimum(np.asarray(w_s, np.int64) >> (WS_FRAC - LUT_BITS), (1 << LUT_BITS) - 1)   # w_s = 2^15 uses the last entry


_TABLES = None


def device_tables(alpha_p=0.05, beta_p=3.0, alpha_d=0.05, beta_d=3.0):
    """The four 1024-entry tables of the fixed-point device model, indexed by the top 10 bits of w_s
    (w = (k + 0.5) / 1024 at the table centre):
      STEP_UP[k]  = rne( alpha_p * exp(-beta_p * w) * 2^15 )          potentiation step, Q1.15
      STEP_DN[k]  = rne( alpha_d * exp(-beta_d * (1 - w)) * 2^15 )    depression step, Q1.15
      NUP[k]      = rne( exp(beta_p * w) / (alpha_p * beta_p) * 2^8 ) cumulative pulse index, Q.8
      NDN[k]      = rne( exp(beta_d * (1 - w)) / (alpha_d * beta_d) * 2^8 )"""
    k = np.arange(1 << LUT_BITS)
    w = (k + 0.5) / (1 << LUT_BITS)
    return {
        "STEP_UP": rne(alpha_p * np.exp(-beta_p * w) * WS_ONE),
        "STEP_DN": rne(alpha_d * np.exp(-beta_d * (1.0 - w)) * WS_ONE),
        "NUP": rne(np.exp(beta_p * w) / (alpha_p * beta_p) * 256),
        "NDN": rne(np.exp(beta_d * (1.0 - w)) / (alpha_d * beta_d) * 256),
    }


def tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = device_tables()
    return _TABLES


def ws_to_wq(w_s):
    """Read-side state: w_q = min(255, (w_s + 64) >> 7)."""
    return np.minimum((np.asarray(w_s, np.int64) + 64) >> (WS_FRAC - W_BITS), W_LEVELS)


def step_q(w_s, gain_q, c2c_q, direction):
    """Size of one pulse in Q1.15 for direction +1 (potentiate) / -1 (depress):
    floor((STEP[idx] * gain_q * (2^14 + c2c_q) + 2^27) / 2^28), gain Q2.14, c2c Q1.14."""
    t = tables()
    idx = _lut_index(w_s)
    base = np.where(np.asarray(direction) > 0, t["STEP_UP"][idx], t["STEP_DN"][idx])
    prod = base * np.asarray(gain_q, np.int64) * (ETA_ONE + np.asarray(c2c_q, np.int64))
    return (prod + (1 << (GAIN_FRAC + ETA_FRAC - 1))) >> (GAIN_FRAC + ETA_FRAC)


def pulse(w_s, n, gain_q, c2c_q_fn, dead=None):
    """Apply signed pulse counts ``n`` one pulse at a time (the saturating law holds exactly), each pulse with a
    fresh c2c sample from ``c2c_q_fn(shape)``; w_s clamped to [0, 2^15].  Returns the new state."""
    w = np.asarray(w_s, np.int64).copy()
    n = np.asarray(n, np.int64).copy()
    if dead is not None:
        n[np.asarray(dead, bool)] = 0
    remaining = n.copy()
    while True:
        active = remaining != 0
        if not active.any():
            break
        direction = np.sign(remaining)
        s = step_q(w, gain_q, c2c_q_fn(w.shape), direction)
        w = np.where(active, np.clip(w + direction * s, 0, WS_ONE), w)
        remaining -= direction
    return w


def pulses_for(w_s, w_target_s):
    """Nominal signed pulse count from w_s to w_target_s (both Q1.15):
    up:   rne8( NUP[idx(target)] - NUP[idx(now)] )   (Q.8 -> integer, round half to even on the 8 fraction bits)
    down: -rne8( NDN[idx(target)] - NDN[idx(now)] )"""
    t = tables()
    i0 = _lut_index(w_s); i1 = _lut_index(w_target_s)
    n_up = rne_shift(t["NUP"][i1] - t["NUP"][i0], 8)
    n_dn = rne_shift(t["NDN"][i1] - t["NDN"][i0], 8)
    return np.where(np.asarray(w_target_s) >= np.asarray(w_s), n_up, -n_dn)


def w_meas_from_sum(g_sum, n_reads: int = VERIFY_READS):
    """Measured state from the sum of n_reads verify currents (each G_hat * 64):
    w_meas_s = clamp( ((g_sum - n*64*GMIN_Q) * 2^15 + n*64*GSTEP*255/2) // (n*64*GSTEP*255), 0, 2^15 ).
    Integer division with round-half-up; values below G_min clamp to 0."""
    den = n_reads * VERIFY_CODE * GSTEP * W_LEVELS
    num = (np.asarray(g_sum, np.int64) - n_reads * VERIFY_CODE * GMIN_Q) * WS_ONE
    num = np.maximum(num, 0)
    return np.minimum((num + den // 2) // den, WS_ONE)


def program_target(w_s, w_target_s, gain_q, read_fn, c2c_q_fn, dead=None, tol_q=TOL_Q, max_iter=MAX_ITER,
                   damping_q=DAMP_Q, log=None):
    """Write-verify toward a band: read (VERIFY_READS averaged), compare, pulse the damped nominal count (at least
    one), repeat until |w_target - w_meas| <= max(tol, half a pulse) for every device or max_iter.

    read_fn(w_s) -> array of VERIFY_READS summed verify currents per device (the PL path incl. read noise).
    Returns (w_s, iterations, total_pulses)."""
    w = np.asarray(w_s, np.int64).copy()
    tgt = np.clip(np.asarray(w_target_s, np.int64), 0, WS_ONE)
    gain_q = np.asarray(gain_q, np.int64)
    dead_m = np.zeros(w.shape, bool) if dead is None else np.asarray(dead, bool)
    total = 0
    for it in range(1, max_iter + 1):
        w_meas = w_meas_from_sum(read_fn(w))
        err = tgt - w_meas
        t = tables(); idx = _lut_index(w)
        res = (np.maximum(t["STEP_UP"][idx], t["STEP_DN"][idx]) * gain_q) >> (GAIN_FRAC + 1)   # half a pulse
        todo = (np.abs(err) > np.maximum(tol_q, res)) & ~dead_m
        if log is not None:
            log.append({"iter": it, "max_abs_err": int(np.abs(err).max()), "todo": int(todo.sum())})
        if not todo.any():
            return w, it - 1, total
        aim = np.clip(w + ((damping_q * err) >> GAIN_FRAC), 0, WS_ONE)
        n = pulses_for(w, aim)
        n = np.where(n == 0, np.sign(err), n)
        n = np.where(todo, n, 0)
        total += int(np.abs(n).sum())
        w = pulse(w, n, gain_q, c2c_q_fn, dead_m)
    return w, max_iter, total
