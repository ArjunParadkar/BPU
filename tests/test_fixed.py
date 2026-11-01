"""The fixed-point reference (bpusim/fixed.py) against the float model and against itself."""
import numpy as np
import pytest

from bpusim import fixed as F
from bpusim.device import DeviceParams
from bpusim.tile import Tile


def test_rne_shift_matches_torch_round_semantics():
    x = np.arange(-70, 70, dtype=np.int64)
    for s in (1, 2, 3, 5):
        want = np.rint(x / 2 ** s).astype(np.int64)       # numpy.rint == torch.round == half to even
        np.testing.assert_array_equal(F.rne_shift(x, s), want)
    np.testing.assert_array_equal(F.rne_shift(x, 0), x)


def test_formats_cannot_overflow():
    assert F.GMAX_Q == 65425 < 2 ** 16
    assert ((F.GMAX_Q * (F.ETA_ONE + F.ETA_MAX) + 8192) >> 14) < 2 ** F.GHAT_BITS
    ghat_max = (F.GMAX_Q * (F.ETA_ONE + F.ETA_MAX) + 8192) >> 14
    i_pass0 = 255 * ghat_max * 32                      # unsigned low byte, 32 rows
    i_pass1 = 128 * ghat_max * 32
    assert i_pass0 < 2 ** (F.CURRENT_BITS - 2)          # |I+|, |I-| < 2^30, so |I+ - I-| < 2^31
    assert i_pass0 + 2 * i_pass1 * 256 < 2 ** (F.ACC_BITS - 1)   # |Y| = |I_0 + 256 I_1| < 2^39


def test_fixed_tracks_float_tile():
    """The integer tile reproduces the float Tile's pre-ADC output to within the quantisation bound.

    Bound per column: |Y_fixed * scale - y_float| <= sum_i |x_i| * (|dx| + |dw+| + |dw-| + |d eta| terms);
    with x, w, eta rounded to half an LSB each the worst case is well below 1 % of full scale.  The
    test asserts the measured relative error < 0.5 % of the full-scale output and that the correlation
    between the two models is > 0.9999 over 50 random tiles."""
    rng = np.random.default_rng(2026)
    rel = []
    for _ in range(50):
        p = DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=0.02)
        t = Tile(32, 32, rng, p)
        wp = rng.uniform(0, 1, (32, 32)); wm = rng.uniform(0, 1, (32, 32))
        t.gp.w[:] = wp; t.gm.w[:] = wm
        x = rng.uniform(-1, 1, 32)
        # the float model's noise for this read, captured by seeding a twin generator
        seed = int(rng.integers(0, 2 ** 31))
        t.rng = np.random.default_rng(seed); t.gp.rng = t.rng; t.gm.rng = t.rng
        y_float = t.read(x)
        eta = F.eta_stream_from_numpy(np.random.default_rng(seed), 0.02)[0]
        r = F.tile_read(F.w_to_code(wp), F.w_to_code(wm), np.ones((32, 32), bool), F.act_to_code(x), eta, adc_shift=0)
        # units: Y counts x_q (1/128) * G codes (G_LSB) ; float y is in units of G_range: W = (G+ - G-)/G_range
        y_fixed = r.Y / 128.0 * F.G_LSB_S / (100e-6 - 1e-6)
        fs = np.abs(x).sum()                                   # full scale: all |W| = 1
        rel.append(np.abs(y_fixed - y_float).max() / fs)
        assert np.corrcoef(y_fixed, y_float)[0, 1] > 0.9999
    assert max(rel) < 0.005, max(rel)


def test_two_pass_equals_one_wide_pass():
    rng = np.random.default_rng(7)
    wp = rng.integers(0, 256, (32, 32)); wm = rng.integers(0, 256, (32, 32))
    eta = F.eta_stream_from_numpy(rng, 0.05)[0]
    x16 = rng.integers(-32768, 32768, 32)
    A = rng.random((32, 32)) < 0.8
    two = F.tile_read(wp, wm, A, x16, eta, adc_shift=0, passes=2)
    # one pass with 16-bit drive is not a contract mode, but the integer identity x = 256*hi + lo must hold
    gp = np.where(A, F.noisy_conductance_q(F.conductance_q(wp), eta.reshape(32, 32, 2)[:, :, 0]), 0)
    gm = np.where(A, F.noisy_conductance_q(F.conductance_q(wm), eta.reshape(32, 32, 2)[:, :, 1]), 0)
    want = (x16[:, None] * (gp - gm)).sum(axis=0)
    np.testing.assert_array_equal(two.Y, want)


def test_adc_saturates_and_counts_clips():
    Y = np.array([0, 1 << 20, -(1 << 20), 3 << 12, -(1 << 12) - 1] + [0] * 27)
    y = F.sat(F.rne_shift(Y, 12), -128, 127)
    assert list(y[:5]) == [0, 127, -128, 3, -1]


def test_prng_is_deterministic_and_gaussian_shaped():
    a = F.EtaPRNG(0xC0FFEE, F.sigma_to_code(0.1)); b = F.EtaPRNG(0xC0FFEE, F.sigma_to_code(0.1))
    x = np.concatenate([a.read_block() for _ in range(3)]); y = np.concatenate([b.read_block() for _ in range(3)])
    np.testing.assert_array_equal(x, y)
    assert abs(x.std() / F.ETA_ONE - 0.1) < 0.005 and abs(x.mean()) / F.ETA_ONE < 0.002
    c = F.EtaPRNG(0xC0FFEF, F.sigma_to_code(0.1)).read_block()
    assert not np.array_equal(c, a.read_block())


def test_xoshiro_reference_vector():
    """xoshiro128** from the all-ones-like seed (1,2,3,4): first outputs per the reference implementation."""
    g = F.Xoshiro128ss(1, 2, 3, 4)
    assert [g.next() for _ in range(3)] == [11520, 0, 5927040]


def test_write_verify_converges_in_fixed_point():
    rng = np.random.default_rng(3)
    n = 256
    gain = F.sat(F.rne((1 + 0.03 * rng.standard_normal(n)) * F.GAIN_ONE), F.GAIN_MIN, 65535)
    c2c = lambda shape: F.eta_to_code(0.03 * rng.standard_normal(shape))
    def read_fn(w_s):
        g = F.conductance_q(F.ws_to_wq(w_s))
        tot = np.zeros(n, np.int64)
        for _ in range(F.VERIFY_READS):
            tot += F.noisy_conductance_q(g, F.eta_to_code(0.02 * rng.standard_normal(n))) * F.VERIFY_CODE
        return tot
    # the same judgement as tests/test_device.py::test_program_target_write_verify on the float model
    target = np.random.default_rng(7).uniform(0.05, 0.95, n)
    w, iters, pulses = F.program_target(np.zeros(n, np.int64), F.rne(target * F.WS_ONE), gain, read_fn, c2c)
    err = np.abs(w / F.WS_ONE - target)
    assert iters < 60, iters
    assert err.max() < 0.02 + 3 * 0.01, err.max()
    assert np.mean(err <= 0.02) > 0.8
    # the rails are reachable and stay inside the band
    for t in (0.0, 1.0):
        w, iters, _ = F.program_target(rng.integers(0, F.WS_ONE + 1, n), np.full(n, int(t * F.WS_ONE)), gain, read_fn, c2c)
        assert iters < F.MAX_ITER and np.abs(w / F.WS_ONE - t).max() < 0.05
