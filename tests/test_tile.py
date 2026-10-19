import numpy as np

from bpusim.device import DeviceParams
from bpusim.tile import Tile


def test_differential_weight_range_and_mask(rng):
    t = Tile(8, 4, rng, DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=0))
    W = rng.uniform(-1, 1, (8, 4))
    t.program(W, tol=0.01)
    assert np.abs(t.W - W).max() < 0.03
    mask = rng.random((8, 4)) < 0.5
    t.set_mask(mask)
    assert np.all(t.W[~mask] == 0)


def test_read_error_scales_as_sigma_over_sqrt_N():
    """Relative error of y = sum_i W_i x_i with read noise sigma is ~ sigma/sqrt(N) when all terms are equal."""
    sigma = 0.05
    for N in (16, 64, 256):
        rng = np.random.default_rng(N)
        t = Tile(N, 1, rng, DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=sigma))
        t.program(np.full((N, 1), 0.8), tol=0.005)
        x = np.ones(N)
        y_true = x @ t.W
        ys = np.array([t.read(x)[0] for _ in range(4000)])
        rel = ys.std() / y_true[0]
        expect = sigma / np.sqrt(N)
        assert abs(rel - expect) / expect < 0.15, (N, rel, expect)


def test_update_moves_weights_in_the_right_direction(rng):
    t = Tile(6, 6, rng, DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=0))
    dW = rng.uniform(-0.3, 0.3, (6, 6))
    t.update(dW)
    assert np.abs(t.W - dW).max() < 0.05
    # saturating: push beyond +1 routes into G- headroom and clips at the rail
    t.update(np.full((6, 6), 2.0))
    assert np.all(t.W <= 1.0 + 1e-9) and t.W.min() > 0.95


def test_batch_read_and_transpose(rng):
    t = Tile(5, 3, rng, DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=0))
    W = rng.uniform(-0.5, 0.5, (5, 3)); t.program(W, tol=0.005)
    X = rng.standard_normal((7, 5))
    np.testing.assert_allclose(t.read(X), X @ t.W, atol=1e-12)
    y = rng.standard_normal(3)
    np.testing.assert_allclose(t.read_T(y), t.W @ y, atol=1e-12)


def test_dead_synapse_reads_zero_once_pruned(rng):
    """A dead device is stuck at G_min.  A pair with a dead G+ reads <= 0, a pair with a dead
    G- keeps conducting through G+; the growth rule prunes both kinds immediately."""
    from bpusim.growth import GrowthRule
    t = Tile(10, 10, rng, DeviceParams(dead_frac=0.2))
    t.program(np.full((10, 10), 0.7))
    assert np.all(t.W[t.gp.dead] <= 1e-9)
    GrowthRule(t).prune()
    assert np.all(t.W[t.dead] == 0.0) and not np.any(t.A & t.dead)


def test_plasticity_gain_kill_switch_and_write_accounting(rng):
    t = Tile(6, 6, rng, DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=0))
    t.plasticity_gain = 0.0                      # V_G = 0: plasticity frozen in hardware
    t.update(np.full((6, 6), 0.3))
    assert np.all(t.W == 0.0) and t.n_writes == 0
    t.plasticity_gain = 1.0
    t.update(np.full((6, 6), 0.3))
    assert t.W.min() > 0.25 and t.n_writes == 36
    assert abs(t.writes_per_device() - 0.5) < 1e-12   # 36 writes over 72 devices
    assert abs(t.writes_per_device(seconds=10.0) - 0.05) < 1e-12


def test_accumulate_and_flush_writes_only_whole_levels(rng):
    """Accumulate-and-flush: nothing is pulsed until a synapse holds one conductance level (1/L);
    the flushed amount is a whole number of levels with the remainder rounded stochastically
    (unbiased), and writes are counted only on pulses."""
    p = DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=0)
    t = Tile(8, 8, rng, p, levels=20)              # one level = 0.05 in weight units
    assert abs(t.level - 0.05) < 1e-12
    for _ in range(4):
        t.accumulate(np.full((8, 8), 0.01))        # 0.04 pending: below one level
    assert t.n_writes == 0 and np.abs(t.W).max() == 0 and np.allclose(t.acc, 0.04)
    t.accumulate(np.full((8, 8), 0.01))            # 0.05 = exactly one level -> every synapse flushes once
    assert t.n_writes == 64 and abs(t.flush_rate - 1 / 5) < 1e-12
    assert np.allclose(t.acc, 0.0, atol=1e-12) and np.abs(t.W - 0.05).max() < 0.02
    # a request of 2.5 levels flushes 2 or 3 levels, 3 with probability one half, mean 2.5
    t2 = Tile(40, 40, np.random.default_rng(1), p, levels=20)
    t2.accumulate(np.full((40, 40), 0.125))
    flushed = 0.125 - t2.acc
    n_lv = np.rint(flushed / 0.05)
    assert set(np.unique(n_lv)) <= {2.0, 3.0} and abs(n_lv.mean() - 2.5) < 0.1
    # the accumulator keeps the un-flushed remainder (+-0.025) and a mask/rewrite clears it
    assert np.allclose(np.abs(t2.acc), 0.025, atol=1e-9)
    t2.program(np.zeros((40, 40)))
    assert np.all(t2.acc == 0)


def test_accumulate_respects_write_quota(rng):
    p = DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=0)
    t = Tile(10, 10, rng, p, levels=10)
    # a budget of 1 write/device/s over dt = 0.1 s allows 0.1 * 200 devices = 20 synapse writes per call
    t.accumulate(rng.uniform(0.1, 0.5, (10, 10)), write_budget=1.0, dt=0.1)
    assert t.n_writes == 20
