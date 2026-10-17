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
