import numpy as np

from bpusim.device import DeviceParams
from bpusim.tile import Tile
from bpusim.growth import GrowthRule, spawn


def test_grow_rule_recovers_gradient_sign(rng):
    """Node perturbation: C_ij ~ -sigma^2 dL/dW_ij.  With L = -(target . y), the
    correlation must be positive for inputs that would help and negative otherwise."""
    n_in, n_out = 6, 3
    t = Tile(n_in, n_out, rng, DeviceParams(), mask=np.zeros((n_in, n_out), bool))
    g = GrowthRule(t, theta_grow=0.0, budget=4, tau_C=2000, tau_mean=50)
    target = np.array([1.0, -1.0, 0.0])
    x = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # only inputs 0,1 are ever on
    L_bar = 0.0
    for _ in range(4000):
        xi = 0.1 * rng.standard_normal(n_out)
        y = t.read(x) + xi
        L = -(target @ y)
        m = -(L - L_bar); L_bar += 0.05 * (L - L_bar)
        g.observe(m, x, y)
    # inputs 0,1 -> output 0 positive, -> output 1 negative, -> output 2 ~ 0, unused inputs ~ 0
    assert g.C[0, 0] > 0 and g.C[1, 0] > 0
    assert g.C[0, 1] < 0 and g.C[1, 1] < 0
    assert abs(g.C[0, 2]) < 0.3 * abs(g.C[0, 0])
    assert np.abs(g.C[2:, :]).max() == 0.0
    grown = g.grow()
    assert grown == 4 and t.n_active == 4
    assert t.A[0, 0] and t.A[1, 0] and t.A[0, 1] and t.A[1, 1]
    # seeded along sign(C)
    assert t.W[0, 0] > 0 and t.W[0, 1] < 0


def test_budget_respected_and_prune(rng):
    t = Tile(5, 5, rng, DeviceParams(sigma_r=0.0))
    g = GrowthRule(t, theta_grow=1e-9, theta_prune=0.05, T_prune=3, budget=10)
    t.A[:] = False
    g.C = rng.standard_normal((5, 5))
    assert g.grow() == 10 and t.n_active == 10
    assert g.grow() == 0
    # all weights are tiny seeds (0.05) -> set below threshold and let the counter run
    t.gp.w[:] = 0.0; t.gm.w[:] = 0.0
    assert g.prune() == 0 and g.prune() == 0
    assert g.prune() == 10 and t.n_active == 0


def test_dead_devices_pruned_immediately(rng):
    t = Tile(20, 20, rng, DeviceParams(dead_frac=0.1))
    t.program(np.full((20, 20), 0.5))
    g = GrowthRule(t, theta_prune=0.01, T_prune=1000)
    n = g.prune()
    assert n == int(t.dead.sum()) and not np.any(t.A & t.dead)


def test_spawn_is_a_noisy_copy(rng):
    t = Tile(8, 8, rng, DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=0))
    t.program(rng.uniform(-0.8, 0.8, (8, 8)), tol=0.005)
    c1 = spawn(t, rng, sigma_copy=0.05)
    c2 = spawn(t, rng, sigma_copy=0.05)
    assert np.corrcoef(t.W.ravel(), c1.W.ravel())[0, 1] > 0.99
    assert np.abs(c1.W - c2.W).max() > 1e-3
    assert np.array_equal(c1.A, t.A)
