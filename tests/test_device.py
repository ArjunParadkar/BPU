import math
import numpy as np
import pytest

from bpusim.device import DeviceArray, DeviceParams


def test_bounds_after_many_pulses(rng):
    d = DeviceArray((200,), rng, DeviceParams(), w_init=0.5)
    d.pulse(np.full(200, 300))
    assert np.all(d.w <= 1.0) and np.all(d.w >= 0.0)
    assert d.w.mean() > 0.95
    d.pulse(np.full(200, -300))
    assert np.all(d.w >= 0.0)
    assert d.w.mean() < 0.05


def test_conductance_map():
    p = DeviceParams()
    d = DeviceArray((3,), np.random.default_rng(1), p, w_init=[0.0, 0.5, 1.0])
    np.testing.assert_allclose(d.G, [p.G_min, p.G_min + 0.5 * p.G_range, p.G_max])
    np.testing.assert_allclose(d.w_from_G(d.G), d.w)


def test_saturating_step_law():
    p = DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=0)
    d = DeviceArray((1,), np.random.default_rng(0), p, w_init=0.0)
    n = 30
    for _ in range(n):
        d.pulse(1)
    # closed form w(n) = ln(1 + alpha beta n) / beta
    w_expect = math.log(1 + p.alpha_p * p.beta_p * n) / p.beta_p
    assert abs(d.w[0] - w_expect) < 0.01


def test_pulses_for_inverts_pulse():
    p = DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=0)
    d = DeviceArray((5,), np.random.default_rng(0), p, w_init=[0.0, 0.2, 0.5, 0.8, 0.95])
    target = np.array([0.6, 0.6, 0.6, 0.6, 0.6])
    n = d.pulses_for(target)
    d.pulse(n)
    assert np.all(np.abs(d.w - target) < 0.02)


def test_sigma_eff():
    p = DeviceParams(sigma_d2d=0.03, sigma_c2c=0.03, sigma_r=0.02)
    assert abs(p.sigma_eff - math.sqrt(0.03**2 + 0.03**2 + 0.02**2)) < 1e-12


def test_read_noise_statistics():
    p = DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=0.05)
    d = DeviceArray((20000,), np.random.default_rng(2), p, w_init=0.5)
    rel = d.read() / d.G - 1
    assert abs(rel.mean()) < 2e-3
    assert abs(rel.std() - 0.05) < 2e-3


def test_c2c_and_d2d_noise_statistics():
    # step size spread across devices from d2d only
    p = DeviceParams(sigma_d2d=0.05, sigma_c2c=0.0, sigma_r=0.0)
    d = DeviceArray((20000,), np.random.default_rng(3), p, w_init=0.3)
    w0 = d.w.copy(); d.pulse(1)
    steps = d.w - w0
    assert abs(steps.std() / steps.mean() - 0.05) < 0.005
    # c2c: the same device stepping repeatedly from the same state shows the spread
    p = DeviceParams(sigma_d2d=0.0, sigma_c2c=0.05, sigma_r=0.0)
    d = DeviceArray((20000,), np.random.default_rng(4), p, w_init=0.3)
    w0 = d.w.copy(); d.pulse(1)
    steps = d.w - w0
    assert abs(steps.std() / steps.mean() - 0.05) < 0.005


def test_retention_relaxes_and_nonvolatile_default():
    p = DeviceParams(tau_ret=1.0, w_inf=0.0, sigma_d2d=0, sigma_c2c=0, sigma_r=0)
    d = DeviceArray((1,), np.random.default_rng(0), p, w_init=1.0)
    d.retain(1.0)
    assert abs(d.w[0] - math.exp(-1)) < 1e-9
    d2 = DeviceArray((1,), np.random.default_rng(0), DeviceParams(), w_init=1.0)
    d2.retain(1e9)
    assert d2.w[0] == 1.0


def test_dead_devices_stuck():
    p = DeviceParams(dead_frac=0.3)
    d = DeviceArray((1000,), np.random.default_rng(5), p, w_init=0.5)
    assert 0.2 < d.dead.mean() < 0.4
    d.pulse(50)
    assert np.all(d.w[d.dead] == 0.0)
    assert np.all(d.G[d.dead] == p.G_min)


def test_program_target_write_verify():
    d = DeviceArray((500,), np.random.default_rng(6), DeviceParams(), w_init=0.0)
    target = np.random.default_rng(7).uniform(0.05, 0.95, 500)
    its = d.program_target(target, tol=0.02)
    assert its < 60
    # the true state is within tol plus the averaged-read uncertainty (~0.01 in w near the top)
    assert np.abs(d.w - target).max() < 0.02 + 3 * 0.01
    assert np.mean(np.abs(d.w - target) <= 0.02) > 0.8
