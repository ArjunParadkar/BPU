import numpy as np
import pytest

from bpusim.attractor import RingAttractor, GridModule, circ_dist, population_vector
from bpusim.device import DeviceParams


@pytest.mark.parametrize("J1,expect_bump", [(1.5, False), (1.9, False), (2.5, True), (3.0, True)])
def test_bump_forms_iff_J1_gt_2(J1, expect_bump):
    r = RingAttractor(32, J0=-0.5, J1=J1, use_tiles=False, rng=np.random.default_rng(0))
    r.settle(800)
    mod = r.bump_modulation()
    assert (mod > 1.0) == expect_bump, (J1, mod)


def test_bump_on_noisy_tile_and_decode():
    r = RingAttractor(32, J0=-0.5, J1=3.0, params=DeviceParams(sigma_d2d=0.03), rng=np.random.default_rng(1))
    r.reset(angle=1.0)
    r.settle(500)
    assert r.bump_modulation() > 1.0
    assert abs(circ_dist(r.decode(), 1.0)) < 0.3


def test_velocity_moves_bump_linearly():
    r = RingAttractor(32, J0=-0.5, J1=3.0, use_tiles=False, rng=np.random.default_rng(0))
    c = r.calibrate_velocity_gain()
    assert abs(c) > 0
    r.reset(0.0); r.settle(500)
    a0 = r.decode()
    ang = r.run(np.full(300, 0.5 / c), record=False)  # command 0.5 rad/s for 0.3 s
    moved = circ_dist(ang[-1], a0)
    assert abs(moved - 0.15) < 0.03


def test_population_vector():
    theta = 2 * np.pi * np.arange(16) / 16
    r = np.exp(np.cos(theta - 2.0))
    assert abs(circ_dist(population_vector(r, theta), 2.0)) < 1e-6


def test_grid_module_forms_pattern_and_velocity_shifts_it():
    g = GridModule(16, 16, use_tiles=False, rng=np.random.default_rng(0))
    g.settle(600)
    S0 = g.sheet().copy()
    assert (S0.max() - S0.min()) / S0.mean() > 1.0          # patterned, not uniform
    ks = g.lattice_wavevectors(2)
    assert len(ks) == 2                                      # two independent lattice directions
    for _ in range(150):
        g.step(vx=0.1)
    dx, dy = g.shift_between(S0, g.sheet())
    assert dx != 0 and dy == 0                               # the lattice translated along x only
    assert np.corrcoef(np.roll(S0, (dx, dy), (0, 1)).ravel(), g.sheet().ravel())[0, 1] > 0.8
