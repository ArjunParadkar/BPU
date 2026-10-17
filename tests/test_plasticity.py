import numpy as np

from bpusim.plasticity import (STDP, EligibilityTrace, stdp_window, three_factor, homeostatic_scaling,
                               oja_update, normalize_columns, SurpriseGate)


def test_stdp_online_matches_window():
    dt = 1e-3
    for lag in (5, -5, 20, -20):
        s = STDP(1, 1, A_plus=0.01, A_minus=0.012, tau_plus=20e-3, tau_minus=25e-3, dt=dt)
        total = 0.0
        t_pre, t_post = (0, lag) if lag > 0 else (-lag, 0)
        for t in range(60):
            total += s.step([t == t_pre], [t == t_post])[0, 0]
        expect = stdp_window(lag * dt, 0.01, 0.012, 20e-3, 25e-3)
        assert abs(total - expect) < 1e-9, (lag, total, expect)


def test_eligibility_trace_decay():
    e = EligibilityTrace((1,), tau_e=1.0, dt=1e-3)
    e.step(1.0 / 1e-3)   # a unit impulse spread over one step
    assert abs(e.e[0] - 1.0) < 1e-6
    for _ in range(1000):
        e.step(0.0)
    assert abs(e.e[0] - np.exp(-1)) < 2e-3


def test_three_factor_threshold():
    e = np.array([[1.0, 0.1], [-1.0, 0.05]])
    dW, n = three_factor(0.1, 0.5, e, theta_w=0.01)
    assert n == 2
    np.testing.assert_allclose(dW, [[0.05, 0.0], [-0.05, 0.0]])
    g = SurpriseGate((2, 2), theta_w=0.01)
    out, k = g(np.full((2, 2), 0.004)); assert k == 0
    out, k = g(np.full((2, 2), 0.004)); assert k == 0
    out, k = g(np.full((2, 2), 0.004)); assert k == 4 and abs(out[0, 0] - 0.012) < 1e-12


def test_homeostatic_scaling_direction():
    G = np.ones((3, 2))
    dG = homeostatic_scaling(G, r=np.array([0.5, 2.0]), r_star=1.0, tau_H=10.0, dt=1.0)
    assert np.all(dG[:, 0] > 0) and np.all(dG[:, 1] < 0)
    np.testing.assert_allclose(dG[:, 0], 0.05)


def test_oja_converges_to_unit_norm():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 1)) * 0.1
    C = np.diag([3.0, 1.0, 0.5, 0.1])
    for _ in range(3000):
        x = rng.multivariate_normal(np.zeros(4), C)
        y = x @ W
        W += oja_update(x, y, W, 0.01)
    assert abs(np.linalg.norm(W) - 1.0) < 0.15
    assert abs(W[0, 0]) > 0.9  # aligned with the principal component
    Wn = normalize_columns(rng.standard_normal((5, 3)), 2.0)
    np.testing.assert_allclose(np.linalg.norm(Wn, axis=0), 2.0)
