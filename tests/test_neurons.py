import numpy as np

from bpusim.neurons import LIF, RateNeurons


def test_lif_threshold_current(rng):
    n = LIF(2, rng, tau_m=20e-3, theta=1.0, R=1.0, dt=1e-4)
    I = np.array([0.9, 1.5])  # below / above theta/R
    spikes = np.zeros(2, int)
    for _ in range(20000):
        spikes += n.step(I)
    assert spikes[0] == 0 and spikes[1] > 10


def test_lif_rate_matches_formula(rng):
    tau, theta, I = 20e-3, 1.0, 2.0
    n = LIF(1, rng, tau_m=tau, theta=theta, dt=1e-5)
    count = 0
    T = 1.0
    for _ in range(int(T / n.dt)):
        count += int(n.step(np.array([I]))[0])
    expect = 1.0 / (tau * np.log(I / (I - theta)))
    assert abs(count / T - expect) / expect < 0.03


def test_rate_relaxes_to_f(rng):
    r = RateNeurons(3, rng, tau=10e-3, dt=1e-3, f="relu")
    for _ in range(200):
        r.step(np.array([-1.0, 0.5, 2.0]))
    np.testing.assert_allclose(r.r, [0.0, 0.5, 2.0], atol=1e-6)


def test_rate_noise_has_requested_scale(rng):
    r = RateNeurons(5000, rng, tau=10e-3, dt=1e-3, f="linear", sigma=0.1)
    r.step(np.zeros(5000))
    # one step: sigma * sqrt(dt/tau)
    assert abs(r.r.std() - 0.1 * np.sqrt(0.1)) < 0.003
