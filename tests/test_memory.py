import numpy as np

from bpusim.device import DeviceParams
from bpusim.memory import ScaffoldMemory, HopfieldAttention

NOISELESS = DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=0)


def _pairs(rng, n, K):
    return rng.choice([-1.0, 1.0], (n, K)), rng.choice([-1.0, 1.0], (n, K))


def test_pseudo_inverse_recall_exact_on_noiseless_tile():
    rng = np.random.default_rng(0)
    P, S = _pairs(rng, 32, 16)
    mem = ScaffoldMemory(32, 32, rng, NOISELESS)
    mem.store(P, S, method="pinv")
    # the pseudo-inverse solution itself is exact
    M = S @ np.linalg.pinv(P)
    np.testing.assert_allclose(M @ P, S, atol=1e-9)
    # and the programmed tile recalls every pair exactly after the sign nonlinearity
    for mu in range(16):
        out = mem.recall(S[:, mu], n_cleanup=0)
        assert np.array_equal(out["content"], S[:, mu])
        assert np.array_equal(out["index"], P[:, mu])
    # continuous reads reproduce the stored content up to programming tolerance
    err = np.abs(mem.W_ps.read(P.T) - S.T).max()
    assert err < 0.25


def test_pseudo_inverse_recall_from_erased_cues():
    rng = np.random.default_rng(1)
    P, S = _pairs(rng, 32, 8)
    mem = ScaffoldMemory(32, 32, rng, NOISELESS)
    mem.store(P, S, method="pinv")
    ok = 0
    for mu in range(8):
        cue = S[:, mu].copy(); cue[rng.random(32) < 0.5] = 0.0
        ok += np.array_equal(mem.recall(cue)["content"], S[:, mu])
    assert ok >= 7


def test_delta_rule_converges_to_pinv():
    rng = np.random.default_rng(2)
    P, S = _pairs(rng, 32, 8)
    mem = ScaffoldMemory(32, 32, rng, NOISELESS)
    info = mem.store(P, S, method="delta", eta=0.02, epochs=40)
    assert info["delta_mse"][-1] < 0.05 * info["delta_mse"][0]
    for mu in range(8):
        assert np.array_equal(mem.recall(S[:, mu], n_cleanup=0)["content"], S[:, mu])


def test_cleanup_projects_onto_scaffold():
    rng = np.random.default_rng(3)
    P, S = _pairs(rng, 32, 6)
    mem = ScaffoldMemory(32, 32, rng, NOISELESS)
    mem.store(P, S)
    p = P[:, 2].copy(); p[:6] *= -1  # 6 flipped index bits
    assert np.array_equal(mem.cleanup(p), P[:, 2])


def test_hopfield_attention_retrieves_pattern():
    rng = np.random.default_rng(4)
    X = rng.choice([-1.0, 1.0], (32, 16))
    att = HopfieldAttention(X, beta=0.5, rng=rng, params=NOISELESS)
    for mu in range(16):
        cue = X[:, mu].copy(); cue[rng.random(32) < 0.5] = 0.0
        out = att.retrieve(cue)
        assert out["argmax"] == mu
        assert np.array_equal(np.sign(out["xi"]), X[:, mu])
    # energy decreases along the update from a corrupted state
    cue = X[:, 0].copy(); cue[rng.random(32) < 0.3] = 0.0
    assert att.energy(att.retrieve(cue)["xi"]) <= att.energy(cue) + 1e-9


def test_noisy_tile_still_recalls_small_sets():
    rng = np.random.default_rng(5)
    P, S = _pairs(rng, 32, 8)
    mem = ScaffoldMemory(32, 32, rng, DeviceParams())  # default 3%/3%/2% noise
    mem.store(P, S)
    ok = sum(np.array_equal(mem.recall(S[:, mu])["content"], S[:, mu]) for mu in range(8))
    assert ok >= 7


def test_hopfield_k1():
    """One stored pair on noise-free tiles: the scaffold loop must recall it exactly from the clean
    cue, and in at least 95 % of trials from a cue with 20 % of its bits flipped."""
    import numpy as np
    from bpusim.memory import ScaffoldMemory
    from bpusim.device import DeviceParams
    rng = np.random.default_rng(0)
    N, K = 32, 1
    P = rng.choice([-1.0, 1.0], (N, K)); S = rng.choice([-1.0, 1.0], (N, K))
    mem = ScaffoldMemory(N, N, rng, DeviceParams(sigma_d2d=0, sigma_c2c=0, sigma_r=0)); mem.store(P, S, method="pinv")
    assert np.array_equal(mem.recall(S[:, 0].copy())["content"], S[:, 0])
    ok = 0; n = 200
    for _ in range(n):
        cue = S[:, 0].copy(); m = rng.random(N) < 0.2; cue[m] = -cue[m]
        ok += np.array_equal(mem.recall(cue)["content"], S[:, 0])
    assert ok / n >= 0.95, ok / n
