import numpy as np

from bpusim.topology import certify, diagrams, pca_reduce
from bpusim.metrics import wasserstein, total_persistence, feature_lifetimes, modularity, best_modularity, \
    mutual_information, task_information


def _ring_cloud(rng, T=1500, N=32):
    a = np.cumsum(rng.normal(0, 0.1, T))
    theta = 2 * np.pi * np.arange(N) / N
    return np.maximum(np.cos(a[:, None] - theta[None, :]) - 0.3, 0) + 0.02 * rng.standard_normal((T, N))


def test_certify_ring():
    rng = np.random.default_rng(0)
    c = certify(_ring_cloud(rng), maxdim=1, n_points=250, n_null=10, rng=rng)
    assert c["verdict"] == "ring" and c["betti"][1] == 1 and c["betti"][0] == 0


def test_fragmented_cloud_is_not_a_ring():
    """Four tight clusters on a circle: H0 reports extra components, so no ring certificate."""
    rng = np.random.default_rng(5)
    theta = 2 * np.pi * np.arange(32) / 32
    a = np.repeat(2 * np.pi * np.arange(4) / 4, 300) + 0.02 * rng.standard_normal(1200)
    X = np.maximum(np.cos(a[:, None] - theta[None, :]) - 0.3, 0) + 0.01 * rng.standard_normal((1200, 32))
    c = certify(X, maxdim=1, n_points=200, n_null=8, rng=rng)
    assert c["betti"][0] >= 1 and not c["is_ring"]


def test_certify_torus():
    rng = np.random.default_rng(1)
    T = 2500
    a = np.cumsum(rng.normal(0, 0.15, T)); b = np.cumsum(rng.normal(0, 0.15, T))
    tx = 2 * np.pi * np.arange(8) / 8
    X = (np.exp(np.cos(a[:, None] - tx[None, :]))[:, :, None]
         * np.exp(np.cos(b[:, None] - tx[None, :]))[:, None, :]).reshape(T, -1)
    X += 0.05 * rng.standard_normal(X.shape)
    c = certify(X, maxdim=2, n_points=250, n_null=6, rng=rng)
    assert c["verdict"] == "torus", c["betti"]


def test_null_rejects_unstructured_cloud():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((800, 32))
    c = certify(X, maxdim=1, n_points=200, n_null=10, rng=rng)
    assert c["verdict"] == "none"


def test_pca_reduce_dims():
    X = np.random.default_rng(0).standard_normal((100, 50))
    assert pca_reduce(X, 20).shape == (100, 20)


def test_wasserstein_and_persistence():
    D1 = np.array([[0.0, 1.0], [0.0, 0.1]])
    D2 = np.array([[0.0, 1.0]])
    assert abs(wasserstein(D1, D1)) < 1e-12
    assert abs(wasserstein(D1, D2) - 0.05) < 1e-9  # the (0,0.1) bar matches the diagonal at cost 0.05
    assert abs(total_persistence(D1) - 1.1) < 1e-12
    np.testing.assert_allclose(feature_lifetimes(D1), [1.0, 0.1])


def test_modularity_two_cliques():
    R = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            if i != j:
                R[i, j] = 1; R[i + 3, j + 3] = 1
    R[2, 3] = 1
    Q, labels = best_modularity(R)
    assert Q > 0.3 and len(set(labels[:3])) == 1 and len(set(labels[3:])) == 1
    assert abs(modularity(R, [0, 0, 0, 1, 1, 1]) - Q) < 1e-9


def test_mutual_information():
    a = np.array([0, 0, 1, 1]); b = np.array([1, 1, 0, 0])
    assert abs(mutual_information(a, b) - 1.0) < 1e-12
    assert abs(mutual_information(a, np.array([0, 1, 0, 1]))) < 1e-12
    rng = np.random.default_rng(0)
    task = rng.integers(0, 2, 400)
    Z = np.column_stack([task + 0.05 * rng.standard_normal(400), 0.3 * rng.standard_normal(400)])
    info = task_information(Z, task, n_clusters=4)
    assert info["mi_bits"] > 0.8 and info["mi_shuffle"] < 0.1
