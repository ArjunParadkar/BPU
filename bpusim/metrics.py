"""WhiteBox topology metrics and module-graph metrics.

* SV  (structural velocity)  = W_p(D_t, D_{t+dt}) / dt, the p-Wasserstein
  distance between successive persistence diagrams per unit time.
* TC  (total persistence)    = sum of bar lifetimes of a diagram.
* FPL (feature persistence)  = per-feature lifetimes, sorted descending.
* Q   Newman modularity of a (symmetrised) routing graph.
* MI  mutual information between a module's discretised activity and the task id.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


# ----------------------------------------------------------------- diagrams
def wasserstein(D1, D2, p: int = 2) -> float:
    """p-Wasserstein distance between two finite diagrams (points may match the diagonal)."""
    D1 = np.asarray(D1, float).reshape(-1, 2); D2 = np.asarray(D2, float).reshape(-1, 2)
    D1 = D1[np.isfinite(D1).all(1)]; D2 = D2[np.isfinite(D2).all(1)]
    n, m = len(D1), len(D2)
    if n == 0 and m == 0:
        return 0.0
    # augment with diagonal projections: (n+m) x (n+m) cost matrix
    proj1 = np.column_stack([(D1.sum(1) / 2), (D1.sum(1) / 2)]) if n else np.zeros((0, 2))
    proj2 = np.column_stack([(D2.sum(1) / 2), (D2.sum(1) / 2)]) if m else np.zeros((0, 2))
    A = np.vstack([D1, proj2])      # rows: D1 points then diagonal slots for D2 points
    B = np.vstack([D2, proj1])      # cols: D2 points then diagonal slots for D1 points
    C = np.linalg.norm(A[:, None, :] - B[None, :, :], ord=np.inf, axis=2) ** p
    # diagonal-to-diagonal matches cost zero
    if n and m:
        C[n:, m:] = 0.0
    r, c = linear_sum_assignment(C)
    return float(C[r, c].sum() ** (1.0 / p))


def total_persistence(D) -> float:
    D = np.asarray(D, float).reshape(-1, 2)
    L = D[:, 1] - D[:, 0]
    return float(L[np.isfinite(L)].sum())


def feature_lifetimes(D) -> np.ndarray:
    D = np.asarray(D, float).reshape(-1, 2)
    L = D[:, 1] - D[:, 0]
    return np.sort(L[np.isfinite(L)])[::-1]


def structural_velocity(D_prev, D_next, dt: float, p: int = 2) -> float:
    return wasserstein(D_prev, D_next, p) / dt


def whitebox_track(clouds, dts, maxdim: int = 1, dim: int = 1, n_pca: int = 20, n_points: int = 200,
                   rng=None):
    """SV / TC / FPL along a sequence of activity clouds. Returns dict of lists."""
    from .topology import diagrams, pca_reduce, subsample
    rng = rng or np.random.default_rng(0)
    dg = [diagrams(subsample(pca_reduce(X, n_pca), n_points, rng), maxdim)[dim] for X in clouds]
    SV = [structural_velocity(dg[i], dg[i + 1], dts[i]) for i in range(len(dg) - 1)]
    TC = [total_persistence(D) for D in dg]
    FPL = [feature_lifetimes(D)[:5].tolist() for D in dg]
    return {"SV": SV, "TC": TC, "FPL": FPL}


# ------------------------------------------------------------- module graph
def modularity(R, communities) -> float:
    """Newman modularity Q of the symmetrised adjacency R for a community labelling."""
    R = np.asarray(R, float)
    A = ((R + R.T) > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    k = A.sum(1)
    two_e = A.sum()
    if two_e == 0:
        return 0.0
    c = np.asarray(communities)
    same = c[:, None] == c[None, :]
    return float(((A - np.outer(k, k) / two_e) * same).sum() / two_e)


def best_modularity(R):
    """Greedy (Clauset-Newman-Moore) community detection; returns (Q, labels)."""
    import networkx as nx
    R = np.asarray(R, float)
    A = ((R + R.T) > 0).astype(float)
    np.fill_diagonal(A, 0.0)
    if A.sum() == 0:
        return 0.0, np.arange(len(A))
    G = nx.from_numpy_array(A)
    comms = nx.algorithms.community.greedy_modularity_communities(G)
    labels = np.zeros(len(A), int)
    for i, cset in enumerate(comms):
        for v in cset:
            labels[v] = i
    return modularity(A, labels), labels


def mutual_information(labels_a, labels_b, base: float = 2.0) -> float:
    """Plug-in mutual information between two discrete labellings (bits by default)."""
    a = np.asarray(labels_a); b = np.asarray(labels_b)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((len(ua), len(ub)))
    np.add.at(joint, (ia, ib), 1.0)
    joint /= joint.sum()
    pa = joint.sum(1, keepdims=True); pb = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum() / np.log(base))


def entropy(labels, base: float = 2.0) -> float:
    _, cnt = np.unique(np.asarray(labels), return_counts=True)
    p = cnt / cnt.sum()
    return float(-(p * np.log(p)).sum() / np.log(base))


def task_information(Z: np.ndarray, task: np.ndarray, n_clusters: int = 8, seed: int = 0) -> dict:
    """I(Z; T): discretise a module's activity by k-means, then plug-in MI with the task id.

    Returns MI in bits, the normalised MI (divided by H(T)) and a shuffle baseline.
    """
    from sklearn.cluster import KMeans
    Z = np.asarray(Z, float)
    if Z.std() < 1e-12:
        return {"mi_bits": 0.0, "mi_norm": 0.0, "mi_shuffle": 0.0}
    k = min(n_clusters, len(Z))
    lab = KMeans(k, n_init=4, random_state=seed).fit_predict(Z)
    mi = mutual_information(lab, task)
    rng = np.random.default_rng(seed)
    sh = mutual_information(lab, rng.permutation(task))
    H = entropy(task)
    return {"mi_bits": mi, "mi_norm": mi / H if H > 0 else 0.0, "mi_shuffle": sh}
