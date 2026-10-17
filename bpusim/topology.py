"""Topological certification of attractor manifolds (Lesson 10).

Pipeline
--------
1. Activity point cloud X (T samples x N neurons) -> PCA to <= 20 dimensions.
2. Vietoris-Rips persistent homology (ripser; gudhi fallback) up to ``maxdim``.
3. Null model: circularly shift each neuron's trace by an independent random
   offset (destroys the joint structure, keeps every marginal), recompute the
   diagrams ``n_null`` times, and set the threshold per dimension at the
   99th percentile of the *longest* null feature (the technical document's
   definition; with few null draws this is the largest null lifetime).  The
   99th percentile of the pooled null lifetimes is also recorded.
   A bar must also be at least ``rel`` (default 0.5) of the longest bar in its
   dimension: with a few hundred sample points a large torus always carries
   medium-length "sampling hole" bars that exceed a shift null (whose clouds
   are structureless blobs), so the null alone cannot separate the two
   fundamental loops from the holes; the relative criterion is the usual
   persistence-gap rule.  The count above the null alone is also reported.
4. Certify a *ring* if exactly one H1 bar is significant, no H2 bar is (when
   H2 was computed) and no finite H0 bar is (an extra persistent H0 bar
   means the cloud is fragmented into clusters, the pinning failure
   signature); a *torus* if H1 has exactly two and H2 exactly one significant
   bar and again no extra H0 bar.  Betti-0 is reported as the number of
   significant *finite* H0 bars, i.e. components beyond the one that never
   dies, counting only components that outlive three times the median
   H0 bar (the typical sampling spacing of the max-min sample).
"""
from __future__ import annotations

import numpy as np

try:  # pragma: no cover - import guard
    from ripser import ripser as _ripser
    HAVE_RIPSER = True
except Exception:  # pragma: no cover
    _ripser = None
    HAVE_RIPSER = False
try:  # pragma: no cover
    import gudhi as _gudhi
    HAVE_GUDHI = True
except Exception:  # pragma: no cover
    _gudhi = None
    HAVE_GUDHI = False


def pca_reduce(X: np.ndarray, n_components: int = 20) -> np.ndarray:
    X = np.asarray(X, float)
    Xc = X - X.mean(0, keepdims=True)
    k = min(n_components, Xc.shape[0], Xc.shape[1])
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:k].T


def subsample(X: np.ndarray, n_points: int, rng: np.random.Generator, method: str = "maxmin") -> np.ndarray:
    """Reduce a cloud to n_points by max-min (farthest point) sampling or at random."""
    X = np.asarray(X)
    if X.shape[0] <= n_points:
        return X
    if method == "random":
        return X[rng.choice(X.shape[0], n_points, replace=False)]
    idx = [int(rng.integers(X.shape[0]))]
    d = np.linalg.norm(X - X[idx[0]], axis=1)
    for _ in range(n_points - 1):
        j = int(np.argmax(d))
        idx.append(j)
        d = np.minimum(d, np.linalg.norm(X - X[j], axis=1))
    return X[idx]


def diagrams(X: np.ndarray, maxdim: int = 1) -> list[np.ndarray]:
    """Persistence diagrams [H0, H1, ...] of the Rips filtration of X (finite bars only in H>=1)."""
    X = np.ascontiguousarray(X, dtype=float)
    if HAVE_RIPSER:
        dg = _ripser(X, maxdim=maxdim)["dgms"]
    elif HAVE_GUDHI:
        rc = _gudhi.RipsComplex(points=X)
        st = rc.create_simplex_tree(max_dimension=maxdim + 1)
        st.compute_persistence()
        dg = [np.array([p for (dim, p) in st.persistence_pairs_as_intervals() if dim == d]) if False else
              np.array(st.persistence_intervals_in_dimension(d)) for d in range(maxdim + 1)]
    else:
        raise ImportError("Install ripser (pip install ripser) or gudhi for persistent homology.")
    out = []
    for d, D in enumerate(dg):
        D = np.asarray(D, float).reshape(-1, 2)
        D = D[np.isfinite(D[:, 1])]          # drop the one infinite H0 bar as well
        out.append(D)
    return out


def lifetimes(D: np.ndarray) -> np.ndarray:
    D = np.asarray(D, float).reshape(-1, 2)
    L = D[:, 1] - D[:, 0]
    return L[np.isfinite(L)]


def circular_shift_null(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shift each neuron's time series by an independent random offset."""
    X = np.asarray(X)
    T = X.shape[0]
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        out[:, j] = np.roll(X[:, j], int(rng.integers(T)))
    return out


def certify(X: np.ndarray, maxdim: int = 1, n_pca: int = 20, n_points: int = 300, n_null: int = 20,
            q: float = 99.0, rng: np.random.Generator | None = None, min_null_bars: int = 1,
            rel: float = 0.5, rel_h0: float = 3.0, n_points_h2: int | None = None) -> dict:
    """Certify the topology of an activity cloud against the circular-shift null.

    Returns a dict with the diagrams, the per-dimension thresholds, the
    number of significant bars per dimension (``betti``), and ``verdict`` in
    {"ring", "torus", "none"} plus the boolean flags ``is_ring`` / ``is_torus``.
    """
    rng = rng or np.random.default_rng(0)
    X = np.asarray(X, float)
    # Shared PCA basis so that the null is compared in the same coordinates as the data;
    # subsampling (max-min) is applied after reduction.
    Xr = pca_reduce(X, n_pca)
    dims = range(0, maxdim + 1)

    def _diagrams(Y):
        """H0/H1 on n_points, H2 (expensive) on n_points_h2 points of the same cloud."""
        if maxdim < 2 or not n_points_h2 or n_points_h2 >= n_points:
            return diagrams(subsample(Y, n_points, rng), maxdim)
        d01 = diagrams(subsample(Y, n_points, rng), 1)
        d2 = diagrams(subsample(Y, n_points_h2, rng), 2)
        return [d01[0], d01[1], d2[2]]

    dg = _diagrams(Xr)
    null_L = {d: [] for d in dims}
    null_max = {d: [] for d in dims}
    for _ in range(n_null):
        Xn = pca_reduce(circular_shift_null(X, rng), n_pca)
        dgn = _diagrams(Xn)
        for d in dims:
            L = lifetimes(dgn[d])
            null_L[d].extend(L.tolist())
            null_max[d].append(float(L.max()) if L.size else 0.0)
    thr, thr_pool, betti, betti_null, sig = {}, {}, {}, {}, {}
    for d in dims:
        pool = np.asarray(null_L[d]) if len(null_L[d]) >= min_null_bars else np.zeros(1)
        thr_pool[d] = float(np.percentile(pool, q)) if pool.size else 0.0
        thr[d] = float(np.percentile(null_max[d], q)) if null_max[d] else 0.0
        L = np.sort(lifetimes(dg[d]))[::-1]
        above = L[L > thr[d]]
        betti_null[d] = int(above.size)
        if d == 0:
            # finite H0 bars are the merge distances of the sample; a real fragment is a component
            # that outlives the typical sampling spacing by a wide margin (3x the median bar)
            med = float(np.median(L)) if L.size else 0.0
            sig[d] = above[above > rel_h0 * med]
        else:
            sig[d] = above[above >= rel * L[0]] if L.size else above
        betti[d] = int(sig[d].size)
    no_fragments = betti.get(0, 0) == 0
    is_ring = no_fragments and betti.get(1, 0) == 1 and (maxdim < 2 or betti.get(2, 0) == 0)
    is_torus = no_fragments and maxdim >= 2 and betti.get(1, 0) == 2 and betti.get(2, 0) == 1
    verdict = "torus" if is_torus else ("ring" if is_ring else "none")
    return {
        "verdict": verdict, "is_ring": bool(is_ring), "is_torus": bool(is_torus),
        "betti": {int(k): v for k, v in betti.items()},            # betti[0] = extra (finite) components
        "betti_above_null_only": {int(k): v for k, v in betti_null.items()},
        "rel": rel,
        "threshold": {int(k): v for k, v in thr.items()},
        "threshold_pooled": {int(k): v for k, v in thr_pool.items()},
        "null_max_lifetime": {int(k): float(np.max(v)) if v else 0.0 for k, v in null_max.items()},
        "significant_lifetimes": {int(k): v.tolist() for k, v in sig.items()},
        "top_lifetimes": {int(d): np.sort(lifetimes(dg[d]))[::-1][:5].tolist() for d in dims},
        "diagrams": [D.tolist() for D in dg],
        "n_points": int(min(n_points, X.shape[0])), "n_points_h2": int(n_points_h2 or n_points), "n_null": int(n_null), "q": q,
    }
