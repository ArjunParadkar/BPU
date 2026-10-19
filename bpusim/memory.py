"""L5: memory on tiles.

:class:`ScaffoldMemory`
    Heteroassociation between index patterns p^mu (the scaffold, e.g. grid
    states) and content patterns s^mu, stored on tiles either by the
    pseudo-inverse  W = S P^+  (programmed by write-verify) or by the local
    delta rule  dW = eta (s - W p) p^T  run through the noisy tile.  The
    recall loop is  cue -> index -> attractor cleanup -> content, where the
    cleanup is a pseudo-inverse autoassociator on the index tile iterated
    with a sign nonlinearity (a binary attractor on the scaffold).

:class:`HopfieldAttention`
    Modern Hopfield / attention retrieval  xi_new = X softmax(beta X^T xi)
    with the stored patterns X held as conductances; both the score read
    (X^T xi) and the value read (X a) go through the same noisy tile.
"""
from __future__ import annotations

import numpy as np

from .device import DeviceParams
from .tile import Tile, FloatTile


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


_TIE_RNG = np.random.default_rng(20261018)


def signz(x, rng: np.random.Generator | None = None) -> np.ndarray:
    """sign(x) with exact ties (x == 0) broken at random to +-1 from a fixed-seed generator
    (the earlier version mapped every tie to +1, a bias toward the all-ones pattern)."""
    s = np.sign(np.asarray(x, float))
    z = s == 0
    if z.any():
        s = s.copy(); s[z] = (rng or _TIE_RNG).choice([-1.0, 1.0], int(z.sum()))
    return s


class ScaledTile:
    """A tile that stores W / scale so that an arbitrary real matrix fits in |W| <= 1."""

    def __init__(self, n_in, n_out, rng, params=None, float_ref: bool = False):
        """float_ref: the float32 reference substrate (FloatTile) with the device read noise of ``params``
        injected on every read."""
        self.tile = FloatTile(n_in, n_out, rng, sigma_r=(params or DeviceParams()).sigma_r) if float_ref else Tile(n_in, n_out, rng, params)
        self.scale = 1.0

    def program(self, W, tol=0.01):
        W = np.asarray(W, float)
        self.scale = max(float(np.abs(W).max()), 1e-12)
        self.tile.program(W / self.scale, tol=tol)

    def read(self, x):
        return self.tile.read(x) * self.scale

    def read_T(self, y):
        return self.tile.read_T(y) * self.scale

    def update(self, dW):
        return self.tile.update(np.asarray(dW) / self.scale)

    @property
    def W(self):
        return self.tile.W * self.scale


class ScaffoldMemory:
    def __init__(self, n_index: int, n_content: int, rng: np.random.Generator,
                 params: DeviceParams | None = None):
        self.n_index, self.n_content = n_index, n_content
        self.rng = rng
        self.params = params or DeviceParams()
        self.W_sp = ScaledTile(n_content, n_index, rng, self.params)  # content -> index
        self.W_ps = ScaledTile(n_index, n_content, rng, self.params)  # index -> content
        self.W_pp = ScaledTile(n_index, n_index, rng, self.params)    # index cleanup (autoassociator)
        self.P = None; self.S = None

    # ------------------------------------------------------------------ store
    def store(self, P: np.ndarray, S: np.ndarray, method: str = "pinv", eta: float = 0.1,
              epochs: int = 30, theta_w: float = 0.0, delta_scale: float | None = None) -> dict:
        """Store columns of P (n_index x K) with columns of S (n_content x K)."""
        P = np.asarray(P, float); S = np.asarray(S, float)
        self.P, self.S = P, S
        Pp = np.linalg.pinv(P)                       # K x n_index
        M_ps = S @ Pp                                 # n_content x n_index : s = M_ps p
        M_sp = P @ np.linalg.pinv(S)                  # n_index x n_content : p = M_sp s
        M_pp = P @ Pp                                 # projection onto span(P)
        info = {"method": method}
        if method == "pinv":
            self.W_ps.program(M_ps.T); self.W_sp.program(M_sp.T); self.W_pp.program(M_pp.T)
        elif method == "delta":
            # the delta rule converges to the pseudo-inverse solution from zero weights; the
            # tile's scale is fixed in advance from the known solution's magnitude (hardware
            # would use a fixed full-scale range)
            for st, M in ((self.W_ps, M_ps), (self.W_sp, M_sp), (self.W_pp, M_pp)):
                st.scale = delta_scale or max(float(np.abs(M).max()) * 1.5, 1e-12)
                st.tile.program(np.zeros(st.tile.shape))
            errs = []
            for ep in range(epochs):
                order = self.rng.permutation(P.shape[1])
                e_ep = 0.0
                for mu in order:
                    p, s = P[:, mu], S[:, mu]
                    for st, x, y in ((self.W_ps, p, s), (self.W_sp, s, p), (self.W_pp, p, p)):
                        err = y - st.read(x)
                        dW = eta * np.outer(x, err)
                        if theta_w > 0:
                            dW = np.where(np.abs(dW) >= theta_w, dW, 0.0)
                        st.update(dW)
                        e_ep += float(np.mean(err ** 2))
                errs.append(e_ep / (3 * P.shape[1]))
            info["delta_mse"] = errs
        else:
            raise ValueError(method)
        return info

    # ----------------------------------------------------------------- recall
    def cleanup(self, p, n_iter: int = 3):
        """Attractor cleanup on the index tile: p <- sign(W_pp p), iterated."""
        for _ in range(n_iter):
            p = signz(self.W_pp.read(p))
        return p

    def recall(self, cue: np.ndarray, n_cleanup: int = 3, binary: bool = True) -> dict:
        p_hat = self.W_sp.read(cue)
        p_raw = signz(p_hat) if binary else p_hat
        p_star = self.cleanup(p_raw, n_cleanup) if n_cleanup > 0 else p_raw
        s_hat = self.W_ps.read(p_star)
        s_star = signz(s_hat) if binary else s_hat
        return {"index_raw": p_raw, "index": p_star, "content_raw": s_hat, "content": s_star}

    def direct_recall(self, cue: np.ndarray) -> np.ndarray:
        """Single heteroassociative read index -> content (no scaffold loop), for comparison."""
        return signz(self.W_ps.read(cue))


class HopfieldAttention:
    def __init__(self, X: np.ndarray, beta: float, rng: np.random.Generator, params: DeviceParams | None = None, float_ref: bool = False):
        """X: (d, P) stored patterns as columns.  float_ref: float32 reference substrate (same read noise injected)."""
        self.X = np.asarray(X, float)
        self.d, self.P = self.X.shape
        self.beta = beta
        self.store = ScaledTile(self.d, self.P, rng, params, float_ref=float_ref)
        self.store.program(self.X)

    def retrieve(self, xi: np.ndarray, n_iter: int = 1) -> dict:
        xi = np.asarray(xi, float)
        for _ in range(n_iter):
            scores = self.store.read(xi)                # X^T xi  (noisy read 1)
            a = _softmax(self.beta * scores)
            xi = self.store.read_T(a)                   # X a     (noisy read 2)
        return {"xi": xi, "attention": a, "argmax": int(np.argmax(a))}

    def energy(self, xi):
        s = self.X.T @ xi
        m = s.max()
        return float(-(m + np.log(np.exp(self.beta * (s - m)).sum())) / self.beta + 0.5 * xi @ xi)
