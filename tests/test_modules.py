import numpy as np

from bpusim.device import DeviceParams
from bpusim.modules import ModuleGraph, GraphConfig, layered_routing
from bpusim.metrics import modularity


def _graph(fixed=False, seed=0):
    rng = np.random.default_rng(seed)
    cfg = GraphConfig(n_per_module=8, n_modules=4, n_relax=4, edge_budget=4)
    R = layered_routing(4, [0], [1, 2], [3]) if fixed else None
    return ModuleGraph(cfg, {0: 3}, 2, [0], [3], rng, DeviceParams(), R_fixed=R), rng


def test_forward_shapes_and_eligibilities():
    g, rng = _graph(fixed=True)
    out = g.forward({0: np.array([1.0, 0.0, -1.0])}, record=True)
    assert out["y"].shape == (2,)
    assert out["trace"].shape == (4, 32)
    assert len(out["elig_W"]) == 4 and out["elig_B"][(0, 1)].shape == (8, 8)
    assert g.n_edges == 4


def test_route_statistic_detects_useful_connection():
    """With a graph that has no routes, the only way module 0 (input) can affect the loss is
    through a route 0 -> 3 (output); the node-perturbation statistic must rank 0 -> 3 first."""
    g, rng = _graph(fixed=False)
    g.cfg.theta_e = np.inf   # observe only, never connect
    x = np.array([1.0, 0.5, -0.5])
    for _ in range(400):
        fwd = g.forward({0: x})
        loss = float(np.sum((fwd["y"] - np.array([1.0, -1.0])) ** 2))
        g.learn(fwd, loss, task=0, structure=True)
    C = g.C.copy(); np.fill_diagonal(C, -1)
    # routes into the output module carry the signal; 0 -> 3 is the informative one
    assert C[0, 3] > 0
    into_out = C[:, 3]
    assert np.argmax(into_out) == 0 or C[0, 3] > 0.5 * into_out.max()


def test_learning_reduces_loss_fixed_graph():
    g, rng = _graph(fixed=True)
    xs = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    targets = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    losses = []
    for t in range(1000):
        i = t % 2
        fwd = g.forward({0: xs[i]})
        loss = float(np.sum((fwd["y"] - targets[i]) ** 2))
        g.learn(fwd, loss, task=i, structure=False, target=targets[i])
        losses.append(loss)
    assert np.mean(losses[-100:]) < 0.5 * np.mean(losses[:100])


def test_graph_grows_routes_toward_output():
    g, rng = _graph(fixed=False)
    g.cfg.theta_e = 0.0
    x = np.array([1.0, 0.0, 0.0]); target = np.array([1.0, 0.0])
    for _ in range(300):
        fwd = g.forward({0: x})
        loss = float(np.sum((fwd["y"] - target) ** 2))
        g.learn(fwd, loss, task=0, structure=True, target=target)
    assert g.n_edges >= 1
    # the first route created should end in the output module (only it influences the loss)
    ks, ls = np.nonzero(g.R)
    assert 3 in set(ls.tolist())


def test_modularity_of_layered_routing():
    R = layered_routing(8, [0, 1], [2, 3, 4, 5], [6, 7])
    assert abs(modularity(R, [0] * 8)) < 1e-12
    assert R.sum() == 16
