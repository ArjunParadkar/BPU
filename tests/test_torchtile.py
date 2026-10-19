import math
import numpy as np
import torch

from bpusim.torchtile import TileLinear, TileOptimizer, OpCounter, tile_attention


def test_ideal_matches_linear():
    torch.manual_seed(0)
    t = TileLinear(8, 4, sigma=None)
    x = torch.randn(3, 8)
    assert torch.allclose(t(x), x @ t.W + t.b)


def test_load_and_read_noise_statistics():
    g = torch.Generator().manual_seed(0)
    t = TileLinear(64, 64, bias=False, sigma=0.05, split=(0, 0, 1), scale=1.0, generator=g)
    W = torch.rand(64, 64) * 0.8 - 0.4
    t.load(W, verify=False)
    assert torch.allclose(t.W, W, atol=1e-6)
    reads = torch.stack([t.read_weight() for _ in range(200)])
    rel = (reads.std(0) / W.abs()).mean()
    assert abs(rel - 0.05) < 0.01          # one device of the pair holds the weight, so noise ~ sigma_r
    assert (reads.mean(0) - W).abs().max() < 0.02


def test_pulsed_update_quantised_and_residual_kept():
    g = torch.Generator().manual_seed(1)
    t = TileLinear(16, 16, bias=False, sigma=0.0, scale=0.5, alpha=0.02, generator=g)
    t.load(torch.zeros(16, 16), verify=False)
    dW = torch.full((16, 16), 0.002)        # 0.004 in state units: below half a pulse (0.01)
    r = t.apply_delta(dW)
    assert r.abs().max() == 0 and torch.allclose(t.pending, dW)
    for _ in range(3):
        r = t.apply_delta(dW)               # accumulated 0.008 -> 0.4 pulse -> 0; 0.012 -> 0.6 pulse -> 1 pulse
    assert t.n_pulses > 0 and t.W.max() > 0
    assert torch.allclose(t.W, 0.5 * (t.wp - t.wm))
    # a large change saturates softly inside [-scale, scale]
    t.apply_delta(torch.full((16, 16), 5.0)); t.apply_delta(torch.full((16, 16), 5.0))
    assert t.W.max() <= 0.5 + 1e-6 and t.W.min() > 0.45


def test_optimizer_learns_through_tiles():
    torch.manual_seed(0)
    model = torch.nn.Sequential(TileLinear(10, 32, sigma=0.04, scale=0.5, alpha=0.02), torch.nn.ReLU(), TileLinear(32, 1, sigma=0.04, scale=0.5, alpha=0.02))
    x = torch.randn(256, 10); y = (x[:, :1] * 0.7 + 0.3 * x[:, 1:2] ** 2)
    opt = TileOptimizer(model.modules(), torch.optim.Adam(model.parameters(), lr=3e-3))
    losses = []
    for i in range(300):
        opt.zero_grad(); loss = ((model(x) - y) ** 2).mean(); loss.backward(); opt.step(); losses.append(loss.item())
    assert np.mean(losses[-20:]) < 0.5 * np.mean(losses[:20])


def test_feedback_alignment_backward_uses_B():
    torch.manual_seed(0)
    t = TileLinear(6, 3, sigma=None, feedback_alignment=True)
    x = torch.randn(4, 6, requires_grad=True)
    y = t(x); y.sum().backward()
    expected = torch.ones(4, 3) @ t.B_fa.T
    assert torch.allclose(x.grad, expected, atol=1e-6)
    assert t.W.grad.shape == (6, 3)


def test_counter_and_attention():
    c = OpCounter()
    t = TileLinear(64, 32, sigma=0.02, counter=c)
    t(torch.randn(2, 5, 64))
    assert c.macs == 10 * 64 * 32 and c.reads == 10 * 2 and c.conversions == 10 * 32 * 2
    q = torch.randn(1, 2, 8, 16); out = tile_attention(q, q, q, 0.02, 0.02, counter=c)
    assert out.shape == q.shape and c.cache_writes == 2 * 2 * 8 * 16
    clean = tile_attention(q, q, q, 0.0, 0.0)
    assert (out - clean).abs().mean() < 0.1


def test_level_write_mode_flushes_whole_levels_only():
    g = torch.Generator().manual_seed(2)
    t = TileLinear(16, 16, bias=False, sigma=0.0, scale=0.5, alpha=0.02, generator=g, write_mode="level", levels=25)
    t.load(torch.zeros(16, 16), verify=False)
    lvl = 0.5 / 25                                   # 0.02 in weight units
    for _ in range(3):
        r = t.apply_delta(torch.full((16, 16), 0.006))
    assert r.abs().max() == 0 and t.n_writes == 0 and torch.allclose(t.pending, torch.full((16, 16), 0.018))
    t.apply_delta(torch.full((16, 16), 0.006))       # 0.024 >= one level: flush 1 level (p = 0.8) or 2 (p = 0.2)
    assert t.n_writes == 256 and abs(t.flush_rate - 0.25) < 1e-9
    n_lv = torch.round((0.024 - t.pending) / lvl)
    assert set(n_lv.unique().tolist()) <= {1.0, 2.0}
    assert (t.W > 0).all()
    # the default level count follows sigma_eff: round(1 / 0.04) = 25
    assert TileLinear(4, 4, sigma=0.04).levels == 25
