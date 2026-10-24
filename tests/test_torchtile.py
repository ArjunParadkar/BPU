import math
import numpy as np
import torch

from bpusim.torchtile import TileLinear, TileOptimizer, OpCounter, tile_attention, ReadoutSpec


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
    assert r.abs().max() == 0 and torch.allclose(t.pending, dW, atol=1e-4)   # 16-bit accumulator grid
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


def test_nearest_pulse_default_and_primitives():
    g = torch.Generator().manual_seed(3)
    t = TileLinear(16, 16, bias=False, sigma=0.0, scale=0.5, alpha=0.02, generator=g, readout=None)
    assert t.write_mode == "nearest_pulse" and t.readout is None
    t.load(torch.zeros(16, 16), verify=False)
    dW = torch.full((16, 16), 0.002)                     # 0.004 in state units: a fifth of a pulse (0.02)
    for _ in range(2):
        t.apply_delta(dW)
    assert t.n_pulses == 0 and torch.allclose(t.pending, torch.full((16, 16), 0.004), atol=1e-4)
    t.apply_delta(dW); t.apply_delta(dW)                  # 0.008 -> 0.4 pulse -> 0; 0.010 -> 0.5 -> rounds to 1 pulse
    assert t.n_pulses == 256 and torch.allclose(t.W, torch.full((16, 16), 0.01), atol=1e-6)
    assert torch.allclose(t.pending, torch.full((16, 16), -0.002), atol=1e-4)   # nominal subtracted, remainder kept
    # accumulator resolution: 0.5 / 2^15
    t.pending.zero_(); t.apply_delta(torch.full((16, 16), 1e-6))
    assert float(t.pending.abs().max()) == 0.0
    assert t.set_noise(1.0) == 0.2 and t.primitive_log[-1]["clipped"]
    t.set_noise(0.0); t.set_dropout(0.5)
    W = t.read_weight(); frac = float((W == 0).float().mean())
    assert 0.35 < frac < 0.65


def test_float_write_mode_is_the_reference_arm():
    g = torch.Generator().manual_seed(4)
    t = TileLinear(32, 32, bias=False, sigma=0.05, split=(0, 0, 1), write_mode="float", generator=g)
    W = torch.rand(32, 32) - 0.5; t.load(W)
    assert torch.allclose(t.W, W) and t.readout is None and t.writes_per_device() == 0.0
    reads = torch.stack([t.read_weight() for _ in range(200)])
    assert abs(float((reads.std(0) / W.abs()).mean()) - 0.05) < 0.01
    t.apply_delta(torch.full((32, 32), 0.001))
    assert torch.allclose(t.W, W + 0.001)


def test_default_readout_is_layer_8bit_auto():
    g = torch.Generator().manual_seed(5)
    t = TileLinear(64, 16, bias=False, sigma=0.0, generator=g)
    assert t.readout is not None and t.readout.mode == "layer" and t.readout.bits == 8 and t.readout.auto
    assert abs(t.readout.E_ADC_J - 10e-15 * 256) < 1e-20
    x = torch.randn(8, 64)
    y = t(x); y_ideal = x @ t.W
    assert (y - y_ideal).abs().max() <= float(y_ideal.abs().max()) / 128 + 1e-6   # within one 8-bit step
    assert t.site in t.readout.ranges


def test_float_reference_noise_is_not_clipped_by_the_dial():
    """2026-10-23: the MODULATE dial clips device tiles to [0.005, 0.2]; the float32 reference has no dial.
    Before the fix the reference was clipped too, so every E11 float32 arm above 0.2 ran at 0.2."""
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "experiments"))
    from e11_regularization import set_all_noise
    from e8_nanogpt_tiles import GPT
    f = TileLinear(8, 8, sigma=0.0, split=(0, 0, 1), write_mode="float")
    assert f.set_noise(1.0) == 1.0 and f.sigma_r == 1.0 and not f.primitive_log[-1]["clipped"]
    d = TileLinear(8, 8, sigma=0.0, split=(0, 0, 1))
    assert d.set_noise(1.0) == 0.2 and d.primitive_log[-1]["clipped"]
    for wm, legacy, expect in (("float", False, 0.7), ("nearest_pulse", False, 0.7), ("float", True, 0.2), ("nearest_pulse", True, 0.2)):
        m = GPT(11, 16, 1, 2, 8, 0.0, None, seed=0, tile_kw=dict(split=(0.0, 0.0, 1.0), write_mode=wm))
        set_all_noise(m, 0.7, 0.0, legacy_clip=legacy)
        assert all(abs(t.sigma_r - expect) < 1e-12 for t in m.modules() if isinstance(t, TileLinear)), (wm, legacy)
        assert all(b.attn_sigma_r == 0.7 for b in m.blocks)
    from bpusim.tile import FloatTile
    import numpy as np
    ft = FloatTile(4, 4, np.random.default_rng(0))
    assert ft.set_noise(0.9) == 0.9
