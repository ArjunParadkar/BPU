"""rel_pert: the measured relative perturbation of every layer's output under read noise (E11-cal, E15).

For a trained E8-style GPT (``e8_nanogpt_tiles.GPT``; every Linear a ``TileLinear``, attention through
``tile_attention``) and a noise setting sigma, every read site s (qkv, attention, proj, fc, fc2 per block,
and the head) gets

    rel_pert_s(sigma) = || y_s,noisy - y_s,ideal || / || y_s,ideal ||       (norms summed over all batches)

where y_ideal is the exact float product of the SAME stored weights (tiles: the realised weight
scale * A * (w+ - w-); float32: W) with no read noise and no read-out quantisation, and y_noisy is what
the layer actually returns at sigma (read noise + the 8-bit layer-accumulate converter on tiles).

  local       the site is fed its input from the ideal pass (the clean stream), so the number is that
              site's own perturbation and does not compound;
  cumulative  the whole forward runs noisy and each site's output is compared with the ideal pass's.

On tiles rel_pert(0) > 0 is the substrate's intrinsic perturbation (converter quantisation); on float32
rel_pert(0) = 0 exactly.  The primary scalar is ``mean_local``, the unweighted mean of the local values
over all sites.  The converter's auto-range state is saved before and restored after every measurement,
so measuring does not change the model.
"""
from __future__ import annotations

import contextlib

import numpy as np
import torch

import e8_nanogpt_tiles as e8
from bpusim.torchtile import TileLinear, tile_modules


def set_noise_as_run(model, sigma):
    """What E11 did (``e11_regularization.set_all_noise``): the MODULATE dial on every tile (clipped to
    [0.005, 0.2], on the float32 reference too) and the attention cache at sigma (not clipped)."""
    for t in tile_modules(model):
        t.set_noise(sigma); t.set_dropout(0.0)
    for b in model.blocks:
        b.attn_sigma_r = sigma


def set_noise_unclipped(model, sigma):
    """The nominal setting: every read site at sigma, no dial clip."""
    for t in tile_modules(model):
        t.sigma_r = float(sigma); t.dropout = 0.0
    for b in model.blocks:
        b.attn_sigma_r = sigma


def _ranges(model):
    st = {}
    for t in tile_modules(model):
        if t.readout is not None:
            st[id(t.readout)] = (t.readout, dict(t.readout.ranges))
    for b in model.blocks:
        if b.readout is not None:
            st[id(b.readout)] = (b.readout, dict(b.readout.ranges))
    return st


def _restore(st):
    for ro, r in st.values():
        ro.ranges = r


def site_names(model):
    names = {}
    for i, b in enumerate(model.blocks):
        names[id(b.qkv)] = f"b{i}.qkv"; names[id(b.proj)] = f"b{i}.proj"; names[id(b.fc)] = f"b{i}.fc"; names[id(b.fc2)] = f"b{i}.fc2"
    names[id(model.head)] = "head"
    return names


@contextlib.contextmanager
def _patched(model, mode, store):
    """mode 'local': every site returns its ideal output and records (noisy - ideal) on the clean input;
    'ideal': every site returns and records its ideal output; 'noisy': every site returns and records its
    noisy output."""
    names = site_names(model)
    hooks = []

    def ideal_lin(m, x):
        W = m.W * m.A
        y = x @ W
        return y + m.b if m.b is not None else y

    def hook(m, inp, out):
        nm = names[id(m)]
        if mode == "noisy":
            store.setdefault(nm, []).append(out.detach()); return out
        ide = ideal_lin(m, inp[0])
        if mode == "ideal":
            store.setdefault(nm, []).append(ide.detach()); return ide
        d = store.setdefault(nm, [0.0, 0.0])
        d[0] += float(((out - ide) ** 2).sum()); d[1] += float((ide ** 2).sum())
        return ide

    for t in tile_modules(model):
        hooks.append(t.register_forward_hook(hook))
    orig = e8.tile_attention
    blk_of_site = {}
    for i, b in enumerate(model.blocks):
        blk_of_site[f"attn{id(b)}"] = f"b{i}.attn"
    calls = {"i": 0}

    def att(q, k, v, sigma_w=0.0, sigma_r=0.0, counter=None, causal=True, tile=32, readout=None, site="attn"):
        nm = blk_of_site.get(site)
        if nm is None:      # ideal model branch (sigma None): one name per call order
            nm = f"b{calls['i'] % len(model.blocks)}.attn"; calls["i"] += 1
        noisy = orig(q, k, v, sigma_w, sigma_r, None, causal, tile, readout, site)
        if mode == "noisy":
            store.setdefault(nm, []).append(noisy.detach()); return noisy
        ide = orig(q, k, v, 0.0, 0.0, None, causal, tile, None, site)
        if mode == "ideal":
            store.setdefault(nm, []).append(ide.detach()); return ide
        d = store.setdefault(nm, [0.0, 0.0])
        d[0] += float(((noisy - ide) ** 2).sum()); d[1] += float((ide ** 2).sum())
        return ide

    e8.tile_attention = att
    try:
        yield
    finally:
        e8.tile_attention = orig
        for h in hooks:
            h.remove()


@torch.no_grad()
def measure(model, batches, sigma, setter=set_noise_as_run, cumulative=True):
    """batches: list of LongTensor (B, T) on the model's device.  Returns per-site local (and cumulative)
    rel_pert, the logits' cumulative rel_pert and the scalar summaries.  The model's noise setting is left
    at sigma (callers reset it)."""
    was_training = model.training; model.eval()
    st = _ranges(model)
    setter(model, sigma)
    loc = {}
    with _patched(model, "local", loc):
        for x in batches:
            model(x)
    _restore(st)
    local = {k: float(np.sqrt(v[0] / max(v[1], 1e-30))) for k, v in loc.items()}
    out = {"sigma": float(sigma), "local": local, "mean_local": float(np.mean(list(local.values()))),
           "mean_local_linear": float(np.mean([v for k, v in local.items() if not k.endswith(".attn")])),
           "mean_local_attn": float(np.mean([v for k, v in local.items() if k.endswith(".attn")]))}
    if cumulative:
        num, den = {}, {}
        for x in batches:
            ide, noi = {}, {}
            with _patched(model, "ideal", ide):
                li, _ = model(x)
            _restore(st)
            with _patched(model, "noisy", noi):
                ln, _ = model(x)
            _restore(st)
            ide["logits"] = [li]; noi["logits"] = [ln]
            for k in ide:
                a = torch.cat([t.flatten() for t in ide[k]]); b = torch.cat([t.flatten() for t in noi[k]])
                num[k] = num.get(k, 0.0) + float(((b - a) ** 2).sum()); den[k] = den.get(k, 0.0) + float((a ** 2).sum())
        cum = {k: float(np.sqrt(num[k] / max(den[k], 1e-30))) for k in num}
        out["cumulative"] = cum
        out["logits_cumulative"] = cum.pop("logits")
        out["mean_cumulative"] = float(np.mean(list(cum.values())))
    if was_training:
        model.train()
    return out


def batches_from(data, n_batches, tokens_per_batch, block, device, seed=12345):
    """n_batches batches of tokens_per_batch tokens each (tokens_per_batch / block sequences), fixed seed."""
    g = torch.Generator().manual_seed(seed)
    nseq = max(1, tokens_per_batch // block)
    out = []
    for _ in range(n_batches):
        ix = torch.randint(0, len(data) - block - 1, (nseq,), generator=g)
        out.append(torch.stack([data[i:i + block] for i in ix]).to(device))
    return out
