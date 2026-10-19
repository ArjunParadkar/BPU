"""E8: frontier-model accelerator parity, laptop stand-in for E1.

A nanoGPT-style character-level transformer (6 layers, d = 192, 6 heads,
context 128) is trained on tiny Shakespeare on CPU in three ways:

(a) IDEAL     float weights, standard attention;
(b) NOISE-AWARE  every Linear (qkv, out, mlp fc, mlp proj, lm_head) is a
    TileLinear (differential pairs, mask, read noise per read, write noise
    per pulse, soft-bound quantised pulses) and attention runs in the
    Hopfield form through volatile crossbars (scores = one tile read,
    softmax, values = a second tile read).  Starting from the ideal
    checkpoint of the same seed, the model is fine-tuned under noise for
    ``--ft-steps`` steps at each sigma_eff in {0, .02, .04, .06, .10}
    (equal split over d2d / c2c / read);
(c) DEPLOY    the ideal weights are LOADed with write-verify into tiles at
    each sigma and evaluated with no training.

Reported per sigma and seed: validation loss for (b) and (c), the gap to
the ideal, and the success test "(b) within 2 % of ideal at sigma <= 0.06".
Every MAC, tile read, A/D conversion, pulse and routed event is counted
for one evaluation batch and one training step at sigma = 0.04; wall time
of the ideal model's forward and training step is measured for E8b.
Embeddings and layer norms stay digital (periphery).
"""
from __future__ import annotations

import math
import os
import pathlib
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from common import base_parser, save_results, figure_path, Timer, ROOT
from bpusim.torchtile import TileLinear, TileOptimizer, OpCounter, tile_attention, tile_modules, ReadoutSpec
from bpusim.noise import summarize

SIGMAS = [0.0, 0.02, 0.04, 0.06, 0.10]
DATA = ROOT / "data" / "tinyshakespeare.txt"


# ------------------------------------------------------------------ data
def load_text(path=DATA):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(ids))
    return ids[:n], ids[n:], len(chars)


def get_batch(data, block, batch, gen, device=None):
    ix = torch.randint(0, len(data) - block - 1, (batch,), generator=gen)
    x = torch.stack([data[i:i + block] for i in ix]); y = torch.stack([data[i + 1:i + block + 1] for i in ix])
    if device is not None:
        x, y = x.to(device), y.to(device)
    return x, y


def model_device(model):
    return next(model.parameters()).device


# ----------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, n_head, sigma, counter, fa=False, scale=0.5, gen=None, tile_kw=None):
        super().__init__()
        self.n_head = n_head
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        kw = dict(sigma=sigma, counter=counter, feedback_alignment=fa, scale=scale, generator=gen, **(tile_kw or {}))
        self.qkv = TileLinear(d, 3 * d, **kw)
        self.proj = TileLinear(d, d, **kw)
        self.fc = TileLinear(d, 4 * d, **kw)
        self.fc2 = TileLinear(4 * d, d, **kw)
        self.sigma = sigma; self.counter = counter
        # attention-cache noise follows the tiles' noise split (d2d, c2c, read): cache writes carry the c2c
        # share, cache reads the read share (equal split: sigma / sqrt(3) each, as before)
        sp = np.asarray((tile_kw or {}).get("split", (1.0, 1.0, 1.0)), float); sp = sp / max(np.linalg.norm(sp), 1e-12)
        self.attn_sigma_w = float(sigma) * float(sp[1]) if sigma is not None else 0.0
        self.attn_sigma_r = float(sigma) * float(sp[2]) if sigma is not None else 0.0
        # ReadoutSpec for the attention reads: the architecture default (layer-accumulate 8-bit) on device
        # tiles, None on the ideal model; E8b overrides it per cell with set_readout
        self.readout = ReadoutSpec.default() if (sigma is not None and (tile_kw or {}).get("write_mode") != "float" and (tile_kw or {}).get("readout", "default") == "default") else None

    def attn(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        if self.sigma is None:
            # explicit softmax attention rather than the fused SDPA kernels: the fused efficient-attention
            # backward was found to be wrong on this laptop's RTX 5070 (DECISIONS.md), and T = 128 is small
            y = tile_attention(q, k, v, sigma_w=0.0, sigma_r=0.0, counter=None, causal=True)
        else:
            y = tile_attention(q, k, v, sigma_w=self.attn_sigma_w, sigma_r=self.attn_sigma_r, counter=self.counter, causal=True, readout=self.readout, site=f"attn{id(self)}")
        return self.proj(y.transpose(1, 2).reshape(B, T, C))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc(self.ln2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, vocab, d=192, n_layer=6, n_head=6, block=128, sigma=None, counter=None, fa=False, scale=0.5, seed=0,
                 tile_kw=None, head_kw=None):
        """tile_kw: extra TileLinear arguments for the block layers (write_mode, dfa, ...); head_kw: for the
        output head (which keeps true backprop under DFA because its error is its own output error)."""
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.block = block
        self.tok = nn.Embedding(vocab, d); self.pos = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d, n_head, sigma, counter, fa, scale, gen, tile_kw) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d)
        hk = dict(tile_kw or {}); hk.update(head_kw or {})
        self.head = TileLinear(d, vocab, bias=False, sigma=sigma, counter=counter, feedback_alignment=fa, scale=scale, generator=gen, **hk)
        nn.init.normal_(self.tok.weight, std=0.02); nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) if targets is not None else None
        return logits, loss

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def set_readout(model: GPT, spec):
    """Attach a ReadoutSpec (or None) to every tile and attention block of a tiled model."""
    for m in tile_modules(model):
        m.readout = spec
    for b in model.blocks:
        b.readout = spec


def copy_weights(src: GPT, dst: GPT, verify=True):
    """Copy ideal weights into a (possibly tiled) model; tiles are LOADed by write-verify."""
    sd = src.state_dict()
    with torch.no_grad():
        for (name, m_dst) in dst.named_modules():
            if isinstance(m_dst, TileLinear):
                m_dst.load(sd[name + ".W"], verify=verify)
                if m_dst.b is not None:
                    m_dst.b.copy_(sd[name + ".b"])
        for name, p in dst.named_parameters():
            if ".W" in name or name.endswith(".b") and any(isinstance(m, TileLinear) and n2 == name.rsplit(".", 1)[0] for n2, m in dst.named_modules()):
                continue
            p.copy_(sd[name])


@torch.no_grad()
def evaluate(model, data, block, batch, n_batches, gen):
    model.eval(); tot = 0.0
    dev = model_device(model)
    for _ in range(n_batches):
        x, y = get_batch(data, block, batch, gen, dev)
        _, loss = model(x, y); tot += loss.item()
    model.train()
    return tot / n_batches


def train(model, data, steps, block, batch, lr, gen, tiled: bool, log_every=200, warmup=100, label=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    topt = TileOptimizer(model.modules(), opt) if tiled else None
    t0 = time.time(); losses = []
    dev = model_device(model)
    for step in range(steps):
        for g_ in opt.param_groups:
            g_["lr"] = lr * min(1.0, (step + 1) / warmup) * (0.5 * (1 + math.cos(math.pi * step / steps)) * 0.9 + 0.1)
        x, y = get_batch(data, block, batch, gen, dev)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        (topt or opt).step()
        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            print(f"    {label} step {step + 1}/{steps} loss {np.mean(losses[-log_every:]):.3f} ({time.time() - t0:.0f}s)", flush=True)
    return losses


def count_ops(model_factory, data, block, batch, gen):
    """Op counts for one eval batch and one training step (forward + backward + pulsed update)."""
    counter = OpCounter()
    model = model_factory(counter)
    x, y = get_batch(data, block, batch, gen, model_device(model))
    counter.reset()
    with torch.no_grad():
        model(x, y)
    eval_ops = counter.snapshot()
    counter.reset()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3); topt = TileOptimizer(model.modules(), opt)
    _, loss = model(x, y); loss.backward(); topt.step()
    train_ops = counter.snapshot()
    # the backward pass costs ~2x the forward MACs on the same tiles (transposed reads); count it
    train_ops["macs_backward_estimate"] = 2 * eval_ops["macs"]
    return {"tokens": int(x.numel()), "eval_batch": eval_ops, "train_step": train_ops,
            "n_params": model.n_params(), "n_devices": int(sum(m.n_devices for m in tile_modules(model))),
            "n_tiles": int(sum(m.n_tiles for m in tile_modules(model)))}


def main():
    ap = base_parser("E8 nanoGPT through noisy tiles", default_seeds=3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--ft-steps", type=int, default=500)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--sigmas", type=str, default=",".join(map(str, SIGMAS)))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    seeds = 10 if args.full else args.seeds
    sigmas = [float(s) for s in args.sigmas.split(",")]
    if args.smoke:
        args.steps, args.ft_steps, args.eval_batches, seeds = 30, 10, 2, 1
    train_ids, val_ids, vocab = load_text()
    mk = lambda sigma, counter, seed, fa=False: GPT(vocab, args.d, args.layers, args.heads, args.block, sigma, counter, fa, seed=seed)
    runs = []
    timing = {}
    for seed in range(seeds):
        torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
        ideal = mk(None, None, seed)
        print(f"seed {seed}: ideal model {ideal.n_params() / 1e6:.2f}M params")
        with Timer("  ideal training") as tm:
            losses = train(ideal, train_ids, args.steps, args.block, args.batch, args.lr, gen, tiled=False, label="ideal")
        ideal_val = evaluate(ideal, val_ids, args.block, args.batch, args.eval_batches, torch.Generator().manual_seed(99))
        print(f"  ideal val loss {ideal_val:.4f}")
        # wall-clock for E8b (ideal model, this CPU)
        if seed == 0:
            xb, yb = get_batch(train_ids, args.block, args.batch, gen)
            with torch.no_grad():
                t0 = time.time(); [ideal(xb, yb) for _ in range(5)]; fwd = (time.time() - t0) / 5
            opt = torch.optim.AdamW(ideal.parameters(), lr=0.0)
            t0 = time.time()
            for _ in range(5):
                _, l = ideal(xb, yb); opt.zero_grad(); l.backward(); opt.step()
            trn = (time.time() - t0) / 5
            timing = {"forward_batch_s": fwd, "train_step_s": trn, "tokens_per_batch": int(xb.numel()), "threads": args.threads,
                      "cpu": os.cpu_count()}
        row = {"seed": seed, "ideal_val_loss": ideal_val, "ideal_train_loss_last": float(np.mean(losses[-100:])), "per_sigma": {}}
        sd_ideal = {k: v.clone() for k, v in ideal.state_dict().items()}
        for sigma in sigmas:
            # (c) deploy: load with write-verify, evaluate, no training
            dep = mk(sigma, None, seed); dep.load_state_dict(sd_ideal, strict=False); copy_weights(ideal, dep, verify=True)
            dep_val = evaluate(dep, val_ids, args.block, args.batch, args.eval_batches, torch.Generator().manual_seed(99))
            # (b) noise-aware fine-tune from the ideal checkpoint
            na = mk(sigma, None, seed); na.load_state_dict(sd_ideal, strict=False); copy_weights(ideal, na, verify=True)
            gen_ft = torch.Generator().manual_seed(1000 + seed)
            with Timer(f"  sigma={sigma} noise-aware fine-tune"):
                train(na, train_ids, args.ft_steps, args.block, args.batch, args.ft_lr, gen_ft, tiled=True, warmup=20, label=f"na s={sigma}")
            na_val = evaluate(na, val_ids, args.block, args.batch, args.eval_batches, torch.Generator().manual_seed(99))
            na_clean = None
            if sigma > 0:   # the same weights read without noise (how much the fine-tune moved them)
                for m in tile_modules(na):
                    m.sigma_r = 0.0
                na_clean = evaluate(na, val_ids, args.block, args.batch, args.eval_batches, torch.Generator().manual_seed(99))
            wpd = float(np.mean([m.writes_per_device() for m in tile_modules(na)])) / max(args.ft_steps, 1)
            row["per_sigma"][str(sigma)] = {"deploy_val_loss": dep_val, "noise_aware_val_loss": na_val, "noise_aware_clean_read_val_loss": na_clean,
                                            "deploy_gap_pct": 100 * (dep_val / ideal_val - 1), "noise_aware_gap_pct": 100 * (na_val / ideal_val - 1),
                                            "writes_per_device_per_step": wpd}
            print(f"  sigma={sigma}: deploy {dep_val:.4f} ({100 * (dep_val / ideal_val - 1):+.2f}%)  noise-aware {na_val:.4f} ({100 * (na_val / ideal_val - 1):+.2f}%)  writes/dev/step {wpd:.3g}", flush=True)
        runs.append(row)
    # op counts at sigma = 0.04 (counts do not depend on seed)
    gen_ops = torch.Generator().manual_seed(7)
    ops = count_ops(lambda c: mk(0.04, c, 0), train_ids, args.block, args.batch, gen_ops)
    # summary
    summary = {"ideal_val_loss": summarize([r["ideal_val_loss"] for r in runs])}
    for s in sigmas:
        summary[str(s)] = {k: summarize([r["per_sigma"][str(s)][k] for r in runs]) for k in ("deploy_val_loss", "noise_aware_val_loss", "deploy_gap_pct", "noise_aware_gap_pct")}
    ok = [s for s in sigmas if s <= 0.06 and summary[str(s)]["noise_aware_gap_pct"]["mean"] <= 2.0]
    worst = max(summary[str(s)]["noise_aware_gap_pct"]["mean"] for s in sigmas if s <= 0.06)
    out = {"seeds": seeds, "steps": args.steps, "ft_steps": args.ft_steps, "batch": args.batch, "block": args.block, "d": args.d,
           "layers": args.layers, "heads": args.heads, "sigmas": sigmas, "smoke": args.smoke, "runs": runs, "summary": summary,
           "ops": ops, "timing": timing,
           "verdict": {"success": len(ok) == sum(s <= 0.06 for s in sigmas), "worst_noise_aware_gap_pct_at_sigma_le_0.06": worst,
                       "statement": ("SMOKE RUN. " if args.smoke else "") +
                                    (f"noise-aware tiles stay within 2% of the ideal val loss at every sigma <= 0.06 (worst {worst:+.2f}%)" if len(ok) == sum(s <= 0.06 for s in sigmas)
                                     else f"noise-aware tiles exceed the 2% band at sigma <= 0.06 (worst {worst:+.2f}%); deploy-only gaps: " +
                                     ", ".join(f"s={s}:{summary[str(s)]['deploy_gap_pct']['mean']:+.1f}%" for s in sigmas))}}
    print("E8 verdict:", out["verdict"]["statement"])
    save_results("e8", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e8_nanogpt_tiles"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = res["sigmas"]; S = res["summary"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4))
    base = S["ideal_val_loss"]["mean"]
    for key, lab, mk in (("deploy_val_loss", "deploy (load, no training)", "s"), ("noise_aware_val_loss", "noise-aware fine-tuned", "o")):
        m = [S[str(x)][key]["mean"] for x in s]; e = [S[str(x)][key]["sem"] for x in s]
        ax[0].errorbar(s, m, e, marker=mk, capsize=3, label=lab)
    ax[0].axhline(base, color="k", ls="--", lw=0.8, label="ideal float"); ax[0].axhline(base * 1.02, color="gray", ls=":", lw=0.8, label="+2 %")
    ax[0].set_xlabel("sigma_eff"); ax[0].set_ylabel("val loss (nats/char)"); ax[0].legend(fontsize=8); ax[0].set_title("E8: nanoGPT on noisy tiles", fontsize=10)
    g1 = [S[str(x)]["deploy_gap_pct"]["mean"] for x in s]; g2 = [S[str(x)]["noise_aware_gap_pct"]["mean"] for x in s]
    ax[1].plot(s, g1, "s-", label="deploy"); ax[1].plot(s, g2, "o-", label="noise-aware"); ax[1].axhline(2, color="gray", ls=":")
    ax[1].set_xlabel("sigma_eff"); ax[1].set_ylabel("gap to ideal (%)"); ax[1].legend(fontsize=8); ax[1].set_title(res["verdict"]["statement"][:70], fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e8] figure {path}")


if __name__ == "__main__":
    main()
