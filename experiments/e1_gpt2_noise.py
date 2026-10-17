"""E1: GPT-2 124M under tile noise, with noise-aware fine-tuning on WikiText-2.

Every nn.Linear (and the Conv1D projections GPT-2 uses) is wrapped so that
its weight is read through the tile noise model on every forward pass:

    W_read = W (1 + sigma_d2d zeta) (1 + sigma_r eps_t)

with zeta a fixed per-weight mismatch draw and eps_t fresh per forward, each
carrying sigma/sqrt(2) so the total relative noise is sigma (cycle-to-cycle
write noise is folded into the fine-tuning updates as a fresh multiplicative
factor on each weight change).  For every sigma in the sweep we report
validation perplexity (a) of the pretrained model read through noisy tiles
and (b) after noise-aware fine-tuning at that sigma.

Not meant to run on this laptop in full: use ``--device cuda`` on a GPU
host.  ``--smoke`` builds a tiny random GPT-2 and synthetic tokens and runs
20 steps on CPU to exercise the whole pipeline without any download.
"""
from __future__ import annotations

import argparse
import math
import time

import numpy as np

from common import save_results, figure_path

SIGMAS = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12]


class NoisyWeight:
    """Forward-pre-hook state: fixed mismatch zeta per weight and fresh read noise per call."""

    def __init__(self, module, weight_name, sigma, gen):
        import torch
        self.module, self.weight_name, self.gen = module, weight_name, gen
        self.sigma = sigma
        w = getattr(module, weight_name)
        self.zeta = torch.randn(w.shape, generator=gen, device="cpu").to(w.device)
        self.clean = None

    def set_sigma(self, sigma):
        self.sigma = sigma

    def pre(self, module, inputs):
        import torch
        if self.sigma == 0:
            return
        w = getattr(module, self.weight_name)
        self.clean = w
        s = self.sigma / math.sqrt(2.0)
        eps = torch.randn(w.shape, device=w.device, dtype=w.dtype)
        noisy = w * (1 + s * self.zeta.to(w.dtype)) * (1 + s * eps)
        # swap the parameter for a noisy tensor that still carries the gradient to the clean weight
        del module._parameters[self.weight_name]
        setattr(module, self.weight_name, noisy)

    def post(self, module, inputs, output):
        if self.clean is None:
            return
        if self.weight_name in module.__dict__:
            delattr(module, self.weight_name)
        module._parameters[self.weight_name] = self.clean
        self.clean = None


def wrap_model(model, sigma, seed=0):
    import torch
    from torch import nn
    gen = torch.Generator().manual_seed(seed)
    hooks = []
    try:
        from transformers.pytorch_utils import Conv1D
        kinds = (nn.Linear, Conv1D)
    except Exception:  # pragma: no cover
        kinds = (nn.Linear,)
    for name, m in model.named_modules():
        if isinstance(m, kinds):
            nw = NoisyWeight(m, "weight", sigma, gen)
            m.register_forward_pre_hook(nw.pre)
            m.register_forward_hook(nw.post)
            hooks.append(nw)
    return hooks


def set_sigma(hooks, sigma):
    for h in hooks:
        h.set_sigma(sigma)


def get_data(tokenizer, block, smoke, n_smoke=64):
    import torch
    if smoke:
        g = torch.Generator().manual_seed(0)
        tr = torch.randint(0, tokenizer_vocab(tokenizer), (n_smoke, block), generator=g)
        return tr, tr[: n_smoke // 4]
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    def tok(split):
        text = "\n\n".join(t for t in ds[split]["text"] if t.strip())
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        n = (len(ids) // block) * block
        return ids[:n].view(-1, block)
    return tok("train"), tok("validation")


def tokenizer_vocab(tok):
    return tok if isinstance(tok, int) else len(tok)


def evaluate(model, data, device, batch, max_batches=None):
    import torch
    model.eval(); tot = 0.0; n = 0
    with torch.no_grad():
        for i in range(0, len(data), batch):
            if max_batches is not None and i // batch >= max_batches:
                break
            x = data[i:i + batch].to(device)
            out = model(x, labels=x)
            tot += float(out.loss) * x.numel(); n += x.numel()
    return math.exp(tot / n)


def finetune(model, data, device, batch, steps, lr, c2c):
    import torch
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(1)
    n = len(data)
    for s in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g)
        x = data[idx].to(device)
        loss = model(x, labels=x).loss
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if c2c > 0:   # cycle-to-cycle write noise: each applied update is multiplied by (1 + c2c eps)
            before = [p.detach().clone() for p in model.parameters() if p.grad is not None]
            opt.step()
            with torch.no_grad():
                k = 0
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    d = p - before[k]; p.copy_(before[k] + d * (1 + c2c * torch.randn_like(d))); k += 1
        else:
            opt.step()
        if (s + 1) % 50 == 0 or s == steps - 1:
            print(f"      step {s + 1}/{steps} loss {float(loss):.3f}")
    return model


def build(smoke, device):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    if smoke:
        cfg = GPT2Config(vocab_size=512, n_positions=64, n_embd=64, n_layer=2, n_head=2)
        model = GPT2LMHeadModel(cfg); tok = 512
    else:
        model = GPT2LMHeadModel.from_pretrained("gpt2"); tok = GPT2TokenizerFast.from_pretrained("gpt2")
    return model.to(device), tok


def main():
    ap = argparse.ArgumentParser(description="E1 GPT-2 under tile noise")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--smoke", action="store_true", help="tiny random GPT-2, synthetic tokens, 20 steps on CPU")
    ap.add_argument("--steps", type=int, default=400, help="fine-tuning steps per sigma")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--c2c", type=float, default=0.0, help="write noise on fine-tuning updates (0 = off)")
    ap.add_argument("--sigmas", type=str, default=",".join(map(str, SIGMAS)))
    ap.add_argument("--eval-batches", type=int, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    import torch
    sigmas = [float(s) for s in args.sigmas.split(",")]
    if args.smoke:
        args.steps, args.block, args.batch, args.eval_batches = 20, 64, 4, 2
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    base, tok = build(args.smoke, device)
    train, val = get_data(tok, args.block, args.smoke)
    base_state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    rows = []
    for sigma in sigmas:
        t0 = time.time()
        base.load_state_dict(base_state)
        hooks = wrap_model(base, sigma, seed=0)
        ppl_pre = evaluate(base, val, device, args.batch, args.eval_batches)
        finetune(base, train, device, args.batch, args.steps, args.lr, args.c2c)
        ppl_ft = evaluate(base, val, device, args.batch, args.eval_batches)
        # clean-read perplexity of the noise-aware model (how much the fine-tune moved the weights)
        set_sigma(hooks, 0.0); ppl_clean = evaluate(base, val, device, args.batch, args.eval_batches)
        for h in hooks:
            h.module._forward_pre_hooks.clear(); h.module._forward_hooks.clear()
        rows.append({"sigma": sigma, "ppl_pretrained_noisy": ppl_pre, "ppl_finetuned_noisy": ppl_ft, "ppl_finetuned_clean_read": ppl_clean,
                     "seconds": time.time() - t0})
        print(f"  sigma={sigma}: pretrained@noise {ppl_pre:.2f}  finetuned@noise {ppl_ft:.2f}  finetuned@clean {ppl_clean:.2f}  ({time.time() - t0:.0f}s)")
    out = {"sigmas": sigmas, "rows": rows, "steps": args.steps, "block": args.block, "batch": args.batch, "lr": args.lr,
           "smoke": args.smoke, "device": device, "c2c": args.c2c}
    pre = [r["ppl_pretrained_noisy"] for r in rows]; ft = [r["ppl_finetuned_noisy"] for r in rows]
    i_min = int(np.argmin(ft))
    out["verdict"] = {"best_sigma_after_finetune": sigmas[i_min],
                      "statement": ("SMOKE RUN (random tiny model): numbers are meaningless. " if args.smoke else "") +
                                   f"lowest fine-tuned perplexity at sigma={sigmas[i_min]} ({ft[i_min]:.2f}); pretrained-at-noise rises from {pre[0]:.2f} to {pre[-1]:.2f}"}
    print("E1 verdict:", out["verdict"]["statement"])
    save_results("e1", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e1_gpt2_noise"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = res["sigmas"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(s, [r["ppl_pretrained_noisy"] for r in res["rows"]], "o-", label="pretrained, noisy read")
    ax.plot(s, [r["ppl_finetuned_noisy"] for r in res["rows"]], "s-", label="noise-aware fine-tuned, noisy read")
    ax.plot(s, [r["ppl_finetuned_clean_read"] for r in res["rows"]], "^--", label="noise-aware fine-tuned, clean read")
    ax.set_xlabel("tile noise $\\sigma$"); ax.set_ylabel("WikiText-2 validation perplexity"); ax.set_yscale("log")
    ax.set_title("E1: GPT-2 124M through noisy tiles" + (" (SMOKE)" if res.get("smoke") else ""), fontsize=10); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e1] figure {path}")


if __name__ == "__main__":
    main()
