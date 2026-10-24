"""E1-fix + E16: GPT-2 124M, deployment of an already-trained model on BPU tiles (round four, 2026-10-23).

"Learning vs knowing": a pretrained model is KNOWN structure; variability should be suppressed when it
is deployed.  E1 (results/e1_full) read GPT-2 through a float noise model W(1+s zeta)(1+s eps); it never
ran the tile path.  E1-fix runs it.

Substrates (same algorithm on both):
  tiles    every Conv1D of the 12 blocks (c_attn, attn.c_proj, c_fc, mlp.c_proj) and the LM head is a
           ``TileLinear``: differential pairs, PER-COLUMN scale = max |W| of that output column (a gain
           in the column periphery; no weight is clipped), programmed by write-verify (``load(verify=True)``:
           residual uniform within max(0.01, half a pulse x d2d gain)), sigma_device split equally over
           d2d / c2c / read (sigma / sqrt 3 each; d2d enters through the programming residual, c2c through
           writes, read noise on every read), default read-out = layer-accumulate 8-bit (auto range);
           embeddings, layer norms, biases and the attention score/value products stay digital (as in E1;
           GPT-2's attention is not on the volatile crossbars here).
  float32  the same modules with float weights and the same read noise (sigma / sqrt 3) injected on
           every read; no programming error, no converter.

Arms (perplexity on the full WikiText-2 validation set, 512-token blocks):
  a) baselines: float32 GPT-2 clean; tiles at sigma 0 (programming + converter only).
  b) deploy-only (no fine-tune), sigma in {0, 0.01, 0.02, 0.04, 0.06}, both substrates, 3 programming /
     read-noise seeds.
  c) noise-aware fine-tune (hardware-aware training OFF-CHIP, Joshi et al. 2020): float32 master weights,
     every step the master is programmed into the tiles (fresh write-verify residual), the forward runs
     through the tile read path, the gradient is passed straight through to the master; at the end the
     master is programmed once and evaluated at the same sigma.  sigma in {0.02, 0.04, 0.06} at 3x and 5x
     the E1 budget (1200 and 2000 AdamW steps, batch 8 x 512, lr 2e-5, as E1).  float32 runs the identical
     procedure at sigma in {0, 0.02, 0.04, 0.06}; float32 sigma 0 at N steps is the matched reference.
     Diagnostic: ON-CHIP fine-tune at 0.04 x 2000 steps (tiles programmed once, AdamW proposals realised
     by nearest-pulse writes, ``TileOptimizer``; the on-chip head is untied from the digital embedding).
  d) E16 chip-side handling at sigma_device = 0.04: dial-low deploy (read noise set to the dial minimum
     0.005 by ``set_noise``; d2d / c2c unchanged), read averaging k in {1, 2, 4, 8} (read noise / sqrt k;
     energy: k x the array-read energy of the E8b model, one conversion per output since the k reads
     integrate in analog; the bound with k x conversions too is also given), and the combination
     dial-low + k = 8 + the 5x noise-aware fine-tune at 0.04 (learn high, deploy low).

PRE-REGISTERED (2026-10-23, before any run):
  criterion  some arm at sigma_device = 0.04 lands within 3 % perplexity of its float32 reference
             (deploy-only and E16 deploy arms: float32 pretrained, clean; fine-tuned arms: float32
             sigma 0 fine-tuned for the same number of steps).  Every arm's gap is reported, and the
             energy cost of averaging.
  divergence any perplexity above 2x its float32 reference is reported as "diverged" (with the first
             sigma where it happens), never as a raw number in a verdict.  The same rule fixes the E1
             "pretrained-at-noise" row (results/e1_full, float noise model): diverged from the first
             sigma at which it exceeds 2x the float32 perplexity.
  seeds      deploy arms 3 seeds; fine-tunes 1 seed (as E1), stated.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn

from common import ROOT, save_results, figure_path
from bpusim.torchtile import TileLinear, TileOptimizer, OpCounter
from e1_gpt2_noise import get_data, build

OUT = "results/e1fix"
SIGMAS_DEPLOY = [0.0, 0.01, 0.02, 0.04, 0.06]
SIGMAS_FT = [0.02, 0.04, 0.06]
E1_STEPS = 400
DIAL_LOW = 0.005
KS = [1, 2, 4, 8]


class TileProj(nn.Module):
    """A GPT-2 projection (Conv1D, or the tied LM head) executed through a TileLinear.  ``master`` is the
    original float weight (in, out) orientation via ``get_W``; the bias stays digital."""

    def __init__(self, get_W, bias, in_f, out_f, substrate, sigma, gen, counter=None):
        super().__init__()
        self.get_W = get_W; self.bias = bias
        self.substrate = substrate
        self.tl = TileLinear(in_f, out_f, bias=False, sigma=float(sigma), split=(1.0, 1.0, 1.0), scale=1.0, init_std=0.0,
                             write_mode="float" if substrate == "float32" else "nearest_pulse", generator=gen, counter=counter)
        self.base_sigma_r = self.tl.sigma_r

    @torch.no_grad()
    def program(self):
        W = self.get_W().detach()
        if self.substrate == "tiles":
            sc = W.abs().amax(0, keepdim=True).clamp_min(1e-8)
            self.tl.scale = sc; self.tl.acc_step = sc * 2.0 ** -15
            self.tl.load(W, verify=True)
        else:
            self.tl.load(W)

    def forward(self, x):
        y = self.tl(x)
        return y + self.bias if self.bias is not None else y


def tile_gpt2(model, substrate, sigma, seed, counter=None):
    """Replace every block Conv1D and the LM head by TileProj modules (masters stay in the model)."""
    gen = torch.Generator().manual_seed(1000 + seed)
    dev = model.transformer.wte.weight.device
    projs = []
    for blk in model.transformer.h:
        for parent, name in ((blk.attn, "c_attn"), (blk.attn, "c_proj"), (blk.mlp, "c_fc"), (blk.mlp, "c_proj")):
            conv = getattr(parent, name)
            in_f, out_f = conv.weight.shape
            tp = TileProj(None, conv.bias, in_f, out_f, substrate, sigma, gen, counter).to(dev)
            tp.master = conv                                   # registered child: its weight is the master
            tp.get_W = (lambda c=conv: c.weight)
            setattr(parent, name, tp); projs.append(tp)
    wte = model.transformer.wte
    head = TileProj(lambda: wte.weight.t(), None, wte.weight.shape[1], wte.weight.shape[0], substrate, sigma, gen, counter).to(dev)
    model.lm_head = head; projs.append(head)
    for tp in projs:
        tp.tl.to(dev)
        tp.program()
    return projs


def set_read(projs, sigma_r=None, dial_low=False, k=1):
    """Deploy-time read setting: nominal read noise (sigma_device / sqrt 3), or the dial minimum, divided by sqrt k."""
    for tp in projs:
        if dial_low:
            tp.tl.set_noise(DIAL_LOW)
            tp.tl.sigma_r = tp.tl.sigma_r / math.sqrt(k)
        else:
            tp.tl.sigma_r = (tp.base_sigma_r if sigma_r is None else sigma_r) / math.sqrt(k)


@torch.no_grad()
def ppl(model, data, device, batch, max_batches=None):
    model.eval(); tot = 0.0; n = 0
    for i in range(0, len(data), batch):
        if max_batches is not None and i // batch >= max_batches:
            break
        x = data[i:i + batch].to(device)
        out = model(x, labels=x)
        tot += float(out.loss) * x.numel(); n += x.numel()
    return math.exp(tot / n)


def finetune(model, projs, data, device, args, steps, onchip=False):
    """Off-chip hardware-aware fine-tune (default) or on-chip nearest-pulse fine-tune."""
    model.train()
    tl_params = {id(tp.tl.W) for tp in projs}
    if onchip:
        params = [p for p in model.parameters() if p.requires_grad and not any(p is tp.master.weight for tp in projs if hasattr(tp, "master"))]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        topt = TileOptimizer(model.modules(), opt)
    else:
        params = [p for p in model.parameters() if p.requires_grad and id(p) not in tl_params]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    g = torch.Generator().manual_seed(1)
    n = len(data); t0 = time.time(); losses = []
    wte = model.transformer.wte.weight
    for s in range(steps):
        if not onchip:
            for tp in projs:
                tp.program()                                   # fresh write-verify of the current master
        idx = torch.randint(0, n, (args.batch,), generator=g)
        x = data[idx].to(device)
        loss = model(x, labels=x).loss
        opt.zero_grad(set_to_none=True)
        for tp in projs:
            tp.tl.W.grad = None
        loss.backward()
        if not onchip:                                         # straight-through: tile-weight gradient -> master
            for tp in projs:
                gW = tp.tl.W.grad
                if gW is None:
                    continue
                if hasattr(tp, "master"):
                    tp.master.weight.grad = gW.clone()
                else:
                    wte.grad = gW.t().clone() if wte.grad is None else wte.grad + gW.t()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        (topt if onchip else opt).step()
        losses.append(float(loss))
        if s == 9 or (s + 1) % 200 == 0 or s == steps - 1:
            print(f"      step {s + 1}/{steps} loss {np.mean(losses[-50:]):.3f} ({(time.time() - t0) / (s + 1):.2f} s/step)", flush=True)
    if not onchip:
        for tp in projs:
            tp.program()                                       # deploy: one final programming of the master
    return losses


def ops_per_token(model, projs, val, device):
    c = OpCounter()
    for tp in projs:
        tp.tl.counter = c
    x = val[:1].to(device)
    with torch.no_grad():
        model(x)
    for tp in projs:
        tp.tl.counter = None
    return {k: v / x.numel() for k, v in c.snapshot().items()}


def energy_table(ops):
    import e8b_energy_model as e8b
    base, parts = e8b.bpu_energy(ops, 1, "layer", 8, fom="fom10")
    rows = {}
    for k in KS:
        e_analog = base + (k - 1) * parts["mac_J"]
        e_bound = base + (k - 1) * (parts["mac_J"] + parts["readout_J"])
        rows[str(k)] = {"nJ_per_token_analog_avg": e_analog * 1e9, "nJ_per_token_k_conversions": e_bound * 1e9,
                        "x_vs_k1_analog": e_analog / base, "x_vs_k1_k_conversions": e_bound / base}
    return {"k1_parts_nJ_per_token": {k: v * 1e9 for k, v in parts.items()}, "k1_nJ_per_token": base * 1e9, "averaging": rows,
            "note": "E8b constants, layer-accumulate 8-bit FoM 10 fJ; MACs + conversions + routed events of the tiled projections (attention products digital, not counted)"}


def fresh(args, device, substrate, sigma, seed):
    torch.manual_seed(seed)
    model, _ = build(args.smoke, device)
    projs = tile_gpt2(model, substrate, sigma, seed)
    return model, projs


def run_arm(name, args, device, train, val):
    path = ROOT / OUT / "arms" / f"{name}.json"
    if path.exists() and not args.smoke:
        print(f"[{name}] on disk", flush=True); return
    t0 = time.time(); rec = {"arm": name}
    mb = args.eval_batches
    if name == "deploy":
        model, _ = build(args.smoke, device)
        rec["float32_pretrained_clean"] = ppl(model, val, device, args.batch, mb); del model
        rows = []
        for sub in ("tiles", "float32"):
            for sigma in SIGMAS_DEPLOY:
                for seed in range(args.deploy_seeds):
                    model, projs = fresh(args, device, sub, sigma, seed)
                    r = {"substrate": sub, "sigma": sigma, "seed": seed, "ppl": ppl(model, val, device, args.batch, mb)}
                    if sub == "tiles" and sigma == 0.04:
                        r["dial_low"] = {}
                        for k in KS:
                            set_read(projs, k=k); r[f"avg_k{k}"] = ppl(model, val, device, args.batch, mb)
                            set_read(projs, dial_low=True, k=k); r["dial_low"][str(k)] = ppl(model, val, device, args.batch, mb)
                        if seed == 0:
                            rec["ops_per_token"] = ops_per_token(model, projs, val, device)
                            rec["energy"] = energy_table(rec["ops_per_token"])
                    rows.append(r)
                    print(f"  deploy {sub} sigma={sigma} seed {seed}: ppl {r['ppl']:.2f}" + (f" | dial-low {r['dial_low']}" if 'dial_low' in r else ""), flush=True)
                    del model, projs; torch.cuda.empty_cache() if torch.cuda.is_available() else None
        rec["rows"] = rows
    else:
        # ft_<sub>_s<sigma>_n<steps> | onchip_s<sigma>_n<steps>
        parts = name.split("_")
        onchip = parts[0] == "onchip"
        sub = "tiles" if onchip else parts[1]
        sigma = float(parts[-2][1:]); steps = int(parts[-1][1:])
        if args.smoke:
            steps = 10
        model, projs = fresh(args, device, sub, sigma, 0)
        rec.update({"substrate": sub, "sigma": sigma, "steps": steps, "onchip": onchip, "lr": args.lr, "batch": args.batch, "block": args.block})
        rec["ppl_before"] = ppl(model, val, device, args.batch, mb)
        rec["train_loss"] = finetune(model, projs, train, device, args, steps, onchip=onchip)[::10]
        rec["ppl"] = ppl(model, val, device, args.batch, mb)
        set_read(projs, sigma_r=0.0); rec["ppl_clean_read"] = ppl(model, val, device, args.batch, mb); set_read(projs)
        if sub == "tiles" and sigma == 0.04:
            rec["avg"] = {}; rec["dial_low"] = {}
            for k in KS:
                set_read(projs, k=k); rec["avg"][str(k)] = ppl(model, val, device, args.batch, mb)
                set_read(projs, dial_low=True, k=k); rec["dial_low"][str(k)] = ppl(model, val, device, args.batch, mb)
            set_read(projs)
        if sub == "tiles":
            nw = sum(tp.tl.n_writes for tp in projs); nd = sum(tp.tl.n_devices for tp in projs)
            rec["writes_per_device_total"] = nw / nd
        print(f"  {name}: ppl {rec['ppl_before']:.2f} -> {rec['ppl']:.2f} (clean read {rec['ppl_clean_read']:.2f})" + (f" dial-low {rec['dial_low']}" if "dial_low" in rec else ""), flush=True)
    rec["wall_s"] = time.time() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rec, open(path, "w"), indent=1)


def all_arms():
    a = ["deploy"]
    for n in (3 * E1_STEPS, 5 * E1_STEPS):
        a += [f"ft_tiles_s{s:g}_n{n}" for s in SIGMAS_FT] + [f"ft_float32_s{s:g}_n{n}" for s in [0.0] + SIGMAS_FT]
    a.append(f"onchip_s0.04_n{5 * E1_STEPS}")
    return a


def summarize(args):
    arms = {}
    d = ROOT / OUT / "arms"
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        r = json.load(open(p)); arms[r["arm"]] = r
    out = {"arms": arms, "sigmas_deploy": SIGMAS_DEPLOY, "sigmas_ft": SIGMAS_FT, "dial_low": DIAL_LOW, "ks": KS, "smoke": args.smoke}
    table = []
    fmt = lambda v, ref: "diverged (> 2x float32)" if v > 2 * ref else f"{v:.2f} ({100 * (v / ref - 1):+.1f} %)"
    dep = arms.get("deploy")
    if dep:
        f0 = dep["float32_pretrained_clean"]
        def mean(sub, s, key="ppl"):
            v = [r[key] for r in dep["rows"] if r["substrate"] == sub and r["sigma"] == s]
            return float(np.mean(v)) if v else None
        for sub in ("tiles", "float32"):
            for s in SIGMAS_DEPLOY:
                m = mean(sub, s)
                if m is not None:
                    table.append({"arm": f"deploy {sub} sigma={s:g}", "sigma_device": s, "ppl": m, "ref": f0, "gap_pct": 100 * (m / f0 - 1), "diverged": m > 2 * f0})
        r04 = [r for r in dep["rows"] if r["substrate"] == "tiles" and r["sigma"] == 0.04 and "dial_low" in r]
        for k in KS:
            if r04:
                m = float(np.mean([r[f"avg_k{k}"] for r in r04])); table.append({"arm": f"E16 deploy averaging k={k}", "sigma_device": 0.04, "ppl": m, "ref": f0, "gap_pct": 100 * (m / f0 - 1), "diverged": m > 2 * f0})
                m = float(np.mean([r["dial_low"][str(k)] for r in r04])); table.append({"arm": f"E16 deploy dial-low k={k}", "sigma_device": 0.04, "ppl": m, "ref": f0, "gap_pct": 100 * (m / f0 - 1), "diverged": m > 2 * f0})
        out["float32_pretrained_clean"] = f0
        out["energy"] = dep.get("energy")
        # first sigma where deploy-only exceeds 2x float32
        for sub in ("tiles", "float32"):
            first = next((s for s in SIGMAS_DEPLOY if (mean(sub, s) or 0) > 2 * f0), None)
            out[f"deploy_{sub}_first_diverged_sigma"] = first
    for n in (3 * E1_STEPS, 5 * E1_STEPS):
        ref = arms.get(f"ft_float32_s0_n{n}")
        if not ref:
            continue
        R = ref["ppl"]
        for s in [0.0] + SIGMAS_FT:
            for sub in ("tiles", "float32"):
                a = arms.get(f"ft_{sub}_s{s:g}_n{n}")
                if a:
                    table.append({"arm": f"fine-tune {sub} sigma={s:g} {n} steps", "sigma_device": s, "ppl": a["ppl"], "ref": R, "gap_pct": 100 * (a["ppl"] / R - 1), "diverged": a["ppl"] > 2 * R,
                                  "writes_per_device_total": a.get("writes_per_device_total")})
                    if sub == "tiles" and s == 0.04:
                        for k in KS:
                            table.append({"arm": f"E16 fine-tune {n} + averaging k={k}", "sigma_device": 0.04, "ppl": a["avg"][str(k)], "ref": R, "gap_pct": 100 * (a["avg"][str(k)] / R - 1), "diverged": a["avg"][str(k)] > 2 * R})
                            table.append({"arm": f"E16 fine-tune {n} + dial-low k={k}" + (" (the pre-registered combined arm)" if (k == 8 and n == 5 * E1_STEPS) else ""), "sigma_device": 0.04,
                                          "ppl": a["dial_low"][str(k)], "ref": R, "gap_pct": 100 * (a["dial_low"][str(k)] / R - 1), "diverged": a["dial_low"][str(k)] > 2 * R})
        oc = arms.get(f"onchip_s0.04_n{n}")
        if oc:
            table.append({"arm": f"on-chip nearest-pulse fine-tune sigma=0.04 {n} steps (diagnostic)", "sigma_device": 0.04, "ppl": oc["ppl"], "ref": R, "gap_pct": 100 * (oc["ppl"] / R - 1),
                          "diverged": oc["ppl"] > 2 * R, "writes_per_device_total": oc.get("writes_per_device_total")})
    out["table"] = table
    # E1 old row, corrected
    try:
        e1 = json.load(open(ROOT / "results/e1_full/results.json"))
        f = e1["rows"][0]["ppl_pretrained_noisy"]
        first = next((r["sigma"] for r in e1["rows"] if r["ppl_pretrained_noisy"] > 2 * f), None)
        out["e1_full_pretrained_at_noise_corrected"] = {"float32_ppl": f, "first_sigma_above_2x": first,
                                                       "statement": f"E1 pretrained-at-noise (float noise model, results/e1_full): diverged from sigma = {first:g} (first sigma above 2x the float32 perplexity {f:.2f}); values above that are not reported." if first is not None else "never above 2x"}
    except Exception:
        pass
    at04 = [t for t in table if abs(t["sigma_device"] - 0.04) < 1e-9 and "float32" not in t["arm"]]
    best = min(at04, key=lambda t: t["gap_pct"]) if at04 else None
    met = bool(best and best["gap_pct"] <= 3.0)
    needed = set(all_arms()) - set(arms)
    lines = [f"{t['arm']}: {fmt(t['ppl'], t['ref'])}" for t in at04]
    en = out.get("energy") or {}
    estr = "; ".join(f"k={k}: {v['x_vs_k1_analog']:.2f}x (k conversions {v['x_vs_k1_k_conversions']:.2f}x)" for k, v in en.get("averaging", {}).items())
    out["verdict"] = {"criterion_met": met, "complete": not needed, "missing_arms": sorted(needed), "best_arm_at_0.04": best,
                      "statement": ("SMOKE RUN. " if args.smoke else "") + ("" if not needed else f"PARTIAL ({len(arms)}/{len(all_arms())} arm groups). ")
                                   + f"Criterion (some arm at sigma_device 0.04 within 3 % of its float32 reference): {'MET' if met else 'NOT met'}"
                                   + (f"; best {best['arm']} at {best['gap_pct']:+.1f} %" if best else "") + ". Arms at 0.04: " + "; ".join(lines)
                                   + (f". Energy of read averaging (E8b, per token, vs k=1): {estr}" if estr else "")
                                   + (f". {out['e1_full_pretrained_at_noise_corrected']['statement']}" if "e1_full_pretrained_at_noise_corrected" in out else "")}
    print("E1-fix verdict:", out["verdict"]["statement"])
    save_results("e1fix", out, OUT)
    if not args.no_fig:
        plot(out, figure_path("e1fix_gpt2_deploy"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    dep = res["arms"].get("deploy")
    if dep:
        f0 = res["float32_pretrained_clean"]
        for sub, col in (("tiles", "C0"), ("float32", "k")):
            s = SIGMAS_DEPLOY; m = [np.mean([r["ppl"] for r in dep["rows"] if r["substrate"] == sub and r["sigma"] == x] or [np.nan]) for x in s]
            ax[0].plot(s, m, "o-", color=col, label=f"deploy-only, {sub}")
        ax[0].axhline(f0, color="0.5", ls=":", label="float32 clean"); ax[0].axhline(2 * f0, color="r", ls=":", lw=1, label="2x float32 (diverged above)")
        ax[0].set_yscale("log"); ax[0].set_xlabel("σ_device"); ax[0].set_ylabel("WikiText-2 perplexity"); ax[0].legend(fontsize=7); ax[0].set_title("deploy a pretrained GPT-2 on tiles", fontsize=9)
    T = [t for t in res["table"] if abs(t["sigma_device"] - 0.04) < 1e-9]
    if T:
        ax[1].barh(range(len(T)), [min(t["gap_pct"], 100) for t in T], color=["C2" if t["gap_pct"] <= 3 else "C1" for t in T])
        ax[1].set_yticks(range(len(T))); ax[1].set_yticklabels([t["arm"] for t in T], fontsize=6); ax[1].axvline(3, color="k", ls="--", lw=1)
        ax[1].set_xlabel("gap to the float32 reference, % (capped at 100)"); ax[1].set_title("σ_device = 0.04: every arm (dashed = 3 % criterion)", fontsize=9)
    fig.suptitle(res["verdict"]["statement"][:170], fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e1fix] figure {path}")


def main():
    ap = argparse.ArgumentParser(description="E1-fix: GPT-2 deployment on tiles + E16 chip-side handling")
    ap.add_argument("--arms", default="all", help="comma list of arm names (see all_arms), or 'all'")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--deploy-seeds", type=int, default=3)
    ap.add_argument("--eval-batches", type=int, default=None)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    global OUT
    if args.smoke:
        OUT = "results/hpcc_smoke/e1fix"; args.block, args.batch, args.eval_batches, args.deploy_seeds = 64, 4, 2, 1
    if not args.summarize:
        device = args.device
        _, tok = build(args.smoke, "cpu")
        train, val = get_data(tok, args.block, args.smoke)
        arms = all_arms() if args.arms == "all" else args.arms.split(",")
        for a in arms:
            run_arm(a, args, device, train, val)
    summarize(args)


if __name__ == "__main__":
    main()
