"""E17 (EXPLORATORY, no pass/fail): does the knowledge survive the noise?  (round four, optional)

For each seed s with both runs finished in E15, compare the float32 sigma = 0 model and the tiles model
trained at the tiles sigma* (read from results/e15/results.json) on the SAME 2000 held-out val sequences
(512 tokens each; 8 token positions per sequence are kept, the same positions for both models, giving
16 000 token vectors per layer).  Representations = the residual stream after the embedding and after every
block; both models are read clean (sigma 0; the tiles model through its 8-bit read-out, as it deploys).

  CKA        linear CKA (Kornblith et al. 2019) between matched layers, and the full layer x layer matrix.
  probe      softmax probe for the next token, restricted to the 256 most frequent next tokens of the sample;
             trained on half the tokens of model A's layer, tested on the other half of (a) A itself
             (within), (b) model B's same layer as is (raw transfer: only meaningful if the bases agree;
             the two models do not share an initialisation, so near-chance is expected), and (c) B after an
             orthogonal Procrustes map B -> A fitted on the training half (aligned transfer).  Both directions.
  kNN        10-nearest-neighbour graphs (cosine) of 4000 token vectors per layer; overlap = mean fraction of
             shared neighbours (chance = k / N).
Hypothesis (exploratory, stated before the run): the weights differ but the representational structure
matches, CKA > 0.8 at matched layers (the hardware analogue of representational drift: stable population
geometry over changing synapses).  Reported as numbers, not as a verdict.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F

from common import ROOT, save_results, figure_path
import e15_scratch_lm as e15
from e15_scratch_lm import build, set_read, Tokens

OUT = "results/e17"


def load_model(key, device):
    p = ROOT / e15.OUT / "runs" / key / "weights.pt"
    c = torch.load(p, map_location=device, weights_only=False)
    m = build(c["sub"], c["sigma"], c["seed"], c["cfg"], device)
    missing, unexpected = m.load_state_dict(c["weights"], strict=False)
    assert not unexpected, unexpected
    set_read(m, 0.0); m.eval()
    return m, c["cfg"]


@torch.no_grad()
def reps(model, data, ix, pos, block, device, bs=16):
    """Residual stream after the embedding and after every block at the kept positions: list of (n, d)."""
    out = None; nxt = []
    for i in range(0, len(ix), bs):
        x, y = data.windows(ix[i:i + bs], block)
        x = x.to(device); T = x.shape[1]
        h = model.tok(x) + model.pos(torch.arange(T, device=device))
        hs = [h]
        for b in model.blocks:
            h = b(h); hs.append(h)
        p = pos[i:i + bs]
        sel = [torch.stack([hh[j, p[j]] for j in range(len(p))]).reshape(-1, hh.shape[-1]).float().cpu() for hh in hs]
        out = sel if out is None else [torch.cat([a, b]) for a, b in zip(out, sel)]
        nxt.append(torch.stack([y[j, p[j]] for j in range(len(p))]).reshape(-1))
    return out, torch.cat(nxt)


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = (X.T @ Y).pow(2).sum(); return float(hsic / ((X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()))


def probe_scores(A, B, labels, device, steps=300):
    n = len(labels); idx = torch.randperm(n, generator=torch.Generator().manual_seed(0)); tr, te = idx[: n // 2], idx[n // 2:]
    mu, sd = A[tr].mean(0), A[tr].std(0) + 1e-6
    za = ((A - mu) / sd).to(device); zb_raw = ((B - B[tr].mean(0)) / (B[tr].std(0) + 1e-6)).to(device)
    # orthogonal Procrustes B -> A on the training half
    U, _, Vt = torch.linalg.svd(zb_raw[tr].T @ za[tr]); R = U @ Vt
    zb_al = zb_raw @ R
    y = labels.to(device); k = int(y.max()) + 1
    W = torch.zeros(za.shape[1], k, device=device, requires_grad=True); b = torch.zeros(k, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=1e-2)
    for _ in range(steps):
        opt.zero_grad(); F.cross_entropy(za[tr] @ W + b, y[tr]).backward(); opt.step()
    acc = lambda Z: float(((Z[te] @ W + b).argmax(1) == y[te]).float().mean())
    return {"within": acc(za), "raw_transfer": acc(zb_raw), "aligned_transfer": acc(zb_al), "chance_majority": float((y[te] == y[tr].mode().values).float().mean())}


def knn_overlap(A, B, k=10, n=4000):
    A = F.normalize(A[:n], dim=1); B = F.normalize(B[:n], dim=1)
    na = (A @ A.T).fill_diagonal_(-2).topk(k, 1).indices; nb = (B @ B.T).fill_diagonal_(-2).topk(k, 1).indices
    shared = [(len(set(a.tolist()) & set(b.tolist()))) / k for a, b in zip(na, nb)]
    return {"overlap": float(np.mean(shared)), "chance": k / n}


def main():
    ap = argparse.ArgumentParser(description="E17 representation comparison (exploratory)")
    ap.add_argument("--data", default="data/e15")
    ap.add_argument("--n-seq", type=int, default=2000)
    ap.add_argument("--pos-per-seq", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    global OUT
    if args.smoke:
        e15.OUT = "results/hpcc_smoke/e15"; OUT = "results/hpcc_smoke/e17"; args.data = "data/e15_smoke"; args.n_seq = 64
    r15 = json.load(open(ROOT / e15.OUT / "results.json"))
    star = r15["hypotheses"]["tiles_sigma_star"]["star"]
    val = Tokens(ROOT / args.data / "val.bin")
    pairs = []
    for seed in range(r15["seeds_requested"]):
        ka, kb = e15.run_key("float32", 0.0, seed), e15.run_key("tiles", star, seed)
        if not ((ROOT / e15.OUT / "runs" / ka / "weights.pt").exists() and (ROOT / e15.OUT / "runs" / kb / "weights.pt").exists()):
            continue
        ma, cfg = load_model(ka, args.device); mb, _ = load_model(kb, args.device)
        g = torch.Generator().manual_seed(17)
        ix = torch.randint(0, len(val) - cfg["block"] - 2, (args.n_seq,), generator=g).tolist()
        pos = torch.randint(0, cfg["block"], (args.n_seq, args.pos_per_seq), generator=g)
        RA, nxt = reps(ma, val, ix, pos, cfg["block"], args.device); RB, _ = reps(mb, val, ix, pos, cfg["block"], args.device)
        top = torch.bincount(nxt).topk(min(256, int(nxt.max()) + 1)).indices
        keep = torch.isin(nxt, top); remap = {int(t): i for i, t in enumerate(top)}
        lab = torch.tensor([remap[int(t)] for t in nxt[keep]])
        L = len(RA)
        row = {"seed": seed, "float32": ka, "tiles": kb, "layers": L,
               "cka_matched": [cka(RA[l], RB[l]) for l in range(L)],
               "cka_matrix": [[cka(RA[i], RB[j]) for j in range(L)] for i in range(L)],
               "probe_f32_to_tiles": [probe_scores(RA[l][keep], RB[l][keep], lab, args.device) for l in range(L)],
               "probe_tiles_to_f32": [probe_scores(RB[l][keep], RA[l][keep], lab, args.device) for l in range(L)],
               "knn": [knn_overlap(RA[l], RB[l]) for l in range(L)]}
        pairs.append(row)
        print(f"seed {seed}: CKA matched {np.round(row['cka_matched'], 3).tolist()}", flush=True)
        del ma, mb
    res = {"exploratory": True, "tiles_sigma_star": star, "pairs": pairs, "n_seq": args.n_seq, "pos_per_seq": args.pos_per_seq}
    if pairs:
        ck = np.mean([p["cka_matched"] for p in pairs], 0)
        res["cka_matched_mean"] = ck.tolist()
        al = np.mean([[q["aligned_transfer"] / max(q["within"], 1e-9) for q in p["probe_f32_to_tiles"]] for p in pairs], 0)
        kn = np.mean([[q["overlap"] for q in p["knn"]] for p in pairs], 0)
        res["statement"] = (("SMOKE RUN. " if args.smoke else "") + f"EXPLORATORY, no pass/fail. {len(pairs)} seed pair(s), float32 sigma 0 vs tiles sigma* {star:g}: "
                            f"matched-layer CKA {', '.join(f'{x:.2f}' for x in ck)} (layers 0 = embedding .. {len(ck) - 1}); {int((ck > 0.8).sum())}/{len(ck)} layers above 0.8; "
                            f"aligned probe transfer / within {', '.join(f'{x:.2f}' for x in al)}; kNN overlap {', '.join(f'{x:.2f}' for x in kn)} (chance {pairs[0]['knn'][0]['chance']:.4f}).")
    else:
        res["statement"] = "no seed with both runs finished"
    print("E17:", res["statement"])
    save_results("e17", res, OUT)
    if pairs and not args.no_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        ax[0].plot(res["cka_matched_mean"], "o-"); ax[0].axhline(0.8, ls=":", color="k"); ax[0].set_xlabel("layer"); ax[0].set_ylabel("linear CKA, matched layers")
        im = ax[1].imshow(np.mean([p["cka_matrix"] for p in pairs], 0), vmin=0, vmax=1); fig.colorbar(im, ax=ax[1]); ax[1].set_xlabel("tiles layer"); ax[1].set_ylabel("float32 layer")
        for key, lab in (("within", "within"), ("raw_transfer", "raw transfer"), ("aligned_transfer", "Procrustes-aligned transfer")):
            ax[2].plot(np.mean([[q[key] for q in p["probe_f32_to_tiles"]] for p in pairs], 0), "o-", label=lab)
        ax[2].set_xlabel("layer"); ax[2].set_ylabel("next-token probe accuracy (top-256)"); ax[2].legend(fontsize=7)
        fig.suptitle(res["statement"][:170], fontsize=7); fig.tight_layout(); fig.savefig(figure_path("e17_representations"), dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
