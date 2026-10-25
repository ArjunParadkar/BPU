"""E8 full: matched float32 fine-tune control (EXPLORATORY / POST HOC, 2026-10-24).

E8 at full size (results/e8_full, d 384, 6 layers, 5000 ideal steps, 1000 fine-tune steps, 3 seeds) missed its 2 %
criterion: the noise-aware tile fine-tune ended +6.6 % (sigma 0) ... +3.9 % (sigma 0.06) above the ideal model, which
was NOT trained for those 1000 extra steps.  The ideal model is heavily overfit (train ~0.57, val ~2.2), so the gap
may be continued overfitting rather than the substrate.  This control runs the same 1000 extra steps without tiles:

  ideal_continued   the ideal float model fine-tuned for --ft-steps steps exactly as the tile arm (lr --ft-lr, warmup
                    20, cosine, the same batch generator seed 1000 + seed), no noise.
  float32_twin@s    the float32 twin (TileLinear write_mode="float": same read / write noise injected on float weights,
                    no pulses, no converter) at each sigma of the E8 grid, fine-tuned the same way.

Each arm is evaluated on the same validation batches as E8 (generator seed 99), as deployed and with a clean read.
The ideal model is retrained per seed with the E8 code; its val loss is checked against results/e8_full.
This cannot change the E8 verdict (criterion not met); it only informs the reading.  Arms are saved one by one
(results/e8_full_f32ctl/parts) so the job resumes after an interruption.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from common import base_parser, save_results, ROOT
from bpusim.torchtile import tile_modules
import e8_nanogpt_tiles as e8

OUT = "results/e8_full_f32ctl"


def main():
    ap = base_parser("E8 float32 fine-tune control", default_seeds=3)
    ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--ft-steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--d", type=int, default=384); ap.add_argument("--layers", type=int, default=6); ap.add_argument("--heads", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--ft-lr", type=float, default=3e-4)
    ap.add_argument("--eval-batches", type=int, default=40); ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--sigmas", type=str, default=",".join(map(str, e8.SIGMAS)))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out_dir = args.out if args.out and args.out != "results" else OUT
    seeds = args.seeds; sigmas = [float(s) for s in args.sigmas.split(",")]
    if args.smoke:
        args.steps, args.ft_steps, args.eval_batches, seeds, args.d = 30, 10, 2, 1, 48
        out_dir = "results/hpcc_smoke/e8_f32ctl"
    parts = ROOT / out_dir / "parts"; parts.mkdir(parents=True, exist_ok=True)
    ref = ROOT / "results/e8_full/results.json"
    ref = json.load(open(ref)) if ref.exists() else None
    train_ids, val_ids, vocab = e8.load_text()
    mk = lambda sigma, seed, kw=None: e8.GPT(vocab, args.d, args.layers, args.heads, args.block, sigma, None, False, seed=seed, tile_kw=kw)
    ev = lambda m: e8.evaluate(m, val_ids, args.block, args.batch, args.eval_batches, torch.Generator().manual_seed(99))
    for seed in range(seeds):
        arms = ["ideal_continued"] + [f"float32_twin@{s:g}" for s in sigmas]
        if all((parts / f"s{seed}_{a}.json").exists() for a in arms):
            print(f"seed {seed}: all arms on disk", flush=True); continue
        torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
        ideal = mk(None, seed)
        t0 = time.time()
        losses = e8.train(ideal, train_ids, args.steps, args.block, args.batch, args.lr, gen, tiled=False, label="ideal")
        ideal_val = ev(ideal)
        e8_val = ref["runs"][seed]["ideal_val_loss"] if ref and seed < len(ref["runs"]) else None
        print(f"seed {seed}: ideal val {ideal_val:.4f} (E8 full {e8_val}) in {time.time() - t0:.0f}s", flush=True)
        sd = {k: v.clone() for k, v in ideal.state_dict().items()}
        for arm in arms:
            p = parts / f"s{seed}_{arm}.json"
            if p.exists():
                print(f"  {arm}: on disk", flush=True); continue
            t0 = time.time()
            if arm == "ideal_continued":
                m = mk(None, seed); m.load_state_dict(sd); tiled = False; sigma = None
            else:
                sigma = float(arm.split("@")[1])
                m = mk(sigma, seed, {"write_mode": "float"}); m.load_state_dict(sd, strict=False); e8.copy_weights(ideal, m, verify=False); tiled = True
            before = ev(m)
            e8.train(m, train_ids, args.ft_steps, args.block, args.batch, args.ft_lr, torch.Generator().manual_seed(1000 + seed), tiled=tiled, warmup=20, label=arm)
            val = ev(m); clean = None
            if sigma:
                for t in tile_modules(m):
                    t.sigma_r = 0.0
                clean = ev(m)
            rec = {"seed": seed, "arm": arm, "sigma": sigma, "ideal_val_loss": ideal_val, "e8_full_ideal_val_loss": e8_val,
                   "ideal_train_loss_last": float(np.mean(losses[-100:])), "val_before_ft": before, "val_after_ft": val, "val_after_ft_clean_read": clean,
                   "gap_pct_vs_ideal": 100 * (val / ideal_val - 1), "wall_s": time.time() - t0}
            json.dump(rec, open(p, "w"), indent=1)
            print(f"  {arm}: before {before:.4f} after {val:.4f} ({rec['gap_pct_vs_ideal']:+.2f}% vs ideal) clean {clean} [{rec['wall_s']:.0f}s]", flush=True)
    # summary
    recs = [json.load(open(f)) for f in sorted(parts.glob("*.json"))]
    summ = {}
    for a in sorted({r["arm"] for r in recs}):
        g = [r["gap_pct_vs_ideal"] for r in recs if r["arm"] == a]
        summ[a] = {"gap_pct_mean": float(np.mean(g)), "gap_pct_sd": float(np.std(g, ddof=1)) if len(g) > 1 else 0.0, "n": len(g)}
    tile = {}
    if ref:
        for s in sigmas:
            tile[f"{s:g}"] = ref["summary"][str(s)]["noise_aware_gap_pct"]["mean"]
    res = {"exploratory": True, "post_hoc": True, "seeds": seeds, "sigmas": sigmas, "ft_steps": args.ft_steps, "records": recs, "summary": summ,
           "e8_full_tile_noise_aware_gap_pct": tile,
           "verdict": {"statement": "EXPLORATORY control for E8 full (cannot change its verdict). Gap of each 1000-step fine-tune vs the un-fine-tuned ideal: "
                       + "; ".join(f"{a} {v['gap_pct_mean']:+.2f}% (n={v['n']})" for a, v in summ.items())
                       + (" | E8 tile noise-aware: " + ", ".join(f"s={k} {v:+.2f}%" for k, v in tile.items()) if tile else "")}}
    print(res["verdict"]["statement"], flush=True)
    save_results("e8_f32ctl", res, out_dir)


if __name__ == "__main__":
    main()
