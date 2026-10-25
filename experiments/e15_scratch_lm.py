"""E15: train a small frontier-type language model FROM SCRATCH on tiles (round four, "learning vs knowing").

Learning vs knowing: variability should help while structure is forming (training from scratch) and be
suppressed once it is formed (deployment).  E11 showed it on a 2.7M character model; E15 asks it of a
decoder-only GPT at web scale.

Model: the E8 GPT (``e8_nanogpt_tiles.GPT``) at 8 layers, d = 512, 8 heads, context 512, GPT-2 tokenizer
(vocab 50257): 25.2M parameters in the blocks + a 25.7M untied LM head on tiles + a 25.7M digital token
embedding (77M total, 51M on tiles; ~2.4e17 training FLOPs at 0.8B tokens).  Every Linear (qkv, proj, fc,
fc2, head) is a ``TileLinear``; embeddings and layer norms are digital.
  tiles    nearest-pulse writes (16-bit accumulator, nominal-effect subtraction), layer-accumulate 8-bit
           read-out (auto range), attention through the volatile crossbars (``tile_attention``: score and
           value reads through the read-out); read noise sigma_r is the ONLY device noise (split (0, 0, 1),
           as E11), set with the MODULATE primitive.
  float32  float weights, ideal updates, the same multiplicative read noise injected on every weight read
           and on the attention cache reads (not clipped: the reference has no dial).  Its attention is
           computed with torch's fused SDPA kernel on the noisy K / V reads (mathematically the same product
           as ``tile_attention`` without a converter; the memory saving is what lets it run at batch 64 x 512).
Data: ``data/e15`` (``e15_data.py``): OpenWebText (or FineWeb-Edu, see meta.json), held-out val split
(documents not in train), TinyStories and Python as distribution-shift sets.  Matched across arms: the same
token budget, the same batch sequence (batch b of step t of seed s is drawn from (s, t, b) only), the same
optimiser (AdamW, betas 0.9 / 0.95, weight decay 0 so the noise is the only regulariser, lr 6e-4, 2 % warm-up,
cosine to 10 %, clip 1.0), batch 64 x 512 tokens (micro-batches with gradient accumulation; tiles are
written once per optimiser step).

Arms: float32 at sigma in {0, 0.1, 0.2, 0.4}; tiles at sigma in {0, 0.02, 0.04, 0.06, 0.1}; "learn high,
deploy low" = the tiles sigma = 0.04 run (trained at 0.04, identical training) evaluated with the dial at
its minimum 0.005.  3 seeds if the budget allows, else 2 (stated in results.json and MONDAY2.md); if 2 do
not fit, the TOKEN budget is cut, for every arm alike, not the arms.

Evaluation per run (end of training): loss on val / TinyStories / Python at three read settings: clean
(dial off, sigma 0), as deployed (the training sigma), dial-low (0.005); perplexity = exp(loss); shift gaps
= shift-set loss - val loss; rel_pert (``relpert.measure``, 16 batches of 256 tokens) at the training sigma,
0 and 0.005; total writes per device against the 1e6 endurance; ops per token for the E8b energy
projection.  A short val curve is recorded at every 10 % checkpoint.

PRE-REGISTERED (2026-10-23, before any run; primary metric = val loss at the CLEAN read, as in E11;
sigma* = argmin over the arm's grid of the seed-mean, bootstrap CI over seeds, ``e11.bootstrap_argmin``):
  H1  tiles sigma* > 0 with the 95 % CI excluding 0.
  H2  tiles sigma* within the native band [0.02, 0.06] (point estimate; the CI is reported).
  H3  tiles at sigma* reach val loss <= float32 at sigma = 0 (seed means; per-seed wins reported).
  H4  learn-high / deploy-low (tiles trained at 0.04, read at 0.005) is at least as good as fixed sigma*
      (tiles trained and read at sigma*): val loss of the first <= val loss of the second (seed means).
Reported, no threshold: float32 sigma* with CI; shift-gap slopes (python - val, tinystories - val) per
substrate with CI; rel_pert at each optimum; writes / device vs 1e6; E8b projected energy per training token
(layer-accumulate 8-bit, FoM 10 fJ; forward array + converter + routing energy x 3 for forward + two backward
passes, cache writes, measured write pulses x 100 pJ).  Negatives are reported as negatives.

Running it: ``--worker`` processes the run list (seeds outer, arms inner) under per-run lock directories
(results/e15/runs/<run>/lock, stale after 45 min without a heartbeat), checkpoints every 10 % of the tokens
and when the job's ``--segment-minutes`` budget is nearly spent, so a chain of 6-hour jobs (submit.sh e15)
resumes exactly where the last one stopped; finished runs keep their final weights (weights.pt, for E17)
and drop the full checkpoint.  ``--timing`` measures throughput on synthetic tokens; ``--summarize`` builds
results/e15/results.json from the finished runs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import socket
import time
import types

import numpy as np
import torch
import torch.nn.functional as F

from common import ROOT, save_results, figure_path
import e8_nanogpt_tiles as e8
from e8_nanogpt_tiles import GPT
from e11_regularization import set_all_noise, bootstrap_argmin, bootstrap_slope
from bpusim.torchtile import TileLinear, TileOptimizer, OpCounter, tile_modules, write_budget_report
from relpert import measure, set_noise_unclipped, _ranges, _restore

F32_SIGMAS = [0.0, 0.1, 0.2, 0.4]
TILE_SIGMAS = [0.0, 0.02, 0.04, 0.06, 0.1]
DIAL_LOW = 0.005
LEARN_HIGH = 0.04
OUT = "results/e15"
_ORIG_TA = e8.tile_attention
_ORIG_ATTN = e8.Block.attn


def arms():
    return [("float32", s) for s in F32_SIGMAS] + [("tiles", s) for s in TILE_SIGMAS]


def run_key(sub, sigma, seed):
    return f"{sub}_s{sigma:g}_seed{seed}"


# ------------------------------------------------------------------ model
def sdpa_attn(self, x):
    """float32 reference attention: noisy K / V reads, fused SDPA (falls back to the E8 path while relpert patches it)."""
    if e8.tile_attention is not _ORIG_TA:
        return _ORIG_ATTN(self, x)
    B, T, C = x.shape
    q, k, v = self.qkv(x).split(C, dim=2)
    hd = C // self.n_head
    q, k, v = (t.view(B, T, self.n_head, hd).transpose(1, 2) for t in (q, k, v))
    sr = self.attn_sigma_r
    if sr > 0:
        k = k * (1 + sr * torch.randn_like(k)); v = v * (1 + sr * torch.randn_like(v))
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return self.proj(y.transpose(1, 2).reshape(B, T, C))


def build(sub, sigma, seed, cfg, device):
    torch.manual_seed(seed)
    tile_kw = dict(split=(0.0, 0.0, 1.0), write_mode="float" if sub == "float32" else "nearest_pulse")
    model = GPT(cfg["vocab"], cfg["d"], cfg["layers"], cfg["heads"], cfg["block"], sigma, None, seed=seed, tile_kw=tile_kw).to(device)
    if sub == "float32":
        for b in model.blocks:
            b.attn = types.MethodType(sdpa_attn, b)
    set_all_noise(model, sigma, 0.0)
    return model


def set_read(model, sigma):
    """Evaluation read setting (no clip on the float32 reference; tiles: 0 = dial off, else the dial)."""
    set_all_noise(model, sigma, 0.0)


# ------------------------------------------------------------------ data
class Tokens:
    def __init__(self, path, in_ram=False):
        # 2026-10-25: in_ram reads the whole file into memory.  The training file (1.7 GB) sits on the cluster's
        # network filesystem, and 64 random 513-token windows per step through a memmap were a measurable part of
        # the 3.5x gap between the timing job (a 4 MB synthetic file, always in the page cache) and the real runs.
        self.arr = np.fromfile(path, dtype=np.uint16) if in_ram else np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.arr)

    def windows(self, ix, block):
        a = np.stack([np.asarray(self.arr[i:i + block + 1], dtype=np.int64) for i in ix])
        t = torch.from_numpy(a)
        return t[:, :-1].contiguous(), t[:, 1:].contiguous()


def train_batch(data, seed, step, m, micro, block, device):
    g = torch.Generator().manual_seed(seed * 10_000_000 + step * 256 + m)
    ix = torch.randint(0, len(data) - block - 2, (micro,), generator=g).tolist()
    x, y = data.windows(ix, block)
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def eval_loss(model, data, n_batches, micro, block, device, seed=99):
    model.eval(); tot = 0.0
    g = torch.Generator().manual_seed(seed)
    for _ in range(n_batches):
        ix = torch.randint(0, len(data) - block - 2, (micro,), generator=g).tolist()
        x, y = data.windows(ix, block)
        _, loss = model(x.to(device), y.to(device)); tot += float(loss)
    model.train()
    return tot / n_batches


# ------------------------------------------------------------------ checkpoints
def tile_extras(model):
    ex = {"tiles": [], "blocks": []}
    for t in tile_modules(model):
        r = {"n_pulses": t.n_pulses, "n_writes": t.n_writes, "n_acc_calls": t.n_acc_calls, "primitive_log": t.primitive_log[-20:]}
        if t.readout is not None:
            r["ranges"] = [v.detach().cpu() if torch.is_tensor(v) else v for v in t.readout.ranges.values()]
        ex["tiles"].append(r)
    for b in model.blocks:
        if b.readout is not None:
            pre = f"attn{id(b)}"
            ex["blocks"].append({k[len(pre):]: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in b.readout.ranges.items()})
        else:
            ex["blocks"].append(None)
    return ex


def restore_extras(model, ex, device):
    for t, r in zip(tile_modules(model), ex["tiles"]):
        t.n_pulses, t.n_writes, t.n_acc_calls = r["n_pulses"], r["n_writes"], r["n_acc_calls"]
        t.primitive_log = list(r["primitive_log"])
        if t.readout is not None and r.get("ranges"):
            v = r["ranges"][0]
            t.readout.ranges = {t.site: v.to(device) if torch.is_tensor(v) else v}
    for b, r in zip(model.blocks, ex["blocks"]):
        if r is not None and b.readout is not None:
            b.readout.ranges = {f"attn{id(b)}{k}": (v.to(device) if torch.is_tensor(v) else v) for k, v in r.items()}


def save_ckpt(path, model, opt, state):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "state": state, "extras": tile_extras(model),
                "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------ one run (resumable)
def train_step(model, opt, topt, train, seed, step, accum, micro, block, device, sync_each=False):
    """One optimiser step over ``accum`` micro-batches.  The loss is accumulated on the device and read once per
    step (sync_each=True reproduces the original per-micro-batch float(loss), kept for the throughput diagnostic)."""
    topt.zero_grad(); tot = 0.0 if sync_each else torch.zeros((), device=device)
    for m in range(accum):
        x, y = train_batch(train, seed, step, m, micro, block, device)
        _, loss = model(x, y)
        (loss / accum).backward()
        tot = tot + (float(loss) if sync_each else loss.detach()) / accum
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    topt.step()
    return float(tot)


def lr_at(step, steps, lr):
    warm = max(1, int(0.02 * steps))
    return lr * min(1.0, (step + 1) / warm) * (0.9 * 0.5 * (1 + math.cos(math.pi * min(step, steps) / steps)) + 0.1)


def train_run(sub, sigma, seed, args, cfg, data, deadline, heartbeat):
    key = run_key(sub, sigma, seed)
    d = ROOT / OUT / "runs" / key; d.mkdir(parents=True, exist_ok=True)
    ck = d / "ckpt.pt"
    device = args.device
    model = build(sub, sigma, seed, cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    topt = TileOptimizer(model.modules(), opt)
    tokens_per_step = args.batch_seqs * cfg["block"]
    steps = args.tokens // tokens_per_step
    accum = args.batch_seqs // args.micro
    state = {"step": 0, "losses": [], "curve": [], "wall_s": 0.0, "segments": 0, "steps": steps, "tokens_per_step": tokens_per_step}
    if ck.exists():
        # 2026-10-25: load on the CPU.  With map_location=device the saved RNG states landed on CUDA and
        # torch.set_rng_state refused them ("RNG state must be a torch.ByteTensor"): every resume crashed.
        # load_state_dict copies weights / optimizer state to the parameters' device.
        c = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); state = c["state"]
        restore_extras(model, c["extras"], device)
        torch.set_rng_state(c["rng_cpu"].cpu().to(torch.uint8))
        if c["rng_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([r.cpu().to(torch.uint8) for r in c["rng_cuda"]])
        print(f"[{key}] resumed at step {state['step']}/{steps}", flush=True)
    assert state["steps"] == steps, f"token budget changed for {key}: {state['steps']} vs {steps}"
    state["segments"] += 1
    marks = {int(steps * k / 10) for k in range(1, 11)}
    model.train(); t0 = time.time(); t_seg = time.time()
    step = state["step"]; step0 = step
    while step < steps:
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(step, steps, args.lr)
        tot = train_step(model, opt, topt, data["train"], seed, step, accum, args.micro, cfg["block"], device)
        state["losses"].append(tot); step += 1; state["step"] = step
        if step % 100 == 0 or step == 10:
            heartbeat()
            dt = time.time() - t0
            print(f"[{key}] step {step}/{steps} loss {np.mean(state['losses'][-100:]):.3f} {(time.time() - t_seg) / max(1, step - step0):.2f} s/step "
                  f"({(time.time() - t_seg) / 3600:.2f} h this segment)", flush=True)
        if step in marks:
            st = _ranges(model)
            v = eval_loss(model, data["val"], args.curve_batches, args.micro, cfg["block"], device)
            _restore(st)
            state["curve"].append({"step": step, "tokens": step * tokens_per_step, "val_as_trained": v})
            state["wall_s"] += time.time() - t0; t0 = time.time()
            save_ckpt(ck, model, opt, state)
            print(f"[{key}] checkpoint at {100 * step / steps:.0f} % (val {v:.3f})", flush=True)
        if time.time() > deadline and step < steps:
            state["wall_s"] += time.time() - t0
            save_ckpt(ck, model, opt, state)
            print(f"[{key}] segment budget spent at step {step}; checkpointed", flush=True)
            return "paused"
    state["wall_s"] += time.time() - t0
    final = finish(model, sub, sigma, seed, state, args, cfg, data)
    json.dump(final, open(d / "final.json", "w"), indent=1)
    keep = {k: v for k, v in model.state_dict().items() if not any(k.endswith(s) for s in (".gp", ".gm", ".wp", ".wm", ".pending", ".A"))}
    torch.save({"weights": keep, "cfg": cfg, "sub": sub, "sigma": sigma, "seed": seed}, d / "weights.pt")
    if ck.exists():
        ck.unlink()
    print(f"[{key}] done: val clean {final['eval']['val']['clean']:.3f}", flush=True)
    return "done"


def finish(model, sub, sigma, seed, state, args, cfg, data):
    out = {"substrate": sub, "sigma": sigma, "seed": seed, "steps": state["steps"], "tokens": state["steps"] * state["tokens_per_step"],
           "wall_s": state["wall_s"], "segments": state["segments"], "train_loss_last1pct": float(np.mean(state["losses"][-max(1, len(state["losses"]) // 100):])),
           "train_curve": state["losses"][::max(1, len(state["losses"]) // 200)], "val_curve": state["curve"], "eval": {}, "relpert": {}}
    st = _ranges(model)
    settings = {"clean": 0.0, "deployed": sigma, "dial_low": DIAL_LOW}
    for name in ("val", "tinystories", "python"):
        if data.get(name) is None:
            continue
        out["eval"][name] = {}
        for sname, s in settings.items():
            set_read(model, s); out["eval"][name][sname] = eval_loss(model, data[name], args.eval_batches, args.micro, cfg["block"], args.device); _restore(st)
    for sname in settings:
        out["eval"].setdefault("gap_python", {})[sname] = out["eval"]["python"][sname] - out["eval"]["val"][sname] if "python" in out["eval"] else None
        out["eval"].setdefault("gap_tinystories", {})[sname] = out["eval"]["tinystories"][sname] - out["eval"]["val"][sname] if "tinystories" in out["eval"] else None
    vt = torch.from_numpy(np.asarray(data["val"].arr[:2_000_000], dtype=np.int64))
    from relpert import batches_from
    rb = batches_from(vt, args.relpert_batches, 256, min(256, cfg["block"]), args.device)
    for sname, s in settings.items():
        r = measure(model, rb, s, set_noise_unclipped, cumulative=False); _restore(st)
        out["relpert"][sname] = {"sigma": s, "mean_local": r["mean_local"], "mean_local_linear": r["mean_local_linear"], "mean_local_attn": r["mean_local_attn"], "local": r["local"]}
    set_read(model, sigma)
    out["write_budget"] = write_budget_report(tile_modules(model))
    out["pulses_total"] = int(sum(t.n_pulses for t in tile_modules(model) if not t.float_weights))
    if sub == "tiles":
        out["ops_per_token_forward"] = ops_per_token(model, data["val"], cfg, args)
    return out


def ops_per_token(model, data, cfg, args):
    c = OpCounter()
    for t in tile_modules(model):
        t.counter = c
    for b in model.blocks:
        b.counter = c
    x, y = data.windows([0], cfg["block"])
    st = _ranges(model)
    with torch.no_grad():
        model(x.to(args.device), y.to(args.device))
    _restore(st)
    for t in tile_modules(model):
        t.counter = None
    for b in model.blocks:
        b.counter = None
    return {k: v / x.numel() for k, v in c.snapshot().items()}


# ------------------------------------------------------------------ worker (locks + segments)
def claim(d, stale_s=2700):
    lock = d / "lock"
    try:
        lock.mkdir(parents=True)
    except FileExistsError:
        hb = lock / "heartbeat"
        age = time.time() - (hb.stat().st_mtime if hb.exists() else lock.stat().st_mtime)
        if age < stale_s:
            return None
        print(f"  stale lock on {d.name} ({age / 60:.0f} min), taking it over", flush=True)
    (lock / "owner").write_text(f"{os.environ.get('SLURM_JOB_ID', 'local')} {socket.gethostname()} {time.strftime('%F %T')}\n")
    (lock / "heartbeat").touch()
    return lock


def release(lock):
    for f in lock.iterdir():
        f.unlink()
    lock.rmdir()


def run_list(args):
    seeds = range(args.seeds)
    sel = None if args.arms == "all" else {a for a in args.arms.split(",")}
    return [(sub, s, sd) for sd in seeds for sub, s in arms() if sel is None or f"{sub}_s{s:g}" in sel]


def load_data(args):
    root = ROOT / args.data
    data = {"train": Tokens(root / "train.bin", in_ram=getattr(args, "data_in_ram", True)), "val": Tokens(root / "val.bin")}
    for n in ("tinystories", "python"):
        p = root / f"{n}.bin"
        data[n] = Tokens(p) if p.exists() and p.stat().st_size > 4 * 513 else None
    return data


def worker(args, cfg):
    t_start = time.time()
    deadline = t_start + args.segment_minutes * 60 - args.margin_minutes * 60
    data = load_data(args)
    print(f"worker {os.environ.get('SLURM_JOB_ID', 'local')} on {socket.gethostname()}: train {len(data['train']) / 1e6:.0f}M tokens, budget {args.tokens / 1e6:.0f}M per run, "
          f"{args.seeds} seeds, segment {args.segment_minutes} min", flush=True)
    while time.time() < deadline - min(600, 0.2 * args.segment_minutes * 60):
        picked = None
        for sub, s, sd in run_list(args):
            d = ROOT / OUT / "runs" / run_key(sub, s, sd)
            if (d / "final.json").exists():
                continue
            lock = claim(d)
            if lock is None:
                continue
            picked = (sub, s, sd, lock); break
        if picked is None:
            print("no unclaimed work left", flush=True); break
        sub, s, sd, lock = picked
        try:
            r = train_run(sub, s, sd, args, cfg, data, deadline, heartbeat=lambda: (lock / "heartbeat").touch())
        finally:
            release(lock)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        if r == "paused":
            break
    summarize(args)


# ------------------------------------------------------------------ timing
def timing(args, cfg):
    dev = args.device
    cuda = str(dev).startswith("cuda") and torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if cuda else "cpu"
    N = 2_000_000
    root = ROOT / "data" / ("e15_smoke" if args.smoke else "e15_timing"); root.mkdir(parents=True, exist_ok=True)
    p = root / "synthetic.bin"
    if not p.exists():
        np.random.default_rng(0).integers(0, cfg["vocab"], N, dtype=np.uint16).tofile(p)
    data = Tokens(p)
    rows = []
    for sub, sigma in (("float32", 0.1), ("tiles", 0.04)):
        for micro in args.timing_micros:
            try:
                torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
                model = build(sub, sigma, 0, cfg, dev)
                opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95)); topt = TileOptimizer(model.modules(), opt)
                accum = args.batch_seqs // micro
                times = []
                for step in range(args.timing_steps + 1):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.time(); topt.zero_grad()
                    for m in range(accum):
                        x, y = train_batch(data, 0, step, m, micro, cfg["block"], dev)
                        _, loss = model(x, y); (loss / accum).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); topt.step()
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    if step > 0:
                        times.append(time.time() - t0)
                spp = float(np.median(times)); tps = args.batch_seqs * cfg["block"] / spp
                mem = torch.cuda.max_memory_allocated() / 2 ** 30 if cuda else None
                row = {"substrate": sub, "sigma": sigma, "micro": micro, "s_per_step": spp, "tokens_per_s": tps, "peak_mem_GiB": mem,
                       "hours_per_run_at_0.8B": 0.8e9 / tps / 3600, "n_params": sum(p.numel() for p in model.parameters())}
                del model, opt, topt
            except torch.cuda.OutOfMemoryError as e:   # noqa: F841
                row = {"substrate": sub, "sigma": sigma, "micro": micro, "oom": True}
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            rows.append(row); print(row, flush=True)
    best = {}
    for r in rows:
        if not r.get("oom") and (r["substrate"] not in best or r["tokens_per_s"] > best[r["substrate"]]["tokens_per_s"]):
            best[r["substrate"]] = r
    plan = {}
    if len(best) == 2:
        h_f = best["float32"]["hours_per_run_at_0.8B"]; h_t = best["tiles"]["hours_per_run_at_0.8B"]
        for seeds in (3, 2):
            plan[f"{seeds}_seeds_gpu_hours_at_0.8B"] = seeds * (len(F32_SIGMAS) * h_f + len(TILE_SIGMAS) * h_t)
    out = {"gpu": name, "cfg": cfg, "batch_seqs": args.batch_seqs, "rows": rows, "best": best, "plan": plan,
           "note": "9 trainings per seed (the learn-high/deploy-low arm reuses the tiles 0.04 run); plan = GPU-hours on this GPU type"}
    tag = name.replace(" ", "_").replace("/", "_")
    save_results("e15_timing", out, (f"results/hpcc_smoke/e15_timing/{tag}" if args.smoke else f"results/e15_timing/{tag}"))



# ------------------------------------------------------------------ throughput diagnostic on the real path (2026-10-25)
def diag(args, cfg):
    """Seconds per optimiser step on the REAL training data and the real step function, per arm and variant:
    memmap + per-micro sync (as the first segments ran), in-RAM data + per-micro sync, in-RAM + one sync per step."""
    import subprocess
    dev = args.device
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:   # noqa: BLE001
        smi = repr(e)
    print(smi, flush=True)
    root = ROOT / args.data
    rows = []
    variants = (("memmap_sync", False, True), ("ram_sync", True, True), ("ram_nosync", True, False))
    for sub, sigma in (("float32", 0.0), ("float32", 0.4), ("tiles", 0.04)):
        for vname, in_ram, sync in variants:
            train = Tokens(root / "train.bin", in_ram=in_ram)
            model = build(sub, sigma, 0, cfg, dev)
            opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0); topt = TileOptimizer(model.modules(), opt)
            accum = args.batch_seqs // args.micro; times = []
            for step in range(args.diag_steps):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.time()
                train_step(model, opt, topt, train, 7, 10_000 + step + 1000 * len(rows), accum, args.micro, cfg["block"], dev, sync_each=sync)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times.append(time.time() - t0)
            spp = float(np.median(times[args.diag_steps // 4:]))
            row = {"substrate": sub, "sigma": sigma, "variant": vname, "micro": args.micro, "s_per_step": spp,
                   "tokens_per_s": args.batch_seqs * cfg["block"] / spp, "first_steps_s": times[:5]}
            rows.append(row); print(row, flush=True)
            del model, opt, topt, train
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    tag = name.replace(" ", "_").replace("/", "_")
    save_results("e15_diag", {"gpu": name, "host": socket.gethostname(), "nvidia_smi": smi, "rows": rows, "cfg": cfg, "batch_seqs": args.batch_seqs},
                 f"results/e15_diag/{tag}")

# ------------------------------------------------------------------ summary
def summarize(args):
    finals = []
    d = ROOT / OUT / "runs"
    for p in sorted(d.glob("*/final.json")) if d.exists() else []:
        finals.append(json.load(open(p)))
    S = {}
    for f in finals:
        S.setdefault((f["substrate"], f["sigma"]), {})[f["seed"]] = f
    res = {"arms": [f"{a}_s{s:g}" for a, s in arms()], "f32_sigmas": F32_SIGMAS, "tile_sigmas": TILE_SIGMAS, "dial_low": DIAL_LOW, "learn_high": LEARN_HIGH,
           "seeds_requested": args.seeds, "tokens_per_run": args.tokens, "n_finished": len(finals), "summary": {}, "hypotheses": {}}
    for (sub, s), by in sorted(S.items()):
        k = f"{sub}_s{s:g}"
        g = lambda f, a, b: f["eval"].get(a, {}).get(b)
        res["summary"][k] = {"seeds": sorted(by), **{f"{a}_{b}": [g(by[sd], a, b) for sd in sorted(by)] for a in ("val", "tinystories", "python", "gap_python", "gap_tinystories") for b in ("clean", "deployed", "dial_low")},
                             "relpert_deployed": [by[sd]["relpert"]["deployed"]["mean_local"] for sd in sorted(by)],
                             "relpert_clean": [by[sd]["relpert"]["clean"]["mean_local"] for sd in sorted(by)],
                             "writes_per_device_total": [by[sd]["write_budget"].get("writes_per_device_total", 0.0) for sd in sorted(by)],
                             "wall_h": [by[sd]["wall_s"] / 3600 for sd in sorted(by)]}
        res["summary"][k]["val_ppl_clean"] = [math.exp(v) for v in res["summary"][k]["val_clean"] if v is not None]

    def grid(sub, sig, key="val", setting="clean"):
        seeds = sorted(set.intersection(*[set(S.get((sub, s), {})) for s in sig])) if all((sub, s) in S for s in sig) else []
        return seeds, [[S[(sub, s)][sd]["eval"][key][setting] for s in sig] for sd in seeds]
    H = {}
    for sub, sig in (("tiles", TILE_SIGMAS), ("float32", F32_SIGMAS)):
        seeds, C = grid(sub, sig)
        if seeds:
            H[f"{sub}_sigma_star"] = bootstrap_argmin(C, sig); H[f"{sub}_sigma_star"]["n_seeds"] = len(seeds)
            for gk in ("gap_python", "gap_tinystories"):
                if all(S[(sub, s)][sd]["eval"].get(gk, {}).get("clean") is not None for s in sig for sd in seeds):
                    H[f"{sub}_{gk}_slope"] = bootstrap_slope(grid(sub, sig, gk)[1], sig)
            st = H[f"{sub}_sigma_star"]["star"]
            H[f"{sub}_relpert_at_star"] = float(np.mean([S[(sub, st)][sd]["relpert"]["deployed"]["mean_local"] for sd in seeds]))
    v = {}
    t = H.get("tiles_sigma_star"); f0 = S.get(("float32", 0.0), {})
    if t:
        v["H1"] = bool(t["star"] > 0 and t["ci95"][0] > 0)
        v["H2"] = bool(0.02 <= t["star"] <= 0.06)
        st = t["star"]; ts = S[("tiles", st)]
        common = sorted(set(ts) & set(f0))
        if common:
            a = [ts[sd]["eval"]["val"]["clean"] for sd in common]; b = [f0[sd]["eval"]["val"]["clean"] for sd in common]
            v["H3"] = bool(np.mean(a) <= np.mean(b)); H["H3_detail"] = {"tiles_at_star": float(np.mean(a)), "float32_s0": float(np.mean(b)), "seeds_tiles_better": int(sum(x <= y for x, y in zip(a, b))), "n": len(common)}
        lh = S.get(("tiles", LEARN_HIGH), {})
        common = sorted(set(lh) & set(ts))
        if common:
            a = [lh[sd]["eval"]["val"]["dial_low"] for sd in common]; b = [ts[sd]["eval"]["val"]["deployed"] for sd in common]
            v["H4"] = bool(np.mean(a) <= np.mean(b)); H["H4_detail"] = {"learn_high_deploy_low": float(np.mean(a)), "fixed_star": float(np.mean(b)), "seeds_learn_high_better": int(sum(x <= y for x, y in zip(a, b))), "n": len(common)}
    res["hypotheses"] = H; res["verdict"] = {k: v.get(k) for k in ("H1", "H2", "H3", "H4")}
    # energy per training token (E8b constants) from a tiles run
    tr = [f for f in finals if f["substrate"] == "tiles" and f.get("ops_per_token_forward")]
    if tr:
        import e8b_energy_model as e8b
        f = tr[0]; ops = f["ops_per_token_forward"]
        e, parts = e8b.bpu_energy(ops, 1, "layer", 8, fom="fom10")
        fwd = parts["mac_J"] + parts["readout_J"] + parts["route_J"]
        pulses_per_token = {g["substrate"] + f"_s{g['sigma']:g}": g["pulses_total"] / g["tokens"] for g in tr}
        res["energy_per_training_token"] = {k: {"nJ": (3 * fwd + parts["cache_write_J"] + ppt * e8b.E_WRITE) * 1e9, "write_nJ": ppt * e8b.E_WRITE * 1e9} for k, ppt in pulses_per_token.items()}
        res["energy_per_inference_token_nJ"] = e * 1e9
    exp = len(run_list(args)); done = len(finals)
    fmt = lambda b: "SUPPORTED" if b else ("NOT supported" if b is not None else "not yet testable")
    parts = []
    if t:
        parts.append(f"tiles sigma* = {t['star']:g} (CI {t['ci95'][0]:g}..{t['ci95'][1]:g}, {t['n_seeds']} seeds)")
    if H.get("float32_sigma_star"):
        fs = H["float32_sigma_star"]; parts.append(f"float32 sigma* = {fs['star']:g} (CI {fs['ci95'][0]:g}..{fs['ci95'][1]:g})")
    if "H3_detail" in H:
        parts.append(f"val (clean) tiles at sigma* {H['H3_detail']['tiles_at_star']:.3f} vs float32 sigma 0 {H['H3_detail']['float32_s0']:.3f}")
    if "H4_detail" in H:
        parts.append(f"learn high / deploy low {H['H4_detail']['learn_high_deploy_low']:.3f} vs fixed sigma* {H['H4_detail']['fixed_star']:.3f}")
    res["verdict"]["statement"] = (("SMOKE RUN. " if args.smoke else "") + (f"PARTIAL: {done}/{exp} runs finished. " if done < exp else "")
                                   + f"{args.tokens / 1e9:.2f}B tokens per run, {args.seeds} seeds requested. " + "; ".join(parts)
                                   + ". " + "; ".join(f"{k} {fmt(res['verdict'][k])}" for k in ("H1", "H2", "H3", "H4")) + ".")
    print("E15 verdict:", res["verdict"]["statement"], flush=True)
    save_results("e15", res, OUT)
    if finals and not args.no_fig:
        plot(res, figure_path("e15_scratch_lm"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    S = res["summary"]
    for sub, sig, col in (("tiles", TILE_SIGMAS, "C0"), ("float32", F32_SIGMAS, "k")):
        xs = [s for s in sig if f"{sub}_s{s:g}" in S]
        for j, (key, lab) in enumerate((("val_clean", "val, clean"), ("gap_python_clean", "python − val"), ("gap_tinystories_clean", "tinystories − val"))):
            m = [np.mean([v for v in S[f"{sub}_s{s:g}"][key] if v is not None] or [np.nan]) for s in xs]
            ax[j].plot(xs, m, "o-", color=col, label=f"{sub}: {lab}")
    if "tiles_s0.04" in S:
        ax[0].plot([LEARN_HIGH], [np.mean(S["tiles_s0.04"]["val_dial_low"])], "*", ms=12, color="C3", label="tiles: learn 0.04, deploy 0.005")
    for a, t in zip(ax, ("val loss vs training σ", "shift gap: Python", "shift gap: TinyStories")):
        a.set_xlabel("training read noise σ"); a.set_title(t, fontsize=9); a.legend(fontsize=7)
    fig.suptitle(res["verdict"]["statement"][:170], fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="E15 from-scratch LM on tiles")
    ap.add_argument("--worker", action="store_true"); ap.add_argument("--timing", action="store_true"); ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--diag", action="store_true", help="throughput on the real data and step function (2026-10-25)")
    ap.add_argument("--diag-steps", type=int, default=40)
    ap.add_argument("--no-data-in-ram", dest="data_in_ram", action="store_false")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default="data/e15")
    ap.add_argument("--tokens", type=int, default=800_000_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default="all", help="comma list like tiles_s0.04,float32_s0, or 'all'")
    ap.add_argument("--batch-seqs", type=int, default=64)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--eval-batches", type=int, default=64)
    ap.add_argument("--curve-batches", type=int, default=16)
    ap.add_argument("--relpert-batches", type=int, default=16)
    ap.add_argument("--segment-minutes", type=float, default=360)
    ap.add_argument("--margin-minutes", type=float, default=25)
    ap.add_argument("--timing-steps", type=int, default=6)
    ap.add_argument("--timing-micros", type=str, default="8,16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    args.timing_micros = [int(x) for x in str(args.timing_micros).split(",")]
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True   # A100: TF32 for both substrates alike
    cfg = dict(vocab=50257, d=512, layers=8, heads=8, block=512)
    global OUT
    if args.smoke:
        cfg = dict(vocab=50257, d=32, layers=2, heads=2, block=32)
        OUT = "results/hpcc_smoke/e15"; args.data = "data/e15_smoke"
        args.tokens, args.batch_seqs, args.micro, args.seeds = 20 * 4 * 32, 4, 2, 1
        args.eval_batches = args.curve_batches = args.relpert_batches = 2; args.timing_steps = 2; args.timing_micros = [2]
        root = ROOT / args.data; root.mkdir(parents=True, exist_ok=True)
        for n, sz in (("train", 50000), ("val", 20000), ("tinystories", 5000), ("python", 5000)):
            if not (root / f"{n}.bin").exists():
                np.random.default_rng(len(n)).integers(0, 50257, sz, dtype=np.uint16).tofile(root / f"{n}.bin")
    if args.diag:
        diag(args, cfg)
    elif args.timing:
        timing(args, cfg)
    elif args.worker:
        worker(args, cfg)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
