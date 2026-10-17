"""E9: trainability under drift, where an FPGA-style accelerator is frozen.

The E8 architecture (nanoGPT, 6 layers, d = 192) is trained once on tiny
Shakespeare as an ideal float model, then LOADed into noisy tiles
(sigma_eff = 0.04) and exposed to a non-stationary character stream:
Shakespeare for ``--seg`` steps, then a different corpus (this repository's
own Python source, concatenated), then Shakespeare again.  Three
deployments see the identical stream:

(i)   FROZEN    weights loaded once, no updates (a fixed accelerator / FPGA
      bitstream);
(ii)  BPU-LOCAL the on-chip three-factor rule: eligibility e_ij = <pre_i post_j>
      over the batch (Hebbian trace), broadcast modulator
      m = -(loss - running mean loss), surprise-gated sparse pulses
      (|eta m e| >= theta_w), homeostatic scaling of each output unit's
      column toward the population-mean activity; no gradient through the
      tiles;
(iii) BPU-FA    feedback alignment: the backward pass through every tile
      uses a fixed random feedback matrix instead of W^T, updates are
      pulsed through the tiles (the stronger local alternative).

Reported: loss along the stream for the three deployments, recovery time
after each switch (steps until the 100-step mean loss closes 90 % of the
gap between its value at the switch and its best value in the segment),
writes per device per second against the 6e-3 endurance budget (chip time
per step is a stated assumption, ``--step-seconds``), and the success test
"(ii) or (iii) tracks the shift within 300 steps while (i) does not, with
writes within budget".
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import torch

from common import base_parser, save_results, figure_path, Timer, ROOT
from bpusim.torchtile import TileLinear, TileOptimizer, tile_modules
from e8_nanogpt_tiles import GPT, get_batch, train, evaluate, copy_weights, DATA


def load_corpora():
    shakes = pathlib.Path(DATA).read_text(encoding="utf-8")
    files = sorted((ROOT / "bpusim").glob("*.py")) + sorted((ROOT / "experiments").glob("*.py")) + sorted((ROOT / "tests").glob("*.py"))
    code = "\n\n".join(f.read_text(encoding="utf-8") for f in files)
    code = (code * (len(shakes) // max(len(code), 1) + 1))[:len(shakes)]   # same length as Shakespeare
    chars = sorted(set(shakes) | set(code))
    stoi = {c: i for i, c in enumerate(chars)}
    enc = lambda t: torch.tensor([stoi[c] for c in t], dtype=torch.long)
    return enc(shakes), enc(code), len(chars)


class LocalThreeFactor:
    """On-chip local learning for TileLinear layers: Hebbian eligibility x scalar modulator x surprise gate."""

    def __init__(self, model, eta=1e-3, theta_w=0.0, tau_L=50, tau_H=200.0, r_star_mode="population"):
        self.tiles = tile_modules(model)
        self.eta, self.theta_w, self.tau_L, self.tau_H = eta, theta_w, tau_L, tau_H
        self.L_bar = None; self.L_dev = None
        self.cache = {}
        self.r_bar = {id(t): None for t in self.tiles}
        for t in self.tiles:
            t.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        x = inp[0].detach().reshape(-1, inp[0].shape[-1]); y = out.detach().reshape(-1, out.shape[-1])
        self.cache[id(module)] = (x, y)

    @torch.no_grad()
    def step(self, loss: float):
        if self.L_bar is None:
            self.L_bar, self.L_dev = loss, abs(loss) + 1e-3
        m = float(np.clip(-(loss - self.L_bar) / (self.L_dev + 1e-6), -3, 3))
        self.L_bar += (loss - self.L_bar) / self.tau_L
        self.L_dev += (abs(loss - self.L_bar) - self.L_dev) / self.tau_L
        writes = 0
        for t in self.tiles:
            x, y = self.cache[id(t)]
            e = x.T @ y / x.shape[0]                       # eligibility <pre_i post_j>
            dW = self.eta * m * e
            # homeostatic scaling of each output column toward the population-mean activity
            r = y.abs().mean(0)
            rb = self.r_bar[id(t)]
            self.r_bar[id(t)] = r.clone() if rb is None else rb + (r - rb) / self.tau_H
            r_star = self.r_bar[id(t)].mean()
            dW = dW + (1.0 / self.tau_H) * t.W * ((r_star - self.r_bar[id(t)]) / (r_star + 1e-9))[None, :]
            t.apply_delta(dW, theta_w=self.theta_w)
            writes += t.n_writes
        return m


def recovery_time(curve, switch_idx, window=100, frac=0.9):
    """Steps after switch_idx until the moving average closes `frac` of the gap to its segment minimum."""
    seg = np.asarray(curve[switch_idx:], float)
    if len(seg) < window + 2:
        return None
    ma = np.convolve(seg, np.ones(window) / window, mode="valid")
    start, best = ma[0], ma.min()
    if start - best <= 1e-6:
        return 0
    target = start - frac * (start - best)
    hit = np.nonzero(ma <= target)[0]
    return int(hit[0]) if hit.size else None


def run_stream(model, segments, block, batch, gen, mode, opt=None, local=None, log_every=200):
    losses, writes = [], []
    step = 0; t0 = time.time()
    for seg_i, data in enumerate(segments):
        for _ in range(len(data)):
            pass
        n_steps = data["steps"]; ids = data["ids"]
        for _ in range(n_steps):
            x, y = get_batch(ids, block, batch, gen)
            if mode == "frozen":
                with torch.no_grad():
                    _, loss = model(x, y)
            elif mode == "local":
                with torch.no_grad():
                    _, loss = model(x, y)
                local.step(loss.item())
            else:  # feedback alignment
                _, loss = model(x, y)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            losses.append(loss.item())
            writes.append(sum(t.n_writes for t in tile_modules(model)))
            step += 1
            if step % log_every == 0:
                print(f"    {mode} step {step} seg {seg_i} loss {np.mean(losses[-log_every:]):.3f} ({time.time() - t0:.0f}s)", flush=True)
    return losses, writes


def main():
    ap = base_parser("E9 trainability under drift", default_seeds=1)
    ap.add_argument("--seg", type=int, default=2000, help="steps per stream segment")
    ap.add_argument("--pretrain-steps", type=int, default=1000)
    ap.add_argument("--sigma", type=float, default=0.04)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--lr-fa", type=float, default=3e-4)
    ap.add_argument("--eta-local", type=float, default=1e-3)
    ap.add_argument("--theta-w", type=float, default=0.0)
    ap.add_argument("--step-seconds", type=float, default=0.01, help="assumed chip time per training step for the write-rate conversion")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.smoke:
        args.seg, args.pretrain_steps = 30, 30
    shakes, code, vocab = load_corpora()
    n = int(0.9 * len(shakes)); shakes_tr = shakes[:n]
    segments = [{"name": "shakespeare", "ids": shakes_tr, "steps": args.seg}, {"name": "python_source", "ids": code, "steps": args.seg},
                {"name": "shakespeare", "ids": shakes_tr, "steps": args.seg}]
    results = {}
    seed = 0
    torch.manual_seed(seed); gen = torch.Generator().manual_seed(seed)
    ideal = GPT(vocab, 192, 6, 6, args.block, None, None, seed=seed)
    with Timer("pretrain ideal on Shakespeare"):
        train(ideal, shakes_tr, args.pretrain_steps, args.block, 24, 1e-3, gen, tiled=False, label="pretrain")
    base_loss = {"shakespeare": evaluate(ideal, shakes[n:], args.block, args.batch, 20, torch.Generator().manual_seed(5)),
                 "python_source": evaluate(ideal, code, args.block, args.batch, 20, torch.Generator().manual_seed(5))}
    print(f"ideal val loss: {base_loss}")
    curves = {}
    for mode in ("frozen", "local", "fa"):
        model = GPT(vocab, 192, 6, 6, args.block, args.sigma, None, fa=(mode == "fa"), seed=seed)
        model.load_state_dict(ideal.state_dict(), strict=False); copy_weights(ideal, model, verify=True)
        opt = local = None
        if mode == "fa":
            opt = TileOptimizer(model.modules(), torch.optim.AdamW(model.parameters(), lr=args.lr_fa, betas=(0.9, 0.95), weight_decay=0.0), theta_w=args.theta_w)
        elif mode == "local":
            local = LocalThreeFactor(model, eta=args.eta_local, theta_w=args.theta_w)
        gen_s = torch.Generator().manual_seed(100)
        with Timer(f"stream: {mode}"):
            losses, writes = run_stream(model, segments, args.block, args.batch, gen_s, mode, opt, local)
        n_dev = sum(t.n_devices for t in tile_modules(model))
        wpd = writes[-1] / n_dev; total_s = len(losses) * args.step_seconds
        rec = [recovery_time(losses, k * args.seg) for k in (1, 2)]
        seg_means = [float(np.mean(losses[k * args.seg:(k + 1) * args.seg][-200:])) for k in range(3)]
        curves[mode] = {"loss": losses[::max(1, len(losses) // 600)], "loss_full_len": len(losses), "writes_cum": writes[::max(1, len(writes) // 600)],
                        "writes_per_device_total": wpd, "writes_per_device_per_s": wpd / total_s if total_s else None,
                        "recovery_steps_after_switch": rec, "segment_end_loss": seg_means}
        print(f"  {mode}: segment-end losses {np.round(seg_means, 3)}, recovery {rec}, writes/device/s {wpd / total_s:.3g}")
    frozen_ok = curves["frozen"]["recovery_steps_after_switch"]
    tracks = {m: all(r is not None and r <= 300 for r in curves[m]["recovery_steps_after_switch"]) and
              curves[m]["segment_end_loss"][1] < curves["frozen"]["segment_end_loss"][1] - 0.05 for m in ("local", "fa")}
    within_budget = {m: (curves[m]["writes_per_device_per_s"] or 0) <= 6e-3 for m in ("local", "fa")}
    success = any(tracks[m] and within_budget[m] for m in tracks)
    out = {"seg": args.seg, "sigma": args.sigma, "pretrain_steps": args.pretrain_steps, "batch": args.batch, "step_seconds": args.step_seconds,
           "ideal_val_loss": base_loss, "segments": [s["name"] for s in segments], "curves": curves, "tracks_within_300": tracks,
           "writes_within_budget": within_budget, "smoke": args.smoke,
           "verdict": {"success": success, "statement": ("SMOKE RUN. " if args.smoke else "") +
                       f"frozen: python-segment loss {curves['frozen']['segment_end_loss'][1]:.3f} (no recovery by construction); "
                       f"local three-factor: {curves['local']['segment_end_loss'][1]:.3f}, recovery {curves['local']['recovery_steps_after_switch']}, "
                       f"{curves['local']['writes_per_device_per_s']:.3g} writes/dev/s; "
                       f"feedback alignment: {curves['fa']['segment_end_loss'][1]:.3f}, recovery {curves['fa']['recovery_steps_after_switch']}, "
                       f"{curves['fa']['writes_per_device_per_s']:.3g} writes/dev/s; budget 6e-3 at {args.step_seconds} s/step; "
                       + ("SUCCESS: a local rule tracks the shift within 300 steps inside the write budget" if success else
                          "criterion not met: " + ", ".join(f"{m} tracks={tracks[m]} budget={within_budget[m]}" for m in tracks))}}
    print("E9 verdict:", out["verdict"]["statement"])
    save_results("e9", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e9_trainability"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    seg = res["seg"]
    for mode, lab in (("frozen", "frozen (FPGA-style)"), ("local", "BPU local three-factor"), ("fa", "BPU feedback alignment")):
        c = res["curves"][mode]; L = np.array(c["loss"]); x = np.linspace(0, c["loss_full_len"], len(L))
        k = max(1, len(L) // 60); sm = np.convolve(L, np.ones(k) / k, mode="valid")
        ax[0].plot(x[:len(sm)], sm, label=lab)
    for k in (1, 2):
        ax[0].axvline(k * seg, color="k", ls=":", lw=0.8)
    ax[0].set_xlabel("stream step"); ax[0].set_ylabel("loss (nats/char)"); ax[0].legend(fontsize=8)
    ax[0].set_title("E9: Shakespeare -> Python source -> Shakespeare", fontsize=10)
    for mode in ("local", "fa"):
        c = res["curves"][mode]; w = np.array(c["writes_cum"], float); x = np.linspace(0, c["loss_full_len"], len(w))
        ax[1].plot(x, w / max(w[-1], 1), label=f"{mode}: {c['writes_per_device_per_s']:.2g} writes/dev/s")
    ax[1].set_xlabel("stream step"); ax[1].set_ylabel("cumulative writes (normalised)"); ax[1].legend(fontsize=8)
    ax[1].set_title(f"write budget 6e-3 /dev/s at {res['step_seconds']} s/step", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e9] figure {path}")


if __name__ == "__main__":
    main()
