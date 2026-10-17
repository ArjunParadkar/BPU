"""E2: the local grow rule versus RigL versus random regrowth on sparse MNIST MLPs.

A 784-256-10 MLP is trained at 10% density.  Every ``--update-every`` steps
a fraction of the active synapses with the smallest |w| is dropped (identical
for all methods) and the same number is regrown by one of three criteria:

* ``rigl``   : largest |dL/dw| among inactive synapses (dense gradient, Evci et al. 2020),
* ``local``  : largest |C_ij| = |< m x_i xi_j >| accumulated since the last
               update, where xi_j is the noise injected into the layer's
               pre-activation and m = -(L - L_bar) is the broadcast
               modulator (node perturbation; no backward pass through the
               absent synapse, no dense gradient),
* ``random`` : uniform among inactive synapses.

Weights are trained with SGD in all three cases so the comparison isolates
the *regrowth criterion*.  Equal synapse budget throughout.  PyTorch; CPU by
default, ``--device cuda`` if available.  MNIST is downloaded as raw IDX
files (no torchvision dependency).
"""
from __future__ import annotations

import argparse
import gzip
import os
import pathlib
import time
import urllib.request

import numpy as np

from common import save_results, figure_path, Timer, ROOT

MNIST_URLS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}


def load_mnist(data_dir=None):
    d = pathlib.Path(data_dir or ROOT / "data" / "mnist"); d.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, fn in FILES.items():
        path = d / fn
        if not path.exists():
            for base in MNIST_URLS:
                try:
                    urllib.request.urlretrieve(base + fn, path); break
                except Exception as e:  # pragma: no cover
                    print(f"  download failed from {base}: {e}")
            if not path.exists():
                raise RuntimeError("could not download MNIST")
        with gzip.open(path, "rb") as f:
            raw = f.read()
        if "x" in key:
            out[key] = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
        else:
            out[key] = np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)
    return out


def synthetic_mnist(rng, n_train=6000, n_test=1000):
    """Stand-in data for --smoke runs without network access: 10 Gaussian class prototypes."""
    protos = rng.standard_normal((10, 784)) * 0.5
    def make(n):
        y = rng.integers(0, 10, n); x = protos[y] + rng.standard_normal((n, 784)) * 0.8
        return (x - x.min()) / (x.max() - x.min()), y
    xtr, ytr = make(n_train); xte, yte = make(n_test)
    return {"train_x": xtr.astype(np.float32), "train_y": ytr, "test_x": xte.astype(np.float32), "test_y": yte}


def run_method(method, data, seed, device, epochs, density, update_every, drop_frac, batch, lr, noise_sigma):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    dims = [(784, 256), (256, 10)]
    Ws, masks, Cs = [], [], []
    for (i, o) in dims:
        W = (torch.randn(i, o) * (2.0 / i) ** 0.5).to(device)
        m = (torch.rand(i, o) < density).to(device)
        W = W * m
        Ws.append(W.requires_grad_(True)); masks.append(m); Cs.append(torch.zeros(i, o, device=device))
    xtr = torch.tensor(data["train_x"], device=device); ytr = torch.tensor(data["train_y"], device=device)
    xte = torch.tensor(data["test_x"], device=device); yte = torch.tensor(data["test_y"], device=device)
    n = len(xtr); steps_per_epoch = n // batch
    L_bar = None; step = 0; hist = []
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for b in range(steps_per_epoch):
            idx = perm[b * batch:(b + 1) * batch]
            x, y = xtr[idx], ytr[idx]
            # forward with node-perturbation noise on each layer's pre-activation (used by 'local')
            h_in = x
            acts, xis = [], []
            for li, (W, m) in enumerate(zip(Ws, masks)):
                acts.append(h_in)
                u = h_in @ (W * m)
                xi = noise_sigma * torch.randn_like(u) if method == "local" else torch.zeros_like(u)
                xis.append(xi)
                u = u + xi
                h_in = F.relu(u) if li < len(Ws) - 1 else u
            logits = h_in
            losses = F.cross_entropy(logits, y, reduction="none")
            loss = losses.mean()
            grads = torch.autograd.grad(loss, Ws)
            with torch.no_grad():
                for W, g, m in zip(Ws, grads, masks):
                    W -= lr * g * m
                if method == "local":
                    Lb = losses.detach() if L_bar is None else L_bar
                    mod = -(losses.detach() - Lb.mean())
                    L_bar = losses.detach() if L_bar is None else 0.9 * L_bar + 0.1 * losses.detach()
                    for li in range(len(Ws)):
                        Cs[li] += (acts[li] * mod[:, None]).T @ xis[li] / batch
            step += 1
            if step % update_every == 0 and ep < epochs - 1:   # no topology change in the last epoch (RigL cool-down)
                with torch.no_grad():
                    for li, (W, m) in enumerate(zip(Ws, masks)):
                        n_active = int(m.sum()); k = int(drop_frac * n_active)
                        if k == 0:
                            continue
                        # drop: smallest |w| among active
                        score = torch.where(m, W.abs(), torch.full_like(W, float("inf")))
                        drop_idx = torch.topk(score.flatten(), k, largest=False).indices
                        m.view(-1)[drop_idx] = False
                        W.view(-1)[drop_idx] = 0.0
                        # grow
                        inactive = ~m
                        inactive.view(-1)[drop_idx] = False   # do not regrow what was just dropped
                        if method == "rigl":
                            xb, yb = xtr[perm[:1024]], ytr[perm[:1024]]
                            # dense gradient of the loss w.r.t. all weights of this layer
                            h = xb
                            for lj, (Wj, mj) in enumerate(zip(Ws, masks)):
                                if lj == li:
                                    Wd = (Wj * mj).detach().requires_grad_(True); h = h @ Wd
                                else:
                                    h = h @ (Wj * mj)
                                h = F.relu(h) if lj < len(Ws) - 1 else h
                            with torch.enable_grad():
                                lg = F.cross_entropy(h, yb)
                                gd = torch.autograd.grad(lg, Wd)[0]
                            gscore = torch.where(inactive, gd.abs(), torch.full_like(W, -1.0))
                        elif method == "local":
                            gscore = torch.where(inactive, Cs[li].abs(), torch.full_like(W, -1.0))
                        else:
                            gscore = torch.where(inactive, torch.rand_like(W), torch.full_like(W, -1.0))
                        grow_idx = torch.topk(gscore.flatten(), k, largest=True).indices
                        m.view(-1)[grow_idx] = True
                        W.view(-1)[grow_idx] = 0.0   # new synapses start at zero (RigL convention)
                        Cs[li].zero_()
        with torch.no_grad():
            h = xte
            for lj, (Wj, mj) in enumerate(zip(Ws, masks)):
                h = h @ (Wj * mj); h = F.relu(h) if lj < len(Ws) - 1 else h
            acc = float((h.argmax(1) == yte).float().mean())
        hist.append(acc)
        print(f"    {method} seed {seed} epoch {ep + 1}/{epochs}: test acc {acc:.4f} density {[round(float(m.float().mean()), 3) for m in masks]} ({time.time() - t0:.0f}s)")
    return {"test_acc": hist, "final_acc": hist[-1], "n_active": int(sum(int(m.sum()) for m in masks))}


def main():
    ap = argparse.ArgumentParser(description="E2 grow rule vs RigL vs random")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--density", type=float, default=0.1)
    ap.add_argument("--update-every", type=int, default=100)
    ap.add_argument("--drop-frac", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true", help="synthetic data, 1 epoch, 1 seed")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()
    seeds = 10 if args.full else args.seeds
    epochs = 8 if args.full else args.epochs
    if args.smoke:
        data = synthetic_mnist(np.random.default_rng(0)); seeds = 1; epochs = 1
    else:
        with Timer("MNIST"):
            data = load_mnist()
    methods = ["local", "rigl", "random"]
    runs = {m: [] for m in methods}
    with Timer("E2"):
        for seed in range(seeds):
            for m in methods:
                runs[m].append(run_method(m, data, seed, args.device, epochs, args.density, args.update_every,
                                          args.drop_frac, args.batch, args.lr, args.noise))
    from bpusim.noise import summarize
    summary = {m: summarize([r["final_acc"] for r in runs[m]]) for m in methods}
    out = {"seeds": seeds, "epochs": epochs, "density": args.density, "update_every": args.update_every,
           "drop_frac": args.drop_frac, "noise_sigma": args.noise, "smoke": args.smoke, "runs": runs, "summary": summary}
    gap = summary["local"]["mean"] - summary["rigl"]["mean"]
    out["verdict"] = {"local_minus_rigl": gap, "local_minus_random": summary["local"]["mean"] - summary["random"]["mean"],
                      "statement": "local grow rule %s RigL (%.4f vs %.4f test accuracy, random %.4f) at %.0f%% density" % (
                          "matches" if abs(gap) <= 2 * max(summary["local"]["sem"], summary["rigl"]["sem"], 1e-4) else ("beats" if gap > 0 else "trails"),
                          summary["local"]["mean"], summary["rigl"]["mean"], summary["random"]["mean"], 100 * args.density)}
    print("E2 verdict:", out["verdict"]["statement"])
    save_results("e2", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e2_grow_vs_rigl"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m, runs in res["runs"].items():
        A = np.array([r["test_acc"] for r in runs])
        ep = np.arange(1, A.shape[1] + 1)
        ax.errorbar(ep, A.mean(0), A.std(0) / np.sqrt(len(A)), marker="o", capsize=3, label=m)
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy"); ax.set_title(f"E2: regrowth criterion at {100 * res['density']:.0f}% density", fontsize=10)
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e2] figure {path}")


if __name__ == "__main__":
    main()
