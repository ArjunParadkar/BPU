"""E8b: energy and latency projection from E8's measured op counts.

BPU side (L7 model, projection from the device model, constants stated):
  E_MAC   = V^2 G t        with V = 0.2 V, G = 10 uS, t = 10 ns  -> 4 fJ per MAC
  E_ADC   = 2^b * 1 pJ     at b = 8 bits, one conversion per tile output per read (tile boundaries only)
  E_write = 100 pJ         per device pulse
  E_route = 10 pJ          per routed event (a vector crossing a tile / layer boundary)
  latency = reads in series * 1 us  (one tile read per layer boundary, pipelined across tokens)

Baseline (literature constants + measured wall time on this laptop):
  3.7 pJ per FP32 multiply (Horowitz, ISSCC 2014) for every MAC, plus
  640 pJ per 32-bit DRAM weight fetch, counted once per weight per forward
  pass with no cache reuse, and an optimistic case with 10x cache reuse.
  Wall time per forward batch and per training step as measured by E8 on
  this CPU (ideal float model).

Scaling: MACs and weight fetches scale with parameter count; the table and
plot cover 10M (measured), 124M, 1.3B, 7B and 70B parameters.  BPU numbers
are projections; nothing on the BPU side is measured on hardware.
"""
from __future__ import annotations

import numpy as np

from common import base_parser, save_results, figure_path, load_results

V, G, T_READ = 0.2, 10e-6, 10e-9
E_MAC = V ** 2 * G * T_READ            # 4e-15 J
E_ADC = (2 ** 8) * 1e-12               # 256 pJ
E_WRITE = 100e-12
E_ROUTE = 10e-12
T_READ_SERIES = 1e-6
E_FP32_MAC = 3.7e-12
E_DRAM_FETCH = 640e-12
SIZES = {"10M (E8, measured ops)": 10e6, "124M (GPT-2 small)": 124e6, "1.3B": 1.3e9, "7B": 7e9, "70B": 70e9}


def bpu_energy(ops, tokens):
    e = ops["macs"] * E_MAC + ops["conversions"] * E_ADC + ops["events"] * E_ROUTE + ops["pulses"] * E_WRITE + ops.get("cache_writes", 0) * E_WRITE * 0.1
    parts = {"mac_J": ops["macs"] * E_MAC, "adc_J": ops["conversions"] * E_ADC, "route_J": ops["events"] * E_ROUTE,
             "write_J": ops["pulses"] * E_WRITE, "cache_write_J": ops.get("cache_writes", 0) * E_WRITE * 0.1}
    return e / tokens, {k: v / tokens for k, v in parts.items()}


def baseline_energy(macs, n_weights, tokens, reuse=1.0):
    return (macs * E_FP32_MAC + n_weights * E_DRAM_FETCH / reuse) / tokens


def main():
    ap = base_parser("E8b energy and latency projection", default_seeds=1)
    args = ap.parse_args()
    e8 = load_results("e8")
    ops_eval = e8["ops"]["eval_batch"]; ops_train = e8["ops"]["train_step"]; tokens = e8["ops"]["tokens"]
    n_params = e8["ops"]["n_params"]; n_tile_params = e8["ops"]["n_devices"] // 2
    timing = e8.get("timing", {})
    # per token at the measured size
    bpu_eval, parts_eval = bpu_energy(ops_eval, tokens)
    train_ops = dict(ops_train); train_ops["macs"] = ops_train["macs"] + ops_train.get("macs_backward_estimate", 0)
    bpu_train, parts_train = bpu_energy(train_ops, tokens)
    # the weights fetched per forward pass: every tile weight once (no reuse) -- digital baseline
    base_eval = baseline_energy(ops_eval["macs"], n_tile_params, tokens, reuse=1.0)
    base_eval_opt = baseline_energy(ops_eval["macs"], n_tile_params, tokens, reuse=10.0)
    base_train = baseline_energy(3 * ops_eval["macs"], 3 * n_tile_params, tokens, reuse=1.0)
    base_train_opt = baseline_energy(3 * ops_eval["macs"], 3 * n_tile_params, tokens, reuse=10.0)
    # latency: reads in series per token = tile reads per token along one path
    layers = e8["layers"]; reads_in_series = layers * 4 + 1 + layers * 2   # qkv, proj, fc, fc2 per layer + head + 2 attention reads
    latency_token = reads_in_series * T_READ_SERIES
    # measured wall time per token on this CPU
    cpu_fwd_token = timing.get("forward_batch_s", float("nan")) / max(timing.get("tokens_per_batch", 1), 1)
    cpu_train_token = timing.get("train_step_s", float("nan")) / max(timing.get("tokens_per_batch", 1), 1)
    # scaling table: MACs per token ~ 2 * params (+ attention, scaled with the measured ratio)
    macs_per_token_meas = ops_eval["macs"] / tokens
    ratio_macs = macs_per_token_meas / (2 * n_tile_params)
    conv_per_mac = ops_eval["conversions"] / ops_eval["macs"]; ev_per_mac = ops_eval["events"] / ops_eval["macs"]
    table = []
    for name, P in SIZES.items():
        macs = 2 * P * ratio_macs
        ops = {"macs": macs, "conversions": macs * conv_per_mac, "events": macs * ev_per_mac, "pulses": 0, "cache_writes": 0}
        e_bpu, _ = bpu_energy(ops, 1)
        table.append({"model": name, "params": P, "macs_per_token": macs, "bpu_J_per_token": e_bpu,
                      "digital_J_per_token_no_reuse": baseline_energy(macs, P, 1, 1.0), "digital_J_per_token_10x_reuse": baseline_energy(macs, P, 1, 10.0),
                      "ratio_no_reuse": baseline_energy(macs, P, 1, 1.0) / e_bpu, "ratio_10x_reuse": baseline_energy(macs, P, 1, 10.0) / e_bpu})
    out = {"constants": {"E_MAC_J": E_MAC, "E_ADC_J": E_ADC, "E_WRITE_J": E_WRITE, "E_ROUTE_J": E_ROUTE, "t_read_series_s": T_READ_SERIES,
                         "E_FP32_MAC_J": E_FP32_MAC, "E_DRAM_FETCH_J": E_DRAM_FETCH},
           "measured_ops": e8["ops"], "cpu_timing": timing,
           "per_token_10M": {"bpu_eval_J": bpu_eval, "bpu_eval_parts_J": parts_eval, "bpu_train_step_J": bpu_train, "bpu_train_parts_J": parts_train,
                             "digital_eval_J_no_reuse": base_eval, "digital_eval_J_10x_reuse": base_eval_opt,
                             "digital_train_J_no_reuse": base_train, "digital_train_J_10x_reuse": base_train_opt,
                             "bpu_latency_s_per_token": latency_token, "reads_in_series": reads_in_series,
                             "cpu_forward_s_per_token": cpu_fwd_token, "cpu_train_s_per_token": cpu_train_token},
           "scaling": table,
           "verdict": {"statement": (f"PROJECTION (device model constants, not hardware): at 10M params the BPU model gives {bpu_eval * 1e9:.2f} nJ/token for inference "
                                     f"vs {base_eval * 1e9:.1f} nJ/token digital with no weight reuse ({base_eval / bpu_eval:.0f}x) and {base_eval_opt * 1e9:.1f} nJ/token with 10x reuse "
                                     f"({base_eval_opt / bpu_eval:.0f}x); A/D conversion is {100 * parts_eval['adc_J'] / bpu_eval:.0f}% of the BPU budget; "
                                     f"projected latency {latency_token * 1e6:.0f} us/token vs {cpu_fwd_token * 1e6:.0f} us/token measured on this CPU; "
                                     f"training step: BPU {bpu_train * 1e9:.1f} nJ/token (writes {100 * parts_train['write_J'] / bpu_train:.0f}%) vs digital {base_train * 1e9:.0f} nJ/token.")}}
    print("E8b:", out["verdict"]["statement"])
    save_results("e8b", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e8b_energy"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = res["scaling"]; P = [r["params"] for r in t]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].loglog(P, [r["bpu_J_per_token"] for r in t], "o-", label="BPU (projection)")
    ax[0].loglog(P, [r["digital_J_per_token_no_reuse"] for r in t], "s-", label="digital, no weight reuse")
    ax[0].loglog(P, [r["digital_J_per_token_10x_reuse"] for r in t], "^--", label="digital, 10x cache reuse")
    ax[0].set_xlabel("parameters"); ax[0].set_ylabel("energy per token (J), inference"); ax[0].legend(fontsize=8); ax[0].set_title("E8b: energy per token vs model size", fontsize=10)
    p = res["per_token_10M"]["bpu_eval_parts_J"]
    ax[1].bar(list(p.keys()), [v * 1e9 for v in p.values()]); ax[1].set_ylabel("nJ per token (10M model)"); ax[1].set_title("BPU inference energy breakdown (projection)", fontsize=10)
    ax[1].tick_params(axis="x", labelsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e8b] figure {path}")


if __name__ == "__main__":
    main()
