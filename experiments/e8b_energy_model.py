"""E8b: energy per token -- measured digital control vs the BPU read-out modes.

DIGITAL CONTROL (measured, not Horowitz constants): the same nanoGPT
(6 layers, d = 192, E8 architecture) runs on the GPU in this laptop with
every Linear at INT8 (W8A8: per-channel int8 weights, per-token dynamic int8
activations, int32 accumulation through ``torch._int_mm``; attention,
layer norms, GELU and softmax stay FP32 as on any INT8 accelerator).  Energy
= GPU board power (nvidia-smi, sampled at 10 Hz during a steady-state
inference loop) x wall time / tokens, reported both with the idle power
included and with it subtracted.  FP32 on the same GPU is measured the same
way.  The CPU INT8 path (``torch.ao.quantization.quantize_dynamic``) is timed
but its energy cannot be measured here (RAPL counters are root-only), so it
is reported as time only.  A published H100 INT8 figure is included and
labelled as published peak throughput, not a measurement.

BPU (projection from the device model; nothing is measured on hardware):
  E_MAC   = V^2 G t = 4 fJ   (0.2 V, 10 uS, 10 ns)
  read-out modes x converter precision, the 3 x 2 table:
    per-tile ADC      one conversion per 32-row tile partial sum
    per-layer ADC     analog accumulation across the tiles of a layer, one
                      conversion per layer output
    LIF read-out      no ADC: a LIF neuron per layer output, spike count
                      (<= 2^b - 1 per token) encodes the value; 10 pJ per
                      spike event (router) + 5 pJ per LIF update (one per
                      spike: the integration itself is analog)
    8-bit: 256 pJ per conversion, 4-bit: 16 pJ per conversion
  periphery: router 10 pJ per routed event, write driver 100 pJ per pulse,
  volatile cache writes 10 pJ.
The accuracy of every mode is measured in simulation: the E8 noise-aware
model at sigma = 0.04 (ideal pre-training 1500 steps, 300 noise-aware
fine-tuning steps, the E8 protocol) is evaluated with the read-out emulated
(converter ranges calibrated on one batch, values clip at the rails) and the
validation-loss gap to the ideal float model is reported with the measured
spike rate.  Scaling to 1T parameters: MACs proportional to parameters,
per-layer conversions proportional to layers x d_model, per-tile conversions
proportional to layers x d_model^2 / 32 (one conversion per 32-row block).
"""
from __future__ import annotations

import math
import os
import subprocess
import threading
import time

import numpy as np
import torch
from torch import nn

from common import base_parser, save_results, figure_path, load_results, Timer
from bpusim.torchtile import TileLinear, OpCounter, ReadoutSpec, tile_modules
from e8_nanogpt_tiles import GPT, load_text, get_batch, train, evaluate, copy_weights, count_ops, set_readout, model_device

V, G, T_READ = 0.2, 10e-6, 10e-9
E_MAC = V ** 2 * G * T_READ            # 4e-15 J
FOM_W = {"fom10": 10e-15, "fom50": 50e-15}   # Walden figure of merit, J per conversion-step (Murmann ADC survey: 10 fJ = state of the art, 50 fJ = conservative)
BITS = [6, 8]
E_ADC = {(f, b): FOM_W[f] * 2 ** b for f in FOM_W for b in BITS}   # E_ADC = FoM_W x 2^b
E_SPIKE = 10e-12                       # router event per spike
E_LIF_UPDATE = 5e-12                   # per LIF update (one per spike; the integration is analog)
E_WRITE = 100e-12
E_ROUTE = 10e-12
E_CACHE_WRITE = 10e-12
T_READ_SERIES = 1e-6
MODES = ["tile", "layer", "lif"]
# (layers, d_model) of standard dense configurations; MACs per token ~ 12 L d^2 (+ attention, scaled as measured)
SIZES = {"2.7M (E8, measured)": (6, 192), "124M (GPT-2 small)": (12, 768), "1.3B": (24, 2048), "7B": (32, 4096), "70B": (80, 8192), "1T (dense, 140 x 24576)": (140, 24576)}
H100_INT8_TOPS, H100_W = 1979.0, 700.0   # NVIDIA H100 SXM datasheet, dense INT8 peak; published, not measured
H100_UTIL = 0.40                         # a realistic sustained utilisation for the "H100 at 40 %" control row


# ------------------------------------------------------------------ INT8 digital control on the GPU
class Int8Linear(nn.Module):
    """W8A8 linear: int8 per-output-channel weights, per-token dynamic int8 activations, int32 accumulate."""

    def __init__(self, W: torch.Tensor, b: torch.Tensor | None):
        super().__init__()
        self.out_features = W.shape[1]
        pad_out = (-W.shape[1]) % 8                                        # _int_mm wants an output width that is a multiple of 8
        W = torch.nn.functional.pad(W, (0, pad_out))
        s_w = W.abs().amax(0).clamp_min(1e-8) / 127.0                     # (out,)
        self.register_buffer("Wq", torch.round(W / s_w).clamp(-128, 127).to(torch.int8).contiguous())
        self.register_buffer("s_w", s_w.float())
        self.register_buffer("b", b.detach().float() if b is not None else None)

    def forward(self, x):
        shp = x.shape; x2 = x.reshape(-1, shp[-1])
        s_x = x2.abs().amax(1, keepdim=True).clamp_min(1e-8) / 127.0
        xq = torch.round(x2 / s_x).clamp(-128, 127).to(torch.int8)
        n = xq.shape[0]; pad = (-n) % 16                                    # _int_mm wants multiples of 16 rows
        if pad:
            xq = torch.nn.functional.pad(xq, (0, 0, 0, pad))
        y = (torch._int_mm(xq, self.Wq)[:n].float() * s_x * self.s_w)[:, :self.out_features]
        if self.b is not None:
            y = y + self.b
        return y.reshape(*shp[:-1], -1)


def to_int8(model: GPT) -> GPT:
    """Replace every TileLinear of an ideal model by an Int8Linear (same weights)."""
    for name, m in list(model.named_modules()):
        for cname, c in list(m.named_children()):
            if isinstance(c, TileLinear):
                setattr(m, cname, Int8Linear(c.W.detach(), c.b))
    return model


class PowerSampler:
    """Samples nvidia-smi board power at ~10 Hz on a thread."""

    def __init__(self, interval=0.1):
        self.interval = interval; self.samples = []; self._stop = threading.Event(); self.t = None

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"], text=True, timeout=2)
                self.samples.append(float(out.strip().split("\n")[0]))
            except Exception:
                pass
            time.sleep(self.interval)

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True); self.t.start(); return self

    def __exit__(self, *a):
        self._stop.set(); self.t.join()

    @property
    def mean(self):
        return float(np.mean(self.samples)) if self.samples else float("nan")


def measure_gpu(model, data, block, batch, seconds, label):
    """Steady-state inference loop; returns J/token (board power incl. idle, and above idle), tokens/s."""
    dev = model_device(model); gen = torch.Generator().manual_seed(3)
    xs = [get_batch(data, block, batch, gen, dev)[0] for _ in range(8)]
    with torch.no_grad():
        for x in xs[:4]:
            model(x)
    torch.cuda.synchronize()
    with PowerSampler() as ps:
        t0 = time.time(); n_tok = 0; i = 0
        with torch.no_grad():
            while time.time() - t0 < seconds:
                model(xs[i % len(xs)]); n_tok += xs[i % len(xs)].numel(); i += 1
            torch.cuda.synchronize()
        dt = time.time() - t0
    print(f"  {label}: {n_tok / dt:.0f} tokens/s at {ps.mean:.1f} W ({len(ps.samples)} samples)", flush=True)
    return {"tokens_per_s": n_tok / dt, "power_W": ps.mean, "J_per_token": ps.mean * dt / n_tok, "seconds": dt, "batch": batch, "tokens_per_batch": int(xs[0].numel()), "n_power_samples": len(ps.samples)}


def idle_power(seconds=3.0):
    with PowerSampler() as ps:
        time.sleep(seconds)
    return ps.mean


def measure_cpu_int8(model_cpu, data, block, batch, seconds=8.0):
    import torch.ao.quantization as q
    m = q.quantize_dynamic(model_cpu, {nn.Linear}, dtype=torch.qint8) if any(isinstance(x, nn.Linear) for x in model_cpu.modules()) else model_cpu
    gen = torch.Generator().manual_seed(3); x = get_batch(data, block, batch, gen)[0]
    with torch.no_grad():
        m(x); t0 = time.time(); n = 0
        while time.time() - t0 < seconds:
            m(x); n += x.numel()
    return {"tokens_per_s": n / (time.time() - t0), "energy": "not measurable here (RAPL counters are root-only)"}


# ------------------------------------------------------------------ BPU side
def bpu_energy(ops, tokens, mode, bits, spikes_per_conversion=0.0, training=False, fom="fom10"):
    conv_tile, conv_layer = ops["conversions"], ops["conversions_out"]
    parts = {"mac_J": ops["macs"] * E_MAC, "route_J": ops["events"] * E_ROUTE, "cache_write_J": ops.get("cache_writes", 0) * E_CACHE_WRITE,
             "write_J": ops.get("pulses", 0) * E_WRITE if training else 0.0}
    if mode == "tile":
        parts["readout_J"] = conv_tile * E_ADC[(fom, bits)]
    elif mode == "layer":
        parts["readout_J"] = conv_layer * E_ADC[(fom, bits)]
    else:   # LIF: no ADC, the FoM does not enter; the window is 2^b - 1 spikes
        parts["readout_J"] = conv_layer * spikes_per_conversion * (E_SPIKE + E_LIF_UPDATE)
    return sum(parts.values()) / tokens, {k: v / tokens for k, v in parts.items()}


def h100_J_per_token(macs_per_token, util=1.0):
    return 2 * macs_per_token * H100_W / (H100_INT8_TOPS * 1e12 * util)


def main():
    ap = base_parser("E8b energy: measured INT8 control vs BPU read-out modes", default_seeds=1)
    ap.add_argument("--steps", type=int, default=1500); ap.add_argument("--ft-steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=24); ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--sigma", type=float, default=0.04); ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--measure-batch", type=int, default=128, help="batch for the GPU energy loop (tokens = batch x block)")
    ap.add_argument("--measure-seconds", type=float, default=15.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-measure", action="store_true", help="skip the GPU power measurement (e.g. while the GPU is busy)")
    ap.add_argument("--skip-train", action="store_true", help="reuse results/e8b/checkpoint.pt")
    ap.add_argument("--reuse-measurement", action="store_true", help="take the GPU measurement block from the existing results/e8b/results.json instead of measuring")
    ap.add_argument("--calib-pct", type=float, default=99.9, help="converter range = this percentile of |value| on the calibration batch (100 = max); the max-range result is recorded as well")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    if args.smoke:
        args.steps, args.ft_steps, args.eval_batches, args.measure_seconds = 20, 10, 2, 2.0
    dev = args.device
    train_ids, val_ids, vocab = load_text()
    from common import _out_dir
    ckpt = _out_dir("e8b", args.out) / "checkpoint.pt"
    # ---- the E8 protocol at sigma = 0.04: ideal, then noise-aware fine-tune
    gen = torch.Generator().manual_seed(0)
    ideal = GPT(vocab, 192, 6, 6, args.block, None, None, seed=0).to(dev)
    na = GPT(vocab, 192, 6, 6, args.block, args.sigma, None, seed=0).to(dev)
    if args.skip_train and ckpt.exists():
        sd = torch.load(ckpt, map_location=dev); ideal.load_state_dict(sd["ideal"]); na.load_state_dict(sd["noise_aware"])
    else:
        with Timer("ideal training"):
            train(ideal, train_ids, args.steps, args.block, args.batch, 1e-3, gen, tiled=False, label="ideal")
        na.load_state_dict(ideal.state_dict(), strict=False); copy_weights(ideal, na, verify=True)
        with Timer(f"noise-aware fine-tune at sigma={args.sigma}"):
            train(na, train_ids, args.ft_steps, args.block, args.batch, 3e-4, torch.Generator().manual_seed(1000), tiled=True, warmup=20, label="na")
        ckpt.parent.mkdir(parents=True, exist_ok=True); torch.save({"ideal": ideal.state_dict(), "noise_aware": na.state_dict()}, ckpt)
    ev = lambda m: evaluate(m, val_ids, args.block, args.batch, args.eval_batches, torch.Generator().manual_seed(99))
    ideal_val = ev(ideal); na_val = ev(na)
    print(f"ideal val loss {ideal_val:.4f}; noise-aware tiles at sigma={args.sigma}: {na_val:.4f} ({100 * (na_val / ideal_val - 1):+.2f}%)", flush=True)
    # ---- accuracy of each read-out mode (emulated on the noise-aware model)
    accuracy = {}; accuracy_maxrange = {}
    cal_x, _ = get_batch(train_ids, args.block, args.batch, torch.Generator().manual_seed(11), dev)
    for calib, store in ((args.calib_pct, accuracy), (100.0, accuracy_maxrange)):
        for mode in MODES:
            for bits in BITS:
                spec = ReadoutSpec(mode, bits, calib_pct=calib); set_readout(na, spec)
                spec.calibrating = True
                with torch.no_grad():
                    na(cal_x)
                spec.calibrating = False; spec.reset_counts()
                loss = ev(na)
                spk = spec.spikes / max(spec.spike_sites, 1)
                store[f"{mode}_{bits}"] = {"val_loss": loss, "gap_pct_vs_ideal": 100 * (loss / ideal_val - 1), "gap_pct_vs_noise_aware_unquantised": 100 * (loss / na_val - 1),
                                          "spikes_per_output_per_token": spk if mode == "lif" else None, "calib_pct": calib}
                print(f"  readout {mode:5s} {bits}-bit (range = {calib:g}th pct): val loss {loss:.4f} ({100 * (loss / ideal_val - 1):+.2f}% vs ideal)" + (f", {spk:.2f} spikes per output per token" if mode == "lif" else ""), flush=True)
        if calib == 100.0 and args.calib_pct >= 100.0:
            break
    set_readout(na, None)
    # ---- op counts
    ops_all = count_ops(lambda c: GPT(vocab, 192, 6, 6, args.block, args.sigma, c, seed=0).to(dev), train_ids, args.block, args.batch, torch.Generator().manual_seed(7))
    ops_eval = ops_all["eval_batch"]; tokens = ops_all["tokens"]; n_params = ops_all["n_params"]
    ops_train = dict(ops_all["train_step"]); ops_train["macs"] = ops_train["macs"] + ops_train.get("macs_backward_estimate", 0)
    # writes per training step from E8's fine-tuning average (one Adam step rarely crosses a pulse)
    try:
        e8 = load_results("e8")
        wps = float(np.mean([r["per_sigma"][s]["writes_per_device_per_step"] for r in e8["runs"] for s in r["per_sigma"] if float(s) > 0]))
    except Exception:
        wps = 4e-4
    ops_train["pulses"] = max(ops_train.get("pulses", 0), wps * ops_all["n_devices"] * 1.5)
    # ---- the 3 read-out x 2 FoM x 2 bits table (inference) + training step
    table = {}
    for mode in MODES:
        for fom in FOM_W:
            for bits in BITS:
                spk = accuracy[f"{mode}_{bits}"]["spikes_per_output_per_token"] or 0.0
                e, parts = bpu_energy(ops_eval, tokens, mode, bits, spk, fom=fom)
                e_tr, parts_tr = bpu_energy(ops_train, tokens, mode, bits, spk, training=True, fom=fom)
                table[f"{mode}_{fom}_{bits}"] = {"mode": mode, "fom": fom, "fom_fJ_per_step": FOM_W[fom] * 1e15, "bits": bits, "E_ADC_pJ": E_ADC[(fom, bits)] * 1e12 if mode != "lif" else None,
                                                 "nJ_per_token": e * 1e9, "parts_nJ": {k: v * 1e9 for k, v in parts.items()},
                                                 "train_step_nJ_per_token": e_tr * 1e9, "train_parts_nJ": {k: v * 1e9 for k, v in parts_tr.items()},
                                                 "gap_pct_vs_ideal": accuracy[f"{mode}_{bits}"]["gap_pct_vs_ideal"], "spikes_per_output_per_token": spk if mode == "lif" else None}
    # ---- digital control: measured on the GPU
    digital = {"gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "none"}
    if args.reuse_measurement:
        prev = load_results("e8b", args.out)["digital_control"]
        for k in ("idle_power_W", "fp32", "int8", "gpu"):
            if k in prev:
                digital[k] = prev[k]
        print("  GPU measurement reused from results.json", flush=True)
    elif dev == "cuda" and not args.skip_measure:
        digital["idle_power_W"] = idle_power()
        print(f"  GPU idle power {digital['idle_power_W']:.1f} W", flush=True)
        fp32 = GPT(vocab, 192, 6, 6, args.block, None, None, seed=0).to(dev); fp32.load_state_dict(ideal.state_dict())
        digital["fp32"] = measure_gpu(fp32, train_ids, args.block, args.measure_batch, args.measure_seconds, "GPU FP32")
        int8 = to_int8(fp32)
        int8_val = ev(int8)
        digital["int8"] = measure_gpu(int8, train_ids, args.block, args.measure_batch, args.measure_seconds, "GPU INT8 (W8A8, torch._int_mm)")
        digital["int8"]["val_loss"] = int8_val; digital["int8"]["gap_pct_vs_ideal"] = 100 * (int8_val / ideal_val - 1)
        for k in ("fp32", "int8"):
            d = digital[k]; d["nJ_per_token_incl_idle"] = d["J_per_token"] * 1e9
            d["nJ_per_token_above_idle"] = (d["power_W"] - digital["idle_power_W"]) * d["seconds"] / (d["tokens_per_s"] * d["seconds"]) * 1e9
        print(f"  GPU INT8: {digital['int8']['nJ_per_token_incl_idle']:.1f} nJ/token incl. idle, {digital['int8']['nJ_per_token_above_idle']:.1f} above idle; val loss {int8_val:.4f} ({digital['int8']['gap_pct_vs_ideal']:+.2f}%)", flush=True)
    cpu_model = GPT(vocab, 192, 6, 6, args.block, None, None, seed=0); cpu_model.load_state_dict(ideal.state_dict())
    # the CPU INT8 path needs nn.Linear modules: swap the ideal tiles for Linear layers with the same weights
    for name, m in list(cpu_model.named_modules()):
        for cname, c in list(m.named_children()):
            if isinstance(c, TileLinear):
                lin = nn.Linear(c.in_features, c.out_features, bias=c.b is not None)
                with torch.no_grad():
                    lin.weight.copy_(c.W.T)
                    if c.b is not None:
                        lin.bias.copy_(c.b)
                setattr(m, cname, lin)
    digital["cpu_int8_torch_ao"] = measure_cpu_int8(cpu_model, train_ids, args.block, 32, 4.0 if not args.smoke else 1.0)
    digital["h100_int8_published"] = {"source": "NVIDIA H100 SXM datasheet: 1979 TOPS dense INT8, 700 W (peak, published, not measured)",
                                      "pJ_per_op_at_peak": H100_W / (H100_INT8_TOPS * 1e12) * 1e12,
                                      "nJ_per_token_at_peak_for_this_model": h100_J_per_token(ops_eval["macs"] / tokens) * 1e9,
                                      "utilisation_for_40pct_row": H100_UTIL,
                                      "nJ_per_token_at_40pct_util_for_this_model": h100_J_per_token(ops_eval["macs"] / tokens, H100_UTIL) * 1e9}
    # ---- scaling to 1T
    macs_per_token_meas = ops_eval["macs"] / tokens
    L0, d0 = SIZES["2.7M (E8, measured)"]
    ratio_macs = macs_per_token_meas / (12 * L0 * d0 ** 2)
    conv_layer_per_Ld = ops_eval["conversions_out"] / tokens / (L0 * d0)
    conv_tile_per_Ld2 = ops_eval["conversions"] / tokens / (L0 * d0 ** 2 / 32)
    ev_per_L = ops_eval["events"] / tokens / L0
    cache_per_Ld = ops_eval.get("cache_writes", 0) / tokens / (L0 * d0)
    scaling = []
    for name, (L, d) in SIZES.items():
        P = 12 * L * d ** 2
        ops = {"macs": ratio_macs * 12 * L * d ** 2, "conversions": conv_tile_per_Ld2 * L * d ** 2 / 32, "conversions_out": conv_layer_per_Ld * L * d,
               "events": ev_per_L * L, "cache_writes": cache_per_Ld * L * d, "pulses": 0}
        row = {"model": name, "layers": L, "d_model": d, "params_approx": P, "macs_per_token": ops["macs"], "conversions_per_token_layer_mode": ops["conversions_out"]}
        for mode in MODES:
            for fom in FOM_W:
                for bits in BITS:
                    spk = accuracy[f"{mode}_{bits}"]["spikes_per_output_per_token"] or 0.0
                    row[f"{mode}_{fom}_{bits}_nJ_per_token"] = bpu_energy(ops, 1, mode, bits, spk, fom=fom)[0] * 1e9
        if "int8" in digital:
            # digital control scaled by MACs from the measured INT8 energy of this model
            row["gpu_int8_measured_scaled_nJ_per_token"] = digital["int8"]["nJ_per_token_incl_idle"] * ops["macs"] / macs_per_token_meas
        row["h100_int8_peak_nJ_per_token"] = h100_J_per_token(ops["macs"]) * 1e9
        row["h100_int8_40pct_nJ_per_token"] = h100_J_per_token(ops["macs"], H100_UTIL) * 1e9
        row["bpu_over_h100_40pct"] = {k: row[f"{k}_nJ_per_token"] / row["h100_int8_40pct_nJ_per_token"] for k in ("layer_fom10_8", "layer_fom50_8", "tile_fom10_8", "lif_fom10_8")}
        scaling.append(row)
    layers = 6; reads_in_series = layers * 4 + 1 + layers * 2
    lossless = {k: v for k, v in table.items() if v["gap_pct_vs_ideal"] <= 2.0}
    best = min(lossless.values(), key=lambda r: r["nJ_per_token"]) if lossless else min(table.values(), key=lambda r: r["nJ_per_token"])
    g_int8 = digital.get("int8", {}).get("nJ_per_token_incl_idle")
    h_peak = digital["h100_int8_published"]["nJ_per_token_at_peak_for_this_model"]; h_40 = digital["h100_int8_published"]["nJ_per_token_at_40pct_util_for_this_model"]
    r1t = scaling[-1]["bpu_over_h100_40pct"]["layer_fom10_8"]; r0 = scaling[0]["bpu_over_h100_40pct"]["layer_fom10_8"]
    stmt = (f"PROJECTION on the BPU side (device-model constants, E_ADC = FoM_W x 2^b with FoM_W 10 / 50 fJ per conversion-step), MEASURED on the digital side. Same 2.7M-parameter nanoGPT. "
            + (f"GPU INT8 measured ({digital['gpu']}): {g_int8:.0f} nJ/token incl. idle power, {digital['int8']['nJ_per_token_above_idle']:.0f} above idle, val loss gap {digital['int8']['gap_pct_vs_ideal']:+.2f}%; FP32 {digital['fp32']['nJ_per_token_incl_idle']:.0f} nJ/token. " if g_int8 else "GPU energy not measured in this run. ")
            + f"H100 INT8 published: {h_peak:.1f} nJ/token at peak, {h_40:.1f} at 40% utilisation (not measured). BPU inference nJ/token (val-loss gap vs ideal): "
            + "; ".join(f"{m} {f.replace('fom', 'FoM ')} fJ {b}-bit {table[f'{m}_{f}_{b}']['nJ_per_token']:.0f} ({table[f'{m}_{f}_{b}']['gap_pct_vs_ideal']:+.1f}%)" for m in MODES for f in FOM_W for b in BITS)
            + f". Cheapest lossless cell (gap <= 2%): {best['mode']} {best['fom']} {best['bits']}-bit at {best['nJ_per_token']:.0f} nJ/token = {best['nJ_per_token'] / h_40:.2f}x H100@40%"
            + (f", {g_int8 / best['nJ_per_token']:.0f}x below the measured laptop GPU" if g_int8 else "")
            + f"; MACs alone {table['layer_fom10_8']['parts_nJ']['mac_J']:.1f} nJ/token; LIF spike rate {accuracy['lif_8']['spikes_per_output_per_token']:.1f} (8-bit window) / {accuracy['lif_6']['spikes_per_output_per_token']:.1f} (6-bit) spikes per output per token; converter ranges at the {args.calib_pct:g}th percentile. "
            + f"Scaling: conversions grow as layers x d_model while MACs grow as parameters (~ layers x d_model^2), so the BPU / H100@40% ratio for the per-layer 8-bit FoM-10 cell falls from {r0:.2f} at 2.7M to {r1t:.3f} at 1T: the relative advantage grows with model size because the converter cost per MAC shrinks as 1/d_model.")
    out = {"constants": {"E_MAC_J": E_MAC, "E_ADC_J": {str(k): v for k, v in E_ADC.items()}, "E_SPIKE_J": E_SPIKE, "E_LIF_UPDATE_J": E_LIF_UPDATE, "E_WRITE_J": E_WRITE, "E_ROUTE_J": E_ROUTE,
                         "E_CACHE_WRITE_J": E_CACHE_WRITE, "t_read_series_s": T_READ_SERIES, "lif_energy_accounting": "per spike (router 10 pJ + LIF update 5 pJ); the per-time-step alternative is 2^b updates per output per token"},
           "protocol": {"steps": args.steps, "ft_steps": args.ft_steps, "sigma": args.sigma, "batch": args.batch, "block": args.block, "device": dev, "smoke": args.smoke},
           "ideal_val_loss": ideal_val, "noise_aware_val_loss": na_val, "readout_accuracy": accuracy, "readout_accuracy_max_range": accuracy_maxrange,
           "calib_pct": args.calib_pct, "measured_ops": ops_all, "table_3x2": table,
           "digital_control": digital, "scaling": scaling, "latency": {"bpu_s_per_token": reads_in_series * T_READ_SERIES, "reads_in_series": reads_in_series},
           "fom_W_J_per_step": {k: v for k, v in FOM_W.items()}, "bits": BITS, "modes": MODES,
           "verdict": {"statement": stmt}}
    print("E8b:", stmt)
    save_results("e8b", out, args.out)
    if not args.no_fig:
        plot(out, figure_path("e8b_energy"))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    t = res["table_3x2"]; keys = list(t)
    x = np.arange(len(keys)); vals = [t[k]["nJ_per_token"] for k in keys]
    ax[0].bar(x, vals, color=["C0" if k.endswith("_8") else "C1" for k in keys])
    for i, k in enumerate(keys):
        ax[0].text(i, vals[i] * 1.05, f"{t[k]['gap_pct_vs_ideal']:+.1f}%", ha="center", fontsize=7)
    g = res["digital_control"].get("int8", {}).get("nJ_per_token_incl_idle")
    if g:
        ax[0].axhline(g, color="k", ls="--", lw=1, label=f"GPU INT8 measured {g:.0f} nJ/token")
        ax[0].axhline(res["digital_control"]["int8"]["nJ_per_token_above_idle"], color="k", ls=":", lw=1, label="GPU INT8 above idle")
    ax[0].axhline(res["digital_control"]["h100_int8_published"]["nJ_per_token_at_peak_for_this_model"], color="gray", ls="-.", lw=1, label="H100 INT8 published peak")
    ax[0].axhline(res["digital_control"]["h100_int8_published"]["nJ_per_token_at_40pct_util_for_this_model"], color="gray", ls="-", lw=1, label="H100 INT8 at 40 % utilisation")
    ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels([k.replace("_fom", " ").replace("_", " ") + "b" for k in keys], fontsize=6, rotation=35)
    ax[0].set_ylabel("nJ per token, inference (2.7M model)"); ax[0].legend(fontsize=6); ax[0].set_title("E8b: read-out x FoM x bits (labels: val-loss gap vs ideal)", fontsize=9)
    sc = res["scaling"]; P = [r["params_approx"] for r in sc]
    for k in ("layer_fom10_8", "layer_fom50_8", "tile_fom10_8", "lif_fom10_8"):
        ax[1].loglog(P, [r[f"{k}_nJ_per_token"] for r in sc], "o-", label=f"BPU {k.replace('_fom', ' FoM').replace('_', ' ')}b (projection)")
    if "gpu_int8_measured_scaled_nJ_per_token" in sc[0]:
        ax[1].loglog(P, [r["gpu_int8_measured_scaled_nJ_per_token"] for r in sc], "s--", color="k", label="GPU INT8 measured, scaled by MACs")
    ax[1].loglog(P, [r["h100_int8_peak_nJ_per_token"] for r in sc], "^:", color="gray", label="H100 INT8 published peak")
    ax[1].loglog(P, [r["h100_int8_40pct_nJ_per_token"] for r in sc], "^-", color="gray", label="H100 INT8 at 40 %")
    ax[1].set_xlabel("parameters"); ax[1].set_ylabel("nJ per token"); ax[1].legend(fontsize=6); ax[1].set_title("scaling to 1T (MACs ~ params, conversions ~ layers x d)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"[e8b] figure {path}")


if __name__ == "__main__":
    main()
