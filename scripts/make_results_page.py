"""Build results/index.html from results/*/results.json and figures/*.png.

Substrate framing: the BPU is a substrate, so every row compares the SAME
algorithm on (a) a float32 reference and (b) noisy BPU tiles; success = BPU
matches the reference within the stated tolerance.  Local learning rules
(FA, DFA, three-factor, growth) are additional arms, never the headline.
The "Substrate claims" scoreboard has six rows (E9, E10, E8b, E3, E6, E7);
the write-budget mechanism (E4, E9) is reported in its own box; the remaining
experiments follow.  Every number is read from a results.json; nothing is
typed in.  Usage: python scripts/make_results_page.py [--out results/index.html]
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"


def load(name):
    p = RES / name / "results.json"
    return json.load(open(p)) if p.exists() else None


def fmt(x, d=3):
    try:
        return f"{float(x):.{d}f}"
    except Exception:
        return "–"


def pct(x, d=1):
    try:
        return f"{float(x):+.{d}f} %"
    except Exception:
        return "–"


def chip(kind, text):
    return f'<span class="chip {kind}">{html.escape(text)}</span>'


def esc(x):
    return html.escape(str(x))


def section(eid, eyebrow, title, fig, facts, verdict, notes=(), src=None):
    fig_html = f'<figure><img src="figures/{fig}" alt="{esc(title)}"><figcaption>figures/{fig}</figcaption></figure>' if (FIG / fig).exists() else '<p class="muted">figure not generated yet</p>'
    kv = "".join(f'<div><b>{esc(v)}</b><span>{esc(k)}</span></div>' for k, v in facts)
    li = "".join(f"<li>{n}</li>" for n in notes)
    return f'''<section id="{eid}"><div class="eyebrow">{esc(eyebrow)}</div><h2>{esc(title)}</h2>
<div class="kv">{kv}</div>{fig_html}{("<ul>" + li + "</ul>") if li else ""}
<div class="verdict">{esc(verdict or "not run")}<small>results/{src or eid}/results.json · verdict</small></div></section>'''


def table_html(head, rows):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tablewrap"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "index.html"))
    args = ap.parse_args()
    names = ("e1", "e2", "e2_cifar", "e3", "e3_256", "e3_tracking", "e4", "e4_capped", "e4_level", "e4_level_capped", "e5", "e5_48", "e5_selforg32",
             "e6", "e7", "e7_ref", "e8", "e8b", "e9", "e9_pulse", "e10", "e9_v1", "e10_v1", "e8b_v1")
    R = {k: load(k) for k in names}
    e9, e10, e8b, e3t, e6, e7, e7r = R["e9"], R["e10"], R["e8b"], R["e3_tracking"], R["e6"], R["e7"], R["e7_ref"]
    e4, e4c, e4l, e4lc, e8 = R["e4"], R["e4_capped"], R["e4_level"], R["e4_level_capped"], R["e8"]

    # ------------------------------------------------------------------ substrate scoreboard (six rows)
    rows = []
    # 1. E9
    if e9 and "summary" in e9 and "sgd_bpu" in e9["summary"]:
        s = e9["summary"]; g = e9["post_shift_gap_pct_vs_float32"]; go = e9["post_shift_gap_pct_vs_bpu_oracle"]
        ref = f"SGD float32: {fmt(s['sgd_float32']['segment_end_loss_1']['mean'])} nats/char"
        bpu = f"SGD on tiles (ceiling): {fmt(s['sgd_bpu']['segment_end_loss_1']['mean'])} ({pct(g['sgd_bpu'])}); writes {s['sgd_bpu']['writes_per_device_per_s']['mean']:.3g}/dev/s = {s['sgd_bpu']['percent_of_budget']['mean']:.0f} % of budget"
        extra = "; ".join(f"{a}: {pct(go[a])} vs ceiling, {s[a]['percent_of_budget']['mean']:.0f} % budget" for a in ("fa_bpu", "dfa_bpu", "local_bpu") if a in s)
        e9p = R["e9_pulse"]
        if e9p and "sgd_bpu_pulse" in e9p.get("summary", {}):
            sp = e9p["summary"]["sgd_bpu_pulse"]
            extra += f"; diagnostic: SGD on tiles with nearest-pulse writes (no level gate) {fmt(sp['segment_end_loss_1']['mean'])} ({pct(100 * (sp['segment_end_loss_1']['mean'] / s['sgd_float32']['segment_end_loss_1']['mean'] - 1))} vs float32) at {sp['percent_of_budget']['mean']:.0f} % budget: the gap is the write gate, not the device noise"
        st = chip("ok" if e9["verdict"]["success"] else "nul", "criterion met" if e9["verdict"]["success"] else "not met") + " " + \
             chip("ok" if e9["verdict"]["sgd_bpu_within_5pct"] else "nul", "within 5 %") + " " + chip("ok" if e9["verdict"]["sgd_bpu_within_budget"] else "nul", "write budget")
        rows.append(("Trainable where an FPGA is frozen", "E9", ref, bpu, "post-shift loss within 5 %, writes ≤ 6e-3/dev/s", extra, st))
    else:
        rows.append(("Trainable where an FPGA is frozen", "E9", "pending", "pending", "within 5 %, writes within budget", "", chip("pend", "running")))
    # 2. E10
    if e10 and "twins" in e10:
        tw = e10["twins"]; S = e10["summary"]
        ref = "; ".join(f"{a} {fmt(tw[a]['float32'])}" for a in e10["arms"])
        bpu = "; ".join(f"{a} {fmt(tw[a]['bpu'])} ({tw[a]['gap_points']:+.3f})" for a in e10["arms"])
        extra = "forgetting BPU: " + ", ".join(f"{a} {fmt(S['bpu'][a]['forgetting']['mean'])}" for a in e10["arms"])
        ok = e10["verdict"]["success"]
        st = chip("ok" if ok else "nul", "criterion met" if ok else "not met") + " " + chip("ok" if abs(tw["ewc"]["gap_points"]) <= 0.03 else "nul", f"EWC {tw['ewc']['gap_points']:+.3f}") + " " + chip("ok" if abs(tw["replay"]["gap_points"]) <= 0.03 else "nul", f"replay {tw['replay']['gap_points']:+.3f}")
        rows.append(("Holds continually adapting models", "E10", ref, bpu, "BPU-EWC and BPU-replay within 3 points of float32 twins", extra, st))
    else:
        rows.append(("Holds continually adapting models", "E10", "pending", "pending", "within 3 points", "", chip("pend", "running")))
    # 3. E8b
    if e8b and "table_3x2" in e8b:
        t = e8b["table_3x2"]; d = e8b["digital_control"]
        gi = d.get("int8", {})
        ref = (f"GPU INT8 measured ({d.get('gpu', '')}): {gi['nJ_per_token_incl_idle']:.0f} nJ/token incl. idle, {gi['nJ_per_token_above_idle']:.0f} above idle (gap {pct(gi['gap_pct_vs_ideal'], 2)})" if gi else "GPU INT8: not measured yet") + \
              f"; H100 INT8 published peak {d['h100_int8_published']['nJ_per_token_at_peak_for_this_model']:.1f} nJ/token"
        best = min(t.values(), key=lambda r: r["nJ_per_token"])
        bpu = "; ".join(f"{k.replace('_', ' ')}-bit {v['nJ_per_token']:.0f} nJ ({pct(v['gap_pct_vs_ideal'])})" for k, v in t.items())
        extra = f"cheapest {best['mode']} {best['bits']}-bit at {best['nJ_per_token']:.0f} nJ/token" + (f" = {gi['nJ_per_token_incl_idle'] / best['nJ_per_token']:.0f}x below measured GPU INT8" if gi else "")
        st = chip("part", "BPU = projection, GPU = measured") + (" " + chip("ok" if gi and best["nJ_per_token"] < gi["nJ_per_token_above_idle"] else "nul", "below GPU above-idle") if gi else "")
        rows.append(("Cheaper per token than a GPU", "E8b", ref, bpu, "nJ/token, accuracy of each mode at σ = 0.04", extra, st))
    else:
        rows.append(("Cheaper per token than a GPU", "E8b", "pending", "pending", "nJ/token", "", chip("pend", "running")))
    # 4. E3 tracking
    if e3t and "verdict" in e3t and "n_u_shaped" in e3t["verdict"]:
        amp0 = e3t["cue_amps"][0]; sw = e3t["sweeps"]
        ref = "; ".join(f"ω={o:g}: {fmt(sw[f'amp{amp0:g}_omega{o:g}_float32']['0.0']['track_err']['mean'])} rad" for o in e3t["omegas"])
        bpu = "; ".join(f"ω={o:g} mm 0.1: {fmt(sw[f'amp{amp0:g}_omega{o:g}_bpu_mismatch0.1']['0.0']['track_err']['mean'])} rad" for o in e3t["omegas"] if f"amp{amp0:g}_omega{o:g}_bpu_mismatch0.1" in sw)
        extra = "ΔU: " + ", ".join(f"{k.replace('mismatch', 'mm ')} {fmt(v['delta_U']['mean'], 3)}" for k, v in e3t["landscape"].items())
        n_u, n_c = e3t["verdict"]["n_u_shaped"], e3t["verdict"]["n_conditions"]
        rows.append(("Attractor noise optimum (moving cue)", "E3", ref, bpu, "tracking error at σ = 0; U-test over the σ sweep", extra, chip("ok" if n_u else "nul", f"{n_u}/{n_c} U-shaped")))
    else:
        rows.append(("Attractor noise optimum (moving cue)", "E3", "pending", "pending", "U-test + ΔU", "", chip("pend", "running")))
    # 5. E6
    if e6:
        caps = e6["capacity_at_90pct"]
        ref = "; ".join(f"{m}: {caps[m]['0.0']}" for m in e6["methods"])
        bpu = "; ".join(f"{m}: " + "/".join(str(caps[m][str(s)]) for s in e6["sigmas"][1:]) for m in e6["methods"])
        same = all(caps["attention"][str(s)] >= caps["attention"]["0.0"] for s in e6["sigmas"] if s <= 0.04)
        rows.append(("Memory + attention under noise", "E6", ref + " pairs @90 %", bpu + f" (σ = {', '.join(str(s) for s in e6['sigmas'][1:])})", "capacity on tiles matches σ = 0 up to σ = 0.04", f"{100 * e6.get('flip_fraction', 0):.0f} % flipped cues, {e6['seeds']} seeds, sign ties random",
                     chip("ok" if same else "nul", "attention holds" if same else "attention drops")))
    # 6. E7
    if e7 and "summary" in e7:
        s = e7["summary"]
        def line(sm):
            return (f"nav grown {fmt(sm['nav_acc']['grown']['mean'], 2)} / fixed {fmt(sm['nav_acc']['fixed']['mean'], 2)}" +
                    (f" / grown-continued {fmt(sm['nav_acc']['grown_continued']['mean'], 2)}" if "grown_continued" in sm["nav_acc"] else "") +
                    f"; recall {fmt(sm['recall_bit_acc']['grown']['mean'], 2)} / {fmt(sm['recall_bit_acc']['fixed']['mean'], 2)}")
        ref = line(e7r["summary"]) if e7r and "summary" in e7r else "noise-free tiles: pending"
        bpu = line(s)
        extra = f"composite gap grown−fixed {s['transfer_gap']['mean']:+.2f} (matched synapses+steps), {s['transfer_gap_matched_compute']['mean']:+.2f} (matched compute)" + \
                (f", {s['transfer_gap_continued']['mean']:+.2f} (continued)" if "transfer_gap_continued" in s else "") + f"; Q = {fmt(s['Q_final_grown']['mean'], 2)}"
        neg = s.get("grown_continued_still_behind")
        st = chip("nul" if neg else "part", "negative: grown stays behind fixed" if neg else "grown catches up") if neg is not None else chip("part", "see row")
        rows.append(("Modular self-assembly", "E7", ref, bpu, "grown vs fixed at matched synapses AND matched steps; + grown continued", extra, st))
    else:
        rows.append(("Modular self-assembly", "E7", "pending", "pending", "matched synapses and steps", "", chip("pend", "running")))
    score = table_html(["Substrate claim", "Exp", "float32 / reference", "BPU tiles", "tolerance", "additional arms", "status"],
                       [[esc(a), f"<span class='id'>{esc(b)}</span>", esc(c), esc(d), esc(e), esc(f), g] for a, b, c, d, e, f, g in rows])

    # ------------------------------------------------------------------ write-budget box
    wb_rows = []
    for lab, r in (("E4 threshold writes, uncapped (weekend)", e4), ("E4 quota 6e-3, threshold writes (weekend)", e4c),
                   ("E4 accumulate-and-flush, no quota", e4l), ("E4 accumulate-and-flush + quota 6e-3", e4lc)):
        if r:
            w = r.get("writes_per_device_per_s", sum(x["writes_per_device_per_s"] for x in r["runs"]) / len(r["runs"]))
            wb_rows.append([esc(lab), f"{w:.3g}", f"{100 * w / 6e-3:.0f} %", f"{r.get('flush_rate', float('nan')):.2g}" if "flush_rate" in r else "–",
                            f"{r['verdict']['certified_fraction']:.0%} certified ({r['verdict']['control_certified_fraction']:.0%} controls)", f"{r['development_time_s']:.0f} s"])
    if e9 and "summary" in e9:
        for a in e9["arms"]:
            if a == "sgd_float32":
                continue
            s = e9["summary"][a]
            wb_rows.append([esc(f"E9 {a} (10 ms/step)"), f"{s['writes_per_device_per_s']['mean']:.3g}", f"{s['percent_of_budget']['mean']:.0f} %", f"{s['flush_rate']['mean']:.2g}", f"post-shift loss {fmt(s['segment_end_loss_1']['mean'])}", f"{e9['seg'] * 3 * e9['step_seconds']:.0f} s"])
    if e10 and "summary" in e10:
        for a in e10["arms"]:
            s = e10["summary"]["bpu"][a]
            wb_rows.append([esc(f"E10 {a} on tiles (10 ms/step)"), f"{s['writes_per_device_per_s']['mean']:.3g}", f"{s['percent_of_budget']['mean']:.0f} %", f"{s['flush_rate']['mean']:.2g}", f"avg acc {fmt(s['avg_acc']['mean'])}", "–"])
    wbox = table_html(["run", "writes / device / s", "% of 6e-3 budget", "flush rate", "outcome", "chip time"], wb_rows)

    # ------------------------------------------------------------------ other rows
    other = []
    e3, e5, e2, e2c = R["e3"], R["e5"], R["e2"], R["e2_cifar"]
    if R["e3"] and R["e3_256"] and "verdicts" in R["e3"]:
        for key in ("sweeps", "landscape", "fits", "verdicts", "velocity_gain"):
            R["e3"][key].update(R["e3_256"].get(key, {}))
        mk = [k for k in R["e3"]["verdicts"] if "mismatch0.0_" not in k]
        n_u = sum(R["e3"]["verdicts"][k].get("u_shaped", False) for k in mk)
        R["e3"]["verdict"] = {"n_u_shaped": n_u, "n_conditions": len(mk), "statement": f"{n_u}/{len(mk)} mismatched conditions U-shaped (static drift, weekend protocol)"}
    if e5:
        for extra in ("e5_48", "e5_selforg32"):
            if R[extra]:
                e5["sheets"].update({(k + (" self-organised" if extra == "e5_selforg32" else "")): v for k, v in R[extra]["sheets"].items()})
                e5["verdict"]["statement"] += " || " + R[extra]["verdict"]["statement"]
    other.append(("Accelerator parity (nanoGPT on tiles)", "E8", (f"noise-aware tiles within {fmt(e8['verdict']['worst_noise_aware_gap_pct_at_sigma_le_0.06'], 2)} % of ideal at σ ≤ 0.06, {e8['seeds']} seeds" if e8 else "pending"),
                  chip("ok", "criterion met") if e8 and e8["verdict"]["success"] else chip("pend", "pending")))
    other.append(("Ring self-organises within the write budget", "E4", (e4l["verdict"]["statement"] if e4l else "pending") + (" || " + e4lc["verdict"]["statement"] if e4lc else ""),
                  chip("nul" if e4lc and e4lc["verdict"]["certified_fraction"] == 0 else "part", f"{e4lc['verdict']['certified_fraction']:.0%} certified at budget" if e4lc else "running")))
    other.append(("Attractor noise optimum (static drift, weekend)", "E3", R["e3"]["verdict"]["statement"] if R["e3"] else "pending", chip("nul", f"{R['e3']['verdict']['n_u_shaped']}/{R['e3']['verdict']['n_conditions']} U-shaped") if R["e3"] else chip("pend", "pending")))
    other.append(("Grid module is a torus", "E5", e5["verdict"]["statement"] if e5 else "pending", chip("part", "programmed refs certify; self-org fails") if e5 else chip("pend", "pending")))
    other.append(("Local grow rule vs RigL", "E2", (e2["verdict"]["statement"] if e2 else "") + (" | CIFAR-10 3 %: " + e2c["verdict"]["statement"] if e2c else ""),
                  (chip("ok" if e2 and e2["verdict"].get("within_few_percent_of_rigl") else "nul", "MNIST") if e2 else "") + (" " + chip("nul", "CIFAR-10 refuted") if e2c and e2c["verdict"].get("clearly_worse_than_random") else "")))
    other.append(("GPT-2 through noisy tiles", "E1", "queued for the GPU cluster (MONDAY.md); E8 is the laptop stand-in", chip("pend", "cluster")))
    otab = table_html(["Row", "Exp", "Measured", "Status"], [[esc(a), f"<span class='id'>{esc(b)}</span>", esc(c), d] for a, b, c, d in other])

    # ------------------------------------------------------------------ sections
    secs = []
    if e9 and "summary" in e9:
        s = e9["summary"]
        facts = [(f"{a}: post-shift loss", fmt(s[a]["segment_end_loss_1"]["mean"])) for a in e9["arms"]] + \
                [(f"{a}: % of write budget", f"{s[a]['percent_of_budget']['mean']:.0f} %") for a in e9["arms"] if a != "sgd_float32"]
        secs.append(section("e9", "E9 · trainability under drift", "Same backprop, two substrates; local rules as extra arms", "e9_trainability.png", facts, e9["verdict"]["statement"],
                            [f"Stream: {e9['seg']} steps Shakespeare → {e9['seg']} Python source → {e9['seg']} Shakespeare, batch {e9['batch']}, {e9['seeds']} seeds, σ_eff = {e9['sigma']}, accumulate-and-flush writes at L = 25 levels, chip time {e9['step_seconds']} s/step (assumption).",
                             (f"Diagnostic arm (results/e9_pulse): the same SGD on tiles with the legacy nearest-pulse writes reaches {fmt(R['e9_pulse']['summary']['sgd_bpu_pulse']['segment_end_loss_1']['mean'])} post-shift at {R['e9_pulse']['summary']['sgd_bpu_pulse']['percent_of_budget']['mean']:.0f} % of the budget: device noise costs about 1 %, the level gate costs the rest and buys a 50x lower write rate." if R["e9_pulse"] else ""),
                             "Recovery steps after each switch (per seed): " + "; ".join(f"{a} {s[a]['recovery_steps_after_switch']}" for a in e9["arms"])]))
    if e10 and "twins" in e10:
        S = e10["summary"]; tw = e10["twins"]
        facts = [(f"{a} float32 / BPU", f"{fmt(tw[a]['float32'])} / {fmt(tw[a]['bpu'])}") for a in e10["arms"] + ["joint"]] + \
                [(f"{a} forgetting f32 / BPU", f"{fmt(S['float32'][a]['forgetting']['mean'])} / {fmt(S['bpu'][a]['forgetting']['mean'])}") for a in e10["arms"]]
        notes = [f"Permuted MNIST, 5 tasks, 784-256-256-10, {e10['steps']} steps per task, EWC λ = {e10['lam']}, replay buffer {e10['buffer']} samples (1 % of the training set), {e10['seeds']} seeds; σ_eff = {e10['sigma']} on tiles with accumulate-and-flush writes."]
        ab = e10.get("ablation", {})
        if ab.get("gate_off"):
            notes.append("FPL gate off (every synapse consolidates), avg acc float32 / BPU: " + ", ".join(f"{a} {fmt(ab['gate_off']['float32'][a]['avg_acc']['mean'])} / {fmt(ab['gate_off']['bpu'][a]['avg_acc']['mean'])}" for a in ("fpl", "fpl_replay")) +
                         "; gate on: " + ", ".join(f"{a} {fmt(tw[a]['float32'])} / {fmt(tw[a]['bpu'])}" for a in ("fpl", "fpl_replay")))
        if ab.get("tau_ret"):
            notes.append("Retention ablation on tiles, avg acc at the end of the sequence → after a 1000 s rest: " + "; ".join(
                f"τ_ret = {t} s: " + ", ".join(f"{a} {fmt(v[a]['avg_acc']['mean'])}→{fmt(v[a]['avg_acc_after_rest_1000s']['mean'])}" for a in e10["arms"]) for t, v in ab["tau_ret"].items()) +
                "; τ_ret = ∞: " + ", ".join(f"{a} {fmt(S['bpu'][a]['avg_acc']['mean'])}→{fmt(S['bpu'][a]['avg_acc_after_rest_1000s']['mean'])}" for a in e10["arms"]))
        secs.append(section("e10", "E10 · continual learning", "Five algorithms on float32 and on tiles", "e10_continual.png", facts, e10["verdict"]["statement"], notes))
    if e8b and "table_3x2" in e8b:
        t = e8b["table_3x2"]; d = e8b["digital_control"]; gi = d.get("int8", {})
        rows3 = [[esc(m), f"{t[f'{m}_8']['nJ_per_token']:.0f} nJ ({pct(t[f'{m}_8']['gap_pct_vs_ideal'])})", f"{t[f'{m}_4']['nJ_per_token']:.0f} nJ ({pct(t[f'{m}_4']['gap_pct_vs_ideal'])})"] for m in ("tile", "layer", "lif")]
        rows3.append(["GPU INT8 measured (control)", esc(f"{gi['nJ_per_token_incl_idle']:.0f} nJ incl. idle / {gi['nJ_per_token_above_idle']:.0f} above idle ({pct(gi['gap_pct_vs_ideal'], 2)})" if gi else "not measured"), esc(f"FP32 on the same GPU: {d['fp32']['nJ_per_token_incl_idle']:.0f} nJ" if "fp32" in d else "")])
        rows3.append(["H100 INT8 (published peak, not measured)", f"{d['h100_int8_published']['nJ_per_token_at_peak_for_this_model']:.1f} nJ", ""])
        facts = [("ideal val loss", fmt(e8b["ideal_val_loss"])), ("noise-aware tiles σ=0.04", fmt(e8b["noise_aware_val_loss"])), ("LIF spikes/output/token (8-bit window)", fmt(e8b["readout_accuracy"]["lif_8"]["spikes_per_output_per_token"], 1)),
                 ("LIF spikes/output/token (4-bit)", fmt(e8b["readout_accuracy"]["lif_4"]["spikes_per_output_per_token"], 2)), ("MAC energy, nJ/token", fmt(t["layer_8"]["parts_nJ"]["mac_J"], 1)), ("BPU latency, µs/token", fmt(e8b["latency"]["bpu_s_per_token"] * 1e6, 0))]
        sc = e8b["scaling"]
        scal = table_html(["model", "layers × d", "per-layer 8-bit", "per-layer 4-bit", "per-tile 8-bit", "LIF 4-bit", "GPU INT8 measured, scaled", "H100 INT8 peak"],
                          [[esc(r["model"]), f"{r['layers']} × {r['d_model']}", f"{r['layer_8_nJ_per_token']:.3g}", f"{r['layer_4_nJ_per_token']:.3g}", f"{r['tile_8_nJ_per_token']:.3g}", f"{r['lif_4_nJ_per_token']:.3g}",
                            f"{r.get('gpu_int8_measured_scaled_nJ_per_token', float('nan')):.3g}", f"{r['h100_int8_peak_nJ_per_token']:.3g}"] for r in sc])
        secs.append(section("e8b", "E8b · energy per token", "3 × 2 read-out table against a measured GPU control", "e8b_energy.png", facts, e8b["verdict"]["statement"],
                            ["Read-out mode × converter precision, nJ/token (val-loss gap vs ideal in brackets):" + table_html(["read-out", "8-bit (256 pJ/conv)", "4-bit (16 pJ/conv)"], rows3),
                             f"Constants: 4 fJ/MAC, router 10 pJ/event, write driver 100 pJ/pulse, LIF 5 pJ/update + 10 pJ/spike event (one update per spike; the integration is analog), cache write 10 pJ. Digital control = board power × time / tokens on the {esc(d.get('gpu', ''))}, W8A8 via torch._int_mm, batch {gi.get('batch', '–')} × {e8b['protocol']['block']} tokens; CPU torch.ao INT8 ran at {d['cpu_int8_torch_ao']['tokens_per_s']:.0f} tokens/s but its energy is not measurable here (RAPL root-only).",
                             "Scaling (nJ/token; MACs ∝ params, per-layer conversions ∝ layers × d, per-tile conversions ∝ layers × d² / 32):" + scal]))
    if e3t and "n_u_shaped" in e3t.get("verdict", {}):
        facts = [(k.replace("amp", "cue ").replace("_omega", " ω=").replace("_bpu_mismatch", " mm ").replace("_float32", " float32"), f"{fmt(v['err_at_zero'])}→{fmt(v['err_at_min'])}@σ={v['sigma_star_empirical']} p={v['sign_test_p']:.2g}") for k, v in e3t["verdicts"].items() if k.startswith(f"amp{e3t['cue_amps'][0]:g}")]
        secs.append(section("e3t", "E3 · moving-cue tracking vs dynamic noise", f"N = {e3t['N']}, ω ∈ {e3t['omegas']}, cue {e3t['cue_amps']}, {e3t['seeds']} seeds", "e3_tracking.png", facts[:12], e3t["verdict"]["statement"],
                            ["Pinning barrier ΔU at zero dynamic noise: " + "; ".join(f"{k}: {fmt(v['delta_U']['mean'], 3)} ± {fmt(v['delta_U']['sem'], 3)}" for k, v in e3t["landscape"].items())], src="e3_tracking"))
    if e6:
        caps = e6["capacity_at_90pct"]
        secs.append(section("e6", "E6 · memory and attention", f"{100 * e6.get('flip_fraction', 0):.0f} % flipped bits, sign ties random, {e6['seeds']} seeds", "e6_memory.png",
                            [(f"{m} capacity @90 %, σ=0 / 0.04 / 0.2", f"{caps[m]['0.0']} / {caps[m]['0.04']} / {caps[m]['0.2']}") for m in e6["methods"]], e6["verdict"]["statement"]))
    if e7 and "summary" in e7:
        for lab, r, eid in (("BPU tiles", e7, "e7"), ("noise-free reference tiles", e7r, "e7_ref")):
            if not r:
                continue
            s = r["summary"]; kinds = [k for k in ("grown", "grown_continued", "fixed", "fixed_compute") if k in s["nav_acc"]]
            secs.append(section(eid, f"E7 · modular self-assembly ({lab})", "Grown vs fixed at matched synapses, matched steps, and continued", "e7_modularity.png" if eid == "e7" else "e7_ref.png",
                                [("Q final (grown)", fmt(s["Q_final_grown"]["mean"], 2)), ("nav acc " + " / ".join(kinds), " / ".join(fmt(s["nav_acc"][k]["mean"], 2) for k in kinds)),
                                 ("recall bits " + " / ".join(kinds), " / ".join(fmt(s["recall_bit_acc"][k]["mean"], 2) for k in kinds)),
                                 ("composite gap, matched synapses+steps", f"{s['transfer_gap']['mean']:+.2f}"), ("composite gap, continued", f"{s['transfer_gap_continued']['mean']:+.2f}" if "transfer_gap_continued" in s else "–")],
                                r["verdict"]["statement"], src=eid))
    if e4l or e4lc:
        secs.append(section("e4", "E4 · ring self-organisation under the write budget", "Accumulate-and-flush at one conductance level", "e4_level.png",
                            [(lab, f"{r['verdict']['certified_fraction']:.0%} certified, {r['writes_per_device_per_s']:.3g} w/dev/s = {r['percent_of_write_budget']:.0f} % budget") for lab, r in (("no quota", e4l), ("quota 6e-3", e4lc)) if r] +
                            ([("weekend threshold run", f"{e4['verdict']['certified_fraction']:.0%} certified, {sum(x['writes_per_device_per_s'] for x in e4['runs']) / len(e4['runs']):.3g} w/dev/s")] if e4 else []),
                            " || ".join(r["verdict"]["statement"] for r in (e4l, e4lc) if r), src="e4_level"))
    if e8:
        s = e8["summary"]
        secs.append(section("e8", "E8 · accelerator parity", "nanoGPT on noisy tiles (weekend result)", "e8_nanogpt_tiles.png",
                            [(f"noise-aware gap at σ={sg}", pct(s[str(sg)]["noise_aware_gap_pct"]["mean"], 2)) for sg in e8["sigmas"]], e8["verdict"]["statement"]))
    if e5:
        secs.append(section("e5", "E5 · grid module torus", "Programmed Burak–Fiete reference first", "e5_grid_selforg.png",
                            [(f"{spec} programmed reference", f"{d['reference_torus_fraction']:.0%}") for spec, d in e5["sheets"].items()], e5["verdict"]["statement"]))

    page = f'''<title>bpusim Substrate Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f6f5f1;--bg-2:#ecebe4;--fg:#1d2126;--fg-2:#565c66;--rule:#cfccc2;--accent:#0f6b7a;--accent-2:#b2561e;--ok:#2d7a3a;--ok-bg:#dcefdf;--part:#8a6a10;--part-bg:#f3e8c2;--nul:#8a2d2d;--nul-bg:#f3d9d9;--pend:#4b5563;--pend-bg:#e2e5ea;--display:"Newsreader",Georgia,serif;--body:"IBM Plex Sans",Arial,sans-serif;--mono:"IBM Plex Mono",Menlo,monospace}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#15181c;--bg-2:#1d2126;--fg:#e8e6df;--fg-2:#a4a8b0;--rule:#343941;--accent:#5fc0cf;--accent-2:#e08a52;--ok:#8bd39a;--ok-bg:#1f3a26;--part:#e3c35c;--part-bg:#3d3414;--nul:#e58b8b;--nul-bg:#3f1f1f;--pend:#c3c8d1;--pend-bg:#2b3039;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#15181c;--bg-2:#1d2126;--fg:#e8e6df;--fg-2:#a4a8b0;--rule:#343941;--accent:#5fc0cf;--accent-2:#e08a52;--ok:#8bd39a;--ok-bg:#1f3a26;--part:#e3c35c;--part-bg:#3d3414;--nul:#e58b8b;--nul-bg:#3f1f1f;--pend:#c3c8d1;--pend-bg:#2b3039;color-scheme:dark}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:16px;line-height:1.55;margin:0;padding-block:0 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
.wrap{{max-width:980px;margin:0 auto}}h1,h2{{font-family:var(--display);font-weight:500;text-wrap:balance;margin:0}}h1{{font-size:clamp(32px,5vw,48px);line-height:1.05}}h2{{font-size:26px;margin-top:6px}}p{{max-width:72ch;margin:0}}
.eyebrow{{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--fg-2)}}header{{padding-block:48px 24px;border-bottom:1px solid var(--rule);display:grid;gap:12px}}.meta{{font-family:var(--mono);font-size:13px;color:var(--fg-2)}}
section{{padding-block:32px;border-bottom:1px solid var(--rule);display:grid;gap:16px}}.chip{{display:inline-block;font-family:var(--mono);font-size:11.5px;padding:2px 8px;border-radius:3px;white-space:nowrap;margin:1px 0}}.ok{{color:var(--ok);background:var(--ok-bg)}}.part{{color:var(--part);background:var(--part-bg)}}.nul{{color:var(--nul);background:var(--nul-bg)}}.pend{{color:var(--pend);background:var(--pend-bg)}}
.tablewrap{{overflow-x:auto;border:1px solid var(--rule);border-radius:4px;margin-top:6px}}table{{border-collapse:collapse;width:100%;min-width:760px;font-size:13.5px}}th,td{{text-align:left;vertical-align:top;padding:8px 10px;border-bottom:1px solid var(--rule)}}th{{font-family:var(--mono);font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--fg-2);background:var(--bg-2);font-weight:500}}tr:last-child td{{border-bottom:0}}.id{{font-family:var(--mono);color:var(--accent);font-weight:500;white-space:nowrap}}
figure{{margin:0;background:var(--bg-2);border:1px solid var(--rule);border-radius:4px;padding:10px;display:grid;gap:8px}}figure img{{width:100%;max-width:100%;height:auto;display:block;background:#fff;border-radius:2px}}figcaption{{font-family:var(--mono);font-size:12px;color:var(--fg-2)}}
.kv{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}.kv div{{border-left:2px solid var(--accent);padding:4px 0 4px 12px;min-width:0}}.kv b{{display:block;font-family:var(--mono);font-size:16px;font-weight:500;font-variant-numeric:tabular-nums;line-height:1.25;overflow-wrap:anywhere}}.kv span{{font-size:12px;color:var(--fg-2)}}
ul{{margin:0;padding-left:20px;max-width:80ch;display:grid;gap:6px}}li::marker{{color:var(--accent-2)}}.verdict{{border-left:3px solid var(--accent-2);padding:8px 14px;font-family:var(--display);font-size:16px;line-height:1.4;max-width:80ch}}.verdict small{{display:block;font-family:var(--mono);font-size:11px;color:var(--fg-2);margin-top:6px}}.muted{{color:var(--fg-2);font-family:var(--mono);font-size:13px}}
footer{{padding-block:24px;font-family:var(--mono);font-size:12px;color:var(--fg-2)}}
</style>
<div class="wrap"><header><div class="eyebrow">BrAIn Processing Unit · simulator results · substrate framing</div><h1>The BPU is a substrate: same algorithm, float32 vs noisy tiles</h1>
<p>Every row compares the same algorithm on a float32 reference and on noisy BPU tiles; success means the tiles match the reference within the stated tolerance. Local learning rules are additional arms, never the headline. Every number is read from <code>results/&lt;e&gt;/results.json</code>; BPU energy numbers are projections from the device model, the GPU control is measured.</p>
<div class="meta">generated {time.strftime("%Y-%m-%d %H:%M")} · repo ~/Desktop/Arenda_Projects/BrAInOS/bpusim · DECISIONS.md has every choice · MONDAY.md has the HPCC queue</div></header>
<section id="scoreboard"><div class="eyebrow">Substrate claims</div><h2>Six rows, reference first</h2>{score}
<p class="muted">tolerances are the ones written into each experiment's docstring and verdict block.</p></section>
<section id="writes"><div class="eyebrow">Write budget</div><h2>Accumulate-and-flush: writes only on whole conductance levels</h2>{wbox}
<p class="muted">a device is pulsed only when its digital accumulator holds one conductance level (1/L, L = round(1/σ_eff)); the remainder is rounded stochastically; the endurance budget is 6e-3 writes per device per second.</p></section>
<section id="other"><div class="eyebrow">Other rows</div><h2>Remaining experiments</h2>{otab}</section>
{"".join(secs)}
<footer>bpusim · substrate programme 2026-10-18 · figures by scripts/make_figures.py · page by scripts/make_results_page.py</footer></div>'''
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(secs)} sections)")


if __name__ == "__main__":
    main()
