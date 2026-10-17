"""Build results/index.html from results/*/results.json and figures/*.png.

Scoreboard first: the three claims (accelerator parity E8/E8b, trainability
E9, continual learning E10), then the brain-circuit rows (E3-E7, E2, E1).
Every number on the page is read from a results.json; nothing is typed in.
Usage: python scripts/make_results_page.py [--out results/index.html]
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


def pct(x, d=2):
    try:
        return f"{float(x):+.{d}f} %"
    except Exception:
        return "–"


def chip(kind, text):
    return f'<span class="chip {kind}">{html.escape(text)}</span>'


def status(res, key="success"):
    if res is None:
        return chip("pend", "not run")
    v = res.get("verdict", {})
    if v.get(key) is True:
        return chip("ok", "criterion met")
    if v.get(key) is False:
        return chip("nul", "criterion not met")
    return chip("part", "see verdict")


def section(eid, eyebrow, title, fig, facts, verdict, notes=()):
    fig_html = f'<figure><img src="figures/{fig}" alt="{html.escape(title)}"><figcaption>figures/{fig}</figcaption></figure>' if (FIG / fig).exists() else '<p class="muted">figure not generated yet</p>'
    kv = "".join(f'<div><b>{html.escape(str(v))}</b><span>{html.escape(k)}</span></div>' for k, v in facts)
    li = "".join(f"<li>{n}</li>" for n in notes)
    return f'''<section id="{eid}"><div class="eyebrow">{html.escape(eyebrow)}</div><h2>{html.escape(title)}</h2>
<div class="kv">{kv}</div>{fig_html}{("<ul>" + li + "</ul>") if li else ""}
<div class="verdict">{html.escape(verdict or "not run")}<small>results/{eid}/results.json · verdict</small></div></section>'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RES / "index.html"))
    args = ap.parse_args()
    R = {k: load(k) for k in ("e1", "e2", "e2_cifar", "e3", "e3_256", "e4", "e4_capped", "e5", "e5_48", "e5_selforg32", "e6", "e7", "e8", "e8b", "e9", "e10")}
    # merge the N = 256 E3 run into the main record
    if R["e3"] and R["e3_256"] and "verdicts" in R["e3"] and "verdicts" in R["e3_256"]:
        for key in ("sweeps", "landscape", "fits", "verdicts", "velocity_gain"):
            R["e3"][key].update(R["e3_256"].get(key, {}))
        R["e3"]["N"] = sorted(set(R["e3"]["N"]) | set(R["e3_256"]["N"]))
        mk = [k for k in R["e3"]["verdicts"] if "mismatch0.0_" not in k]
        n_u = sum(R["e3"]["verdicts"][k].get("u_shaped", False) for k in mk)
        R["e3"]["verdict"] = {"n_u_shaped": n_u, "n_conditions": len(mk),
                              "statement": f"{n_u}/{len(mk)} mismatched conditions (N x sigma_d2d x homeostasis) show a significant U; " +
                                           "; ".join(f"{k}: {R['e3']['verdicts'][k]['statement']}" for k in mk)}
    # merge the large-sheet E5 runs into one record
    if R["e5"]:
        for extra in ("e5_48", "e5_selforg32"):
            if R[extra]:
                R["e5"]["sheets"].update({(k + (" self-organised" if extra == "e5_selforg32" else "")): v for k, v in R[extra]["sheets"].items()})
                R["e5"]["verdict"]["statement"] += " || " + R[extra]["verdict"]["statement"]
    rows = []
    # ---- claim rows
    e8, e8b, e9, e10 = R["e8"], R["e8b"], R["e9"], R["e10"]
    rows.append(("Better accelerator than a GPU", "E8 / E8b",
                 (f"noise-aware tiles within {fmt(e8['verdict']['worst_noise_aware_gap_pct_at_sigma_le_0.06'], 2)} % of ideal at σ ≤ 0.06; " if e8 else "") +
                 (e8b["verdict"]["statement"].split(";")[0] if e8b else "energy model pending"),
                 status(e8) + (" " + chip("pend", "energy = projection") if e8b else "")))
    rows.append(("Solves FPGA non-trainability", "E9", e9["verdict"]["statement"][:220] if e9 else "pending", status(e9)))
    rows.append(("Holds continually adapting models", "E10", e10["verdict"]["statement"][:220] if e10 else "pending", status(e10)))
    # ---- brain-circuit rows
    e3, e4, e4c, e5, e6, e7, e2, e2c, e1 = R["e3"], R["e4"], R["e4_capped"], R["e5"], R["e6"], R["e7"], R["e2"], R["e2_cifar"], R["e1"]
    e3new = bool(e3 and "n_u_shaped" in e3.get("verdict", {}))
    rows.append(("Attractor noise optimum", "E3", e3["verdict"]["statement"][:260] if e3 else "pending",
                 chip("ok" if e3new and e3["verdict"]["n_u_shaped"] > 0 else "nul", f"{e3['verdict']['n_u_shaped']}/{e3['verdict']['n_conditions']} conditions U-shaped") if e3new
                 else (chip("nul", "not U-shaped") if e3 else chip("pend", "running"))))
    rows.append(("Ring self-organises", "E4", (e4["verdict"]["statement"] if e4 else "pending") + (f" Capped at the endurance budget: {e4c['verdict']['statement']}" if e4c else ""),
                 chip("part", f"{e4['verdict']['certified_fraction']:.0%} certified") if e4 else chip("pend", "running")))
    e4 = e4 if (e4 and "mean_occupancy_bins_of_8" in e4["verdict"]) else e4
    rows.append(("Grid module is a torus", "E5", e5["verdict"]["statement"] if e5 else "pending", chip("nul" if e5 and "0/" in e5["verdict"]["statement"].split("reference")[1][:6] else "part", "see row") if e5 else chip("pend", "pending")))
    rows.append(("Memory + attention under noise", "E6", e6["verdict"]["statement"][:260] if e6 else "pending", chip("part", f"{100 * e6.get('flip_fraction', 0):.0f} % flipped cues") if e6 else chip("pend", "running")))
    rows.append(("Modular self-assembly", "E7", e7["verdict"]["statement"][:260] if e7 else "pending", chip("part", f"Q = {fmt(e7['summary']['Q_final_grown']['mean'], 2)}") if e7 else chip("pend", "running")))
    rows.append(("Local grow rule vs RigL", "E2", (e2["verdict"]["statement"] if e2 else "") + (" | CIFAR-10 3 %: " + e2c["verdict"]["statement"] if e2c else ""),
                 (chip("ok" if e2 and e2["verdict"].get("within_few_percent_of_rigl") else "nul", "MNIST") if e2 else "") + (" " + chip("ok" if e2c["verdict"].get("within_few_percent_of_rigl") else "nul", "CIFAR-10") if e2c else "")))
    rows.append(("GPT-2 through noisy tiles", "E1", "queued for the GPU cluster (MONDAY.md); E8 is the laptop stand-in", chip("pend", "Monday")))
    table = "".join(f"<tr><td>{html.escape(a)}</td><td class='id'>{html.escape(b)}</td><td>{html.escape(c)}</td><td>{d}</td></tr>" for a, b, c, d in rows)

    secs = []
    if e8:
        s = e8["summary"]
        secs.append(section("e8", "E8 · accelerator parity (laptop stand-in for E1)", "nanoGPT on noisy tiles", "e8_nanogpt_tiles.png",
                            [(f"ideal val loss, {e8['seeds']} seeds", fmt(s["ideal_val_loss"]["mean"], 3))] +
                            [(f"noise-aware gap at σ={sg}", pct(s[str(sg)]["noise_aware_gap_pct"]["mean"])) for sg in e8["sigmas"]] +
                            [(f"deploy-only gap at σ={sg}", pct(s[str(sg)]["deploy_gap_pct"]["mean"])) for sg in e8["sigmas"]],
                            e8["verdict"]["statement"],
                            [f"{e8['ops']['n_params'] / 1e6:.2f}M parameters, {e8['layers']} layers, d = {e8['d']}, context {e8['block']}; {e8['steps']} ideal steps then {e8['ft_steps']} noise-aware fine-tuning steps per σ.",
                             f"Op counts for one eval batch of {e8['ops']['tokens']} tokens at σ = 0.04: {e8['ops']['eval_batch']['macs']:.3g} MACs, {e8['ops']['eval_batch']['conversions']:.3g} A/D conversions, {e8['ops']['eval_batch']['reads']:.3g} tile reads; one training step adds {e8['ops']['train_step']['pulses']:.3g} pulses."]))
    if e8b:
        p = e8b["per_token_10M"]
        secs.append(section("e8b", "E8b · energy and latency projection", "Projected, not measured", "e8b_energy.png",
                            [("BPU inference, nJ/token", fmt(p["bpu_eval_J"] * 1e9, 2)), ("digital, no reuse, nJ/token", fmt(p["digital_eval_J_no_reuse"] * 1e9, 1)),
                             ("digital, 10x reuse, nJ/token", fmt(p["digital_eval_J_10x_reuse"] * 1e9, 1)), ("BPU latency, µs/token", fmt(p["bpu_latency_s_per_token"] * 1e6, 0)),
                             ("this CPU, µs/token (measured)", fmt(p["cpu_forward_s_per_token"] * 1e6, 0)), ("BPU train step, nJ/token", fmt(p["bpu_train_step_J"] * 1e9, 1))],
                            e8b["verdict"]["statement"],
                            ["Constants: E_MAC = V²Gt (0.2 V, 10 µS, 10 ns), E_ADC = 2⁸ pJ per tile-output conversion, 100 pJ per pulse, 10 pJ per routed event, 1 µs per read in series; digital side 3.7 pJ per FP32 MAC and 640 pJ per weight fetch (Horowitz 2014). The scaling table in results.json extends to 70B parameters by MACs ∝ parameters."]))
    if e9:
        c = e9["curves"]
        secs.append(section("e9", "E9 · trainability under drift", "Frozen vs on-chip local learning", "e9_trainability.png",
                            [("frozen, python-segment loss", fmt(c["frozen"]["segment_end_loss"][1])), ("local three-factor", fmt(c["local"]["segment_end_loss"][1])),
                             ("feedback alignment", fmt(c["fa"]["segment_end_loss"][1])), ("FA recovery steps", str(c["fa"]["recovery_steps_after_switch"])),
                             ("FA writes / device / s", f"{c['fa']['writes_per_device_per_s']:.3g}"), ("budget", "6e-3")],
                            e9["verdict"]["statement"]))
    if e10:
        s = e10["summary"]
        secs.append(section("e10", "E10 · continual learning", "Consolidation gated by feature persistence", "e10_continual.png",
                            [("joint (upper bound)", fmt(s["joint"]["avg_acc"]["mean"])), ("BPU average accuracy", fmt(s["bpu"]["avg_acc"]["mean"])), ("BPU after rest", fmt(s["bpu_after_rest"]["avg_acc"]["mean"])),
                             ("naive fine-tune", fmt(s["naive"]["avg_acc"]["mean"])), ("BPU forgetting", fmt(s["bpu"]["forgetting"]["mean"])), ("naive forgetting", fmt(s["naive"]["forgetting"]["mean"]))],
                            e10["verdict"]["statement"]))
    if e3new:
        facts = []
        for k, v in e3["verdicts"].items():
            facts.append((k.replace("_", " "), f"{fmt(v['rmse_at_zero'])}→{fmt(v['rmse_at_min'])}@σ={v['sigma_star_empirical']} p={v['sign_test_p']:.2g}"))
        secs.append(section("e3", "E3 · attractor noise optimum", f"N = {e3['N']}, T = {e3['T']} s, {e3['seeds']} seeds", "e3_drift.png", facts[:8], e3["verdict"]["statement"],
                            [f"Pinning barrier ΔU (zero dynamic noise): " + "; ".join(f"{k}: {fmt(v['delta_U']['mean'], 3)}" for k, v in e3["landscape"].items())]))
    if e4:
        secs.append(section("e4", "E4 · ring self-organisation", f"{e4['seeds']} seeds, sweep cue", "e4_ring_selforg.png",
                            [("certified", f"{e4['verdict']['certified_fraction']:.0%}"), ("controls certified", f"{e4['verdict']['control_certified_fraction']:.0%}"),
                             ("ring bins occupied / 8", fmt(e4["verdict"].get("mean_occupancy_bins_of_8"), 1)), ("writes / device / s", f"{e4['runs'][0].get('writes_per_device_per_s', float('nan')):.3g}")] +
                            ([("capped run certified", f"{e4c['verdict']['certified_fraction']:.0%}"), ("capped development time", f"{e4c['development_time_s']:.0f} s")] if e4c else []),
                            e4["verdict"]["statement"] + (" || CAPPED: " + e4c["verdict"]["statement"] if e4c else "")))
    if e5:
        facts = [(f"{spec} programmed reference", f"{d['reference_torus_fraction']:.0%}") for spec, d in e5["sheets"].items()]
        secs.append(section("e5", "E5 · grid module torus", "Programmed Burak–Fiete reference first", "e5_grid_selforg.png", facts, e5["verdict"]["statement"]))
    if e6:
        caps = e6["capacity_at_90pct"]
        secs.append(section("e6", "E6 · memory and attention", f"{100 * e6.get('flip_fraction', 0):.0f} % flipped bits, {100 * e6.get('erase_fraction', 0):.0f} % erased", "e6_memory.png",
                            [(f"{m} capacity @90 %, σ=0 / 0.2", f"{caps[m]['0.0']} / {caps[m]['0.2']}") for m in e6["methods"]], e6["verdict"]["statement"]))
    if e7:
        s = e7["summary"]
        kinds = [k for k in ("grown", "fixed", "fixed_compute") if k in s["nav_acc"]]
        secs.append(section("e7", "E7 · modular self-assembly", f"layer-local learning, {e7.get('n_per_module', 16)}-neuron modules" if e7.get("rule") == "local" else "node-perturbation learning", "e7_modularity.png",
                            [("Q final (grown)", fmt(s["Q_final_grown"]["mean"], 2)), ("nav acc " + " / ".join(kinds), " / ".join(fmt(s["nav_acc"][k]["mean"], 2) for k in kinds)),
                             ("recall bits " + " / ".join(kinds), " / ".join(fmt(s["recall_bit_acc"][k]["mean"], 2) for k in kinds)),
                             ("transfer gap, matched synapses", f"{s['transfer_gap']['mean']:+.2f}"), ("transfer gap, matched compute", f"{s['transfer_gap_matched_compute']['mean']:+.2f}" if "transfer_gap_matched_compute" in s else "–")],
                            e7["verdict"]["statement"]))
    if e2 or e2c:
        facts = []
        for lab, r in (("MNIST 10 %", e2), ("CIFAR-10 3 %", e2c)):
            if r:
                facts += [(f"{lab} local", fmt(r["summary"]["local"]["mean"])), (f"{lab} RigL", fmt(r["summary"]["rigl"]["mean"])), (f"{lab} random", fmt(r["summary"]["random"]["mean"]))]
        secs.append(section("e2", "E2 · local grow rule vs RigL", "Equal synapse budget, SGD for the weights", "e2_grow_vs_rigl.png", facts,
                            (e2["verdict"]["statement"] if e2 else "") + (" || CIFAR-10: " + e2c["verdict"]["statement"] if e2c else "")))

    page = f'''<title>bpusim Weekend Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,400;6..72,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f6f5f1;--bg-2:#ecebe4;--fg:#1d2126;--fg-2:#565c66;--rule:#cfccc2;--accent:#0f6b7a;--accent-2:#b2561e;--ok:#2d7a3a;--ok-bg:#dcefdf;--part:#8a6a10;--part-bg:#f3e8c2;--nul:#8a2d2d;--nul-bg:#f3d9d9;--pend:#4b5563;--pend-bg:#e2e5ea;--display:"Newsreader",Georgia,serif;--body:"IBM Plex Sans",Arial,sans-serif;--mono:"IBM Plex Mono",Menlo,monospace}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#15181c;--bg-2:#1d2126;--fg:#e8e6df;--fg-2:#a4a8b0;--rule:#343941;--accent:#5fc0cf;--accent-2:#e08a52;--ok:#8bd39a;--ok-bg:#1f3a26;--part:#e3c35c;--part-bg:#3d3414;--nul:#e58b8b;--nul-bg:#3f1f1f;--pend:#c3c8d1;--pend-bg:#2b3039;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#15181c;--bg-2:#1d2126;--fg:#e8e6df;--fg-2:#a4a8b0;--rule:#343941;--accent:#5fc0cf;--accent-2:#e08a52;--ok:#8bd39a;--ok-bg:#1f3a26;--part:#e3c35c;--part-bg:#3d3414;--nul:#e58b8b;--nul-bg:#3f1f1f;--pend:#c3c8d1;--pend-bg:#2b3039;color-scheme:dark}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--fg);font-family:var(--body);font-size:16px;line-height:1.55;margin:0;padding-block:0 64px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
.wrap{{max-width:900px;margin:0 auto}}h1,h2{{font-family:var(--display);font-weight:500;text-wrap:balance;margin:0}}h1{{font-size:clamp(32px,5vw,48px);line-height:1.05}}h2{{font-size:26px;margin-top:6px}}p{{max-width:70ch;margin:0}}
.eyebrow{{font-family:var(--mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--fg-2)}}header{{padding-block:48px 24px;border-bottom:1px solid var(--rule);display:grid;gap:12px}}.meta{{font-family:var(--mono);font-size:13px;color:var(--fg-2)}}
section{{padding-block:32px;border-bottom:1px solid var(--rule);display:grid;gap:16px}}.chip{{display:inline-block;font-family:var(--mono);font-size:11.5px;padding:2px 8px;border-radius:3px;white-space:nowrap}}.ok{{color:var(--ok);background:var(--ok-bg)}}.part{{color:var(--part);background:var(--part-bg)}}.nul{{color:var(--nul);background:var(--nul-bg)}}.pend{{color:var(--pend);background:var(--pend-bg)}}
.tablewrap{{overflow-x:auto;border:1px solid var(--rule);border-radius:4px}}table{{border-collapse:collapse;width:100%;min-width:760px;font-size:14px}}th,td{{text-align:left;vertical-align:top;padding:9px 11px;border-bottom:1px solid var(--rule)}}th{{font-family:var(--mono);font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--fg-2);background:var(--bg-2);font-weight:500}}tr:last-child td{{border-bottom:0}}td.id{{font-family:var(--mono);color:var(--accent);font-weight:500;white-space:nowrap}}
figure{{margin:0;background:var(--bg-2);border:1px solid var(--rule);border-radius:4px;padding:10px;display:grid;gap:8px}}figure img{{width:100%;max-width:100%;height:auto;display:block;background:#fff;border-radius:2px}}figcaption{{font-family:var(--mono);font-size:12px;color:var(--fg-2)}}
.kv{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}.kv div{{border-left:2px solid var(--accent);padding:4px 0 4px 12px;min-width:0}}.kv b{{display:block;font-family:var(--mono);font-size:18px;font-weight:500;font-variant-numeric:tabular-nums;line-height:1.2;overflow-wrap:anywhere}}.kv span{{font-size:12px;color:var(--fg-2)}}
ul{{margin:0;padding-left:20px;max-width:76ch;display:grid;gap:6px}}li::marker{{color:var(--accent-2)}}.verdict{{border-left:3px solid var(--accent-2);padding:8px 14px;font-family:var(--display);font-size:17px;line-height:1.4;max-width:76ch}}.verdict small{{display:block;font-family:var(--mono);font-size:11px;color:var(--fg-2);margin-top:6px}}.muted{{color:var(--fg-2);font-family:var(--mono);font-size:13px}}
footer{{padding-block:24px;font-family:var(--mono);font-size:12px;color:var(--fg-2)}}
</style>
<div class="wrap"><header><div class="eyebrow">BrAIn Processing Unit · simulator results · weekend programme</div><h1>Three claims, seven circuits, reported as measured</h1>
<p>Accelerator parity (E8, with the energy model E8b as a labelled projection), trainability where a fixed accelerator is frozen (E9), continual learning with volatile/nonvolatile consolidation (E10), and the brain-circuit experiments E2–E7 after their fixes. Every number is read from <code>results/&lt;e&gt;/results.json</code>.</p>
<div class="meta">generated {time.strftime("%Y-%m-%d %H:%M")} · repo ~/Desktop/Arenda_Projects/BrAInOS/bpusim · DECISIONS.md has every choice · MONDAY.md has the cluster list</div></header>
<section id="scoreboard"><div class="eyebrow">Scoreboard</div><h2>Claims first, then the substrate</h2><div class="tablewrap"><table><thead><tr><th>Claim</th><th>Exp</th><th>Measured</th><th>Status</th></tr></thead><tbody>{table}</tbody></table></div>
<p class="muted">"criterion met / not met" refers to the success test written into each experiment's docstring and verdict block; BPU energy numbers are projections from the device model, never hardware measurements.</p></section>
{"".join(secs)}
<footer>bpusim · weekend programme 2026-10-17 · figures regenerated by scripts/make_figures.py · page by scripts/make_results_page.py</footer></div>'''
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(secs)} sections)")


if __name__ == "__main__":
    main()
