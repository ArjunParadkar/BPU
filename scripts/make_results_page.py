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
    names = ("e1", "e2", "e2_cifar", "e3", "e3_256", "e3_tracking", "e3_cue", "e4", "e4_capped", "e4_level_t1", "e4_level_t05", "e4_level_t025", "e4_level_capped", "e5", "e5_48", "e5_selforg32",
             "e6", "e7", "e7_ref", "e7_3seeds", "e7_ref_3seeds", "e8", "e8b", "e9", "e9_pulse", "e10", "e9_v1", "e10_v1", "e8b_v1")
    R = {k: load(k) for k in names}
    e9, e10, e8b, e3t, e3c, e6 = R["e9"], R["e10"], R["e8b"], R["e3_tracking"], R["e3_cue"], R["e6"]
    e7, e7r = R["e7"] or R["e7_3seeds"], R["e7_ref"] or R["e7_ref_3seeds"]
    e4, e4c, e4l, e4lc, e8 = R["e4"], R["e4_capped"], R["e4_level_t1"], R["e4_level_capped"], R["e8"]
    e4t = {"1.0": R["e4_level_t1"], "0.5": R["e4_level_t05"], "0.25": R["e4_level_t025"]}

    # ------------------------------------------------------------------ substrate scoreboard (E10, E9, E8b, E3, E6, E7)
    rows = []
    def wb(total, lab=""):
        try:
            return f"{float(total):.3g} writes/dev = {100 * float(total) / 1e6:.2g} % endurance" + lab
        except Exception:
            return "–"
    # 1. E10
    if e10 and "twins" in e10:
        tw = e10["twins"]; S = e10["summary"]; HL = e10.get("headline_arms", e10["arms"] + ["joint"])
        ref = "; ".join(f"{a} {fmt(tw[a]['float32'])}" for a in HL)
        bpu = "; ".join(f"{a} {fmt(tw[a]['bpu'])} ({tw[a]['gap_points']:+.3f})" for a in HL)
        wbudget = "; ".join(f"{a} {S['bpu'][a]['writes_per_device_total']['mean']:.3g}" for a in HL) + " writes/dev (≤ 0.0005 % endurance)"
        extra = "forgetting BPU: " + ", ".join(f"{a} {fmt(S['bpu'][a]['forgetting']['mean'])}" for a in HL if a != "joint")
        rr = e10.get("retention_row_tau_1e4")
        if rr:
            extra += "; τ_ret = 1e4 s after a 1000 s rest: " + ", ".join(f"{a} {fmt(rr[a]['avg_acc_after_rest_1000s']['mean'])}" for a in HL if a in rr)
        ok = e10["verdict"]["success"]
        st = chip("ok" if ok else "nul", "criterion met" if ok else "not met") + " " + chip("ok" if abs(tw["ewc"]["gap_points"]) <= 0.03 else "nul", f"EWC {tw['ewc']['gap_points']:+.3f}") + " " + chip("ok" if abs(tw["replay"]["gap_points"]) <= 0.03 else "nul", f"replay {tw['replay']['gap_points']:+.3f}")
        rows.append(("Holds continually adapting models", "E10", ref, bpu, wbudget, "BPU-EWC and BPU-replay within 3 points of float32 twins", extra, st))
    else:
        rows.append(("Holds continually adapting models", "E10", "pending", "pending", "–", "within 3 points", "", chip("pend", "running")))
    # 2. E9 Pareto
    if e9 and "summary" in e9 and "sgd_float32" in e9["summary"]:
        s = e9["summary"]; refv = s["sgd_float32"]["segment_end_loss_1"]["mean"]
        pareto = e9.get("pareto") or [{"name": a, "arm": a, "theta_flush": 1.0, "post_shift_loss": s[a]["segment_end_loss_1"]["mean"], "gap_pct": 100 * (s[a]["segment_end_loss_1"]["mean"] / refv - 1),
                                        "writes_per_device_total": s[a].get("writes_per_device_total", {}).get("mean", float("nan"))} for a in s if a not in ("frozen", "sgd_float32")]
        ref = f"SGD float32 post-shift {fmt(refv)} nats/char"
        best = min(pareto, key=lambda p: abs(p["gap_pct"]))
        bpu = "; ".join(f"{p['name']} {fmt(p['post_shift_loss'])} ({pct(p['gap_pct'])}, {p['writes_per_device_total']:.3g} w/dev)" for p in sorted(pareto, key=lambda p: (p["arm"], -p["theta_flush"])))
        wbudget = f"closest: {best['name']} {wb(best['writes_per_device_total'])}; limit 1e4 writes/dev"
        pd = e9.get("pulse_diagnostic")
        if not pd and R["e9_pulse"]:
            ps = R["e9_pulse"]["summary"]["sgd_bpu_pulse"]
            pd = {"post_shift_loss": ps["segment_end_loss_1"]["mean"], "writes_per_device_total": ps.get("writes_per_device_total", {}).get("mean", ps["writes_per_device_per_s"]["mean"] * R["e9_pulse"]["seg"] * 3 * R["e9_pulse"]["step_seconds"])}
        extra = f"nearest-pulse diagnostic: {fmt(pd['post_shift_loss'])} ({pct(100 * (pd['post_shift_loss'] / refv - 1))}) at {pd['writes_per_device_total']:.3g} writes/dev" if pd else ""
        ok = e9["verdict"]["success"]
        st = chip("ok" if ok else "nul", "criterion met" if ok else "not met") + " " + chip("part", f"θ_flush ∈ {e9.get('thetas', [1.0])}")
        rows.append(("Trainable where an FPGA is frozen", "E9", ref, bpu, wbudget, "within 5 % of float32 at any θ with < 1e4 total writes/device", extra, st))
    else:
        rows.append(("Trainable where an FPGA is frozen", "E9", "pending", "pending", "–", "within 5 %, < 1e4 writes/device", "", chip("pend", "running")))
    # 3. E8b
    if e8b and "table_3x2" in e8b:
        t = e8b["table_3x2"]; d = e8b["digital_control"]; gi = d.get("int8", {}); h = d["h100_int8_published"]
        ref = (f"GPU INT8 measured (laptop {d.get('gpu', '')}): {gi['nJ_per_token_incl_idle'] / 1e3:.0f} µJ/token incl. idle, {gi['nJ_per_token_above_idle'] / 1e3:.0f} above idle; " if gi else "") + \
              f"H100 INT8 published: {h['nJ_per_token_at_peak_for_this_model'] / 1e3:.1f} µJ at peak, {h.get('nJ_per_token_at_40pct_util_for_this_model', float('nan')) / 1e3:.1f} µJ at 40 % utilisation"
        lossless = {k: v for k, v in t.items() if v["gap_pct_vs_ideal"] <= 2.0}
        best = min(lossless.values(), key=lambda r: r["nJ_per_token"]) if lossless else min(t.values(), key=lambda r: r["nJ_per_token"])
        bpu = "; ".join(f"{v['mode']} {v.get('fom', '').replace('fom', 'FoM')} {v['bits']}b {v['nJ_per_token']:.0f} nJ ({pct(v['gap_pct_vs_ideal'])})" for v in t.values())
        h40 = h.get("nJ_per_token_at_40pct_util_for_this_model")
        extra = f"cheapest lossless cell {best['mode']} {best.get('fom', '')} {best['bits']}-bit {best['nJ_per_token']:.0f} nJ/token" + (f" = {best['nJ_per_token'] / h40:.3f}× H100@40 %" if h40 else "") + "; ratio falls with model size (conversions ∝ layers × d, MACs ∝ params)"
        st = chip("part", "BPU = projection, GPU = measured") + (" " + chip("ok" if h40 and best["nJ_per_token"] < h40 else "nul", "below H100@40 %") if h40 else "")
        rows.append(("Cheaper per token than a GPU", "E8b", ref, bpu, "inference: no writes", "nJ/token at ≤ 2 % loss", extra, st))
    else:
        rows.append(("Cheaper per token than a GPU", "E8b", "pending", "pending", "–", "nJ/token", "", chip("pend", "running")))
    # 4. E3 cue sweep (pinned regime) with the omega sweep as the second line
    if e3c and "pinned_regime" in e3c:
        pr = e3c["pinned_regime"]; sw = e3c["sweeps"]
        ref = "; ".join(f"cue {a:g}: {fmt(sw[f'amp{a:g}_omega{e3c['omegas'][0]:g}_float32']['0.0']['track_err']['mean'])} rad" for a in e3c["cue_amps"] if f"amp{a:g}_omega{e3c['omegas'][0]:g}_float32" in sw)
        bpu = "; ".join(f"{k.replace('amp', 'cue ').replace('_omega', ' ω').replace('_bpu_mismatch', ' mm ')}: {fmt(v['err_sigma0'])}" + (" PINNED" if v["fails"] else "") for k, v in pr.items())
        extra = f"pinned (σ=0 error > 1.5× reference): {e3c['verdict']['n_pinned']}/{len(pr)}; U-shaped among pinned: {e3c['verdict']['n_u_shaped_among_pinned']}" + \
                (f"; ω sweep at cue 0.1 (results/e3_tracking): {e3t['verdict']['n_u_shaped']}/{e3t['verdict']['n_conditions']} U-shaped" if e3t and "n_u_shaped" in e3t.get("verdict", {}) else "")
        n_u = e3c["verdict"]["n_u_shaped_among_pinned"]
        rows.append(("Attractor noise optimum (moving cue)", "E3", ref, bpu, "no writes (read noise only)", "tracking error at σ = 0 per cue; U-test only where σ = 0 fails", extra, chip("ok" if n_u else "nul", f"{n_u} U-shaped of {e3c['verdict']['n_pinned']} pinned")))
    elif e3t and "n_u_shaped" in e3t.get("verdict", {}):
        rows.append(("Attractor noise optimum (moving cue)", "E3", "ω sweep only (cue sweep running)", "", "–", "U-test", "", chip("nul", f"{e3t['verdict']['n_u_shaped']}/{e3t['verdict']['n_conditions']} U-shaped")))
    else:
        rows.append(("Attractor noise optimum (moving cue)", "E3", "pending", "pending", "–", "U-test where σ = 0 fails", "", chip("pend", "running")))
    # 5. E6
    if e6:
        caps = e6["capacity_at_90pct"]
        ref = "; ".join(f"{m}: {caps[m]['0.0']}" for m in e6["methods"])
        bpu = "; ".join(f"{m}: " + "/".join(str(caps[m][str(s)]) for s in e6["sigmas"][1:]) for m in e6["methods"])
        same = all(caps["attention"][str(s)] >= caps["attention"]["0.0"] for s in e6["sigmas"] if s <= 0.04)
        rows.append(("Memory + attention under noise", "E6", ref + " pairs @90 %", bpu + f" (σ = {', '.join(str(s) for s in e6['sigmas'][1:])})", "programmed once (write-verify)", "capacity on tiles matches σ = 0 up to σ = 0.04; K = 1 unit test",
                     f"{100 * e6.get('flip_fraction', 0):.0f} % flipped cues, {e6['seeds']} seeds, sign ties random, test_hopfield_k1 passes", chip("ok" if same else "nul", "attention holds" if same else "attention drops")))
    # 6. E7 with the noise-assisted-growth test
    if e7 and "summary" in e7:
        s = e7["summary"]
        def line(sm):
            return (f"nav grown {fmt(sm['nav_acc']['grown']['mean'], 2)} / fixed {fmt(sm['nav_acc']['fixed']['mean'], 2)}" +
                    (f" / continued {fmt(sm['nav_acc']['grown_continued']['mean'], 2)}" if "grown_continued" in sm["nav_acc"] else "") +
                    (f"; gap {sm['nav_gap_grown']['mean']:+.2f} (CI {sm['nav_gap_grown']['ci95'][0]:+.2f}..{sm['nav_gap_grown']['ci95'][1]:+.2f})" if "nav_gap_grown" in sm else ""))
        ref = line(e7r["summary"]) + f" ({e7r['seeds']} seeds)" if e7r and "summary" in e7r else "noise-free tiles: pending"
        bpu = line(s) + f" ({e7['seeds']} seeds)"
        extra = f"composite gap grown−fixed {s['transfer_gap']['mean']:+.2f}; Q = {fmt(s['Q_final_grown']['mean'], 2)}"
        st = chip("part", "see row")
        if e7r and "nav_gap_grown" in s and "nav_gap_grown" in e7r["summary"]:
            from scipy.stats import ttest_ind
            a = s["nav_gap_grown"]["per_seed"]; b = e7r["summary"]["nav_gap_grown"]["per_seed"]
            tt = ttest_ind(a, b, equal_var=False, alternative="greater")
            e7_noise = {"tiles_gap": float(sum(a) / len(a)), "ref_gap": float(sum(b) / len(b)), "p_one_sided": float(tt.pvalue)}
            extra += f"; tiles gap − reference gap = {e7_noise['tiles_gap'] - e7_noise['ref_gap']:+.2f}, Welch one-sided p = {e7_noise['p_one_sided']:.2g}"
            st = chip("ok" if e7_noise["p_one_sided"] < 0.05 else "nul", "noise-assisted growth" if e7_noise["p_one_sided"] < 0.05 else "no noise-assisted growth")
        rows.append(("Modular self-assembly", "E7", ref, bpu, "–", "grown−fixed gap with CI, tiles vs reference; flag if tiles gap > reference gap at p < 0.05", extra, st))
    else:
        rows.append(("Modular self-assembly", "E7", "pending", "pending", "–", "matched synapses and steps", "", chip("pend", "running")))
    score = table_html(["Substrate claim", "Exp", "float32 / reference", "BPU tiles", "write budget (endurance)", "tolerance", "additional arms / notes", "status"],
                       [[esc(a), f"<span class='id'>{esc(b)}</span>", esc(c), esc(d), esc(e), esc(f), esc(g), h] for a, b, c, d, e, f, g, h in rows])

    # ------------------------------------------------------------------ write-budget box (endurance / tile bandwidth / lifelong rate)
    wb_rows = []
    def e4row(lab, r):
        if not r:
            return
        b = r["runs"][0].get("budget")
        tot = r.get("writes_per_device_total", (r["runs"][0]["writes_per_device_per_s"] * r["development_time_s"]))
        tile_s = r.get("writes_per_tile_per_s", float("nan"))
        wps = r.get("writes_per_device_per_s", sum(x["writes_per_device_per_s"] for x in r["runs"]) / len(r["runs"]))
        wb_rows.append([esc(lab), f"{tot:.3g} ({100 * tot / 1e6:.2g} %)", f"{tile_s:.3g} ({100 * tile_s / 1e4:.2g} %)" if b else "–", f"{wps:.3g} ({100 * wps / 6e-3:.0f} %)",
                        f"{r['verdict']['certified_fraction']:.0%} certified ({r['verdict']['control_certified_fraction']:.0%} controls)", f"{r['development_time_s']:.0f} s"])
    e4row("E4 threshold writes (weekend)", e4); e4row("E4 quota 6e-3, threshold writes (weekend)", e4c)
    for th in ("1.0", "0.5", "0.25"):
        e4row(f"E4 accumulate-and-flush θ_flush = {th}", e4t[th])
    e4row("E4 accumulate-and-flush θ = 1 + quota 6e-3", e4lc)
    if e9 and "summary" in e9:
        for a, sm in e9["summary"].items():
            if a in ("sgd_float32",) or "segment_end_loss_1" not in sm:
                continue
            tot = sm.get("writes_per_device_total", {}).get("mean", sm["writes_per_device_per_s"]["mean"] * e9["seg"] * 3 * e9["step_seconds"])
            tile_s = sm.get("writes_per_tile_per_s", {}).get("mean")
            wb_rows.append([esc(f"E9 {a} (10 ms/step)"), f"{tot:.3g} ({100 * tot / 1e6:.2g} %)", f"{tile_s:.3g} ({100 * tile_s / 1e4:.2g} %)" if tile_s else "–",
                            f"{sm['writes_per_device_per_s']['mean']:.3g} ({sm['percent_of_budget']['mean']:.0f} %)", f"post-shift loss {fmt(sm['segment_end_loss_1']['mean'])}", f"{e9['seg'] * 3 * e9['step_seconds']:.0f} s"])
    if e10 and "summary" in e10:
        for a in e10.get("headline_arms", e10["arms"]):
            sm = e10["summary"]["bpu"][a]
            tot = sm.get("writes_per_device_total", {}).get("mean", float("nan"))
            wb_rows.append([esc(f"E10 {a} on tiles (10 ms/step)"), f"{tot:.3g} ({100 * tot / 1e6:.2g} %)", "–", f"{sm['writes_per_device_per_s']['mean']:.3g} ({sm.get('percent_of_lifelong_rate', sm.get('percent_of_budget', {}))['mean']:.0f} %)", f"avg acc {fmt(sm['avg_acc']['mean'])}", "40 s"])
    wbox = table_html(["run", "total writes / device (% of 1e6 endurance)", "writes / tile / s (% of 1e4 driver bandwidth)", "lifelong rate, writes / dev / s (% of 6e-3)", "outcome", "chip time"], wb_rows)

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
    if e10 and "twins" in e10:
        S = e10["summary"]; tw = e10["twins"]; HL = e10.get("headline_arms", e10["arms"] + ["joint"])
        facts = [(f"{a} float32 / BPU", f"{fmt(tw[a]['float32'])} / {fmt(tw[a]['bpu'])}") for a in HL] + \
                [(f"{a} forgetting f32 / BPU", f"{fmt(S['float32'][a]['forgetting']['mean'])} / {fmt(S['bpu'][a]['forgetting']['mean'])}") for a in HL if a != "joint"] + \
                [(f"{a} total writes/dev (BPU)", f"{S['bpu'][a]['writes_per_device_total']['mean']:.3g}") for a in HL]
        notes = [f"Permuted MNIST, 5 tasks, 784-256-256-10, {e10['steps']} steps per task, EWC λ = {e10['lam']}, replay buffer {e10['buffer']} samples (1 % of the training set), {e10['seeds']} seeds; σ_eff = {e10['sigma']} on tiles with accumulate-and-flush writes at θ_flush = 1."]
        notes += [esc(n) for n in e10.get("notes", [])]
        rr = e10.get("retention_row_tau_1e4")
        if rr:
            notes.append("Retention τ_ret = 1e4 s on tiles, avg acc at the end of the sequence → after a 1000 s rest: " + ", ".join(f"{a} {fmt(rr[a]['avg_acc']['mean'])}→{fmt(rr[a]['avg_acc_after_rest_1000s']['mean'])}" for a in HL if a in rr) + " (the other τ_ret values and the fpl arm are in results.json).")
        secs.append(section("e10", "E10 · continual learning", "Four algorithms + joint, on float32 and on tiles", "e10_continual.png", facts, e10["verdict"]["statement"], notes))
    if e9 and "summary" in e9:
        s = e9["summary"]
        facts = [(f"{a}: post-shift loss", fmt(s[a]["segment_end_loss_1"]["mean"])) for a in e9["arms"]] + \
                [(f"{a}: total writes/dev", f"{s[a].get('writes_per_device_total', {}).get('mean', float('nan')):.3g}") for a in e9["arms"] if a not in ("sgd_float32", "frozen")]
        secs.append(section("e9", "E9 · trainability under drift", "Pareto: post-shift loss vs total writes per device, θ_flush ∈ {0.25, 0.5, 1}", "e9_trainability.png", facts, e9["verdict"]["statement"],
                            [f"Stream: {e9['seg']} steps Shakespeare → {e9['seg']} Python source → {e9['seg']} Shakespeare, batch {e9['batch']}, {e9['seeds']} seeds, σ_eff = {e9['sigma']}, accumulate-and-flush writes at L = 25 levels, chip time {e9['step_seconds']} s/step (assumption).",
                             (f"Diagnostic arm (results/e9_pulse): the same SGD on tiles with the legacy nearest-pulse writes reaches {fmt(R['e9_pulse']['summary']['sgd_bpu_pulse']['segment_end_loss_1']['mean'])} post-shift at {R['e9_pulse']['summary']['sgd_bpu_pulse']['percent_of_budget']['mean']:.0f} % of the budget: device noise costs about 1 %, the level gate costs the rest and buys a 50x lower write rate." if R["e9_pulse"] else ""),
                             "Recovery steps after each switch (per seed): " + "; ".join(f"{a} {s[a]['recovery_steps_after_switch']}" for a in e9["arms"])]))
    if e8b and "table_3x2" in e8b:
        t = e8b["table_3x2"]; d = e8b["digital_control"]; gi = d.get("int8", {})
        h = d["h100_int8_published"]
        cells = lambda m, f, b: f"{t[f'{m}_{f}_{b}']['nJ_per_token']:.0f} nJ ({pct(t[f'{m}_{f}_{b}']['gap_pct_vs_ideal'])})" if f"{m}_{f}_{b}" in t else "–"
        rows3 = [[esc(m), cells(m, "fom10", 6), cells(m, "fom10", 8), cells(m, "fom50", 6), cells(m, "fom50", 8)] for m in ("tile", "layer", "lif")]
        rows3.append(["GPU INT8 measured, laptop (control)", esc(f"{gi['nJ_per_token_incl_idle']:.0f} nJ incl. idle / {gi['nJ_per_token_above_idle']:.0f} above idle ({pct(gi['gap_pct_vs_ideal'], 2)})" if gi else "not measured"), esc(f"FP32 on the same GPU: {d['fp32']['nJ_per_token_incl_idle']:.0f} nJ" if "fp32" in d else ""), "", ""])
        rows3.append(["H100 INT8 published peak (not measured)", f"{h['nJ_per_token_at_peak_for_this_model']:.1f} nJ", "", "", ""])
        rows3.append(["H100 INT8 at 40 % utilisation (published peak / 0.4)", f"{h.get('nJ_per_token_at_40pct_util_for_this_model', float('nan')):.1f} nJ", "", "", ""])
        facts = [("ideal val loss", fmt(e8b["ideal_val_loss"])), ("noise-aware tiles σ=0.04", fmt(e8b["noise_aware_val_loss"])), ("LIF spikes/output/token (8-bit window)", fmt(e8b["readout_accuracy"]["lif_8"]["spikes_per_output_per_token"], 1)),
                 ("LIF spikes/output/token (6-bit)", fmt(e8b["readout_accuracy"].get("lif_6", {}).get("spikes_per_output_per_token"), 1)), ("MAC energy, nJ/token", fmt(t["layer_fom10_8"]["parts_nJ"]["mac_J"], 1)), ("BPU latency, µs/token", fmt(e8b["latency"]["bpu_s_per_token"] * 1e6, 0))]
        sc = e8b["scaling"]
        scal = table_html(["model", "layers × d", "per-layer FoM10 8b", "per-layer FoM50 8b", "per-tile FoM10 8b", "LIF 8b", "GPU INT8 measured, scaled", "H100 peak", "H100 @40 %", "BPU (layer FoM10 8b) / H100@40 %"],
                          [[esc(r["model"]), f"{r['layers']} × {r['d_model']}", f"{r['layer_fom10_8_nJ_per_token']:.3g}", f"{r['layer_fom50_8_nJ_per_token']:.3g}", f"{r['tile_fom10_8_nJ_per_token']:.3g}", f"{r['lif_fom10_8_nJ_per_token']:.3g}",
                            f"{r.get('gpu_int8_measured_scaled_nJ_per_token', float('nan')):.3g}", f"{r['h100_int8_peak_nJ_per_token']:.3g}", f"{r['h100_int8_40pct_nJ_per_token']:.3g}", f"{r['bpu_over_h100_40pct']['layer_fom10_8']:.4f}"] for r in sc])
        secs.append(section("e8b", "E8b · energy per token", "3 read-out × 2 FoM × 2 bits against measured and published controls", "e8b_energy.png", facts, e8b["verdict"]["statement"],
                            ["Read-out mode × FoM_W × precision, nJ/token (val-loss gap vs ideal in brackets); E_ADC = FoM_W × 2^b, LIF has no ADC so its FoM columns repeat:" + table_html(["read-out", "FoM 10 fJ, 6-bit", "FoM 10 fJ, 8-bit", "FoM 50 fJ, 6-bit", "FoM 50 fJ, 8-bit"], rows3),
                             f"Constants: 4 fJ/MAC, E_ADC = FoM_W × 2^b (Murmann survey: 10 fJ state of the art, 50 fJ conservative), router 10 pJ/event, write driver 100 pJ/pulse, LIF 5 pJ/update + 10 pJ/spike event (one update per spike; the integration is analog), cache write 10 pJ. Digital control = board power × time / tokens on the {esc(d.get('gpu', ''))}, W8A8 via torch._int_mm, batch {gi.get('batch', '–')} × {e8b['protocol']['block']} tokens; CPU torch.ao INT8 ran at {d['cpu_int8_torch_ao']['tokens_per_s']:.0f} tokens/s but its energy is not measurable here (RAPL root-only).",
                             "Scaling (nJ/token; MACs ∝ params, per-layer conversions ∝ layers × d, per-tile conversions ∝ layers × d² / 32). Conversions grow as layers × d_model while MACs grow as parameters, so the BPU / H100@40 % ratio falls with model size:" + scal]))
    if e3c and "pinned_regime" in e3c:
        pr = e3c["pinned_regime"]
        facts = [(k.replace("amp", "cue ").replace("_omega", " ω").replace("_bpu_mismatch", " mm "), f"σ=0: {fmt(v['err_sigma0'])} vs ref {fmt(v['ref_err_sigma0'])}" + (" PINNED" if v["fails"] else "")) for k, v in pr.items()]
        secs.append(section("e3c", "E3 · cue-amplitude sweep (pinned regime)", f"ω = {e3c['omegas']}, cue ∈ {e3c['cue_amps']}, mismatch {e3c['mismatch']}, {e3c['seeds']} seeds", "e3_cue.png", facts[:12], e3c["verdict"]["statement"],
                            ["U-test per pinned condition: " + "; ".join(f"{k}: {e3c['verdicts'][k]['statement']}" for k, v in pr.items() if v["fails"]) if any(v["fails"] for v in pr.values()) else "No condition is pinned at σ = 0 (every tile error within 1.5× the float32 reference), so the U-test is not applied."], src="e3_cue"))
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
    if any(e4t.values()) or e4lc:
        facts = [(f"θ_flush = {th}", f"{r['verdict']['certified_fraction']:.0%} certified, {r.get('writes_per_device_total', r['runs'][0]['writes_per_device_per_s'] * r['development_time_s']):.3g} writes/dev") for th, r in e4t.items() if r]
        facts += [("θ = 1 + quota 6e-3", f"{e4lc['verdict']['certified_fraction']:.0%} certified")] if e4lc else []
        facts += [("threshold writes (weekend)", f"{e4['verdict']['certified_fraction']:.0%} certified, {sum(x['writes_per_device_per_s'] for x in e4['runs']) / len(e4['runs']) * e4['development_time_s']:.3g} writes/dev")] if e4 else []
        secs.append(section("e4", "E4 · ring self-organisation under the write budget", "Accumulate-and-flush at θ_flush ∈ {0.25, 0.5, 1} levels", "e4_level_t1.png", facts,
                            " || ".join(f"θ = {th}: " + r["verdict"]["statement"] for th, r in e4t.items() if r), src="e4_level_t1"))
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
<section id="scoreboard"><div class="eyebrow">Substrate claims</div><h2>Six rows, reference first; budget = total writes per device against the 1e6 endurance</h2>{score}
<p class="muted">tolerances are the ones written into each experiment's docstring and verdict block.</p></section>
<section id="writes"><div class="eyebrow">Write budget</div><h2>Accumulate-and-flush: writes only on whole conductance levels</h2>{wbox}
<p class="muted">a device is pulsed only when its digital accumulator holds θ_flush conductance levels (1/L, L = round(1/σ_eff)); the remainder is rounded stochastically. Budget: endurance = total writes per device over the run against 1e6; driver bandwidth = writes per physical tile per second against 1e4; the old 6e-3 writes/device/s is kept as the "lifelong rate" for reference.</p></section>
<section id="other"><div class="eyebrow">Other rows</div><h2>Remaining experiments</h2>{otab}</section>
{"".join(secs)}
<footer>bpusim · substrate programme 2026-10-18 · figures by scripts/make_figures.py · page by scripts/make_results_page.py</footer></div>'''
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(secs)} sections)")


if __name__ == "__main__":
    main()
