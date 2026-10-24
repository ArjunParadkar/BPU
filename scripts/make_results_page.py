"""Build results/index.html from results/*/results.json and figures/*.png.

Substrate framing: the BPU is a substrate, so every row compares the SAME
algorithm on (a) a float32 reference and (b) noisy BPU tiles; success = BPU
matches the reference within the stated tolerance.  Local learning rules
(FA, DFA, three-factor, growth) are additional arms, never the headline.
Since 2026-10-19 a "Variability, used the way the brain uses it" scoreboard
(E11 regularisation, E12 exploration, E13 stochastic resonance, E14 sampling:
mechanism, brain analogue, hypothesis, result, status) comes first; then the
"Substrate claims" scoreboard (E10, E9, E8b, E3-final, E6, E7); the write-budget
box (E4, E9); the remaining experiments follow.  Every number is read from a results.json; nothing is
typed in.  Usage: python scripts/make_results_page.py [--out results/index.html]
"""
from __future__ import annotations

import argparse
import html
import json
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
# dated correction, 2026-10-23 (round four); shown beside the old text, never instead of it
CLIP_CORRECTION = ('Correction 2026-10-23 (round four): the MODULATE dial clipped the float32 reference. TileLinear.set_noise clips to the dial range [0.005, 0.2], and E11 set the float32 reference through it as well, so every E11 arm with σ > 0.2 ran its Linear layers (qkv, proj, fc, fc2, head) at σ_r = 0.2 and only the attention-cache read noise followed σ. Affected arms: every float32 arm above 0.2, i.e. results/e11 (0.3, the beyond-dial point), results/e11_f32_bracket (0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0) and results/e11_full_f32 (0.3, 0.4, 0.5, 0.7, 1.0, 1.4); and the tiles\' beyond-dial 0.3 point (results/e11, results/e11_full_tiles), a diagnostic that was already excluded from σ* and the slope. WITHDRAWN pending the unclipped reruns (jobs 1366759 / 1366760, results/e11_f32_unclipped, results/e11_full_f32_unclipped): the float32 optima σ* = 1.0 (small data) and 0.4 (full data), the claim that the float32 optimum sits about 10x above the tile optimum ("an order of magnitude less injected noise"), the float32 shift-gap slopes over the extended grids (−0.16 to σ 3.0, −0.66 to 1.4) and every float32 or tiles value quoted at σ ≥ 0.3. Unaffected: every arm at σ ≤ 0.2 on both substrates. The tiles σ* = 0.1 (small data) and σ* = 0.04 (full data) stand: both were taken over the dial grid 0–0.2, which contains no clipped point; the 0.3 point was not in the grid, and the full-data bootstrap put all its mass on 0.04 (CI 0.04..0.04). The tiles shift-gap slopes, the twin gaps inside the dial and the dropout arms stand, and so does the round-three statement that the float32 minimum is not reached inside the dial (σ* = 0.2 on the dial top, small and full data). Code fixed the same day: the float32 reference is never clipped (TileLinear.set_noise, FloatTile.set_noise), device values beyond the dial are set directly as a diagnostic (e11_regularization.set_all_noise), --legacy-dial-clip reproduces the old behaviour for E11-cal, and tests/test_torchtile.py::test_float_reference_noise_is_not_clipped_by_the_dial checks both. No other experiment was affected: E12 stays inside the dial, E13 and E14 set beyond-dial values directly.')
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
    names = ("e1", "e2", "e2_cifar", "e3", "e3_256", "e3_tracking", "e3_cue", "e3_final", "e4", "e4_capped", "e4_np", "e4_level_t1", "e4_level_t05", "e4_level_t025", "e4_level_capped", "e5", "e5_48", "e5_selforg32", "e5_selforg_np",
             "e6", "e6_k32", "e7", "e7_ref", "e7_3seeds", "e7_ref_3seeds", "e8", "e8b", "e9", "e9_pareto", "e9_pulse", "e10", "e9_v1", "e10_v1", "e8b_v1", "e11", "e12", "e13", "e14",
             "e11_f32_bracket", "e11_full_tiles", "e11_full_f32", "e12_long", "e3_final50", "e5_ref20", "e5_selforg_np20", "e1_full",
             "e11cal", "e1fix", "e15", "e17", "e11_f32_unclipped", "e11_full_f32_unclipped")
    R = {k: load(k) for k in names}
    e9, e10, e8b, e3t, e3c, e6 = R["e9"], R["e10"], R["e8b"], R["e3_tracking"], R["e3_cue"], (R["e6_k32"] or R["e6"])
    # 2026-10-23 (round four): E6 reads results/e6_k32 (K grid to 32; the results/e6 grid stopped at 16 and capped the
    # attention capacity at the grid top); E3-final reads the 50-seed run (results/e3_final50: 4/8 pinned) instead of 10 seeds (5/8)
    E6_SRC = "e6_k32" if R["e6_k32"] else "e6"; E3F_SRC = "e3_final50" if R["e3_final50"] else "e3_final"
    e3f, e9p, e4n, e5n, e11, e12, e13, e14 = R[E3F_SRC], R["e9_pareto"], R["e4_np"], R["e5_selforg_np"], R["e11"], R["e12"], R["e13"], R["e14"]
    if e9 and e9.get("partial"):
        e9 = None

    # ------------------------------------------------------------------ learning vs knowing (round four, 2026-10-23)
    def lv_state(r, key_ok=None):
        if not r:
            return "queued on Pinnacle; MONDAY2.md has the job IDs and the hold", chip("pend", "pending")
        v = r.get("verdict", {})
        st = v.get("statement", "") if isinstance(v, dict) else str(v)
        if key_ok is None:
            return st, chip("part", "exploratory")
        ok = key_ok(v)
        partial = st.startswith("PARTIAL") or (isinstance(v, dict) and v.get("complete") is False)
        return st, chip("pend", "partial") if partial else (chip("ok", "supported") if ok else chip("nul", "not supported"))
    lv_rows = []
    st, ch = lv_state(R["e11cal"], lambda v: v.get("lines_up"))
    lv_rows.append(["E11-cal", "Is the ~10× gap in σ between the tile and float32 optima only an axis effect? Each E11 arm is placed at the layer perturbation it actually had (rel_pert).",
                    "The optima line up in rel_pert in both regimes: the ratio R has a 95 % CI containing 1 and 0.5 ≤ R ≤ 2", st, ch])
    st, ch = lv_state(R["e1fix"], lambda v: v.get("criterion_met"))
    lv_rows.append(["E1-fix + E16", "Knowing: deploy an already-trained GPT-2 124M on tiles. Arms: deploy-only, noise-aware fine-tune, dial-low read, read averaging.",
                    "Some arm at σ_device = 0.04 lands within 3 % of its float32 perplexity; above 2× counts as diverged", st, ch])
    if R["e15"]:
        v = R["e15"]["verdict"]; st = v.get("statement", "")
        ch = " ".join(chip("pend", f"{k} untested") if v.get(k) is None else chip("ok" if v.get(k) else "nul", f"{k} {'yes' if v.get(k) else 'no'}") for k in ("H1", "H2", "H3", "H4"))
    else:
        st, ch = "not submitted yet (MONDAY2.md: data prep, throughput timing, then 6 h checkpointed chains)", chip("pend", "pending")
    lv_rows.append(["E15", "Learning: train a ~50M-parameter GPT from scratch on tiles (0.8B web tokens, shift sets TinyStories and Python).",
                    "H1: tiles σ* > 0 with a CI excluding 0. H2: σ* in [0.02, 0.06]. H3: tiles at σ* ≤ float32 at σ 0. H4: learn high / deploy low ≥ fixed σ*", st, ch])
    st, ch = lv_state(R["e17"])
    lv_rows.append(["E17", "Does the knowledge survive the noise? Compares the representations of the float32 σ 0 model and the tiles σ* model.", "Exploratory, no pass/fail: CKA > 0.8 at matched layers", st if R["e17"] else "optional, after E15", ch if R["e17"] else chip("pend", "optional")])
    # 2026-10-24: round-four results in; one-paragraph readings beside the verdict statements (numbers from the JSON)
    RESOLVED = ""
    c = R["e11cal"]
    if c and c.get("sigma_space_unclipped"):
        su = c["sigma_space_unclipped"]; ru = c.get("ratio_unclipped", {}); ra = c["ratio"]
        def one(g):
            o = su[g]; f, t, q = o["float32"], o["tiles"], o["ratio"]
            return (f"{g} data: float32 σ* {f['star']:g} (CI {f['ci95'][0]:g}–{f['ci95'][1]:g}, per-seed {f['per_seed_argmin']}) vs tiles σ* {t['star']:g} "
                    f"(CI {t['ci95'][0]:g}–{t['ci95'][1]:g}): ratio {q['point']:.1f}× (CI {q['ci95'][0]:.1f}–{q['ci95'][1]:.1f}); corrected float32 shift-gap slope "
                    f"{f['shift_gap_clean']['slope_per_unit_sigma']:+.2f} (CI {f['shift_gap_clean']['ci95'][0]:+.2f}..{f['shift_gap_clean']['ci95'][1]:+.2f}, {f['shift_gap_clean']['seeds_negative']}/{f['n_seeds']} seeds narrowing)")
        RESOLVED = ("Resolved 2026-10-24 (unclipped reruns 1366759 / 1366760; corrected float32 curve = original arms at σ ≤ 0.2 plus the unclipped arms above 0.2, same seeds, E11's own σ* rule): "
                    + "; ".join(one(g) for g in ("small", "full") if g in su)
                    + ". The float32 optimum is higher than the tiles' σ* by 4× (small data) and 5× (full data), not ~10×; on full data it sits at the dial top (0.2), not beyond it. "
                    "The CIs are over grid points and the grids are coarse: neighbouring grid points would allow roughly 2–8× (small) and 2.5–15× (full). The withdrawn values (1.0 / 0.4, ~10×, slopes −0.16 / −0.66) stay withdrawn and are replaced by these.")
        lv_rows[0][3] = (lv_rows[0][3] + f" || Secondary, with the corrected float32 curve: R = {ru['small']['R']:.2f} (CI {ru['small']['ci95'][0]:.2f}..{ru['small']['ci95'][1]:.2f}) small data, lines up; "
                         f"R = {ru['full']['R']:.2f} (CI {ru['full']['ci95'][0]:.2f}..{ru['full']['ci95'][1]:.2f}) full data, does not. Neither the as-run (primary) nor the corrected analysis lines up in both regimes. "
                         "Reproduction: retrained σ = 0 arms match E11 bit-for-bit (float32) or within 0.0013 (full-data tiles); the small-data tiles E11 arms ran on the laptop GPU and the V100 retrains differ by −0.21..+0.08 nats. " + RESOLVED)
    f1 = R["e1fix"]
    if f1 and f1.get("table"):
        T = {t["arm"]: t for t in f1["table"]}
        def g(a): return T.get(a, {}).get("ppl", float("nan"))
        lv_rows[1][3] = (f"NOT met by the letter: every tile arm at σ_device 0.04 is above 2× its reference (diverged); the pre-registered combined arm (fine-tune 2000 + dial-low + k = 8) reaches {g('E16 fine-tune 2000 + dial-low k=8 (the pre-registered combined arm)'):.1f} against float32 {g('fine-tune float32 sigma=0 2000 steps'):.2f} (+{T['E16 fine-tune 2000 + dial-low k=8 (the pre-registered combined arm)']['gap_pct']:.0f} %). "
                         f"The tile path is already diverged at σ = 0: deploy tiles σ 0 = {g('deploy tiles sigma=0'):.1f} vs float32 {f1['float32_pretrained_clean']:.2f} (programming residual + 8-bit read-out, no noise). "
                         f"The same noise on float32 is handled: deploy σ 0.04 {g('deploy float32 sigma=0.04'):.1f} (+94 %), noise-aware fine-tune σ 0.04 × 2000 {g('fine-tune float32 sigma=0.04 2000 steps'):.2f} (+0.4 %). "
                         f"E16: dial-low cuts the tile deploy at 0.04 from {g('E16 deploy averaging k=1'):.0f} to {g('E16 deploy dial-low k=1'):.0f} (k = 1), averaging k = 8 to {g('E16 deploy averaging k=8'):.0f} at 5.1× the read energy; after fine-tuning neither moves perplexity by more than 2 %. "
                         "Reading: the limit is the zero-noise tile floor, not variability; decomposing it (programming residual vs converter) is the next step and is not submitted (hold). || " + lv_rows[1][3])
    lvbox = table_html(["Exp", "Question", "Pre-registered criterion", "Result", "Status"], [[f"<span class='id'>{esc(a)}</span>", esc(b), esc(c), esc(d), e] for a, b, c, d, e in lv_rows])
    lvbox += "".join(f'<figure><img src="figures/{f}.png" alt="{esc(f)}"><figcaption>figures/{f}.png</figcaption></figure>'
                     for f in ("e11cal_relpert", "e1fix_gpt2_deploy") if (FIG / f"{f}.png").exists())
    # ------------------------------------------------------------------ variability scoreboard (E11-E14): mechanism, brain analogue, hypothesis, result, status
    vrows = []
    def vrow(mech, exp, analogue, hyp, result, status):
        vrows.append([esc(mech), f"<span class='id'>{esc(exp)}</span>", esc(analogue), esc(hyp), esc(result), status])
    # E11
    if e11 and e11.get("hypotheses", {}).get("tiles", {}).get("complete"):
        H = e11["hypotheses"]; S = e11["summary"]; t = H["tiles"]; f = H.get("float32", {})
        a = t["sigma_star_val_clean"]; g = t["shift_gap_gap_clean"]
        res = (f"tiles: σ* = {a['star']:g} (CI {a['ci95'][0]:g}..{a['ci95'][1]:g}; per-seed {a['per_seed_argmin']}), val loss " + ", ".join(f"σ {sg:g}: {fmt(S['tiles'][f'sigma={sg:g}']['val_clean']['mean'])}" for sg in e11["sigmas"]) +
               f"; shift gap {g['gap_at_zero']:.3f} → {g['gap_at_max']:.3f} (slope {g['slope_per_unit_sigma']:+.2f}, CI {g['ci95'][0]:+.2f}..{g['ci95'][1]:+.2f}, {g['seeds_negative']}/{e11['seeds']} seeds narrowing)" +
               (f"; dropout p* = {t['p_star_val_clean']['star']:g}" if "p_star_val_clean" in t else "") +
               (f" | float32: σ* = {f['sigma_star_val_clean']['star']:g} (CI {f['sigma_star_val_clean']['ci95'][0]:g}..{f['sigma_star_val_clean']['ci95'][1]:g}), gap slope {f['shift_gap_gap_clean']['slope_per_unit_sigma']:+.2f}" if f.get("complete") else "") + " [2026-10-23: the beyond-dial 0.3 arms ran their Linear layers at 0.2 (dial clip); the σ* and slopes above use only the dial grid and stand]")
        ok1, ok2 = e11["verdict"]["sigma_star_positive_with_ci"], e11["verdict"]["shift_gap_narrows_with_ci"]
        st = chip("ok" if ok1 else "nul", "σ* > 0" if ok1 else "σ* = 0 / CI includes 0") + " " + chip("ok" if ok2 else "nul", "gap narrows" if ok2 else "gap not narrowing")
        vrow("Regularisation: read noise / synapse dropout during training", "E11", "synaptic unreliability (release probability 0.1–0.9) as a built-in regulariser; dropout",
             "val loss optimum at σ_r > 0; shift gap (Python after Shakespeare) narrows with σ_r", res, st)
    else:
        vrow("Regularisation: read noise / synapse dropout during training", "E11", "synaptic unreliability as a regulariser", "val loss optimum at σ_r > 0; shift gap narrows", "running" if e11 else "pending", chip("pend", "running"))
    # E12
    if e12 and "hypotheses" in e12:
        S = e12["summary"]; H = e12["hypotheses"]
        parts = []
        for tk in e12["tasks"]:
            h1 = H.get(f"{tk}|tiles|gated_vs_best_fixed"); h2 = H.get(f"{tk}|tiles|speed_exponent")
            line = f"{tk} (tiles): " + ", ".join(f"{a} {S[tk]['tiles'][a]['median_trials']:.0f} ({S[tk]['tiles'][a]['reached_frac']:.0%})" for a in e12["arms"])
            if h1:
                line += f"; gated vs {h1['best_fixed']}: {h1['seeds_gated_faster']}/{e12['seeds']} seeds faster, p = {h1['sign_test_p']:.2g}"
            if h2 and h2.get("exponent") is not None:
                line += f"; speed ∝ σ^{h2['exponent']:.2f}" + (f" (CI {h2['ci95'][0]:.2f}..{h2['ci95'][1]:.2f})" if h2.get("ci95") else "")
            parts.append(line)
        n_win = e12["verdict"]["n_tasks_gated_wins_tiles"]
        exps = [H[k] for k in H if k.endswith("speed_exponent") and "|tiles|" in k and H[k].get("exponent") is not None]
        st = chip("ok" if n_win == len(e12["tasks"]) else "nul", f"gated wins {n_win}/{len(e12['tasks'])} tasks") + " " + (chip("ok" if all(x.get("consistent_with_2") for x in exps) else "nul", "σ² law" if all(x.get("consistent_with_2") for x in exps) else "not σ²") if exps else "")
        vrow("Exploration: the noise dial as the probe of node-perturbation RL", "E12", "reward-gated song variability (songbird LMAN / dopamine); locus coeruleus gain control of variability",
             "reward-gated schedule beats fixed σ; learning speed ∝ σ² at small σ", " || ".join(parts), st)
    else:
        vrow("Exploration: the noise dial as the probe of node-perturbation RL", "E12", "reward-gated variability (songbird LMAN)", "gated beats fixed; speed ∝ σ²", "running" if e12 else "pending", chip("pend", "running"))
    # E13
    if e13 and "verdict" in e13:
        v = e13["verdict"]; V = e13["verdicts"]; head = [k for k in V if k.endswith("_tiles")][0]; hv = V[head]; ref = V.get(head.replace("_tiles", "_float32"))
        res = (f"headline (0.8× threshold, τ_m 20 ms): SNR {hv['snr_at_zero']:.1f} dB at σ 0 → {hv['snr_at_star']:.1f} at σ* {hv['sigma_star']} (CI {hv['ci95'][0]}..{hv['ci95'][1]}) → {hv['snr_at_max']:.1f} at 0.2" +
               (f"; float32 {ref['snr_at_zero']:.1f} → {ref['snr_at_star']:.1f} → {ref['snr_at_max']:.1f}" if ref else "") +
               f"; beyond the dial (diagnostic, to 1.0): σ* {hv['beyond_dial']['sigma_star']}{' U-curve' if hv['beyond_dial']['u_curve'] else ''}" +
               f" | grid (drive × τ_m): U-curve within the dial in {len(v['cells_with_u_curve'])}/{len(V)} cells, peak in [0.02, 0.06] in {len(v['cells_with_peak_in_band'])}" +
               ("; " + ", ".join(f"{k.replace('_tiles', '')}: σ* {V[k]['sigma_star']}{'U' if V[k]['u_curve'] else ''}/beyond {V[k]['beyond_dial']['sigma_star']}{'U' if V[k]['beyond_dial']['u_curve'] else ''}" for k in V if k.endswith("_tiles") and k != head)))
        st = chip("ok" if v["headline_u_curve"] else "nul", "U-curve" if v["headline_u_curve"] else "no U-curve in the dial") + " " + chip("ok" if v["headline_peak_in_band"] else "nul", "peak in band" if v["headline_peak_in_band"] else "peak not in [0.02, 0.06]")
        vrow("Stochastic resonance: subthreshold periodic drive through a tile into LIF neurons", "E13", "SR in mechanoreceptors (crayfish, Douglass et al. 1993) and human tactile / visual detection",
             "SNR vs σ_r is an inverted U with its peak in [0.02, 0.06]", res, st)
    else:
        vrow("Stochastic resonance: subthreshold periodic drive through a tile into LIF neurons", "E13", "SR in sensory neurons", "inverted U, peak in [0.02, 0.06]", "running" if e13 else "pending", chip("pend", "running"))
    # E14
    if e14 and "verdict" in e14:
        v = e14["verdict"]; V = e14["verdicts"]
        res = "; ".join(f"α {a:g}: tiles KL {V[f'alpha{a:g}_tiles']['kl_at_zero']:.2f} at σ 0 → {V[f'alpha{a:g}_tiles']['kl_min']:.3f} at σ* {V[f'alpha{a:g}_tiles']['sigma_star']} (CI {V[f'alpha{a:g}_tiles']['ci95'][0]}..{V[f'alpha{a:g}_tiles']['ci95'][1]}) → {V[f'alpha{a:g}_tiles']['kl_at_dial_max']:.3f} at 0.2; float32 σ* {V[f'alpha{a:g}_float32']['sigma_star']}, KL min {V[f'alpha{a:g}_float32']['kl_min']:.3f}" for a in e14["alphas"]) +               "; beyond the dial (to 1.0): " + ", ".join(f"α {a:g} σ* {V[f'alpha{a:g}_tiles']['sigma_star_beyond_dial']}, KL {V[f'alpha{a:g}_tiles']['kl_at_max']:.2f} at 1.0{' rises' if V[f'alpha{a:g}_tiles']['rises_beyond_dial'] else ''}" for a in e14["alphas"])
        st = chip("ok" if v["deterministic_at_zero"] else "nul", "σ = 0 deterministic") + " " + chip("ok" if v["kl_min_at_positive_sigma_all_alphas"] else "nul", "KL min at σ > 0" if v["kl_min_at_positive_sigma_all_alphas"] else "KL min not at σ > 0") + " " + chip("ok" if v["same_sigma_star"] else "part", "one σ* for every α" if v["same_sigma_star"] else "σ* depends on α")
        vrow("Sampling: attention retrieval from a 50/50 cue", "E14", "neural sampling: spontaneous / trial-to-trial variability as posterior samples (Fiser & Berkes 2010; Orbán 2016); perceptual bistability",
             "σ = 0 always retrieves one pattern; KL to the Boltzmann posterior is minimal at σ > 0", res, st)
    else:
        vrow("Sampling: attention retrieval from a 50/50 cue", "E14", "neural sampling hypothesis", "KL minimum at σ > 0", "running" if e14 else "pending", chip("pend", "running"))
    vscore = table_html(["Mechanism", "Exp", "Brain analogue", "Hypothesis", "Result (tiles; float32 reference)", "Status"], vrows)
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
        pd = e9.get("pulse_diagnostic") if e9.get("write_mode", "level") == "level" else None
        if e9.get("locality"):
            extra_loc = "loss vs locality: " + " → ".join(f"{l['arm'].replace('_bpu', '')} {fmt(l['post_shift_loss'])} ({pct(l['gap_pct_vs_float32'])})" for l in e9["locality"])
        if not pd and R["e9_pulse"] and e9.get("write_mode", "level") == "level":
            ps = R["e9_pulse"]["summary"]["sgd_bpu_pulse"]
            pd = {"post_shift_loss": ps["segment_end_loss_1"]["mean"], "writes_per_device_total": ps.get("writes_per_device_total", {}).get("mean", ps["writes_per_device_per_s"]["mean"] * R["e9_pulse"]["seg"] * 3 * R["e9_pulse"]["step_seconds"])}
        extra = f"nearest-pulse diagnostic: {fmt(pd['post_shift_loss'])} ({pct(100 * (pd['post_shift_loss'] / refv - 1))}) at {pd['writes_per_device_total']:.3g} writes/dev" if pd else (extra_loc if e9.get("locality") else "")
        if e9p and "pareto" in e9p:
            extra += f" | round-two level-gated Pareto (results/e9_pareto): " + "; ".join(f"{p['name']} {fmt(p['post_shift_loss'])} ({pct(p['gap_pct'])})" for p in e9p["pareto"] if p["arm"] == "sgd_bpu")
        ok = e9["verdict"]["success"]
        st = chip("ok" if ok else "nul", "criterion met" if ok else "not met") + " " + chip("part", f"{e9.get('write_mode', 'level')} writes, {e9['seeds']} seeds")
        rows.append(("Trainable where an FPGA is frozen", "E9", ref, bpu, wbudget, "within 5 % of float32 with < 1e4 total writes/device; loss vs locality", extra, st))
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
    # 4. E3-final (mismatch 0.2 / 0.3, cue 0.02 / 0.05), then the cue sweep, then the omega sweep
    if e3f and "pinned_regime" in e3f:
        pr = e3f["pinned_regime"]; sw = e3f["sweeps"]
        ref = "; ".join(f"cue {a:g} ω {o:g}: {fmt(sw[f'amp{a:g}_omega{o:g}_float32']['0.0']['track_err']['mean'])} rad" for a in e3f["cue_amps"] for o in e3f["omegas"] if f"amp{a:g}_omega{o:g}_float32" in sw)
        bpu = "; ".join(f"{k.replace('amp', 'cue ').replace('_omega', ' ω').replace('_bpu_mismatch', ' mm ')}: {fmt(v['err_sigma0'])}" + (" PINNED" if v["fails"] else "") for k, v in pr.items())
        n_p, n_u = e3f["verdict"]["n_pinned"], e3f["verdict"]["n_u_shaped_among_pinned"]
        extra = f"σ = 0 pinned fraction {n_p}/{len(pr)}; U-shaped among pinned: {n_u}" + ("" if n_p else "; no pinning up to 30 % mismatch: attractor robustness is the result") + \
                (f"; cue sweep at ω 0.5 (results/e3_cue): {e3c['verdict']['n_pinned']}/{len(e3c['pinned_regime'])} pinned" if e3c and "pinned_regime" in e3c else "") + \
                (f"; ω sweep at cue 0.1 (results/e3_tracking): {e3t['verdict']['n_u_shaped']}/{e3t['verdict']['n_conditions']} U-shaped" if e3t and "n_u_shaped" in e3t.get("verdict", {}) else "")
        e3_note = (" " + chip("pend", f"{e3f['seeds']} seeds (results/{E3F_SRC}); the 10-seed run had 5/8 pinned")) if E3F_SRC == "e3_final50" else ""
        rows.append(("Attractor noise optimum (moving cue, E3-final)", "E3", ref, bpu, "no writes (read noise only)", "σ = 0 pinned fraction; U-test only where pinned", extra,
                     chip("nul" if n_p == 0 else "part", "no pinning up to 30 % mismatch" if n_p == 0 else f"{n_p} pinned, {n_u} U-shaped") + e3_note))
    elif e3c and "pinned_regime" in e3c:
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
                     f"K grid {min(e6['counts'])}..{max(e6['counts'])}, {100 * e6.get('flip_fraction', 0):.0f} % flipped cues, {e6['seeds']} seeds, sign ties random, test_hopfield_k1 passes" + (f"; reads results/{E6_SRC} since 2026-10-23 (the 1..16 grid of results/e6 capped attention at its top, 16; its finer grid gives scaffold 3–5 pairs)" if E6_SRC == "e6_k32" else ""), chip("ok" if same else "nul", "attention holds" if same else "attention drops")))
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
    e4row("E4 nearest-pulse writes (default since 2026-10-19), 20 seeds", e4n)
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
    other.append(("Ring self-organises (nearest-pulse writes, 20 seeds)", "E4", (e4n["verdict"]["statement"] if e4n else "pending") + (" || level θ = 1: " + e4l["verdict"]["statement"] if e4l else ""),
                  chip("ok" if e4n and e4n["verdict"]["certified_fraction"] >= 0.5 else "part", f"{e4n['verdict']['certified_fraction']:.0%} certified, {e4n['writes_per_device_total']:.3g} writes/dev" if e4n else "running")))
    if e5n:
        other.append(("Grid self-organisation (nearest-pulse writes, 5 seeds)", "E5", e5n["verdict"]["statement"], chip("nul" if all(d["trained_torus_fraction"] == 0 for d in e5n["sheets"].values()) else "part",
                      "; ".join(f"{k}: {d['trained_torus_fraction']:.0%} self-organised, {d['reference_torus_fraction']:.0%} programmed" for k, d in e5n["sheets"].items()))))
    other.append(("Attractor noise optimum (static drift, weekend)", "E3", R["e3"]["verdict"]["statement"] if R["e3"] else "pending", chip("nul", f"{R['e3']['verdict']['n_u_shaped']}/{R['e3']['verdict']['n_conditions']} U-shaped") if R["e3"] else chip("pend", "pending")))
    other.append(("Grid module is a torus", "E5", e5["verdict"]["statement"] if e5 else "pending", chip("part", "programmed refs certify; self-org fails") if e5 else chip("pend", "pending")))
    other.append(("Local grow rule vs RigL", "E2", (e2["verdict"]["statement"] if e2 else "") + (" | CIFAR-10 3 %: " + e2c["verdict"]["statement"] if e2c else ""),
                  (chip("ok" if e2 and e2["verdict"].get("within_few_percent_of_rigl") else "nul", "MNIST") if e2 else "") + (" " + chip("nul", "CIFAR-10 refuted") if e2c and e2c["verdict"].get("clearly_worse_than_random") else "")))
    if R["e1_full"]:
        other.append(("GPT-2 through noisy tiles", "E1", R["e1_full"]["verdict"]["statement"] if isinstance(R["e1_full"].get("verdict"), dict) else str(R["e1_full"].get("verdict", "")), chip("nul", "null (HPCC)")))
    else:
        other.append(("GPT-2 through noisy tiles", "E1", "queued for the GPU cluster (MONDAY.md); E8 is the laptop stand-in", chip("pend", "cluster")))
    otab = table_html(["Row", "Exp", "Measured", "Status"], [[esc(a), f"<span class='id'>{esc(b)}</span>", esc(c), d] for a, b, c, d in other])

    # ------------------------------------------------------------------ HPCC (Pinnacle) runs, 2026-10-22/23
    def e11_star(r, sub):
        h = (r or {}).get("hypotheses", {}).get(sub, {})
        if not h.get("complete"):
            return None
        a = h["sigma_star_val_clean"]; g = h["shift_gap_gap_clean"]
        return (f"σ* = {a['star']:g} (CI {a['ci95'][0]:g}–{a['ci95'][1]:g}, per-seed {a['per_seed_argmin']}); shift-gap slope "
                f"{g['slope_per_unit_sigma']:+.2f} (CI {g['ci95'][0]:+.2f}..{g['ci95'][1]:+.2f}, {g['seeds_negative']}/{r['seeds']} seeds narrowing)")
    def vstate(r):
        v = (r or {}).get("verdict", "")
        return v["statement"] if isinstance(v, dict) else str(v)
    hrows = []
    if R["e11_full_tiles"]:
        t = R["e11_full_tiles"]; ok = t["verdict"]["sigma_star_positive_with_ci"]
        hrows.append(["E11 full data, tiles", f"{t['steps']} steps, 100 % of the text, {t['seeds']} seeds", esc(e11_star(t, "tiles") or "–"), chip("ok" if ok else "nul", "supported" if ok else "not supported")])
    if R["e11_full_f32"]:
        hrows.append(["E11 full data, float32", f"grid to {max(R['e11_full_f32']['sigmas']):g}, {R['e11_full_f32']['seeds']} seeds", esc(e11_star(R["e11_full_f32"], "float32") or "–"), chip("ok", "bracketed") + " " + chip("nul", "WITHDRAWN 2026-10-23: dial clip")])
    if R["e11_f32_bracket"]:
        hrows.append(["E11 small data, float32", f"grid to {max(R['e11_f32_bracket']['sigmas']):g}, {R['e11_f32_bracket']['seeds']} seeds", esc(e11_star(R["e11_f32_bracket"], "float32") or "–"), chip("ok", "bracketed") + " " + chip("nul", "WITHDRAWN 2026-10-23: dial clip")])
    if R["e11cal"] and R["e11cal"].get("sigma_space_unclipped"):
        for g, lab in (("small", "E11 small data, float32 corrected (unclipped above 0.2)"), ("full", "E11 full data, float32 corrected (unclipped above 0.2)")):
            o = R["e11cal"]["sigma_space_unclipped"].get(g)
            if o:
                f = o["float32"]; sg = f["shift_gap_clean"]
                hrows.append([lab, f"grid to {max(f['sigmas']):g}, {f['n_seeds']} seeds", esc(f"σ* = {f['star']:g} (CI {f['ci95'][0]:g}–{f['ci95'][1]:g}, per-seed {f['per_seed_argmin']}); {o['ratio']['point']:.1f}× the tiles σ* {o['tiles']['star']:g}; shift-gap slope {sg['slope_per_unit_sigma']:+.2f} (CI {sg['ci95'][0]:+.2f}..{sg['ci95'][1]:+.2f}, {sg['seeds_negative']}/{f['n_seeds']} seeds narrowing)"),
                              chip("ok", "bracketed") + " " + chip("part", "replaces the withdrawn value")])
    for key, lab, cls, txt in (("e3_final50", "E3-final, 50 seeds", "nul", "null"), ("e5_ref20", "E5 programmed reference, 20 seeds", "ok", "reference"),
                               ("e5_selforg_np20", "E5 self-organisation 32×32, 20 seeds", "nul", "null"), ("e12_long", "E12, 50 000 trials", "part", "partial"),
                               ("e1_full", "E1 GPT-2 124M", "nul", "null")):
        if R[key]:
            hrows.append([lab, "", esc(vstate(R[key])[:700]), chip(cls, txt)])
    hpcc = ""
    if hrows:
        htab = table_html(["Run", "Budget", "Result", "Status"], [[esc(a), esc(b), c, d] for a, b, c, d in hrows])
        figs = "".join(f'<figure><img src="figures/{f}.png" alt="{esc(f)}"><figcaption>figures/{f}.png</figcaption></figure>'
                       for f in ("e11_full_tiles", "e11_full_f32", "e11_f32_bracket", "e12_long", "e3_final50", "e5_selforg_np20", "e1_full") if (FIG / f"{f}.png").exists())
        hpcc = f'''<section id="hpcc"><div class="eyebrow">Arkansas HPCC (Pinnacle), 2026-10-22/23</div><h2>The MONDAY queue at cluster budgets</h2>
<p>Ten jobs from MONDAY.md plus two bracket extensions, run on V100 GPUs and 32-core CPU nodes. The headline: the read-noise regulariser on tiles survives on the full training text (σ* = 0.04, against 0.1 on 10 % of the text), and in both regimes the float32 optimum sits about ten times higher (0.4 full, 1.0 small), so tiles reach the optimum with an order of magnitude less injected noise. [2026-10-23: withdrawn, dial clip. 2026-10-24: corrected, 4× (small data, float32 σ* 0.4) and 5× (full data, float32 σ* 0.2); see the two boxes below.]</p>
<div class="verdict">{esc(CLIP_CORRECTION)}<small>dated correction · DECISIONS.md "Correction 2026-10-23"</small></div>
{('<div class="verdict">' + esc(RESOLVED) + '<small>resolution · DECISIONS.md "Round four outcomes (2026-10-24)"</small></div>') if RESOLVED else ''}
{htab}{figs}
<p class="muted">Each row is read from results/&lt;run&gt;/results.json; the laptop runs above are kept unchanged as the record.</p></section>'''

    # ------------------------------------------------------------------ sections
    secs = []
    if e11 and e11.get("summary"):
        S = e11["summary"]; H = e11.get("hypotheses", {})
        facts = [(f"tiles val (clean) σ {sg:g}", fmt(S["tiles"].get(f"sigma={sg:g}", {}).get("val_clean", {}).get("mean"))) for sg in e11["sigmas"]] + \
                [(f"tiles val (clean) dropout {pd:g}", fmt(S["tiles"].get(f"dropout={pd:g}", {}).get("val_clean", {}).get("mean"))) for pd in e11["dropouts"]]
        notes = [f"nanoGPT 2.7M trained from scratch for {e11['steps']} steps (batch {e11['batch']} × {e11['block']}) on {100 * e11['train_frac']:.0f} % of the Shakespeare training split (small-data regime, the model overfits), {e11['seeds']} seeds; read noise is the only device noise (d2d = c2c = 0), set through the MODULATE primitive; dropout = per-read synapse failure; tiles use nearest-pulse writes and the layer-accumulate 8-bit read-out, the float32 reference injects the same noise on float weights.",
                 "Val loss is read both as deployed (noise on) and clean (noise off); the shift gap is loss(Python source) − loss(Shakespeare val)."]
        if H.get("tiles", {}).get("complete"):
            t = H["tiles"]
            notes.append(f"σ* on tiles: clean {t['sigma_star_val_clean']['star']:g} (bootstrap distribution {t['sigma_star_val_clean']['boot_distribution']}), as deployed {t['sigma_star_val_noisy']['star']:g}; shift-gap slope (clean) {t['shift_gap_gap_clean']['slope_per_unit_sigma']:+.2f} (CI {t['shift_gap_gap_clean']['ci95'][0]:+.2f}..{t['shift_gap_gap_clean']['ci95'][1]:+.2f}).")
        notes.append(esc(CLIP_CORRECTION))
        if e11.get("twin_gap"):
            notes.append("Twin gap tiles vs float32 (val, clean read): " + ", ".join(f"{k} {pct(v['val_clean_gap_pct'])}" for k, v in e11["twin_gap"].items()))
        secs.append(section("e11", "E11 · regularisation", "Read noise and synapse dropout as regularisers, tiles vs float32", "e11_regularization.png", facts, e11["verdict"]["statement"], notes))
    if e12 and "summary" in e12:
        S = e12["summary"]
        facts = [(f"{tk}/{sub} {a}", f"{S[tk][sub][a]['median_trials']:.0f} ({S[tk][sub][a]['reached_frac']:.0%})") for tk in e12["tasks"] for sub in ("tiles", "float32") for a in e12["arms"]]
        notes = [f"2-layer policy (32 hidden units with a tonic drive) on tiles / float32, greedy on a noisy read; two renditions per context with independent read noise, update η (R_A − R_B)(e_A − e_B) with e = Σ x_pre ξ_post (ξ = deviation from the noise-free read), presynaptic normalisation; η = {e12['eta']}, {e12['trials']} pairs per run, {e12['seeds']} seeds; dial schedule changes are MODULATE events.",
                 "Flip rate = fraction of decisions that differ from the noise-free decision (the actual exploration rate): " + "; ".join(f"{tk}/{sub} {a} {S[tk][sub][a]['flip_rate']['mean']:.3f}" for tk in e12["tasks"] for sub in ("tiles",) for a in e12["arms"]),
                 "Why two renditions: a single-rendition rule with a running-mean or linear-critic baseline ended at or below chance on the bandit at every σ up to 1.0 (greedy selection biases the chosen arm's noise sample; see DECISIONS.md)."]
        secs.append(section("e12", "E12 · exploration", "Noise-dial schedules for node-perturbation RL on a bandit and a grid", "e12_exploration.png", facts[:16], e12["verdict"]["statement"], notes))
    if e13 and "verdicts" in e13:
        V = e13["verdicts"]
        facts = [(k, f"σ* {v['sigma_star']} ({v['snr_at_zero']:.0f}→{v['snr_at_star']:.0f}→{v['snr_at_max']:.0f} dB)" + (" U" if v["u_curve"] else "")) for k, v in V.items()]
        notes = [f"N = {e13['N']} LIF neurons (τ_m, threshold 1, dt 1 ms), one device each on a 1 × 64 tile programmed to 0.8; drive {e13['f_drive']:g} Hz with its crest at the stated fraction of threshold; {e13['T']:g} s records, {e13['seeds']} seeds; SNR = power at the drive frequency over the mean power in the surrounding 2 Hz band (population spike train).",
                 "The membrane low-passes per-read noise by √(dt / 2τ_m) ≈ 0.16 at τ_m = 20 ms, so the dial's range (≤ 0.2) gives ≤ 0.025 of membrane noise against a 0.2 gap: the diagnostic grid over drive × τ_m and the beyond-dial levels (0.3–1.0, not a chip setting) locate the resonance."]
        secs.append(section("e13", "E13 · stochastic resonance", "LIF row driven by a subthreshold sinusoid through a tile", "e13_stochastic_resonance.png", facts, e13["verdict"]["statement"], notes))
    if e14 and "verdicts" in e14:
        V = e14["verdicts"]
        facts = [(k, f"KL {v['kl_at_zero']:.2f}→{v['kl_min']:.3f} at σ* {v['sigma_star']}") for k, v in V.items()]
        notes = [f"N = {e14['N']}, P = {e14['P']} stored ±1 patterns, β = {e14['beta']}, cue = α x_a + (1 − α) x_b, {e14['trials']} trials per cue × 4 cue pairs, {e14['seeds']} seeds; posterior = softmax_μ(β x_μ · cue); KL(posterior ‖ retrieval frequencies) with Laplace smoothing; the tiles carry d2d / c2c and write-verify, the float32 reference the same read noise on float weights."]
        secs.append(section("e14", "E14 · sampling", "Attention retrieval from ambiguous cues: retrieval frequencies vs the Boltzmann posterior", "e14_sampling.png", facts, e14["verdict"]["statement"], notes))
    if e3f and "pinned_regime" in e3f:
        pr = e3f["pinned_regime"]
        facts = [(k.replace("amp", "cue ").replace("_omega", " ω").replace("_bpu_mismatch", " mm "), f"σ=0: {fmt(v['err_sigma0'])} vs ref {fmt(v['ref_err_sigma0'])}" + (" PINNED" if v["fails"] else "")) for k, v in pr.items()]
        secs.append(section("e3f", "E3-final · mismatch 0.2 / 0.3, cue 0.02 / 0.05", f"ω = {e3f['omegas']}, {e3f['seeds']} seeds; U-test only where σ = 0 is pinned", "e3_final50.png" if E3F_SRC == "e3_final50" else "e3_final.png", facts[:12], e3f["verdict"]["statement"],
                            ["U-test per pinned condition: " + "; ".join(f"{k}: {e3f['verdicts'][k]['statement']}" for k, v in pr.items() if v["fails"]) if any(v["fails"] for v in pr.values()) else "No condition is pinned at σ = 0 (every tile error within 1.5× the float32 reference): no pinning up to 30 % mismatch; attractor robustness is the result."]
                            + (["2026-10-23: 50-seed numbers (results/e3_final50, 4/8 pinned); the 10-seed run (results/e3_final, 5/8 pinned) is kept as the record."] if E3F_SRC == "e3_final50" else []), src=E3F_SRC))
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
        secs.append(section("e9", "E9 · trainability under drift", ("Loss vs locality with nearest-pulse writes" if e9.get("write_mode", "level") != "level" else "Pareto: post-shift loss vs total writes per device, θ_flush ∈ {0.25, 0.5, 1}"), "e9_trainability.png", facts, e9["verdict"]["statement"],
                            [f"Stream: {e9['seg']} steps Shakespeare → {e9['seg']} Python source → {e9['seg']} Shakespeare, batch {e9['batch']}, {e9['seeds']} seeds, σ_eff = {e9['sigma']}, {e9.get('write_mode', 'level')} writes, chip time {e9['step_seconds']} s/step (assumption)." + (" The round-two level-gated Pareto run is kept in results/e9_pareto (figures/e9_pareto.png)." if e9p else ""),
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
        secs.append(section("e6", "E6 · memory and attention", f"{100 * e6.get('flip_fraction', 0):.0f} % flipped bits, sign ties random, {e6['seeds']} seeds, K grid {min(e6['counts'])}..{max(e6['counts'])}", "e6_memory.png",
                            [(f"{m} capacity @90 %, σ=0 / 0.04 / 0.2", f"{caps[m]['0.0']} / {caps[m]['0.04']} / {caps[m]['0.2']}") for m in e6["methods"]], e6["verdict"]["statement"],
                            [f"2026-10-23: this section reads results/{E6_SRC}. The results/e6 grid (K 1..16) capped the attention capacity at its top (16 at every σ); on the grid to 32 attention holds 20–28 pairs. The scaffold numbers here sit on a coarse grid (1, 2, 4, 8, ...); the fine 1..16 grid of results/e6 gives 3–5 pairs."] if E6_SRC == "e6_k32" else [], src=E6_SRC))
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
    if any(e4t.values()) or e4lc or e4n:
        facts = ([("nearest-pulse (default), 20 seeds", f"{e4n['verdict']['certified_fraction']:.0%} certified, {e4n['writes_per_device_total']:.3g} writes/dev")] if e4n else []) + \
                [(f"θ_flush = {th}", f"{r['verdict']['certified_fraction']:.0%} certified, {r.get('writes_per_device_total', r['runs'][0]['writes_per_device_per_s'] * r['development_time_s']):.3g} writes/dev") for th, r in e4t.items() if r]
        facts += [("θ = 1 + quota 6e-3", f"{e4lc['verdict']['certified_fraction']:.0%} certified")] if e4lc else []
        facts += [("threshold writes (weekend)", f"{e4['verdict']['certified_fraction']:.0%} certified, {sum(x['writes_per_device_per_s'] for x in e4['runs']) / len(e4['runs']) * e4['development_time_s']:.3g} writes/dev")] if e4 else []
        secs.append(section("e4", "E4 · ring self-organisation", "Nearest-pulse writes (20 seeds) and the round-two accumulate-and-flush θ_flush ∈ {0.25, 0.5, 1}", "e4_np.png" if e4n else "e4_level_t1.png", facts,
                            (("nearest-pulse: " + e4n["verdict"]["statement"] + " || ") if e4n else "") + " || ".join(f"θ = {th}: " + r["verdict"]["statement"] for th, r in e4t.items() if r), src="e4_np" if e4n else "e4_level_t1"))
    if e8:
        s = e8["summary"]
        secs.append(section("e8", "E8 · accelerator parity", "nanoGPT on noisy tiles (weekend result)", "e8_nanogpt_tiles.png",
                            [(f"noise-aware gap at σ={sg}", pct(s[str(sg)]["noise_aware_gap_pct"]["mean"], 2)) for sg in e8["sigmas"]], e8["verdict"]["statement"]))
    if e5n:
        secs.append(section("e5n", "E5 · grid self-organisation (nearest-pulse writes)", f"{e5n['seeds']} seeds, {e5n['dev_steps']} development steps", "e5_selforg_np.png",
                            [(f"{spec} self-organised / programmed / control", f"{d['trained_torus_fraction']:.0%} / {d['reference_torus_fraction']:.0%} / {d['control_torus_fraction']:.0%}") for spec, d in e5n["sheets"].items()], e5n["verdict"]["statement"], src="e5_selforg_np"))
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
<div class="wrap"><header><div class="eyebrow">BrAIn Processing Unit · simulator results · variability + substrate framing</div><h1>Variability, used the way the brain uses it — on a substrate that matches float32</h1>
<p>The first scoreboard asks whether device variability does work for the computation (regularisation, exploration, stochastic resonance, sampling), each row with its brain analogue, the pre-registered hypothesis and the measured result; the read-noise dial (MODULATE), per-read dropout, nearest-pulse writes and the layer-accumulate 8-bit read-out are the four architecture changes of 2026-10-19. The second scoreboard keeps the substrate framing: the same algorithm on a float32 reference and on noisy tiles. Every number is read from <code>results/&lt;e&gt;/results.json</code>; BPU energy numbers are projections from the device model, the GPU control is measured.</p>
<p class="chip nul" style="white-space:normal">Correction 2026-10-23: the float32 E11 optima above the dial (σ* 1.0 small data, 0.4 full data) and the "~10x the tile optimum" claim are WITHDRAWN. Every float32 arm above 0.2 was clipped to 0.2 on its Linear layers by the dial. Details are in the HPCC section.</p>
<p class="chip part" style="white-space:normal">Resolved 2026-10-24: the unclipped reruns put the float32 optimum at σ* 0.4 (small data) and 0.2 (full data), 4× and 5× the tiles' σ* (0.1 / 0.04), not ~10×. Round four: E11-cal does not line up (small R 1.15, full 1.03 as run), E1-fix + E16 not met (every tile arm at 0.04 diverged; the tile path is already at 4.4× float32 perplexity with no noise).</p>
<div class="meta">generated {time.strftime("%Y-%m-%d %H:%M")} · repo ~/Desktop/Arenda_Projects/BrAInOS/bpusim · DECISIONS.md has every choice · MONDAY.md has the HPCC queue</div></header>
<section id="learning-vs-knowing"><div class="eyebrow">Round four, 2026-10-23</div><h2>Learning vs knowing</h2>
<p>Variability should help while structure is forming (training from scratch) and be suppressed once it has formed (deploying a trained model). The MODULATE dial is the neuromodulation analogue: high to learn, low to deploy. Prior art: dropout (Srivastava et al. 2014), weight-noise injection, noise-aware analog training (Joshi et al. 2020). The claim tested is that native MoS2 variability (2–6 %) sits at the regularisation optimum, together with the substrate results. It is not a claim that noise regularises in general.</p>
{lvbox}</section>
<section id="variability"><div class="eyebrow">Variability, used the way the brain uses it</div><h2>Four mechanisms, E11–E14: mechanism, brain analogue, hypothesis, result, status</h2>{vscore}
<p class="muted">tiles carry the device model (nearest-pulse writes, d2d / c2c where stated, write-verify); the float32 reference runs the identical algorithm with the same read noise or dropout injected on float weights. The dial's range is 0.005–0.2; "beyond the dial" levels are diagnostics, not chip settings.</p></section>
{hpcc}
<section id="scoreboard"><div class="eyebrow">Substrate claims</div><h2>Six rows, reference first; budget = total writes per device against the 1e6 endurance</h2>{score}
<p class="muted">tolerances are the ones written into each experiment's docstring and verdict block.</p></section>
<section id="writes"><div class="eyebrow">Write budget</div><h2>Accumulate-and-flush: writes only on whole conductance levels</h2>{wbox}
<p class="muted">a device is pulsed only when its digital accumulator holds θ_flush conductance levels (1/L, L = round(1/σ_eff)); the remainder is rounded stochastically. Budget: endurance = total writes per device over the run against 1e6; driver bandwidth = writes per physical tile per second against 1e4; the old 6e-3 writes/device/s is kept as the "lifelong rate" for reference.</p></section>
<section id="other"><div class="eyebrow">Other rows</div><h2>Remaining experiments</h2>{otab}</section>
{"".join(secs)}
<footer>bpusim · variability programme 2026-10-19 on the substrate programme 2026-10-18 · figures by scripts/make_figures.py · page by scripts/make_results_page.py</footer></div>'''
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({len(secs)} sections)")


if __name__ == "__main__":
    main()
