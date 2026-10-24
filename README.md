# bpusim

Simulator for the **BrAIn Processing Unit (BPU)**: a non-von Neumann
architecture on stochastic MoS2 memristor synapses.  The package implements
the equations of *The BPU from First Principles* layer by layer and the
experiment scripts reproduce E1-E7 of the paper.  Everything from E3 to E7
runs on a laptop CPU; E1 and E2 use PyTorch and accept `--device cuda`.

```
bpusim/                     library (NumPy; see module docstrings)
  device.py      L0  memristor: w in [0,1], saturating pulses, d2d/c2c/read noise, retention, dead mask, write-verify
  tile.py        L2  differential-pair crossbar with mask A; read(x), update(dW) -> pulses, accumulate(dW) write budget, program()
  torchtile.py   L2  PyTorch tiles for E8-E10 (CPU or CUDA): level/pulse writes, FA, DFA, read-out emulation (ADC / LIF)
  neurons.py         LIF and rate neurons with additive noise
  plasticity.py  L1  STDP (window + online traces), eligibility trace, three-factor + surprise gate, homeostasis, Oja
  growth.py      L3  C_ij = <m x_i xi_j>, grow/prune within a budget, spawn
  attractor.py   L4  RingAttractor, GridModule, SelfOrganizingSheet (+ exploration cues)
  memory.py      L5  ScaffoldMemory (pseudo-inverse / delta rule, recall loop), HopfieldAttention
  modules.py     L8  Module, ModuleGraph (routing R, node-perturbation learning, route growth), layered_routing
  topology.py        ripser/gudhi persistent homology, circular-shift null, ring/torus certification
  metrics.py         SV, TC, FPL (WhiteBox), Newman modularity Q, mutual information
  noise.py           noise configurations and sweep helpers
experiments/   e1_gpt2_noise.py ... e7_modularity.py, common.py
tests/         pytest (67 tests; one pre-existing E7 route test fails, see MONDAY.md)
scripts/       run_substrate.sh, run_weekend.sh, run_all.sh, make_figures.py, make_results_page.py
results/<e>/results.json, figures/<e>.png
DECISIONS.md   every modelling choice and the measurement that forced it
```

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # torch/transformers/datasets only needed for E1/E2
pip install -e .                      # or run with PYTHONPATH=.
pytest -q
```

## Round four (2026-10-23): learning vs knowing

Variability should help while structure is forming (training from scratch) and be suppressed once it is formed
(deploying a trained model); the MODULATE dial is the neuromodulation analogue, high to learn and low to deploy.
Prior art: dropout (Srivastava, Hinton, Krizhevsky, Sutskever & Salakhutdinov 2014, JMLR 15), weight-noise injection
during training (e.g. Murray & Edwards 1994; Graves 2011), and noise-aware training for analog in-memory computing
(Joshi et al. 2020, Nature Communications 11, 2473). The claim here is narrower: native MoS2 variability (2–6 %) sits
at the regularisation optimum, plus the substrate results. It is not a claim that noise regularises in general.

| Exp | Question | Status |
|---|---|---|
| E11-cal | do the tile and float32 E11 optima line up once each run is placed at its measured layer perturbation (rel_pert)? | **does not line up** (2026-10-24). As run (primary): R = 1.15 (CI 1.13–1.16) small data, fails; 1.03 (0.88–1.17) full data, passes; both regimes required. With the corrected float32 curve (secondary): 1.01 (0.995–1.02) small, 0.755 (0.74–0.77) full: still not both |
| E1-fix + E16 | GPT-2 124M deployed on tiles: deploy-only, noise-aware fine-tune, dial-low, read averaging; some arm at σ 0.04 within 3 % of float32? | **not met** (2026-10-24): every tile arm at 0.04 diverged (> 2× reference); best = the pre-registered combined arm (fine-tune 2000 + dial-low + k 8) 69.6 vs 22.33 (+212 %). The tile path is already at 156.6 vs 35.69 with NO noise (programming residual + 8-bit read-out); float32 with the same noise fine-tunes to +0.4 %. The limit is the zero-noise tile floor, not variability |
| E15 | ~50M-parameter GPT trained from scratch on tiles (0.8B tokens): σ* > 0, in [0.02, 0.06], ≤ float32, learn high / deploy low ≥ fixed σ* | code ready, not submitted |
| E17 | representations of the float32 and tiles E15 models (CKA, probe transfer, kNN overlap), exploratory | code ready, after E15 |

Pre-registrations are in each experiment's docstring; design decisions in DECISIONS.md "Round four".

## HPCC runs (2026-10-22/23): the MONDAY queue on the Arkansas HPCC

| Run | Result | Status |
|---|---|---|
| E11 full data, tiles (10 seeds) | σ* = 0.04 on 10/10 seeds, CI excludes 0; val 1.588 → 1.503; shift gap narrows on 10/10 | **supported** |
| E11 float32 brackets | σ* = 1.0 (small data), 0.4 (full data): about 10x the tile optimum in both regimes | bracketed; **WITHDRAWN 2026-10-23** (every arm above 0.2 was clipped to 0.2 on the Linear layers) |
| E11 float32, corrected (unclipped reruns 1366759/60, 2026-10-24) | small data σ* = 0.4 (CI 0.4–0.4; per-seed 0.4 ×4, 0.3), full data σ* = 0.2 (CI 0.2–0.2; 7/10 seeds 0.2, 3/10 0.3): **4× and 5×** the tile σ* (0.1 / 0.04), ratio CIs 4.0–4.0 and 5.0–5.0 over grid points (coarse grids: neighbouring points allow ~2–8× and ~2.5–15×); shift-gap slopes −1.15 (CI −1.17..−1.12) to σ 3.0 and −2.66 (−2.70..−2.61) to 1.4 | bracketed; **replaces the withdrawn values** — the float32 optimum is higher than the tiles', but 4–5×, not ~10×; on full data it is at the dial top, not beyond it |
| E3-final, 50 seeds | 0/8 conditions U-shaped (4 pinned, 0 of those U-shaped) | null (bound) |
| E5 32×32, 20 seeds | self-organised 0/20; programmed reference 26/40 | null (sheet-limited) |
| E12, 50 000 trials | bandit: annealed fastest on tiles; gated not better than fixed; grid task still unlearned | partial |
| E1 GPT-2 124M | best perplexity at σ = 0 (23.38); noise only costs | null |
| E9, 2000-step segments | sgd on tiles +10.0 % vs float32 after the shift (laptop, 1000-step segments: +2.1 %) | **criterion not met** |

**Correction 2026-10-23: the MODULATE dial clipped the float32 reference.** `TileLinear.set_noise` clips to the dial range [0.005, 0.2], and E11 set the float32 reference through it as well, so every E11 arm with σ > 0.2 ran its Linear layers (qkv, proj, fc, fc2, head) at σ_r = 0.2 and only the attention-cache read noise followed σ. Affected arms: every float32 arm above 0.2, i.e. `results/e11` (0.3, the beyond-dial point), `results/e11_f32_bracket` (0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0) and `results/e11_full_f32` (0.3, 0.4, 0.5, 0.7, 1.0, 1.4); and the tiles' beyond-dial 0.3 point (`results/e11`, `results/e11_full_tiles`), a diagnostic that was already excluded from σ* and the slope. **WITHDRAWN pending the unclipped reruns** (jobs 1366759 / 1366760, `results/e11_f32_unclipped`, `results/e11_full_f32_unclipped`): the float32 optima σ* = 1.0 (small data) and 0.4 (full data), the claim that the float32 optimum sits about 10x above the tile optimum ("an order of magnitude less injected noise"), the float32 shift-gap slopes over the extended grids (−0.16 to σ 3.0, −0.66 to 1.4) and every float32 or tiles value quoted at σ ≥ 0.3. **Unaffected:** every arm at σ ≤ 0.2 on both substrates. The tiles σ* = 0.1 (small data) and σ* = 0.04 (full data) stand: both were taken over the dial grid 0–0.2, which contains no clipped point; the 0.3 point was not in the grid, and the full-data bootstrap put all its mass on 0.04 (CI 0.04..0.04). The tiles shift-gap slopes, the twin gaps inside the dial and the dropout arms stand, and so does the round-three statement that the float32 minimum is not reached inside the dial (σ* = 0.2 on the dial top, small and full data). Code fixed the same day: the float32 reference is never clipped (`TileLinear.set_noise`, `FloatTile.set_noise`), device values beyond the dial are set directly as a diagnostic (`e11_regularization.set_all_noise`), `--legacy-dial-clip` reproduces the old behaviour for E11-cal, and `tests/test_torchtile.py::test_float_reference_noise_is_not_clipped_by_the_dial` checks both. No other experiment was affected: E12 stays inside the dial, E13 and E14 set beyond-dial values directly.

**Resolved 2026-10-24** (unclipped reruns 1366759 / 1366760; corrected float32 curve = the original arms at σ ≤ 0.2 plus the unclipped arms above 0.2, same seeds, E11's own σ* rule): float32 σ* = 0.4 on small data (CI 0.4–0.4) and 0.2 on full data (CI 0.2–0.2), i.e. 4× and 5× the tiles' σ* (0.1 / 0.04), not ~10×; corrected float32 shift-gap slopes −1.15 (to σ 3.0) and −2.66 (to 1.4). The withdrawn values stay withdrawn; details in "Round four outcomes (2026-10-24)".

Details and decisions: DECISIONS.md, "HPCC runs". Cluster layer: `scripts/hpcc/` (`submit.sh`, `build_env.sh`).

## Round three (2026-10-19/20): variability, used the way the brain uses it

Four architecture changes (biological analogue in brackets, details in `DECISIONS.md`
"Round three"): write rule default = **nearest_pulse** (16-bit accumulator flushed every
step, no θ_flush gate; plasticity as a continuous process), **`tile.set_noise(σ_r)`** = the
MODULATE primitive, dial range [0.005, 0.2] (neuromodulatory gain control of variability),
**`tile.set_dropout(p)`** per-read synapse failure (probabilistic vesicle release), read-out
default = **layer-accumulate 8-bit at FoM 10 fJ** (dendritic integration, one conversion per
neuron).  E11–E14 ask whether the variability does work; every row has a float32 reference
with the same noise injected on float weights.

| Row | Result (5 seeds unless stated) | Status |
|---|---|---|
| E11 regularisation (nanoGPT 2.7M on 10 % of Shakespeare, σ_r ∈ {0, 0.02, 0.04, 0.06, 0.1} as the only noise; dropout {0.1, 0.3, 0.5}) | tiles: val 3.37 (σ 0) → **2.28 at σ* = 0.1** (CI 0.1..0.1, per-seed 0.1/0.15) → 2.36 at 0.2, 2.37 at 0.3 (beyond dial); float32 3.58 → 2.47 at 0.2 (σ* on the dial top, still falling: 2.41 at 0.3; HPCC bracket 2026-10-23: float32 σ* = 1.0, about 10x the tile optimum) [WITHDRAWN 2026-10-23: dial clip, see the HPCC correction above]; tiles beat float32 by 4–26 % at every σ in the dial (substrate variability shifts the curve left); shift gap tiles 4.52 → 3.85, slope −3.5 (CI −3.8..−3.2, 5/5 seeds), float32 flat inside the dial (HPCC: narrows slowly above it, slope −0.16 to σ 3.0) [withdrawn 2026-10-23: clipped arms]; dropout p* = 0.1 (val 2.43 both substrates), 0.3 / 0.5 hurt | **σ* > 0 with CI ✓** (minimum bracketed on tiles, on the dial top for float32); **shift gap narrows on tiles ✓**, but only above σ* and not on float32; training noise band 0.06–0.15 |
| E12 exploration (node-perturbation RL, two renditions per context; bandit + grid; fixed / annealed / reward-gated σ) | bandit on tiles: fixed 0.1 reaches criterion 3/5 (median 7 042 decisions), annealed 4/5 (9 408), gated 3/5 (13 544); gated vs best fixed 2/5 seeds faster, p = 1; speed ∝ σ^0.07 (CI 0.00..0.17), not σ²; grid not learned in the budget by any arm (≤ 1/5); flip rate (the real exploration rate) 1–5 % of decisions, tiles explore more than float32 | **gated does not beat fixed; not σ²**; the dial's top (0.2) is an order of magnitude too low to be an exploration knob |
| E13 stochastic resonance (LIF row, drive 0.8× threshold through a tile, σ_r 0–0.2; 10 seeds) | headline: **no spike at any dial setting** on tiles or float32; firing starts at σ ≈ 0.7 (SNR 17 dB) and still rises at 1.0 (27.5 dB); diagnostic grid drive × τ_m: 0/10 U-curves within the dial, onset moves into the dial only for τ_m ≤ 5 ms or drive ≥ 0.9 θ | **silent, as predicted before the run** (membrane low-passes read noise by √(dt/2τ_m) ≈ 0.16); the [0.02, 0.06] band is not anchored by SR |
| E14 sampling (Hopfield attention, 8 patterns, cue = blend of two) | α = 0.5: σ = 0 deterministic (KL 2.29), the dial samples the posterior (KL 0.02 from σ = 0.02 on, σ* 0.06, CI 0.03..0.2; float32 the same); α = 0.55 / 0.6: deterministic throughout the dial (KL 0.75 / 0.17 flat), matched only at σ ≈ 0.5–1.0 | **σ = 0 deterministic ✓; KL minimum at σ > 0 at exact ambiguity only**; σ* depends on the cue margin |
| E3-final (ring attractor, mismatch 0.2 / 0.3, cue 0.02 / 0.05, ω 0.5 / 0.1, 10 seeds) | pinned regime found: 5/8 mismatched conditions have σ-0 error > 1.5× float32 (barrier ΔU 0.88 / 1.35); in every pinned condition the error is flat in noise to three decimals, 0/5 U-shaped | **no noise optimum even where pinned**; attractor robustness is the result |
| E6 (K grid 1..16, 10 seeds) | scaffold 3–5 pairs at every σ to 0.2, attention ≥ 16 (grid top), K = 1 exact | unchanged |
| E9 (nearest-pulse default, 5 seeds; loss vs locality) | post-shift loss: float32 1.224; sgd 1.249 (+2.1 %, 18 writes/device); fa 2.404 (+96 %); dfa 2.885 (+136 %); local 2.683 (+119 %) | **met** (sgd within 5 % at 0.0018 % of endurance); every step away from backprop roughly doubles the loss |
| E4 (nearest-pulse, 20 seeds) | **13/20 certified** (2/20 controls), 4.06 writes/device = 0.0004 % endurance, 208 writes/tile/s = 2.1 % bandwidth; vs θ = 1 level rule 3/10 at 0.52 writes/device | certifies 4× as often for 8× the writes, all inside the budgets |
| E5 self-organisation (nearest-pulse, 5 seeds, 16×16 + 32×32) | 0/5 self-organised sheets certify at 16×16 and 0/5 at 32×32 (0/5 controls); programmed Burak–Fiete reference 2/5 at both sizes under the same sampling (800 points); trained sheets fragment into ~90 isolated states (β0 ≈ 99, no H1), the same failure as every earlier run | **not certified**; the write rule is not the blocker, and the certifier's own pass rate on a programmed torus is 40 % at this sample size |

## Round two (2026-10-19): endurance budget, θ_flush, FoM converters, 10-seed E7

Budget redefined: **endurance** = total writes per device over the run against 1e6 cycles,
**driver bandwidth** = writes per 32×32 tile per second against 1e4, the old 6e-3/dev/s kept
as the "lifelong rate". Flush threshold θ_flush ∈ {0.25, 0.5, 1} levels (stochastic rounding of
the remainder). Page order: E10, E9 Pareto, E8b, then the rest. Details in `DECISIONS.md`
("Round two").

| Row | Result (3 seeds unless stated) | Status |
|---|---|---|
| E10 continual (headline: naive, EWC, replay, fpl_replay, joint) | EWC 0.947 / 0.943, replay 0.942 / 0.935 (float32 / BPU); total writes/device 1.8–4.4 (≤ 0.0005 % of endurance); τ_ret = 1e4 s changes nothing except fpl_replay after a 1000 s rest (0.93 → 0.77). FPL alone is diagnostic (gate off beats gate on), not a consolidation policy | **met** |
| E9 Pareto (θ_flush ∈ {0.25, 0.5, 1}) | SGD on tiles: θ = 1 **1.483** (+25 %) at 0.32 writes/device, θ = 0.5 1.426 (+21 %) at 3.5, θ = 0.25 1.954 (+65 %) at 281; FA / DFA / layer-local +112–166 % at every θ; float32 1.184. Nearest-pulse diagnostic (no gate) **1.226 (+3.6 %) at 15.6 writes/device** = 0.0016 % of endurance | **not met by any level-gated arm**; the ungated pulse writes meet both numbers, so under an endurance budget the gate is a cost without a benefit |
| E8b, E_ADC = FoM_W × 2^b | per-layer FoM-10 8-bit **71 nJ/token** (−0.8 % loss) = 0.014× H100 INT8 at 40 % utilisation (5.0 µJ), 6-bit 44 nJ at +1.8 %; per-tile 257 nJ; LIF 11.6 µJ; laptop GPU measured 256 µJ. Ratio BPU/H100@40 % falls from 0.014 (2.7M) to 0.002 (1T) because conversions ∝ layers × d and MACs ∝ params | projection vs measured / published |
| E3 cue sweep (ω = 0.5, cue 0.01–0.1, mismatch 0.05 / 0.1, 10 seeds) | σ = 0 error 0.18 / 0.35 / 0.94 / 1.49 rad (reference) vs 0.18 / 0.36 / 1.06–1.12 / 1.39–1.40 (tiles): weak cues fail on both substrates, mismatch adds ≤ 19 %, **no pinned regime**, 0/8 U-shaped | pinned regime not found at ω = 0.5 |
| E6 (10 seeds, K from 1) | K = 1 recalled exactly at every σ (unit test passes); scaffold 2–4 pairs, attention 20–28 | unchanged |
| E4 θ_flush (10 seeds) | θ = 1: 3/10 certified, 0.52 writes/device; θ = 0.5: 0/10, 3.9; θ = 0.25: 0/10, 2,020 writes/device and 10× the driver bandwidth | θ = 1 only |
| E7 (10 seeds, tiles and noise-free reference) | nav gap grown−fixed −0.38 (CI −0.53..−0.23) on tiles vs −0.24 (−0.44..−0.04) on the reference; continued −0.33 vs −0.11; Welch p(tiles > ref) = 0.85 | **no noise-assisted growth**; negative on both |

## Substrate programme (2026-10-18): same algorithm, float32 vs noisy tiles

The BPU is a **substrate**.  Every row below compares the SAME algorithm on a
float32 reference and on noisy BPU tiles; success means the tiles match the
reference within the stated tolerance.  Our own local rules (feedback
alignment, direct feedback alignment, layer-local three-factor, growth) are
additional arms, never the headline.  Page: `results/index.html`
(`python scripts/make_results_page.py`), recipe `scripts/run_substrate.sh`,
cluster queue `MONDAY.md`, every choice in `DECISIONS.md` (last section).
Three seeds unless stated; the laptop's RTX 5070 ran E9/E10/E8b, the CPU the rest.

| Substrate claim | Exp | float32 / reference | BPU tiles (σ_eff 0.04 unless stated) | tolerance | status |
|---|---|---|---|---|---|
| Trainable where an FPGA is frozen | E9 | SGD post-shift loss **1.184** nats/char | SGD on tiles (ceiling, level-gated writes) **1.483** (+25 %) at **178 %** of the write budget; FA 2.70, DFA 2.89, layer-local 2.70 (+82–95 % vs ceiling, 6–20× budget); frozen 5.98 | within 5 %, writes ≤ 6e-3/dev/s | **not met** |
| Holds continually adapting models | E10 | EWC 0.947, replay 0.942, naive 0.844, FPL 0.840, joint 0.956 | EWC **0.943** (−0.004), replay **0.935** (−0.007), naive 0.793, FPL 0.828, joint 0.957 | EWC and replay within 3 points | **met** |
| Cheaper per token than a GPU | E8b | GPU INT8 **measured**: 256 µJ/token incl. idle, 169 above idle (RTX 5070 laptop, model far from saturating it); H100 INT8 published peak 2.0 µJ | 8-bit per-layer ADC **3.6 µJ** (−0.8 % loss), per-tile 22 µJ, LIF 11.6 µJ; 4-bit cells 0.26–1.4 µJ but +31–53 % loss | projection vs measurement | 47–71× below the laptop GPU, 1.8× above H100 peak at 8 bits |
| Attractor noise optimum (moving cue) | E3 | tracking error 0.036 / 0.182 / 0.372 rad at ω = 0.1 / 0.5 / 1 rad/s | mismatch 0.02: 0.038 / 0.181 / 0.371; 0.05: 0.040 / 0.185 / 0.376; 0.10: **0.062** / 0.181 / 0.373; ΔU 0.11 / 0.22 / 0.41 | U-test over σ ∈ {0 … 0.08} | **0/9 U-shaped**; error flat in σ; tiles ≈ reference except mismatch 0.1 at ω = 0.1 |
| Memory + attention under noise | E6 | σ = 0: attention 20 pairs @ 90 %, scaffold 0 | σ = 0.02–0.2: attention 24 / 24 / 28 / 28 / 28, scaffold ≤ 4 (10 seeds, sign ties random) | capacity holds | attention holds (noise does not limit it) |
| Modular self-assembly | E7 | noise-free tiles: nav grown 0.30 / fixed 0.75 / continued 0.46, recall 0.84 / 0.85 (3 seeds, seed spread ±0.2: the reference is *not* better than the noisy tiles; 10-seed run queued) | nav grown 0.62 / fixed 0.84 at matched synapses+steps; grown **continued** to the fixed step count 0.80 (recall 0.90 vs 0.82); composition at chance for every arm | grown vs fixed, matched synapses AND steps | grown catches up on base tasks; composition null |

**Write budget.**  `Tile.accumulate` / `TileLinear(write_mode="level")`: a
digital per-synapse accumulator, a device is pulsed only when it holds one
conductance level (1/L, L = round(1/σ_eff) = 21–25), stochastic rounding of
the remainder, writes counted only on pulses.  E4 ring self-organisation with
level writes: **3/10 certify at 0.013 writes/device/s = 217 % of the 6e-3
budget** (threshold writes: 5/10 at 1850 %); with the quota on top 0/10.  E9
SGD on tiles: 178 % of budget, and the level gate is what costs the 25 %
(the same SGD with nearest-pulse writes ends at 1.226, +1 % vs float32, at
8665 % of budget: device noise is free, the write budget is not).  E10 arms
run at 7–24× the budget at 10 ms per step.

**Read with care.**  E8b's digital control is a 2.7M-parameter model on a
laptop GPU that it does not saturate (56–64 W of board power), so the 47–71×
is an upper bound on the advantage; against an H100 at published peak the
8-bit BPU projection is 1.8× *worse* and only the 4-bit cells win, which cost
31–53 % in loss without quantisation-aware training.  E10's FPL gate hurts
(gate off: 0.90 / 0.87 vs 0.84 / 0.83 float32 / BPU) and the FPL arms lose
13–16 points after a 1000 s rest; τ_ret = 1e3 s costs every arm 2–17 points
after that rest, ≥ 1e4 s nothing.  E3's error is the cue lag; dynamic noise
never helps.  A replay-buffer bug (mismatched labels) was found and fixed on
the way (`results/e10/results_replay_bug.json`).

## Run

```bash
scripts/run_substrate.sh      # 2026-10-18 programme: E4 (level writes), E6, E3 tracking, E7 (+ reference), E9, E10, E8b (GPU)
scripts/run_all.sh            # E3-E7, 3 seeds, regenerates figures
scripts/run_all.sh --full     # 10 seeds
python experiments/e2_grow_vs_rigl.py --seeds 3            # MNIST (downloads 11 MB), --device cuda optional
python experiments/e1_gpt2_noise.py --smoke                # CPU pipeline check, no downloads
python experiments/e1_gpt2_noise.py --device cuda          # GPU host: GPT-2 124M + WikiText-2
python scripts/make_figures.py                             # figures from existing results.json
```

Every experiment writes `results/<e>/results.json` with a `verdict` block that
states in words what was found, including when the hypothesis was **not**
confirmed.

## Results (3 seeds unless stated; see `results/*/results.json`)

| Exp | What was measured | Outcome (3 seeds unless stated) | Verdict |
|---|---|---|---|
| E1 | GPT-2 124M perplexity through noisy tiles, with noise-aware fine-tuning | **not run here** (GPU host); `--smoke` pipeline check passes on CPU | pending |
| E2 | MNIST MLP at 10 % density: local grow rule vs RigL vs random regrowth, 5 epochs | test acc local 0.914 +- 0.003, RigL 0.923 +- 0.001, random 0.914 +- 0.002 | **success criterion met** (within a few percent of RigL, not worse than random); no advantage over random regrowth |
| E3 | 32-ring on a 32x32 tile, mismatch 0.05, path-integration RMSE vs dynamic noise, **10 seeds** | RMSE 0.38 (sigma 0) -> 0.33 (sigma 0.05, 7/10 seeds better) -> 0.41 (sigma 0.2); no-mismatch control 0.05 -> 0.34 monotonic | **not U-shaped** at p = 0.34; shallow dip inside the error bars |
| E4 | ring self-organised from a sparse random tile (STDP + growth + normalisation), beta1 = 1 vs circular-shift null | 2/3 seeds certified (H1 bars 0.26, 0.20 vs null 0.18, 0.16; third seed 0.10 vs 0.12), no extra β0 components in any seed; 0/3 untrained controls; released bumps cover 6/8 ring bins; 0.11 writes per device per second during development vs the 6e-3 endurance budget | **partial**: ring forms in most seeds, kernel still translation-variant; development is 18x over the write budget |
| E5 | grid self-organisation on 5x6 and 16x16 sheets, torus (1,2,1) vs null | 0/3 + 0/3 self-organised; **programmed DoG reference also 0/3 + 0/3** (5x6: 1-6 H1 bars, 0-1 H2; 16x16: 0-3 H1, 0-1 H2 and 12-39 extra β0 components, i.e. the lattice hops between discrete states instead of sliding); 0/3 controls | **not certified**; the certifier does not pass even the programmed module at these sizes |
| E6 | 32 +-1 pairs on 32x32 tiles, recall from 50 %-erased cues, sigma in {0 .. 0.2} | pseudo-inverse scaffold: >= 90 % exact recall only up to K = 4-8; delta-rule scaffold the same; **attention: 32/32 at every sigma** (0.98-1.00) | device noise is not the limit, the erasure is; attention wins by a wide margin |
| E7 | 8 generic modules, NAV + RECALL, grown module graph vs fixed layered graph at equal synapses | Q(grown) = 0.12 +- 0.01, rises from 0 as 12 routes grow; all modules carry > 0.1 bit about the task; nav accuracy grown 0.15 vs fixed 0.39, recall bits 0.70 vs 0.76; compositional transfer gap +0.01 (p = 1) | **null**: the graph grows and becomes weakly modular but learns the tasks worse than the fixed graph and composes no better |


## Honest summary

* **The device, tile, plasticity, growth, attractor, memory, topology and
  module layers all work and are unit-tested** (54 tests: device bounds and
  noise statistics, read error ~ sigma/sqrt(N), bump iff J1 > 2, exact
  pseudo-inverse recall, ripser certification of a synthetic circle and
  torus, node-perturbation sign recovery, route growth toward the output).
* **E3 does not show the predicted U.** With 10 seeds the mismatched ring's
  error dips from 0.38 to 0.33 rad at sigma = 0.05 and rises again, but 7/10
  paired seeds is p = 0.34.  Mismatch itself is very visible (0.38 vs 0.05 rad
  without it), so pinning is real; the noise-assisted escape is at best a
  small effect at this N and T.  `figures/e3_drift.png` has the curve.
* **E4 is a partial success.** Growing a ring from a sparse random tile
  required seven measured fixes (DECISIONS.md); the result certifies in 2 of
  3 seeds against 0 of 3 controls, with the learned kernel a clean cos-like
  profile, but the free bump still prefers parts of the ring.  Barcodes are
  in the figure and `results/e4/results.json`.
* **E5 is a negative result with a caveat.** No self-organised sheet
  certifies as a torus, and neither does the *programmed* grid module at 5x6
  or 16x16 under the same sampling and null, so the experiment as specified
  cannot distinguish a failed self-organisation from a certifier that needs
  larger sheets / denser sampling.  The 5x6 sheet is too small for a 2D
  attractor at all (the bump hops between columns).
* **E6 is clear.** Half-erased cues defeat the pseudo-inverse scaffold beyond
  ~8 pairs on a 32x32 tile regardless of device noise, while Hopfield
  attention through the same noisy tiles recalls all 32 patterns at every
  noise level up to sigma = 0.2.
* **E7 is null.** Routes grow where the node-perturbation statistic says
  they should (toward the output modules) and Q rises to ~0.12, but the
  local learner is too weak: the grown graph solves the tasks worse than a
  fixed layered graph with the same synapse budget, and neither composes.
* **E2** meets the technical document's success criterion (within a few
  percent of RigL, not worse than random) but shows no advantage over
  random regrowth.  **E1** is ready for a GPU host.
* **Write budget.** Self-organisation in E4 costs 0.11 writes per device per
  second, 18x the endurance budget of 6e-3 from the technical document; a
  developmental phase of this intensity has to be short or sparser.

The certifier follows the technical document: threshold at the 99th
percentile of the longest null feature, and a ring/torus certificate also
requires no extra persistent β0 component (fragmentation by pinning).  The
alignment of the code with the *Technical Document* and the *Patent-Style
Disclosure* is itemised at the end of `DECISIONS.md`.

Timings on this laptop (20-core CPU): E3 23 s, E4 48 s, E5 ~10 min
(16x16 tiles), E6 109 s, E7 100 s; `scripts/run_all.sh` ~15 min with 3
seeds when nothing else is running.  E2 ~1 min for 3 seeds x 5 epochs.

## Reference equations

The code implements exactly the equations listed in the task brief (device
step law and noise model, differential-pair tile, LIF/rate neurons, STDP /
eligibility / three-factor / homeostasis / Oja, covariance-driven growth with
budget and pruning, ring kernel `J0 + J1 cos d` with velocity shift
`-J1 sin d`, difference-of-Gaussians grid sheet, pseudo-inverse / delta-rule
heteroassociation with attractor cleanup, Hopfield attention
`X softmax(beta X^T xi)` through the noisy tile, persistent-homology
certification against a circular-shift null, and the module graph with
routing matrix, modularity and task information).  Where an equation left a
free choice, `DECISIONS.md` records what was chosen and why.

## Weekend programme (2026-10-17): claims E8–E10 and the fixed E2–E7

Full details: `results/index.html` (built by `scripts/make_results_page.py`),
`DECISIONS.md` (end), `MONDAY.md` (cluster jobs), `scripts/run_weekend.sh`.
Long jobs ran as user services; logs are in `logs/`.

### The three claims

**E8 — accelerator parity (laptop stand-in for E1).** nanoGPT (6 layers,
d = 192, 2.72M parameters, context 128) on tiny Shakespeare, 2 seeds:
ideal val loss 1.675. Every Linear and the attention
cache run through the tile model (pulsed soft-bound writes, read noise per
read, Hopfield-form attention through volatile crossbars).

| noise σ_eff | deploy only (load ideal weights, no training) | noise-aware fine-tune (300 steps) |
|---|---|---|
| σ = 0.0 | +0.49 % | -1.09 % |
| σ = 0.02 | +0.53 % | -0.95 % |
| σ = 0.04 | +0.56 % | -0.85 % |
| σ = 0.06 | +0.78 % | -0.71 % |
| σ = 0.1 | +1.48 % | -0.17 % |

**Criterion met**: noise-aware tiles stay within 2 % of the ideal at every
σ ≤ 0.06 (they are slightly *better* because fine-tuning continues
training); deploy-only is within 1 % up to σ = 0.06 and +1.5 % at σ = 0.10.

**E8b — energy and latency, a projection.** With the stated constants
(4 fJ/MAC, 256 pJ per 8-bit conversion, 100 pJ/pulse, 10 pJ/event; digital
3.7 pJ/MAC + 640 pJ per weight fetch) the result hinges on where values are
converted: converting every 32×32 tile output costs 22.3 µJ/token,
**twice the digital baseline** (11.0 µJ/token); accumulating partial
sums in analog across the tiles of a layer and converting once per output
gives 3.6 µJ/token, **3× better**, with conversion still 99 % of the
budget and MAC energy only 11.3 nJ/token. Projected latency
37 µs/token vs 357 µs/token measured on this CPU. Nothing on the BPU
side is measured on hardware; the paper must state it that way.

**E9 — trainability under drift.** Same architecture as E8, LOADed into tiles at σ = 0.04, then a stream of
1000 steps Shakespeare → 1000 Python source → 1000 Shakespeare. Segment-end losses
(nats/char) frozen / local three-factor / feedback alignment: Shakespeare
1.67 / 4.47 / 2.39, Python 6.02 / 4.52 / 2.39, Shakespeare again
1.67 / 4.49 / 2.18. **Feedback alignment through the tiles does adapt**
(it halves its gap on the new corpus within roughly 60 steps and ends 3.6
nats below the frozen model there), but its pulsed, noisy updates also cost
0.7 nats on the original corpus (2.39 vs the frozen 1.67 at the end of each
Shakespeare segment); the Hebbian three-factor rule degrades the model on
every segment; and neither meets the letter of the criterion: 90 % recovery
takes 1226 and 599 steps for FA (limit 300), and both local rules write
1.41–1.69 times per device per second at the assumed 10 ms per step,
about 250× the 6e-3 endurance budget. **Criterion not met**; the frozen
accelerator's failure (6.0 nats on the new corpus, no recovery) is
confirmed, trainability on-chip is shown for FA at an unaffordable write
rate.

**E10 — continual learning.** Permuted MNIST, five tasks, tiled
784-256-256-10, 3 seeds. Average accuracy after the sequence: joint
0.954, BPU with volatile/nonvolatile consolidation 0.790
(0.809 after one τ_f of rest), naive fine-tune 0.845, frozen
0.262; forgetting BPU 0.209 vs naive 0.140. **Criterion not
met**: the persistence-gated transfer forgets more than naive fine-tuning.

### The substrate, after the fixes

* **E3** (N = 128 on 16 tiles, T = 30 s, 20 seeds paired by mismatch draw,
  mismatch 0.02/0.05/0.10, homeostasis off and on): **0/6 mismatched
  conditions U-shaped.** Homeostasis off: RMSE at zero dynamic noise
  0.25 / 0.51 / 1.14 rad for the three mismatch levels (control 0.19), each
  curve flat-to-rising with its best interior point never significant
  (p ≥ 0.82); the measured pinning barrier ΔU grows with mismatch
  (0.11 / 0.23 / 0.41 vs 0.001 without mismatch), so pinning is real but
  dynamic noise does not release it at these N and T. Homeostasis on
  (τ_H = 5 s, read-side gain) destroys path integration outright (RMSE
  ≈ 1.6 rad at every noise level, including the mismatch-free control): a
  dwelling bump adapts its own gain away. The fitted E(D) model puts σ* at
  the lowest or highest swept value in every condition, i.e. no interior
  optimum. N = 256 (homeostasis off, 10 seeds) **0/3 U-shaped** as well: RMSE at zero noise 0.16 / 0.28 / 0.51 rad (control 0.17), best interior points not significant (p ≥ 0.11); ΔU 0.04 / 0.10 / 0.23, i.e. the pinning barrier roughly halves from N = 128 to 256 at fixed relative mismatch (and the zero-noise error with it), consistent with the paper's 1/N scaling of the mismatch landscape, but it is dynamic noise, not size, that was supposed to give an optimum. Nine mismatched conditions over two sizes and two homeostasis settings, none with a noise optimum: the attractor noise-optimum hypothesis is not supported by this simulator at these scales.
* **E4** (10 seeds, sweep cue): 5/10 trained rings certified beta1 = 1 against the circular-shift null (1/10 untrained controls certified); released bumps occupy 4.7/8 bins of the ring on average (8 = a fully explored ring); development 40 s at 0.111 writes/device/s (uncapped). Capped at the endurance
  budget: 0/10 trained rings certified beta1 = 1 against the circular-shift null (1/10 untrained controls certified); released bumps occupy 4.6/8 bins of the ring on average (8 = a fully explored ring); development 720 s at 0.00598 writes/device/s (capped at 0.006).
* **E5**: the programmed Burak–Fiete module at 32×32 (1024 neurons on 1024
  tiles) certifies as a torus in 100% of seeds once the certifier
  uses a persistence-gap rule, an H0 fragment rule and a dense 30 s
  trajectory with frequent turns (DECISIONS.md). The 48×48 reference (one seed, 2304 neurons on 2304 tiles of 5.3 million devices each, 9 h single-threaded) certifies too: H1 bars 5.3 and 5.2 against a null of 0.94, one H2 bar 3.3 against 0.28, no fragments. The certifier therefore passes the programmed module at both sizes the plan named; The one 32×32 self-organisation attempt (12 000 development steps, 100 release kicks) does **not** certify: after development the sheet is uniformly active with no bump (activity on every site, mean 0.29 of the maximum), and the released states form 99 separate clusters, as does the untrained control. A fourth programmed reference computed inside that run, with a different mismatch draw, also failed (78 near-equal H1 bars, the hopping-lattice signature), so the programmed 32×32 module certifies in 3 of 4 draws; the dependence on the mismatch draw is on the Monday list. The 5×6 and 16×16 verdicts from the first version stand: no torus, and BPU-1 tile B (a 5×6 grid) should be reassigned.
* **E6** (20 % of cue bits flipped): Capacity (largest K with >= 90% exact recall) per sigma: pinv_loop: s=0.0->0, s=0.02->4, s=0.04->0, s=0.08->0, s=0.12->4, s=0.2->4; pinv_direct: s=0.0->0, s=0.02->4, s=0.04->0, s=0.08->0, s=0.12->4, s…
* **E7** (layer-local learning, 32-neuron modules, 9000 + 3000 trials):
  Q = 0.11 ± 0.05 and rising, 92 % of modules task-informative;
  base tasks learned (navigation grown 0.62 vs fixed 0.84, recall bits
  0.80 vs 0.82); compositional gap +0.04 at matched synapses,
  +0.03 at matched compute, with the composite task at chance for every
  arm. Node-perturbation arm: Q 0.10, navigation 0.15 / 0.31.
  A write-dropping bug in the first E7 was found and fixed (DECISIONS.md).
* **E2 CIFAR-10** (3072-512-512-10, 3 % density, 5 epochs): local
  0.420, RigL 0.433, random 0.433: refuted by the document's
  criterion (clearly worse than random regrowth).
