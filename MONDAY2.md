# MONDAY2 — round four on Pinnacle: "learning vs knowing"

Brief: Arjun, 2026-10-23 (E11-cal, E1-fix + E16, E15, optional E17, housekeeping). Cluster layer and quirks:
`scripts/hpcc/submit.sh` (header) and `MONDAY.md`. Pre-registrations are in each experiment's docstring and were
written before submission. Every number is read from `results/<e>/results.json`.

## Queue (submitted 2026-10-23 23:1x CDT)

| # | Job (submit.sh mode) | Job IDs | Partition | What it settles |
|---|---|---|---|---|
| 1 | E11-cal `e11cal`: retrain the σ = 0 arms (seeds 0–4, both regimes, both substrates) + σ* robustness checkpoints, measure every layer's rel_pert over the E11 grids as run and unclipped | **1366758** | gpu06 | do the tile and float32 optima line up in rel_pert space |
| 1b | E11 float32 unclipped reruns above the dial top (small data 0.3–3.0, 5 seeds; full data 0.3–1.4, 10 seeds), then `e11cal --analyze` | **1366759** → **1366760** (afterany) | gpu06 | the real float32 curve above 0.2 (see the finding below) |
| 2 | E1-fix lane A `e1fix`: deploy-only arms (both substrates, 3 seeds, E16 dial-low + averaging) + float32 fine-tunes (σ 0/0.02/0.04/0.06 × 1200/2000 steps) + on-chip diagnostic | **1366761** → resume **1366762** | gpu06 | baselines, deploy-only gaps, matched float32 references |
| 2 | E1-fix lane B: tile off-chip noise-aware fine-tunes (σ 0.02/0.04/0.06 × 1200/2000 steps; σ 0.04 also evaluated dial-low / k-averaged) | **1366763** → resume **1366764** | gpu06 | the 3 % criterion at σ_device = 0.04 |

Already in the queue: bpu-e8-full 1364798 (comp72, running). E9 seg2000 1364797 finished (committed 956f9f1).
Compute share at submission: 3 BPU GPU jobs running or pending at once (the resume jobs wait on afterany), 1 CPU.

## Finding made while building E11-cal (2026-10-23): the MODULATE dial clipped the float32 reference

`TileLinear.set_noise` clips to the dial range [0.005, 0.2] and E11 set the float32 reference through it too, so every
E11 arm with σ > 0.2 (float32 brackets to 3.0 and 1.4, and the tiles' beyond-dial 0.3) ran its Linear layers at
σ_r = 0.2; only the attention cache followed σ. The reported float32 optima (1.0 small data, 0.4 full data) are
therefore optima over attention-cache noise at a fixed 0.2 on the Linear layers, not over σ. E11-cal measures the
perturbation each arm actually had ("as run"), and 1b reruns the float32 arms above 0.2 with the noise set directly
(`--float-unclipped`, `results/e11_f32_unclipped`, `results/e11_full_f32_unclipped`). The earlier results are left as
they are; the correction is dated in DECISIONS.md.

## Decisions taken while building

* E11-cal runs on a V100, not CPU: E11 kept no checkpoints (`run_one` deleted the model), so 20 + 4 models have to
  be retrained with the E11 code (~1.5 h on a V100, ~6 h+ on a CPU node). The retrained σ = 0 val loss is checked
  against E11's own run of the same seed.
* E1-fix programs GPT-2 with a per-column scale (max |W| of each output column, a column-periphery gain) so that no
  weight is clipped. The noise-aware fine-tune is off-chip and hardware-aware (Joshi et al. 2020: float32 master, the
  forward runs through the programmed tiles, straight-through gradient, programmed once at the end). The reason: the
  16-bit accumulator's step is scale × 2⁻¹⁵ and one pulse is ~0.02 × scale, so lr 2e-5 updates hardly ever reach a
  pulse. An on-chip nearest-pulse fine-tune at 0.04 × 2000 steps is kept as a diagnostic. GPT-2's attention products
  stay digital (as in E1).
* E1-fix fine-tunes use 1 seed (as E1); deploy arms use 3.

## HOLD (2026-10-23 ~23:20 CDT, coordinator): no NEW Pinnacle submissions

AHPCC's account policy says "Commercial usage is not allowed"; BrAInOS/BPU is BrAIn Corp work, so Arjun decides
whether it qualifies as research. The seven jobs above were submitted BEFORE the hold (23:1x) and are left alone (not
cancelled). Nothing else is submitted. Ready or being prepared locally, not submitted:
* E15 data prep (CPU, comp06), the E15 throughput timing job (one gpu06 + one agpu06, synthetic tokens), and the
  E15 worker chains: code in progress (`experiments/e15_*.py`, submit.sh modes `e15-data`, `e15-timing`, `e15`).
* E17: not written (optional, after E15).
If the answer is no, the queued jobs 1366759–1366764 can be cancelled by Arjun (`scancel` by ID only).

## E15 (next): see the section below once the timing job reports
