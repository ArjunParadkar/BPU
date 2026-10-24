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

## E15 + E17: ready, NOT submitted (hold)

Code: `experiments/e15_data.py` (tokeniser job), `experiments/e15_scratch_lm.py` (`--timing`, `--worker`, `--summarize`),
`experiments/e17_representations.py`. Smoke-tested on the laptop (single process, tiny model): the data job streams OpenWebText, TinyStories and codeparrot-clean-valid (90 s for the smoke sizes); a paused run resumes and
reproduces the uninterrupted run exactly; the summary and the timing paths run; E17 runs on the smoke checkpoints.

Submission recipe once the hold is lifted (on the login node, from ~/bpusim, after syncing the code):
1. `bash scripts/hpcc/submit.sh e15-data` → job D (comp06, one node, ~1 h expected: streaming + tokenising 0.85B tokens).
2. `bash scripts/hpcc/submit.sh e15-timing` (one V100 + one A100 job, synthetic tokens, no data needed). GPU cap: BPU may
   hold 4 GPU jobs at once; E11-cal (1) + E1-fix (2 lanes) hold 3, so submit the timing jobs one at a time or after a
   lane frees.
3. Budget decision (rule fixed now, before the timing numbers exist): per GPU type, hours per seed
   h = 4 x 0.8e9 / tps_float32 + 5 x 0.8e9 / tps_tiles (9 trainings; the dial arm reuses tiles 0.04). Budget B = 120 GPU-h
   (2 lanes x ~60 h of 6 h segments in this queue). 3 seeds if 3h <= B; else 2 seeds if 2h <= B; else 2 seeds and the
   token budget cut to 0.8e9 x B / (2h) for every arm alike. Use the faster GPU type for all lanes (A100 TF32 if
   agpu06 starts within hours, otherwise V100). Record the decision here and in DECISIONS.md before step 4.
4. `E15_TOKENS=<n> E15_SEEDS=<k> E15_LANES=2 E15_PART=<gpu6|agpu6> E15_AFTER=<D> bash scripts/hpcc/submit.sh e15`
   (8 chained 6 h jobs per lane; lanes share the run list through per-run locks; `E15_LONG=1` adds a gpu72 lane).
5. After E15: `python e15_scratch_lm.py --summarize` (any node, or the laptop after copying results/e15/runs/*/final.json),
   then optionally `bash scripts/hpcc/submit.sh e17`.

## HANDOFF (2026-10-23 ~23:25 CDT)

* State: 7 BPU round-four jobs queued (IDs above), all before the hold; nothing else submitted. Commits c8ad4df, f1260d0,
  4999e58 and the round-four code/housekeeping commit after it. Cluster code synced at 702f33b, E15/E17 included (re-sync after edits with `tar czf - bpusim/*.py experiments/*.py tests/*.py scripts/hpcc/submit.sh | ssh pinnacle 'tar xzf - -C ~/bpusim'`).
* Pages: results page v10 (correction) at https://claude.ai/artifact/9C4QR3DKLAnQQkuxpKyHnc; the regenerated page with
  the "Learning vs knowing" section and the E6 / E3 fixes is committed but republished only together with the next
  results; status page v4 at https://claude.ai/artifact/RHXMs9bjcVxud9PdZVXZpn.
* When jobs finish: copy results back (`ssh pinnacle 'cd ~/bpusim && tar czf - results/e11cal results/e11_f32_unclipped
  results/e11_full_f32_unclipped results/e1fix logs/hpcc' | tar xzf - -C .`), read every log, run
  `python experiments/e11cal_noise_calibration.py --analyze` and `python experiments/e1fix_gpt2_deploy.py --summarize`
  locally (reads JSON only), then `make_figures` / `make_results_page`, DECISIONS outcomes, README table, republish.
* If a job fails twice: stop and report (brief rule).

## Correction published (2026-10-23 ~23:45 CDT)
Dated corrections beside the old text in DECISIONS.md (round-three E11 protocol and outcome, HPCC decisions and
outcomes, new section "Correction 2026-10-23"), README (HPCC table, round-three row) and the page generator (header
banner, HPCC section box, float32 rows chipped WITHDRAWN, E11 section note); code fix + test in f1260d0, synced to
~/bpusim on Pinnacle before any queued job started (the queued reruns use the fixed path; E11-cal reproduces the old
clip explicitly with --legacy-dial-clip). Results page v10, status page v4.
