# MONDAY — HPCC queue

On Pinnacle (AHPCC) the queue is `bash scripts/hpcc/submit.sh queue` (env: `scripts/hpcc/build_env.sh`;
cluster quirks in its header).  Submitted 2026-10-22 23:5x CDT as jobs 1364337-1364344 (E11 tiles/float32 split
into results/e11_full_tiles + results/e11_full_f32: merge the two `runs` lists, then `--resummarize`).

Everything below is launched from a clone of this repo with the same venv
(`pip install -r requirements.txt`).  Results land in `results/<e>/results.json`
and are merged into the page with `python scripts/make_figures.py && python
scripts/make_results_page.py`.  Launch under `sbatch` (or `systemd-run --user`
on a workstation); pool-based scripts need `OMP_NUM_THREADS=1` per worker or
they stall the node.  The laptop recipes are `scripts/run_variability.sh` (round three) and
`scripts/run_substrate.sh` (2026-10-18).

Round three (2026-10-19/20) changed the queue: E7 and E10 already ran 10 seeds on the laptop
(round two, `results/e7`, `results/e7_ref`, `results/e10`), and the variability programme left
three questions only a bigger budget answers.  Order = what each job settles per node-hour.

| # | Job | Command | Resources | What it settles |
|---|---|---|---|---|
| 1 | E11 float32 bracket | `python experiments/e11_regularization.py --substrates float32 --sigmas 0,0.1,0.2,0.3,0.4,0.5,0.7 --sigmas-beyond '' --seeds 5 --out results/e11_f32_bracket` | 1 GPU, ~3 h | the float32 σ* sat on the dial top (0.2) and was still falling at 0.3; this brackets it, so "tiles reach the optimum at half the injected noise" becomes a measured shift instead of a lower bound |
| 2 | E11 10 seeds, full data | `python experiments/e11_regularization.py --seeds 10 --train-frac 1.0 --steps 5000 --out results/e11_full` | 1 GPU ≥ 16 GB, ~20 h (or split by `--substrates`) | whether the σ* = 0.1 regulariser survives outside the small-data regime (on 10 % of the text the model overfits by construction); σ* = 0 here is the expected null and a result |
| 3 | E12 long budget | `OMP_NUM_THREADS=1 python experiments/e12_exploration.py --seeds 10 --trials 50000 --eta 2 --procs 16 --out results/e12_long` | CPU node, 16 cores, ~6 h | the grid task was not learned in 10 000 pairs by any arm; this tests the schedules where they can actually separate.  It does NOT test a wider dial: the dial's top (0.2) is a design decision (DECISIONS round three, E12/E14 both point to σ ≈ 1) |
| 4 | E3-final 50 seeds | `OMP_NUM_THREADS=1 python experiments/e3_tracking.py --mismatch 0.2,0.3 --cue-amp 0.05,0.02 --omegas 0.5,0.1 --seeds 50 --procs 32 --out results/e3_final50` | CPU node, 32 cores, ~4 h | sign-test power on the pinned conditions (10 seeds gave flat errors to three decimals, 0/5 U-shaped); 50 seeds turns "no optimum" into a bound |
| 5 | E5 self-org 20 seeds | `OMP_NUM_THREADS=1 python experiments/e5_grid_selforg.py --sheets 32x32 --seeds 20 --points 800 --points-h2 350 --null 8 --steps 12000 --procs 10 --out results/e5_selforg_np20` (nearest-pulse is the default write mode) | CPU node, 10 cores, ~12 h | 0/5 at 16×16 and 32×32 on the laptop with the programmed reference itself at 2/5; run `--reference-only --seeds 20` first: if the reference stays near 40 % the certifier, not the sheet, is the limit at 800 points |
| 6 | E1 full | `python experiments/e1_gpt2_noise.py --device cuda --steps 400 --batch 8 --block 512` (+ `--c2c 0.02` for write noise) | 1 GPU ≥ 16 GB, ~3 h, downloads GPT-2 124M + WikiText-2 | the real accelerator-parity row: perplexity vs σ for GPT-2 on tiles (E8 is the laptop stand-in) |

Done on the laptop, no longer queued: E7 10 seeds both substrates (no noise-assisted growth,
p = 0.85), E10 10 seeds with ablations (criterion met), E9 5 seeds nearest-pulse (met, +2.1 %),
E4 20 seeds (13/20), E13 10 seeds (silent within the dial, as predicted), E14 5 seeds.

Also useful if a node is free

| Job | Command | Resources | What it settles |
|---|---|---|---|
| E9 5 seeds, 2000-step segments | `python experiments/e9_trainability.py --seeds 5 --seg 2000 --device cuda` | 1 GPU, ~8 h | recovery times with longer segments under the nearest-pulse default; the laptop ran `--seg 1000` |
| E8 full size | `python experiments/e8_nanogpt_tiles.py --d 384 --layers 6 --heads 6 --steps 5000 --ft-steps 1000 --seeds 3 --threads 32` | CPU node, 32 cores (E8 has no `--device`; it is CPU-only) | the 10.6M-parameter version of E8 |
| E8b on a data-centre GPU | `python experiments/e8b_energy_model.py --skip-train --measure-batch 512` after copying `results/e8b/checkpoint.pt` | 1 GPU, idle, 5 min | a measured INT8 control on a GPU that the 2.7M model does not leave idle; needs `nvidia-smi` power readings |

Notes
* The torch tile layer (`bpusim/torchtile.py`) runs on CUDA since 2026-10-18; `--device cuda`
  is implemented for E1, E2, E8b, E9 and E10.
* The scripts print progress with `flush=True`; redirect to a log and read it with `grep -v Warning`.
* `tests/test_modules.py::test_graph_grows_routes_toward_output` fails since the weekend's switch to
  layer-local learning (the route statistic no longer points at the output module within 300 trials);
  it is a pre-existing failure, still the only one (72 pass, 2026-10-20).
