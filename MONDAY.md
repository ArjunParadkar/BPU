# MONDAY — HPCC queue

Everything below is launched from a clone of this repo with the same venv
(`pip install -r requirements.txt`).  Results land in `results/<e>/results.json`
and are merged into the page with `python scripts/make_figures.py && python
scripts/make_results_page.py`.  Launch under `sbatch` (or `systemd-run --user`
on a workstation); pool-based scripts need `OMP_NUM_THREADS=1` per worker or
they stall the node.  The laptop runs from 2026-10-18 are the 3-seed / 10-seed
versions of the same commands (`scripts/run_substrate.sh`).

| # | Job | Command | Resources | What it settles |
|---|---|---|---|---|
| 1 | E1 full | `python experiments/e1_gpt2_noise.py --device cuda --steps 400 --batch 8 --block 512` (+ `--c2c 0.02` for write noise) | 1 GPU ≥ 16 GB, ~3 h, downloads GPT-2 124M + WikiText-2 | the real accelerator-parity row: perplexity vs σ for GPT-2 on tiles (E8 is the laptop stand-in) |
| 2 | E3 50 seeds | `OMP_NUM_THREADS=1 python experiments/e3_tracking.py --full --procs 32` | CPU node, 32 cores, ~2 h | the moving-cue U-test with tight sign-test power (10 seeds on the laptop); add `--N 256` for the size scan |
| 3 | E5 self-org 20 seeds | `OMP_NUM_THREADS=1 python experiments/e5_grid_selforg.py --sheets 32x32 --seeds 20 --points 800 --points-h2 350 --procs 10` | CPU node, 10 cores, ~12 h | whether any mismatch draw / seed self-organises a torus at 32×32 (1/1 failed on the laptop); `--reference-only --seeds 10` first to map the sliding/hopping fraction of the programmed module |
| 4 | E7 10 seeds | `OMP_NUM_THREADS=1 python experiments/e7_modularity.py --full --procs 10` and the same with `--noise-free --out results/e7_ref` | CPU node, 10 cores, ~3 h each | grown vs fixed at matched synapses, matched steps and continued, with error bars; the noise-free run is the reference substrate |
| 5 | E10 10 seeds | `python experiments/e10_continual.py --full --device cuda --threads 8` | 1 GPU (any), ~5 h with the ablations (`--no-ablation` for ~2 h) | the EWC / replay twin gaps with error bars, FPL gate and τ_ret ablations |

Also useful if a node is free

| Job | Command | Resources | What it settles |
|---|---|---|---|
| E9 3 seeds, 2000-step segments | `python experiments/e9_trainability.py --seeds 3 --seg 2000 --device cuda` | 1 GPU, ~4 h | recovery times with longer segments; the laptop ran `--seg 1000` |
| E8 full size | `python experiments/e8_nanogpt_tiles.py --d 384 --layers 6 --heads 6 --steps 5000 --ft-steps 1000 --seeds 3` | 1 GPU | the 10.6M-parameter version of E8 (the tile layer now runs on CUDA) |
| E8b on a data-centre GPU | `python experiments/e8b_energy_model.py --skip-train --measure-batch 512` after copying `results/e8b/checkpoint.pt` | 1 GPU, idle, 5 min | a measured INT8 control on a GPU that the 2.7M model does not leave idle; needs `nvidia-smi` power readings |

Notes
* The torch tile layer (`bpusim/torchtile.py`) runs on CUDA since 2026-10-18; `--device cuda`
  is implemented for E1, E2, E8b, E9 and E10.
* The scripts print progress with `flush=True`; redirect to a log and read it with `grep -v Warning`.
* `tests/test_modules.py::test_graph_grows_routes_toward_output` fails since the weekend's switch to
  layer-local learning (the route statistic no longer points at the output module within 300 trials);
  it is a pre-existing failure, not touched on 2026-10-18.
