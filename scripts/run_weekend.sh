#!/usr/bin/env bash
# Weekend programme: new experiments (A) then the fixed ones (B), with timing.
# Everything runs on a laptop CPU; long jobs are better launched as user services
# (systemd-run --user) so they survive the terminal, see MONDAY.md for the cluster list.
#   scripts/run_weekend.sh            default seeds (see each line)
#   scripts/run_weekend.sh --quick    smoke-sized versions of everything (~15 min)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
Q=""; [ "${1:-}" = "--quick" ] && Q=1
export PYTHONWARNINGS=ignore
T0=$(date +%s)
run() { local name=$1; shift; echo "=== $name  ($(date +%H:%M))"; local t=$(date +%s); (cd experiments && "../$PY" "$@"); echo "--- $name took $(( $(date +%s) - t )) s"; }
# ---- A. accelerator / trainability / continual claims
if [ -n "$Q" ]; then
  run E8  e8_nanogpt_tiles.py --smoke
  run E8b e8b_energy_model.py
  run E9  e9_trainability.py --smoke
  run E10 e10_continual.py --smoke
else
  run E8  e8_nanogpt_tiles.py --seeds 3 --steps 1500 --ft-steps 300 --batch 24 --threads 8      # ~3 h
  run E8b e8b_energy_model.py
  run E9  e9_trainability.py --seg 2000 --threads 8                                               # ~2.5 h
  run E10 e10_continual.py --seeds 3 --threads 4                                                  # ~15 min
fi
# ---- B. brain-circuit substrate, fixed
if [ -n "$Q" ]; then
  run E3  e3_drift.py --quick --procs 4
  run E4  e4_ring_selforg.py --seeds 2 --steps 20000 --null 8
  run E5  e5_grid_selforg.py --reference-only --sheets 32x32 --seeds 1 --points 800 --points-h2 350 --null 4
  run E6  e6_memory.py --seeds 1
  run E7  e7_modularity.py --seeds 1 --base-trials 600 --ft-trials 200
  run E2  e2_grow_vs_rigl.py --smoke
else
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 run E3  e3_drift.py --seeds 20 --procs 8 --tau-H 0,5               # ~3 h
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 run E4  e4_ring_selforg.py --seeds 10 --procs 4        # ~5 min
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 run E4c e4_ring_selforg.py --seeds 10 --procs 4 --steps 720000 --write-budget 6e-3 --out results/e4_capped   # ~15 min
  bash scripts/e5_chain.sh                                                                       # ~2.5 h (32x32 ref x3, 48x48 ref x1, self-org 32x32 x1)
  run E6  e6_memory.py --seeds 3                                                                  # ~5 min (20% flips)
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 run E7  e7_modularity.py --seeds 3 --procs 3           # ~1 h
  run E2  e2_grow_vs_rigl.py --dataset cifar10 --density 0.03 --seeds 3 --epochs 5               # ~10 min
fi
echo "=== figures"; $PY scripts/make_figures.py
echo "run_weekend finished in $(( $(date +%s) - T0 )) s"
