#!/usr/bin/env bash
# Variability programme (2026-10-19): E11-E14 + E3-final + the nearest-pulse reruns of E4/E5/E6/E9.
# Every job was launched as a user unit (systemd-run --user --unit bpu-<name> ...), CPU pools at Nice=10
# with OMP_NUM_THREADS=1 per worker, GPU jobs at Nice=2; logs in logs/<name>.log.
#   scripts/run_variability.sh            the full recipe (~6 h GPU, ~4 h CPU on this laptop)
#   scripts/run_variability.sh --quick    smoke-sized versions (~5 min)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
Q=""; [ "${1:-}" = "--quick" ] && Q=1
export PYTHONWARNINGS=ignore OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
run() { local name=$1; shift; echo "=== $name  ($(date +%H:%M))"; local t=$(date +%s); (cd experiments && "../$PY" "$@"); echo "--- $name took $(( $(date +%s) - t )) s"; }
if [ -n "$Q" ]; then
  run E4   e4_ring_selforg.py --seeds 2 --steps 20000 --null 8 --out results/e4_np
  run E6   e6_memory.py --seeds 1 --out results/e6
  run E9   e9_trainability.py --smoke --out results/e9
  run E11  e11_regularization.py --smoke --out results/e11
  run E12  e12_exploration.py --smoke --out results/e12
  run E13  e13_stochastic_resonance.py --smoke --out results/e13
  run E14  e14_sampling.py --smoke --out results/e14
  run E3f  e3_tracking.py --seeds 2 --T 5 --procs 4 --mismatch 0.2,0.3 --cue-amp 0.05,0.02 --omegas 0.5 --out results/e3_final
else
  run E4   e4_ring_selforg.py --seeds 20 --procs 5 --out results/e4_np                                                              # ~15 min
  run E6   e6_memory.py --full --out results/e6                                                                                     # ~1 h (K 1..16, 10 seeds)
  run E5   e5_grid_selforg.py --sheets 16x16,32x32 --seeds 5 --procs 5 --points 800 --points-h2 350 --null 8 --steps 12000 --out results/e5_selforg_np   # ~3 h
  run E3f  e3_tracking.py --mismatch 0.2,0.3 --cue-amp 0.05,0.02 --omegas 0.5,0.1 --seeds 10 --procs 6 --out results/e3_final      # ~2.5 h
  run E12  e12_exploration.py --seeds 5 --trials 10000 --eta 2 --procs 4 --out results/e12                                          # ~1 h
  run E13  e13_stochastic_resonance.py --seeds 10 --procs 4 --out results/e13                                                       # ~1 h
  run E14  e14_sampling.py --seeds 5 --procs 3 --out results/e14                                                                    # ~20 min
  run E9   e9_trainability.py --seeds 5 --threads 4 --out results/e9                                                                # ~4 h GPU
  run E11  e11_regularization.py --seeds 5 --threads 4 --out results/e11                                                            # ~5 h GPU
fi
echo "=== figures"; $PY scripts/make_figures.py; $PY scripts/make_results_page.py
