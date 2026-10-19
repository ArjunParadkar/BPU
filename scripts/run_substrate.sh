#!/usr/bin/env bash
# Substrate programme (2026-10-18): every experiment compares the SAME algorithm on a float32
# reference and on noisy BPU tiles.  Long jobs are better launched as user services
# (systemd-run --user, see the launch() helper) so they survive the terminal; MONDAY.md has the HPCC queue.
#   scripts/run_substrate.sh            full recipe (GPU for E9/E10/E8b if available; ~4 h on this laptop)
#   scripts/run_substrate.sh --quick    smoke-sized versions of everything (~10 min)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
Q=""; [ "${1:-}" = "--quick" ] && Q=1
export PYTHONWARNINGS=ignore
T0=$(date +%s)
run() { local name=$1; shift; echo "=== $name  ($(date +%H:%M))"; local t=$(date +%s); (cd experiments && "../$PY" "$@"); echo "--- $name took $(( $(date +%s) - t )) s"; }
ST="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"     # one BLAS thread per multiprocessing worker (oversubscription stalls the machine)
if [ -n "$Q" ]; then
  run E4   e4_ring_selforg.py --seeds 2 --steps 20000 --null 8 --write-mode level --out results/e4_level
  run E6   e6_memory.py --seeds 1
  run E3   e3_tracking.py --seeds 2 --T 5 --procs 4
  run E7   e7_modularity.py --seeds 1 --base-trials 600 --ft-trials 200
  run E9   e9_trainability.py --smoke
  run E10  e10_continual.py --smoke
  run E8b  e8b_energy_model.py --smoke
else
  env $ST bash -c "$(declare -f run); PY=$PY; run E4   e4_ring_selforg.py --seeds 10 --procs 4 --write-mode level --out results/e4_level"                                           # ~5 min
  env $ST bash -c "$(declare -f run); PY=$PY; run E4c  e4_ring_selforg.py --seeds 10 --procs 4 --write-mode level --steps 720000 --write-budget 6e-3 --out results/e4_level_capped"  # ~20 min
  run E6   e6_memory.py --full                                                                                                       # ~30 min (10 seeds)
  env $ST bash -c "$(declare -f run); PY=$PY; run E3   e3_tracking.py --seeds 10 --procs 8"                                        # ~1.5 h alone
  env $ST bash -c "$(declare -f run); PY=$PY; run E7   e7_modularity.py --seeds 3 --procs 3"                                       # ~1.5 h
  env $ST bash -c "$(declare -f run); PY=$PY; run E7r  e7_modularity.py --seeds 3 --procs 3 --noise-free --out results/e7_ref"     # ~1.5 h (reference substrate)
  run E9   e9_trainability.py --seeds 3 --seg 1000 --threads 4                                                                       # ~2 h on the GPU
  run E10  e10_continual.py --seeds 3 --threads 4                                                                                     # ~2 h on the GPU (with ablations)
  run E8bt e8b_energy_model.py --skip-measure                                                                                         # train + read-out accuracy, shares the GPU
  run E8b  e8b_energy_model.py --skip-train                                                                                           # GPU power measurement: run on an otherwise IDLE GPU
fi
echo "=== figures"; $PY scripts/make_figures.py; $PY scripts/make_results_page.py
echo "run_substrate finished in $(( $(date +%s) - T0 )) s"
