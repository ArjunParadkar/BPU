#!/usr/bin/env bash
# Run E3-E7 end to end on CPU and regenerate all figures.
#   scripts/run_all.sh          3 seeds (target: < 30 min on a laptop CPU)
#   scripts/run_all.sh --full   10 seeds (paper budget)
# E2 (MNIST, PyTorch) and E1 (GPT-2, GPU host) are not part of this script; see README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
FULL=""
SEEDS=3
if [ "${1:-}" = "--full" ]; then FULL="--full"; SEEDS=10; fi
export PYTHONWARNINGS=ignore
T0=$(date +%s)
run() { local name=$1; shift; echo "=== $name ==="; local t=$(date +%s); (cd experiments && "../$PY" "$@") ; echo "--- $name took $(( $(date +%s) - t )) s"; }
run E3 e3_drift.py --seeds $SEEDS $FULL
run E4 e4_ring_selforg.py --seeds $SEEDS $FULL
run E5 e5_grid_selforg.py --seeds $SEEDS $FULL
run E6 e6_memory.py --seeds $SEEDS $FULL
run E7 e7_modularity.py --seeds $SEEDS $FULL
echo "=== figures ==="; $PY scripts/make_figures.py
echo "run_all finished in $(( $(date +%s) - T0 )) s"
