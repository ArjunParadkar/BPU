#!/usr/bin/env bash
# E5 on the large sheets: programmed reference at 32x32 (3 seeds) and 48x48 (1 seed), then one
# self-organisation attempt at 32x32 (only meaningful if the reference certified).
set -u
cd "$(dirname "$0")/../experiments"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore
PY=../.venv/bin/python
$PY e5_grid_selforg.py --reference-only --sheets 32x32 --seeds 3 --procs 3 --points 800 --points-h2 350 --null 8 --out ../results/e5
$PY e5_grid_selforg.py --reference-only --sheets 48x48 --seeds 1 --points 800 --points-h2 350 --null 8 --out ../results/e5_48
$PY e5_grid_selforg.py --sheets 32x32 --seeds 1 --points 800 --points-h2 350 --null 8 --steps 12000 --out ../results/e5_selforg32
