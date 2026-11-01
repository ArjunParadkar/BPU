#!/usr/bin/env bash
# Build the file-driven bit-exact gate testbench with Verilator (cycle-accurate, 2-state).
# usage: scripts/build_gate_sim.sh [out_dir]   -> <out_dir>/Vtb_tile_gate
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$HERE/tests/_build/gate_verilator}
export PATH="$HOME/.local/opt/oss-cad-suite/bin:$PATH"
mkdir -p "$OUT"
cd "$HERE/hdl"
verilator --binary --timing -O2 -j 0 --Mdir "$OUT" -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL \
  --x-assign fast --x-initial fast --top-module tb_tile_gate \
  -GN_MAX=${N_MAX:-10300} -GW_MAX=${W_MAX:-160} \
  bpu_pkg.sv wl_dac.sv gcond.sv xbar_tile.sv col_integrate.sv shift_add.sv sar_adc.sv eta_prng.sv tile_core.sv tb/tb_tile_gate.sv \
  > "$OUT/verilator.log" 2>&1 || { tail -30 "$OUT/verilator.log"; exit 1; }
echo "$OUT/Vtb_tile_gate"
