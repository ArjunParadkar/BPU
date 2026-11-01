# BPU FPGA validation rig — rung ii (one 32×32 tile on a Kria KV260)

The PL emulates the DEVICE MATH of one 32×32 differential memristor tile in digital logic; the PS runs the
sequencer and the write-verify loop.  The one gate: the tile reproduces the bit-accurate reference
`bpusim/fixed.py` EXACTLY — same inputs, same seed, identical output bits, every column, every read.
Nothing downstream (power, timing, device claims) is reported until that passes.  Status and numbers:
[`../docs/RESULTS.md`](../docs/RESULTS.md).  The contract: [`../docs/reference_contract.md`](../docs/reference_contract.md).

```
fpga/
  hdl/        SystemVerilog: wl_dac, gcond, xbar_tile, col_integrate, shift_add, sar_adc, eta_prng, tile_core, bpu_tile_axi
  hdl/tb/     cocotb tests (units, core, AXI) + the file-driven gate testbench tb_tile_gate.sv
  ps/         firmware in C: fixed-point device model, tile sequencer, write-verify, runner, power measurement,
              two HALs (hal_mmap.c for the board, hal_model.c = a C model of the PL for host runs)
  vectors/    golden-vector generator (calls bpusim.fixed), REGISTERED.md, frozen fixture gate_10k.npz
  scripts/    Vivado block design + build (non-interactive), board setup, Yosys/Verilator helpers, table generators
  tests/      pytest: test_units (cocotb), test_gate (10k bit-exact), test_regression (fixture), test_ps (firmware)
```

## Run it

```bash
# once: OSS CAD Suite (Verilator 5.053) in ~/.local/opt/oss-cad-suite, then
cd fpga && uv venv --python 3.12 .venv && . .venv/bin/activate && uv pip install -r requirements.txt
python -m pytest -q ../tests/test_fixed.py          # the reference against the float model
cd tests
python -m pytest -q test_units.py                   # cocotb: every module + core + AXI, bit-exact (~30 s)
python -m pytest -q test_gate.py --gate-vectors 10000   # THE gate: 10 021 vectors, 0 mismatched bits (~15 s)
python -m pytest -q test_regression.py test_ps.py   # frozen fixture; firmware on the host C model
```

The same `runner` binary runs on the board: `make -C ps board CC=aarch64-linux-gnu-gcc`, then
`ps/build/board/runner mmap <vector dir>` after `scripts/board_setup.sh` (see `scripts/VERSIONS.md`).

## Design in one paragraph

Row-serial datapath: one word line per clock, 32 columns in parallel.  Per column and row: `G_q = 655 + 254·w_q`,
`Ĝ = (G·(2¹⁴+η) + 2¹³) ≫ 14`, `term = d·Ĝ` for both devices (128 multipliers → DSP48E2), exact 32-bit column
integrators, differential + shift-add across the two 8-bit DAC passes of a 16-bit activation, then a 32-lane
8-bit SAR ADC with round-half-even range scaling.  Noise comes either from a DMA'd stream (mode A, the gate) or
from 64 on-chip xoshiro128** generators with a 1024-entry inverse-CDF ROM (mode B), filled into the same buffer
before the read so both modes share one datapath.  81 clocks per one-pass read, 118 for two passes (ADC and
result store included, loads excluded).
