# BPU FPGA tile — RESULTS (rung ii)

Last updated 2026-11-01.  Everything below is labelled **simulated** (Verilator, cycle-accurate RTL), **host** (the
C firmware on the C model of the PL) or **hardware**.  There is no hardware row yet: no Vivado licence/install and
no KV260 on the bench on 2026-11-01.  Nothing in the power/timing section is estimated from the simulator; the
Yosys line is a synthesisability check and says so.

## Pass criteria

| # | criterion | status | evidence |
|---|---|---|---|
| 1 | `reference_contract.md` complete and reviewed | **complete, self-reviewed**; second-reader review open | `docs/reference_contract.md`; §0 records that bpusim had no fixed-point contract and that `bpusim/fixed.py` defines it |
| 2 | all unit tests bit-exact | **PASS (simulated)** | cocotb on Verilator 5.053: `wl_dac` (2 000 random + edge codes), `gcond` (2 025), `sar_adc` (2 112 incl. every tie/rail case), `eta_prng` (3 seeds × 3 reads × 2 048 words); `tile_core` stream/verify/prng; AXI wrapper — `fpga/tests/test_units.py`, 8 tests |
| 3 | ≥10k-vector tile gate: 0 mismatched bits | **PASS (simulated)** | 10021 vectors (10000 registered random + 21 corners), 320672 columns × 48 compared bits (y_adc + 40-bit Y), **0 mismatches**; `fpga/tests/_build/gate_10000.log`; seed base 20261101, fixture SHA-256 `f8a58279cadeb013…` |
| 3b | same gate through the firmware path | **PASS (host)** | `ps/build/host/runner model` on 533 vectors: 0 mismatches — the C model of the PL is a third independent implementation and agrees |
| 4 | write-verify converges within bound for all swept targets | **PASS at the contract's operating point (σ_r ≤ 0.02), FAIL above it by construction** | table below; the convergence region equals the resolution prediction cell for cell (`test_ps.py::test_write_verify_sweep_converges`) |
| 5 | noise reproducible + PL stream == bpusim | **PASS (simulated + host)** | RTL prng-mode noise dump == `fixed.EtaPRNG` word for word (3 reads × 2 048), reseed → identical; C PRNG == Python PRNG; same seed twice → identical sweep output |
| 6 | post-route timing closed; Fmax, cycles/pass, throughput | **NOT RUN** (no Vivado) | cycles per read measured on the RTL: **81** (one 8-bit pass), **118** (two passes), mean 99 over the 10k set, ADC + result store included, DMA loads excluded; Fmax/throughput wait for `scripts/vivado_build.tcl` |
| 7 | real board power measured; energy model updated | **NOT RUN** (no board) | `ps/src/measure.c` reads the carrier INA260 through hwmon; nothing measured, nothing fed back into the energy model |

Corner/stress (gate test 5): rails w ∈ {0, 255} on both arrays, all-zero and all-one rows, max |I⁺ − I⁻|
(263 225 280 at S = 0, saturates), η at ±16383, mask all off, S = 31, two-pass rails and the unsigned low byte,
five round-half-even ties — all 21 in every gate run, 0 mismatches.  Regression (gate test 6): the 10k expected
outputs and the stimulus SHA-256 are frozen in `fpga/vectors/fixtures/gate_10k.npz`; `test_regression.py` fails
if the reference, the generator or numpy's stream moves; CI (`.github/workflows/fpga.yml`) runs units, fixture,
the 10k gate and the firmware tests on every push that touches `fpga/` or the reference.

## Write-verify sweep (host, C firmware on the C model; 8 rows × 32 devices per cell, random start states)

Band ε = 0.02 in w (= 1.98 µS), 8 reads averaged per verify, at most 100 iterations.  ✅ = every row converged,
`it` = mean iterations, `p` = pulses per device; ✗ = not converged, worst |Δw| after 100 iterations.

| σ_r \ target | 0 | 0.05 | 0.1 | 0.25 | 0.5 | 0.75 | 0.9 | 0.95 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | ✅ 4 it / 86 p | ✅ 4 it / 71 p | ✅ 4 it / 55 p | ✅ 3 it / 26 p | ✅ 3 it / 13 p | ✅ 4 it / 28 p | ✅ 4 it / 52 p | ✅ 4 it / 66 p | ✅ 4 it / 79 p |
| 0.02 | ✅ 4 it / 89 p | ✅ 5 it / 67 p | ✅ 4 it / 54 p | ✅ 5 it / 29 p | ✅ 5 it / 13 p | ✅ 8 it / 28 p | ✅ 10 it / 59 p | ✅ 7 it / 72 p | ✅ 8 it / 89 p |
| 0.05 | ✅ 4 it / 86 p | ✅ 5 it / 67 p | ✅ 5 it / 56 p | ✅ 7 it / 32 p | ✅ 18 it / 15 p | ✗ 0.036 | ✗ 0.061 | ✗ 0.053 | ✅ 36 it / 100 p |
| 0.1 | ✅ 6 it / 88 p | ✅ 6 it / 72 p | ✅ 7 it / 53 p | ✅ 29 it / 34 p | ✗ 0.037 | ✗ 0.060 | ✗ 0.087 | ✗ 0.081 | ✗ 0.000 |
| 0.2 | ✅ 7 it / 90 p | ✅ 8 it / 77 p | ✅ 16 it / 62 p | ✗ 0.043 | ✗ 0.085 | ✗ 0.131 | ✗ 0.122 | ✗ 0.114 | ✗ 0.000 |

Reading: with relative read noise σ_r on G = G_min + G_range·w, eight averaged reads measure w with standard
error ≈ σ_r·w/√8.  The loop certifies the band on one such measurement, so it converges iff about
2·σ_r·(w + 0.01)/√8 ≤ ε (one-sided at the top rail, where the measurement clips at w = 1).  That boundary
predicts the table exactly.  Above it the loop ping-pongs — bpusim's float `program_target` has the identical
limitation (its default σ_r is 0.02 for this reason).  Fix options, NOT applied because the contract follows
bpusim: more verify reads (√N), or a band that scales with σ_r.  Pulse counts at the operating point:
13–90 per device from a random start, bounded by the table law (NUP/NDN), never the iteration cap.

## Timing and power

| item | value | label |
|---|---|---|
| cycles per 32×32 MAC pass | 32 row clocks + 4 pipeline + 1 fold (37); a one-pass read 81 clocks end-to-end, two-pass 118 | simulated (RTL) |
| Fmax | — | not run (needs Vivado post-route) |
| tile throughput | — | not run |
| board power idle / loaded | — | not run (needs KV260) |
| E_ADC term in the energy model | unchanged (FoM·2^B, FoM 10 fJ) | projected; no FPGA number fed back |
| synthesisability | Yosys 0.70 `synth_xilinx -family xcup` on the first RTL (asynchronous Gaussian ROM): **elaborates and maps, 448 DSP48E2, 131 k LUT, 68 k FF, 0 BRAM** — the 64 async ROM copies landed in LUTs and the constant multiplies in DSPs, over the K26 budget (117 k LUT). Fixed the same day: ROM read made synchronous (BRAM-inferable), `×5`/`×9` as shift-adds; all tests re-passed. The re-synthesis of the fixed RTL was killed by the kernel OOM-killer at Yosys' memory-mapping pass (17.9 GB peak on a 32 GB laptop), so the post-fix cell count is **unknown until Vivado runs**; expected ≈ 192 DSP (128 datapath + 64 σ·z) and 64 BRAM18 | Yosys = synthesisability check only, **not** a Vivado result |

## What the gate does and does not say

It says the RTL, the C firmware model and the Python reference compute the same integer function on
10021 vectors covering both DAC passes, five noise levels, three ADC ranges, masked and unmasked tiles
and the rails.  It does not say anything about MoS2: the emulator models the device MATH (contract §1), which is
the point of rung ii — the architecture and datapath are isolated from device physics.  It also does not say the
bitstream meets timing or what it draws; those rows are empty until the board run.

## Next (hardware)

1. `vivado -mode batch -source fpga/scripts/vivado_bd.tcl -tclargs build/kv260` then `vivado_build.tcl` → `results.json` (Fmax, DSP/BRAM, WNS).
2. On the KV260: `scripts/board_setup.sh`, `make -C fpga/ps board`, `runner mmap <vec10k>` → the same GATE PASS line, now hardware.
3. `measure` → idle / loaded W, dynamic J per read; update the energy model and this file with "measured-on-FPGA" rows.
