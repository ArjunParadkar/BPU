# Pre-registered gate vectors

Registered 2026-11-01, before the first RTL run of the gate testbench.

| item | value |
|---|---|
| generator | `fpga/vectors/gen_vectors.py` at the commit that adds this file |
| reference | `bpusim/fixed.py` (contract `docs/reference_contract.md`) |
| seed base | **20261101** |
| count | **10 000** random vectors (seeds 20261101 … 20271100) + the 21 named corner vectors |
| weight sets | one per 100 consecutive seeds (100 sets) |
| per-seed derivation | passes = 2 if seed odd else 1; S = {17,18,19}[seed % 3] (+8 if two passes); σ_r = {0, 0.02, 0.05, 0.1, 0.2}[seed % 5]; mask Bernoulli(0.8) when seed % 4 == 3 |
| comparison | every bit of `y_adc[7:0]` and `Y[39:0]` for all 32 columns of every vector; zero tolerance |
| frozen fixture | `fpga/vectors/fixtures/gate_10k.npz` (expected `y`, `Y` for the 10 000 seeds + SHA-256 of stimulus+expected) |
| numpy | the fixture was produced with numpy pinned in `fpga/requirements.txt`; PCG64 + ziggurat `standard_normal` is stable across numpy ≥ 1.17 |

The RTL was never consulted when choosing these values.  A mismatch is reported as the first divergent
(vector, column) with both 64-bit result words; the comparison is never loosened.
