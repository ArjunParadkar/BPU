# BPU FPGA tile — reference contract (rung ii)

Status: **complete, self-reviewed 2026-11-01** (the "reviewed" box in RESULTS.md stays open until a
second reader signs it).  Source of truth for every number here: `bpusim/fixed.py`.  If this document
and that file ever disagree, the file wins and this document is the bug.

## 0. Step-0 finding: bpusim had no fixed-point contract to extract

The spec asks to extract "the fixed-point formats at every stage" from bpusim.  They do not exist.
`bpusim.tile.Tile` and `bpusim.device.DeviceArray` are float64 numpy models: continuous state
`w ∈ [0,1]`, continuous conductance `G = G_min + (G_max − G_min)·w`, a float matrix product
`x @ W_read`, read noise `G·(1 + σ_r·N(0,1))` from numpy's PCG64 + ziggurat.  The only integer
quantisation anywhere in bpusim before this work was

| what | where | rule |
|---|---|---|
| 16-bit signed write accumulator | `Tile.ACC_BITS`, `Tile._quantize_acc` | `rint(acc / 2^-15)·2^-15`, clipped to ±(1 − 2^-15) |
| 8-bit read-out, mode `"tile"` | `torchtile.ReadoutSpec._convert` | `clamp(round(v / step), −128, 127)·step`, `step = R / 128`, `round` = half-to-even (torch) |
| 8-bit read-out, mode `"lif"` | same | `floor(|v| / step)` spike count, `step = R / 255` |

So this contract **defines** the fixed-point tile (`bpusim/fixed.py`, integer-only) as the bit-accurate
reference, chooses its formats to carry the resolution the BPU papers assume (256 conductance levels,
8-bit ADC, Walden energy `FoM·2^B`), and ties it back to the float model with a measured, bounded
approximation error (`tests/test_fixed.py::test_fixed_tracks_float_tile`).  The FPGA is gated against
`fixed.py`, bit for bit.  The float model is NOT a gate reference and never will be: two float64
programs on different hardware are not bit-reproducible either.

Deviation from the spec text, stated plainly: the spec's noise model `Ĝ = G(1+η), η ~ N(0, σ_eff²)` with
`σ_eff² = σ_d2d² + σ_c2c² + σ_r²` is the paper's summary; bpusim applies the three sources at three
different places (d2d: a fixed per-device gain on the pulse step; c2c: a fresh factor per pulse; read
`σ_r`: a fresh factor per device per read).  The contract follows bpusim, not the summary: the READ
datapath carries `σ_r` only; d2d and c2c live in the write path (§6).  `σ_eff` is the paper's figure of
merit for level count, not a datapath quantity.

## 1. What is emulated

One 32×32 differential tile: 1024 synapses, each a pair `(G+, G−)` and a connectivity bit `A`.
A read is `y_j = ADC( Σ_i x_i · A_ij · (Ĝ+_ij − Ĝ−_ij) )` for 32 columns.  Digital logic computes the
DEVICE MATH; nothing here is MoS2.  Out of scope at this rung (documented, not forgotten): retention
(`tau_ret`), per-read dropout (`set_dropout`), the `level` write rule, multi-tile routing, and the
`"layer"`/`"lif"` read-out modes (the ADC implements mode `"tile"`).

## 2. Formats (every stage)

| stage | symbol | width / format | range | note |
|---|---|---|---|---|
| activation code | `x_q` | int8 Q1.7 (1 pass) or int16 Q1.15 (2 passes) | [−128,127] / [−32768,32767] | `x = x_q / 2^frac`; conversion from float: round half even, saturate |
| DAC row drive | `d_i` | 9-bit signed per pass | pass 0 of 2: `x_q & 0xFF` ∈ [0,255] (unsigned low byte); pass 1: `x_q >> 8` ∈ [−128,127]; 1-pass: `d = x_q` | so `x_q = 256·d_1 + d_0` exactly |
| device state (read side) | `w_q` | uint8 | [0,255] | `w = w_q / 255`, 256 levels (bpusim's `levels` cap) |
| device state (write side) | `w_s` | uint16 Q1.15 | [0, 32768] | internal to the write path; `w_q = min(255, (w_s + 64) >> 7)` |
| conductance | `G_q` | uint16 | [655, 65425] | `G_q = 655 + 254·w_q`; `G_LSB = (G_max−G_min)/(254·255) = 1.5285 nS`; emulated `G_min = 655·G_LSB = 1.0012 µS` (0.12 % off bpusim's 1 µS, by construction), `G_max = 100.0 µS` |
| read-noise sample | `η_q` | int16 Q1.14 | [−16383, 16383] | `η = η_q / 16384`; gain `(16384 + η_q)` ∈ [1, 32767] |
| noisy conductance | `Ĝ_q` | 17-bit unsigned | [0, 130846] | `Ĝ_q = (G_q·(16384 + η_q) + 8192) >> 14` — round half up, exact, **no saturation** (bpusim does not saturate G either) |
| column current | `I+_j`, `I−_j` | int32 | \|I\| < 2^30 | `Σ_i d_i · A_ij · Ĝ_ij`, exact integer sum over 32 rows (|d| ≤ 255, Ĝ < 2^17) |
| differential | `I_j` | int32 | \|I\| < 2^31 | `I+_j − I−_j` |
| accumulator | `Y_j` | int40 | \|Y\| < 2^39 | `Y_j = Σ_pass I_j^(pass) << (8·pass)`; exact |
| ADC code | `y_j` | int8 | [−128, 127] | `y = sat8(rne(Y / 2^S))`, `S ∈ [0,31]` the range exponent, `rne` = round half to even; `clipped` counts columns where the unsaturated value left [−128,127] |
| σ_r (prng mode) | `σ_q` | uint16 Q2.14 | | `σ_q = rne(σ_r · 16384)` |
| Gaussian LUT | `z_q[k]` | int16 Q3.13, 1024 entries | [−27011, 27011] | `z_q[k] = rne(Φ⁻¹((k+0.5)/1024)·8192)`; tails truncate at ±3.29σ |
| d2d gain | `g_q` | uint16 Q2.14 | [819, 65535] | `g = 1 + σ_d2d·N(0,1)`, clipped at 0.05 like bpusim (`819 = rne(0.05·16384)`); fixed per device |
| c2c sample | `c_q` | int16 Q1.14 | [−16383, 16383] | fresh per pulse |
| pulse step | `s_q` | Q1.15 | | §6 |
| verify tolerance | `TOL_Q` | Q1.15 | 655 | = 0.02 |
| damping | `DAMP_Q` | Q2.14 | 11469 | = 0.7 |

Mapping from bpusim floats (used only by the golden generator, never on the FPGA):
`x_q = sat(rne(x·2^frac))`, `w_q = sat(rne(w·255))`, `η_q = sat(rne(η·16384), ±16383)`.

## 3. Read datapath, in order (one pass)

1. **wl_dac**: `d_i` from `x_q` per the DAC row of §2.  Rows with `d_i = 0` contribute nothing.
2. **conductance**: `G_q = 655 + 254·w_q` for both devices of every synapse (`254·w = (w<<8) − (w<<1)`).
3. **noise**: `Ĝ_q = (G_q·(16384 + η_q) + 8192) >> 14` with the device's sample `η_q` (stream or prng).
   One sample per device per read; the same sample serves every pass of that read (bpusim builds
   `W_read` once per `read()` call).
4. **mask**: `Ĝ_q := 0` where `A_ij = 0`.
5. **xbar_tile + col_integrate**: `I+_j = Σ_{i=0}^{31} d_i·Ĝ+_ij`, `I−_j = Σ_i d_i·Ĝ−_ij`, exact int32.
   The RTL does this row-serially (one row per clock, 32 columns in parallel, DSP48E2 for the
   multiplies); the order of summation is irrelevant because integer addition is exact.
6. **differential**: `I_j = I+_j − I−_j`.
7. **shift_add**: `Y_j += I_j << (8·pass)`; `Y` is cleared at the start of a read.
8. **sar_adc**: `y_j = sat8(rne_shift(Y_j, S))` where `rne_shift(x, S)`:
   `r = floor(x / 2^S)` (arithmetic shift), `rem = x − r·2^S`; `rem > 2^(S−1) → r+1`;
   `rem == 2^(S−1) → r + (r odd)`; else `r`.  `S = 0` → identity.  Then saturate to [−128, 127].
   Equivalent to `torch.round(v / step)` with `step = 2^S` current codes and `R = 2^(S+7)`.

Output per read: 32 × `y_j` (int8) and, for diagnosis, 32 × `Y_j` (int40); in VERIFY result mode
32 × (`I+_j`, `I−_j`) of the last pass.

## 4. Verify read

`write_verify` reads one word line at a time: activation = code 64 on row `i`, 0 elsewhere, one pass,
result mode VERIFY.  Then `I+_j = 64·Ĝ+_ij` exactly for every masked column, so `Ĝ+_ij = I+_j >> 6`.
The PS averages `VERIFY_READS = 8` reads by summing: `w_meas_s = clamp(((Σ − 8·64·655)·2^15 + den/2) // den, 0, 2^15)`,
`den = 8·64·254·255`, with the numerator floored at 0 (a read below `G_min` means `w = 0`).

## 5. Noise source — the make-or-break choice

Two modes, selected by the `MODE` register.  Both are exact.

**A. Stream (the gate mode).**  The reference produces the quantised samples; the FPGA consumes them.
`eta_stream_from_numpy(rng, σ_r)` draws bpusim's own noise (`σ_r·rng.standard_normal((32,32))` for G+,
then the same for G−, the order `DeviceArray.read()` is called in `Tile.W_read`) and quantises it with
`η_q = sat(rne(η·16384), ±16383)`.  Stream order on the wire: index `((row·32 + col)·2 + device)`,
device 0 = G+, 1 = G−; 2048 int16 words per read, little-endian, 4 per 64-bit beat.  In two-pass reads
the block is consumed once.

**B. Matched PRNG (the realistic mode).**  64 independent **xoshiro128\*\*** generators, instance
`m = 2·col + device`, each stepped once per row per read (so a read consumes 32 words per instance).
Seeding: splitmix64 from state `seed64 + m·4·0x9E3779B97F4A7C15` (mod 2^64); the four outputs' low
32 bits are `s0..s3`; an all-zero seed word set gets `s0 = 1`.  Output word → `k = word >> 22`
(top 10 bits) → `z_q = LUT[k]` (Q3.13) → `η_q = clamp((σ_q·z_q + 4096) >> 13, ±16383)` where `>>`
is an arithmetic (floor) shift.  Reference: `fixed.EtaPRNG`; RTL: `fpga/hdl/eta_prng.sv`.
The generator state persists across reads until `RESEED` is written.

## 6. Write path (device model + write-verify, PS side)

Tables (`fixed.device_tables`, 1024 entries, index `k = min(1023, w_s >> 5)`, table centre `w = (k+0.5)/1024`):
`STEP_UP[k] = rne(α_p e^{−β_p w}·2^15)`, `STEP_DN[k] = rne(α_d e^{−β_d(1−w)}·2^15)`,
`NUP[k] = rne(e^{β_p w}/(α_p β_p)·256)`, `NDN[k] = rne(e^{β_d(1−w)}/(α_d β_d)·256)`,
with bpusim's defaults `α = 0.05`, `β = 3`.

* one pulse, direction ±1: `s_q = (STEP[k]·g_q·(16384 + c_q) + 2^27) >> 28`; `w_s := clamp(w_s ± s_q, 0, 32768)`.
  Pulses are applied one at a time, each with a fresh `c_q` (bpusim `DeviceArray.pulse`).  Dead devices: `n = 0`.
* nominal pulse count (`pulses_for`): up `rne_shift(NUP[k₁] − NUP[k₀], 8)`, down `−rne_shift(NDN[k₁] − NDN[k₀], 8)`,
  chosen by `target ≥ now`.
* write-verify (`program_target`), per iteration: 8 verify reads summed → `w_meas_s`; `err = target − w_meas`;
  `res = (max(STEP_UP[k], STEP_DN[k])·g_q) >> 15` (half a pulse); `todo = |err| > max(655, res)` and not dead;
  stop when nothing is to do (return iteration count − 1); else `aim = clamp(w_s + ((11469·err) >> 14))`,
  `n = pulses_for(w_s, aim)`, `n = sign(err)` where it rounded to 0, `n = 0` where not `todo`; pulse; at most 100 iterations.
  The band is `|Δw| ≤ 0.02` (in `w`), i.e. `|ΔG| ≤ 0.02·(G_max − G_min) = 1.98 µS`.

c2c samples in the PS come from the same PRNG construction as §5B (instance ids 64 + device index) so a
programming run is reproducible from `seed64`.

## 7. Rounding / saturation summary

| point | rule |
|---|---|
| float → code (generator only) | round half to even, then saturate |
| `Ĝ_q` | `+8192 >> 14`: round half up; no saturation (17 bits exact) |
| `I`, `Y` | exact; widths chosen so nothing can overflow |
| ADC | round half to even on the shift, saturate to int8, count clips |
| `η_q` (prng) | floor shift with +half (= round half up in two's complement), clamp ±16383 |
| `s_q` (pulse) | `+2^27 >> 28` round half up; `w_s` clamp [0, 32768] |
| `pulses_for` | round half even on 8 fraction bits |
| `w_meas` | integer division round half up, floored at 0, capped at 32768 |

## 8. Golden vectors

A vector = (`seed`, `passes`, `S`, `σ_r`) → PCG64(`seed`) draws, in this order: `w+` (32×32 uniform in
[0,1] → `w_q`), `w−`, mask (all ones for seeds ≡ 0,1,2 mod 4, Bernoulli 0.8 otherwise), `x` (uniform
[−1,1) → `x_q`), then the η stream per §5A.  Expected: `y` (32 int8), `Y` (32 int40).  `fpga/vectors/`
generates the set from `(seed_base, n)`; the frozen fixture stores seeds + expected outputs + a SHA-256
of the regenerated stimulus so the 10k set never has to be committed.  Pre-registration: the seeds are
fixed in `fpga/vectors/REGISTERED.md` before the first FPGA run.
