# DECISIONS

Choices made while building `bpusim`, with the reason for each.  Where a
choice was forced by something measured during the build, the measurement is
given.  Numbers quoted here come from the diagnostic runs in this session;
the experiment results proper are in `results/<e>/results.json`.

## Round four (2026-10-23): learning vs knowing

**Framing.** "Learning vs knowing." Variability should help while structure is forming (training from scratch)
and be suppressed once it is formed (deployment of a trained model). The MODULATE dial is the neuromodulation
analogue: high to learn, low to deploy. Prior art: dropout (Srivastava et al. 2014), weight-noise injection, and
noise-aware analog training (Joshi et al. 2020). The claim here is that native MoS2 variability (2–6 %) sits at the
regularisation optimum, together with the substrate results. We do NOT claim that noise regularises in general.

Rules kept from earlier rounds: the same algorithm on float32 and on tiles; every hypothesis and threshold
pre-registered in the experiment docstring before any run; nearest-pulse writes, layer-accumulate 8-bit read-out
and the MODULATE dial; every number read from `results/<e>/results.json`; negatives reported as negatives. The job
list and IDs are in `MONDAY2.md`.

* **E11-cal** (`experiments/e11cal_noise_calibration.py`, `experiments/relpert.py`). rel_pert(σ) = ||y_noisy − y_ideal|| /
  ||y_ideal||. It is measured per read site (31 sites), locally (clean input) and cumulatively, against the exact
  float product of the same stored weights. The primary scalar is the mean local value. E11 kept no checkpoints, so
  the σ = 0 arms are retrained with the E11 code (seeds 0–4, both regimes, both substrates; reproduction is checked
  against E11's own runs), and seed 0 is also trained at each σ* as a robustness check. Measurements are taken "as
  run" (with the old dial clip) and unclipped. Pre-registered: the optima line up if, in both regimes, R =
  rel_pert*_f32 / rel_pert*_tiles has a CI containing 1 and 0.5 ≤ R ≤ 2. It runs on a V100 rather than a CPU because
  of the retraining. Building it found the dial clip ("Correction 2026-10-23" at the end of this file).
* **E1-fix + E16** (`experiments/e1fix_gpt2_deploy.py`). GPT-2 124M on the actual tile path; E1 used a float noise
  model. Each column is programmed by write-verify with its own scale (max |W| of the column: a periphery gain, so no
  weight is clipped). σ_device is split equally over d2d, c2c and read. The attention products stay digital (as in
  E1). The noise-aware fine-tune is off-chip and hardware-aware (Joshi et al. 2020): a float32 master is
  re-programmed every step, the forward runs through the tiles, a straight-through gradient reaches the master, and
  the tiles are programmed once at the end. The reason it is not on-chip: lr 2e-5 updates are below the 16-bit
  accumulator's resolution relative to one pulse (~0.02 × scale), so on-chip nearest-pulse writes would hardly move
  the weights; that arm is kept as a diagnostic at 0.04 × 2000 steps. Criterion: some arm at σ_device = 0.04 within
  3 % of its matched float32 reference (deploy arms: pretrained float32; fine-tuned arms: float32 σ 0 fine-tuned for
  the same number of steps). A perplexity above 2× its reference is reported as "diverged". Read averaging costs k×
  the array-read energy of the E8b model (one conversion, since the reads integrate in analog); the bound with k×
  conversions is reported too.
* **E15** (`experiments/e15_scratch_lm.py`, `experiments/e15_data.py`). The E8 GPT at 8 layers × 512 × 8 heads,
  context 512, vocab 50257: 25.2M block parameters plus an untied 25.7M head on tiles, and a 25.7M digital embedding.
  The brief's "about 40M" counts it one way; 51M sit on tiles. The brief's dial arm ("σ = 0.04 during training, then
  0.005 at eval") trains identically to the tiles σ = 0.04 arm with the same seed, so it is that run evaluated at the
  dial minimum: 9 trainings per seed, not 10. The float32 reference computes attention with fused SDPA on the noisy
  K / V reads (the same product without a converter, and the memory it saves is needed at batch 64 × 512). Weight
  decay is 0 so the noise is the only regulariser, as in E11. Batch draws depend only on (seed, step, micro-batch), so
  the data are matched across arms. Checkpoints come at every 10 % of tokens and at the end of each 6 h segment; a
  resumed run reproduces the uninterrupted run exactly (checked on the smoke model: same val loss to 1e-15, same
  writes). Hypotheses H1–H4 are in the docstring. Budget rule: measure throughput first; 3 seeds if they fit,
  otherwise 2; if 2 do not fit, cut the tokens for every arm alike, never the arms.
  The rule is fixed before any timing number exists (MONDAY2.md step 3): hours per seed h = 4 × 0.8e9 / tps_f32 + 5 × 0.8e9 /
  tps_tiles, budget 120 GPU-h; 3 seeds if 3h fits, else 2 if 2h fits, else 2 seeds at 0.8e9 × 120 / (2h) tokens.
* **E17** (`experiments/e17_representations.py`). Exploratory, with no pass/fail: per-layer CKA, a next-token probe
  (within-model, raw transfer, Procrustes-aligned transfer) and 10-NN graph overlap between the float32 σ 0 and tiles
  σ* E15 models of the same seed.

## Conventions

* **Units.** Conductance in siemens, `G_min = 1 µS`, `G_max = 100 µS`; a
  differential pair gives `W = (G+ - G-) / (G_max - G_min)` in `[-1, 1]`.
  Anything that needs larger weights (attractor kernels, pseudo-inverse
  matrices) stores `W / scale` on the tile and multiplies the read by
  `scale` (`memory.ScaledTile`, `RingAttractor.scale`).  Noise is
  multiplicative so the scaling is exact.
* **Tile orientation.** `Tile(n_in, n_out)` holds `W[i, j]` from input `i`
  to output `j`; `read(x) = x @ W_read`.  For the ring the kernel is stored
  as `W[i, j] = W(theta_j - theta_i)` (post minus pre) so the antisymmetric
  `W'` has the paper's sign.
* **Randomness.** Every object takes a `numpy.random.Generator`.  E3 uses
  three independent streams per seed (fabricated mismatch, trajectory,
  dynamic noise) so that one seed pairs the same hardware and the same
  velocity trace across every noise level (`RingAttractor.set_dynamic_rng`).
* **Time step.** Rate and LIF neurons use forward Euler with `dt = 1 ms`
  and `tau = 10 ms` (rate) / `20 ms` (LIF).  Additive neuron noise is
  scaled by `sqrt(dt / tau)` so `sigma` is the stationary standard deviation
  of a free unit.

## L0 device

* **Step law.** Implemented exactly as given.  `pulses_for` inverts it in
  closed form (`n = (e^{beta w1} - e^{beta w0}) / (alpha beta)`) and pulses
  are applied one at a time so the law holds exactly with per-pulse
  cycle-to-cycle noise.
* **Device-to-device noise** is a fixed gain on the step size, clipped at
  0.05 so no device steps backwards.  **Dead devices** are stuck at
  `G_min` and ignore pulses (a stuck-off failure).  A pair whose `G-` device
  is dead still conducts through `G+`; the growth rule prunes any pair with
  a dead device immediately, which is where the "dead-device mask" acts.
* **Write-verify** (`program_target`) averages 8 reads per verify step
  (one read of a device near `w = 1` is uncertain by `0.02` in `w` with
  `sigma_r = 0.02`, equal to the usual tolerance), damps the pulse count by
  0.7 to stop measurement noise from ping-ponging between the coarse
  depression step near `w = 1` and the coarse potentiation step near
  `w = 0`, and floors the tolerance at half a pulse step (a device cannot be
  placed more precisely than one pulse; near `w = 0` the potentiation step
  is `alpha_p = 0.05`, larger than `tol = 0.02`).  With these, 500 random
  targets converge in < 60 verify iterations with 100 % of devices within
  tolerance.
* **Tile.update returns the realised change.** A requested change smaller
  than half a pulse is not written; callers that accumulate small changes
  (`plasticity.SurpriseGate`, the self-organising sheet) subtract what was
  realised and keep the residual.  An earlier version forced at least one
  pulse for any nonzero request; that set a noise floor equal to one pulse
  on every learning rule and was removed.

## Neurons and plasticity

* LIF and rate neurons are both provided; the attractor experiments use
  rate neurons (the paper's ring and grid equations are rate equations).
  STDP in E4/E5 is driven by Bernoulli spikes drawn from the rates
  (`spike_rate_scale = 100 Hz` at `r = 1`); a `stdp_mode="rate"` option
  applies the expected STDP change without sampling noise (used only in
  diagnostics; it made no difference to the E4 outcome).
* **STDP window asymmetry.** `A- = 0.5 A+`.  With `A+ = A-` and equal
  time constants, two neurons that are co-active with no causal order get
  zero net change (pre-before-post and post-before-pre pairs cancel), so a
  balanced window cannot build a symmetric excitatory kernel from a moving
  cue; this was measured (learned profile uncorrelated with `cos`).
* **Three-factor rule** is `dW = eta m e` with a write threshold; the
  `SurpriseGate` accumulates sub-threshold updates so that small persistent
  changes are eventually written (write-budget friendly, Lesson 2.7).
* **Homeostatic scaling** is implemented as the paper's rule in
  `plasticity.homeostatic_scaling` and, in the self-organising sheet, as a
  per-neuron factor on the column norm applied once per normalisation
  interval.  It is *off by default in E4/E5* (`tau_H = 0`): with a
  random-walk cue the rule imprinted the cue's coverage statistics on the
  weights (neurons the cue visited less got up to 2x stronger inputs,
  column norms split 0.5-2.0, and every released bump slid to that region);
  with a uniform sweep cue it is unnecessary.  A `tau_H` of 50 s or more is
  the right order if it is enabled (the cue must cover the sheet many times
  per `tau_H`).

## L3 growth

* `C_ij` is an exponential running average of `m x_i xi_j` with `xi_j` the
  deviation of output `j` from its own running mean.  Growth takes the
  largest `|C|` above `theta_grow` up to the budget and a per-call cap, and
  seeds the new pair along `sign(C)` with `w = 0.05` on the matching
  device.  A unit test checks that `C` recovers the sign of the true
  gradient under node perturbation.
* Pruning requires `|W| < theta_prune` for `T_prune` consecutive calls;
  pruned pairs are reset to the bottom so a regrown synapse starts clean.
* `spawn` programs a child tile to `W (1 + 0.05 N(0,1))`; the child also has
  its own fresh fabrication mismatch, so two children of one parent differ.

## L4 attractors

* **Ring.** `J0 = -0.5`, `J1 = 3`, `I = 1`, relu neurons.  A bump forms
  iff `J1 > 2` (tested at 1.5, 1.9, 2.5, 3.0).  The velocity gain `c`
  (rad/s per unit command at `gamma = 1`, about -95 for these parameters)
  is calibrated once on the ideal network and the command is `v / c`, so the
  true integrated angle is `phi0 + int v dt`; any difference between the
  mismatched ring's gain and the ideal one counts as path-integration error.
* **Mismatch** is the paper's `W (1 + sigma zeta_ij)` with `zeta` taken
  from the tile's own fabricated d2d gains (so it is a property of the
  hardware, not a fresh draw) and then written by write-verify.  Open-loop
  programming would attenuate the mismatch through the saturating step
  law, which is not what the paper's model describes.
* **Dynamic noise** in E3 is split half into multiplicative read noise on
  both tiles and half into additive neuron noise.
* **Grid kernel.** A positive-centre DoG with relu neurons and no
  saturation runs away (the self-excitation of a localised bump exceeds
  one).  The default is the zero-centre DoG `a = b = 1`, `s_i = 2 s_e`,
  gain 2: a purely inhibitory surround with uniform drive, which has
  `W_hat(k*) = 3.7 > 1` at nonzero `k` and `W_hat(0) << 1`.  It gives a
  hexagonal multi-bump lattice on 16x16 and a single bump on 5x6, bounded
  activity, and translates along `x` only under `v_x`.  The 2D phase is
  decoded from the Fourier coefficients at the two dominant non-collinear
  wavevectors.
* **A 5x6 sheet is too small** for a clean 2D attractor with these
  kernels (the bump hops between columns when pushed along a row) and the
  programmed module does not certify as a torus either; this is reported
  as a finding, not hidden.

## Self-organisation (E4, E5) - what was tried and why the protocol is what it is

The self-organising sheet is `tile + rate neurons (saturating at r_max = 2)
+ STDP + growth + fixed global inhibition + tonic drive`, developed under a
moving cue with a sub-threshold recurrent gain and then run at the operating
gain.  Each element below was added because its absence was measured to
break the ring:

1. **Saturation `r_max = 2`.** Without it the Hebbian loop (activity ->
   STDP -> stronger weights -> activity) ran away between normalisations
   (rates 1e60 within one second).
2. **Developmental gain `gain_dev = 3` (< bump threshold) during learning,
   `gain = 12` at test.** With a self-sustaining bump already present during
   development, the bump competes with the cue and STDP reinforces wherever
   it parks (column norms split into 0.4 vs 3.4 between two halves of the
   ring).
3. **Uniform-coverage sweep cue** (constant speed, direction reversed every
   lap; 2D: straight motion with an irrational heading, reflected every
   10 s) instead of an OU random walk.  The random walk covered the ring
   ~1.4 times in 30 s with large regional differences, and the learned
   kernel inherited them.
4. **Row and column normalisation.** Column (post-synaptic) normalisation
   alone left presynaptic row sums varying ~4x around the ring, which
   pinned the bump.
5. **Rare normalisation (every 5 s, `eta = 0.02`).** Every normalisation
   rewrites the whole tile through write-verify, and each rewrite carries a
   pulse-resolution error of ~0.02 in `w`; 200 rewrites accumulate ~0.3
   (measured within-offset weight scatter of 50-100 % of the mean), 8
   rewrites ~0.06.
6. **Synapse budget 0.75 N^2.** A 32-ring needs ~0.47 N^2 for the
   near-neighbour kernel alone; the ideal `cos+` kernel on a full
   neighbour mask certifies with up to 30 % multiplicative weight noise
   (H1 bar 0.52 vs null 0.43) but **fails if 20 % of its synapses are
   missing** at zero noise, so the mask must be essentially complete.
7. **Wide cue (`cue_width = 1 rad`).** The learned kernel width follows the
   cue width; a 0.5 rad cue gave a kernel too narrow to sustain a bump.

With all of this, E4 certifies the self-organised ring in 2 of 3 seeds
against 0 of 3 untrained controls, and the released bumps occupy 6 of 8 ring
bins on average (the ideal programmed kernel occupies 8/8 with H1 bar 2.0
vs null 0.85; the learned kernels' bars are 0.2-0.5 vs 0.1-0.3).  The
remaining gap is residual translation variance of the learned kernel: an
honest partial result, not a certified continuous attractor in every seed.

## Topology

* `ripser` with PCA to <= 20 dimensions and max-min subsampling to 300
  points (250 with H2).  `gudhi` is a fallback.
* **Significance rule.** A bar is significant if it exceeds the 99th
  percentile of the pooled null lifetimes **and** the largest lifetime seen
  in any single null diagram.  The pooled percentile alone is dominated by
  the many short null bars and let 5-8 H1 bars through on a clean synthetic
  torus; the extra maximum criterion fixes that while leaving a synthetic
  ring at 1 bar (2.2 vs 0.68) and a synthetic torus at (2, 1).
* The null is the circular shift of each neuron's trace, computed in the
  same PCA basis and with the same subsampling as the data.
* WhiteBox metrics: `SV` uses a 2-Wasserstein distance computed by
  assignment with diagonal projections (no `persim` dependency).

## Memory (E6)

* **"50 % corrupted"** is read as 50 % of the cue entries erased (set to
  0).  Flipping 50 % of +-1 bits would leave zero information in the cue.
  A `--flip` option adds bit flips on top.
* The attractor cleanup on the index is the pseudo-inverse autoassociator
  `P P^+` on its own tile iterated with `sign` (the Personnaz
  pseudo-inverse Hopfield rule), i.e. a binary attractor on the scaffold.
* The delta rule is run through the noisy tile with a full-scale range set
  in advance from the known solution magnitude (hardware would use a fixed
  range); it converges to the pseudo-inverse solution (unit test).
* Attention uses `beta = 0.5` on 32-bit +-1 patterns and both reads go
  through the same tile (`read` for `X^T xi`, `read_T` for `X a`).
* Result worth flagging: the pseudo-inverse scaffold recalls <= 8 pairs
  exactly from half-erased cues while attention recalls all 32 at every
  noise level up to `sigma = 0.2`; device noise is not the limiting factor
  for either, the erasure is.

## Module graph (E7)

* Modules are generic recurrent tiles (16 neurons, 30 % recurrent density),
  relaxed for 8 iterations per trial with `dt/tau = 0.5`; a tonic drive of
  0.5 keeps every module active enough to be probed.
* **Learning rule.** Node perturbation for every inter-module and recurrent
  synapse: eligibility `sum_t x_pre xi_post`, modulator `m = -(L - L_bar)`
  z-scored by a running deviation and clipped to +-3 (without the scaling
  the write size tracked the raw loss and diverged), applied rate
  `eta / sigma_xi^2` because `<m x xi> = sigma_xi^2 * gradient`.  The
  readout uses the delta rule with the output error as the vector
  modulator (the error is available at the output neurons, so this is still
  a local three-factor rule).  Node perturbation's signal-to-noise falls as
  `1/sqrt(#perturbed units)`, which forces `eta = 0.001`; this is the main
  reason the tasks are small.
* **Single-module exploration.** Per trial only one randomly chosen module
  carries the probe noise (a global gating signal), which raises the
  estimate's SNR by ~sqrt(K); the route statistic for target `l` is then
  conditioned on the trials in which `l` was probed (without that it was
  diluted by `1/K` and no route ever crossed `theta_e`).
* **Route creation** uses `||<m xi_l z_k^T>||_F` over all ordered pairs,
  one route per trial above `theta_e = 0.02`, within a budget of 12 routes,
  seeded along the sign of the correlation matrix; routes whose mean `|W|`
  stays below 0.02 for 300 trials are pruned.
* **Equal synapse budget** for the fixed baseline: a layered graph
  (inputs 0,1 -> hidden 2-5 -> outputs 6,7, 16 routes) with projection
  density scaled so its route synapses equal the grown graph's budget.
* Mutual information discretises each module's mean activity per trial
  with k-means (6 clusters) and reports plug-in MI in bits with a shuffle
  baseline.

## E2

* Both layers of a 784-256-10 MLP are 10 % dense; drop 30 % of active
  synapses by smallest `|w|` every 100 steps, regrow the same number by the
  method's score; no topology change in the last epoch (RigL cool-down).
* The local rule's `C_ij` accumulates `m x_i xi_j` over the interval with
  `xi` injected into each layer's pre-activation (`sigma = 0.1`) and
  `m = -(L - L_bar)` per sample; RigL uses the dense gradient on a 1024
  sample batch.  Weights are trained by SGD in all three arms so only the
  regrowth criterion differs.
* MNIST is fetched as raw IDX files (no torchvision in the environment);
  `--smoke` uses synthetic Gaussian-prototype data.

## E1

* Every `nn.Linear` / GPT-2 `Conv1D` weight is read through
  `W (1 + s zeta)(1 + s eps_t)`, `s = sigma / sqrt 2`, via forward hooks
  that swap the parameter for the noisy tensor during the call and put it
  back (gradients flow to the clean weight).  `--c2c` optionally multiplies
  each optimiser update by `(1 + c2c N(0,1))`.
* Not run here beyond `--smoke` (tiny random GPT-2, synthetic tokens, 20
  CPU steps, exercises the full pipeline without downloads).  Default
  budget on a GPU host: 400 AdamW steps per sigma, batch 8, block 512,
  lr 2e-5.

## Budgets

* `scripts/run_all.sh` runs E3-E7 with 3 seeds and regenerates figures;
  `--full` uses 10 seeds.  Timings are in the README (~15 min for 3 seeds;
  E5's 16x16 sheets dominate, so they use 12k development steps and 100
  kick positions instead of 20k and 160).

## Alignment with the two BPU documents (2026-10-16)

Checked against *BPU Technical Document — Architecture, Mathematics, and
Contribution* and *BPU Patent-Style Disclosure* (both in `~/Downloads`).
The equations match the brief and the code.  Changes made to match the
documents where they are more specific than the brief:

* **Certification threshold** is now the technical document's definition:
  the 99th percentile of the *longest* null feature per dimension (the
  pooled 99th percentile is still recorded as `threshold_pooled`).
* **Betti-0 failure signature.**  The document's ring signature is
  (1, 1, 0) and its failure mode is "extra beta0 bars (fragmented by
  pinning)".  `certify` now counts significant finite H0 bars as
  `betti[0]` and a ring or torus certificate requires zero of them.  This
  is stricter than the brief's "exactly one H1 bar" and is applied to E4/E5.
* **Writes per device against the cycle budget** (Section 11 of the
  technical document, budget ~6e-3 writes per device per second):
  `Tile.writes_per_device(seconds)`; E4 reports it next to the budget.
* **Gate-scaled plasticity / kill switch** (claims 5-6): `Tile.plasticity_gain`
  multiplies every write; 0 freezes plasticity.
* **Fast/slow consolidation pair** (Section 5.1, claim 16):
  `plasticity.ConsolidationPair` implements `G_eff = g_f + g_s` with
  volatile decay and replay-gated transfer.  Not used by any experiment.
* **E2 verdict wording** follows the document's criteria: success = within
  a few percent of RigL at equal budget; refuted = clearly worse than random
  regrowth.
* **Attention beta.**  The document uses `beta = 1/sqrt(d)` for the
  transformer identity; E6 uses `beta = 0.5` (`1/sqrt(32) = 0.18` would be
  the literal value).  Kept at 0.5 because the point of E6 is recall
  capacity under noise, not transformer equivalence; the constant is one
  line (`BETA`) in `e6_memory.py`.
* **Hardware note.**  BPU-1 assigns tile B to a 5x6 grid module.  In this
  simulator a 5x6 sheet with the DoG kernels tried does not form a clean 2D
  attractor (the bump hops between columns and the programmed module does
  not certify as a torus under the document's own null); the E5 result
  should inform that tile's design before tape-out.
* **Not implemented** (no experiment needs them): SV-gated inhibition of
  structural change (claim 15; SV is computed, the gate is a one-line
  policy on top), spawn-on-error of whole modules in E7 (claim 3; `growth.spawn`
  exists at tile level), the control-plane/security layer (Section 9).

## Weekend programme (2026-10-17): E8-E10 and the fixes to E2-E7

### Torch tile layer (`bpusim/torchtile.py`)
* `TileLinear` mirrors `tile.Tile` for model-scale work: differential pairs
  `w+`, `w-` in [0, 1], mask, fixed d2d gain per device, fresh read noise per
  device per forward, and **pulsed soft-bound writes**: the optimiser proposes
  a float step, the tile converts it to a whole number of pulses of size
  `alpha exp(-beta w)` (alpha = 0.02 so a device has ~230 levels, about the
  4 bits per device the technical document quotes), applies per-pulse c2c
  noise, keeps the sub-pulse residual in a pending accumulator.  The pulse
  count uses the step size at the start of the write (not one pulse at a
  time), a simplification that matters only for writes of many pulses.
* `LOAD` by write-verify is emulated as programming to within
  `max(tol, half a pulse)` with one uniform error draw per device rather
  than by iterating reads, because 10M devices x 8 verify reads per
  iteration is not affordable in the inner loop of E8.
* Embeddings and layer norms stay digital (periphery); the attention cache is
  a volatile crossbar (write noise when K/V are stored, read noise on both
  reads).  The backward pass of a tile is counted as 2x the forward MACs.
* Feedback alignment is a custom autograd function: forward through the
  noisy read, input gradient through a fixed random matrix, weight gradient
  local.

### E8 protocol
* Ideal baseline 1500 AdamW steps, batch 24, context 128 (2.72M parameters:
  the brief's "~10M" with d = 192 and 6 layers is 2.7M; d = 384 would be 10.6M
  and ~4x slower, so the architecture in the brief was kept and the
  parameter count is reported).  Noise-aware training is **fine-tuning from
  the ideal checkpoint** for 300 steps at each sigma (the E1 protocol), not
  training from scratch at each sigma; deploy mode loads the ideal weights
  and evaluates.  All three share one validation generator so losses are
  paired.

### E9 protocol
* The second corpus is this repository's own Python source, repeated to the
  length of Shakespeare (a plain-text file in the repo, as the brief allows).
* The local three-factor rule uses a purely Hebbian eligibility
  `<pre_i post_j>` with the z-scored, clipped loss modulator and homeostatic
  column scaling; there is no noise-correlation term, so on a transformer it
  is expected to be weak and is reported as it comes out.
* Chip time per training step is an assumption (10 ms by default,
  `--step-seconds`) because the write-rate budget is per second and a
  simulation step has no physical duration.

### E10 protocol
* `DualTileLinear`: a volatile tile (`tau_f` = 2000 steps = 20 s at 10 ms per
  step) in parallel with a nonvolatile tile; learning writes the volatile
  one; consolidation during replay **transfers** weight from fast to slow
  (conserving `G_eff`), gated per postsynaptic unit by its Feature
  Persistence Lifetime.  FPL per unit is operationalised as the unit's own
  H0 death time in a single-linkage filtration of the unit activity vectors
  over the replay set (how long the unit stays a distinct feature); units
  above the median consolidate, the rest fade.  The paper's FPL is per
  topological feature of the activity cloud; this per-unit form is the
  closest quantity that can gate synapses.
* "After rest" numbers evaluate the network after one `tau_f` of no input,
  when the un-consolidated fast weights have decayed.

### Fixes and what they revealed
* **A write-dropping bug made the first E7 weaker than it should have been.**
  The module graph released accumulated updates from the surprise gate and
  the tile rounded sub-pulse requests to zero, so almost nothing below half a
  pulse (0.025 in weight units) was ever written.  With the residual kept
  (`modules.gated_write`), the two-input test graph trains to loss 0.004
  (local rule) / 0.010 (node perturbation) where it stalled at 0.39 before.
  The E7 verdicts are re-run with the fix; the earlier E7 null should be read
  with this in mind.
* **E7 learning rule.** Layer-local losses: every module has its own readout
  to the task target and minimises its own loss with a one-step gradient that
  never leaves the module (its readout, recurrent tile, incoming routes and
  input tile); node perturbation is kept for the route statistic only.
  Baselines: the fixed layered graph at matched synapses and the same trials,
  and at matched compute (trials scaled by the ratio of MACs per trial).
* **Homeostasis in E3.** The paper's rule with tau_H = 5 s applied to a
  running-mean rate with tau_rate = 1 s destroyed path integration on a
  mismatch-free ring (RMSE 1.25 rad vs 0.05 without homeostasis): the gain
  chased the moving bump.  tau_rate = 20 s and a uniform initial running mean
  reduce the damage but a dwelling bump still adapts its own gain away
  (0.2 rad/s drift at zero mismatch), so E3 is run as two arms, tau_H = 5 s
  and off, and both are reported.  The homeostatic gain is a read-side
  per-neuron gain (no device rewrites).
* **E3 pinning landscape** is measured directly: the bump is placed at 48
  positions, its drift velocity at zero dynamic noise is integrated around
  the ring to U(phi), and Delta U = max U - min U.  The fit
  `E(D) = c1 exp(DeltaU/D) + 2 k D T` with `D = sigma^2` is done by
  least squares with a bootstrap over seeds for sigma*.
* **E4 endurance cap.** Capping development at 6e-3 writes per device per
  second by flushing only the largest pending changes each step (a per-tile
  write quota) and running 18x longer (720 s) gives 0/10 certified rings
  against 5/10 uncapped (10 seeds, 1/10 control false positive): the quota
  starves the small depression writes that shape the kernel.  A rate cap is
  not a free substitute for fewer, larger writes; this is reported, not
  tuned away.
* **E5 certifier.** Three changes, each forced by a measured failure on a
  synthetic torus at the experiment's own sampling density (test
  `test_certify_torus_at_experiment_sampling_density`): (1) a bar must also
  be at least half the longest bar in its dimension (sampling-hole bars of a
  large torus exceed the shift null, whose clouds are blobs); (2) H0
  fragments are components that outlive 3x the median H0 bar, not every
  finite bar above the null (max-min sampling makes all merge distances
  similar); (3) H2 may be computed on a smaller subsample than H1
  (`n_points_h2`).  A further finding: for a lattice that sweeps the torus
  along straight lines every neuron's trace is periodic, and the circular
  shift of periodic traces is not a null at all (it keeps the loops), so the
  programmed reference is driven with frequent random turns.
* **E6** now flips 20 % of the cue bits (no erasure) as asked; the 50 %
  erasure result is kept in the git history (commit 0963cca).
* **E5 reference protocol.** With the three certifier changes above, the
  programmed 32x32 module still failed when driven for 3 s along straight
  sweeps (its H1 bars 3.2 and 2.8 against a null of 2.8: periodic traces
  survive a circular shift) or with turns every 0.2 s (four equal loops and no
  H2: the torus was covered by a few line segments).  Thirty seconds of
  trajectory with a random turn every 0.3 s, sampled every 30 ms, covers the
  torus densely and certifies: H1 bars 2.36 and 2.05 against a null of 0.65,
  one H2 bar 1.18 against 0.28, no fragments.  `reference_cloud` uses that.
  The 5x6 and 16x16 verdicts from the first version stand as reported there;
  BPU-1 tile B should be reassigned (a 5x6 sheet cannot host a torus).
* **Thread oversubscription.** Multiprocessing workers each spawned a full
  set of OpenBLAS threads (8 workers x 20 threads on 20 cores), which stalled
  every job on the machine; all pool-based scripts are launched with
  `OMP_NUM_THREADS=1` (see `scripts/run_weekend.sh`, `scripts/e5_chain.sh`).
* **E7 module size.** Doubling modules to 32 neurons without rescaling the
  weights saturated them at `r_max` and both rules collapsed to chance on
  the navigation task (0.18 correct).  Initial weight scales now shrink by
  `sqrt(16/n)` (`modules._scale_for`), after which node perturbation reaches
  0.83 and the local rule 0.97 on the fixed graph in 2000 trials; the local
  learning rate had to drop from 0.01 to 0.001 at this size.  The first
  32-neuron E7 run (results/e7_firsttry) is kept for the record.
* **E10 first result** (3 seeds): BPU consolidation average accuracy 0.79
  (0.81 after rest) vs joint 0.95, naive 0.85, frozen 0.26; BPU forgetting
  0.21 vs naive 0.14.  The criterion is not met: the persistence-gated
  transfer loses the un-gated half of each task's fast weights and does not
  protect old tasks from the next task's fast weights.  Reported as is; a
  gate-fraction / tau_f scan is on the Monday list.
* **E2 CIFAR-10 at 3 % density** (3072-512-512-10, 5 epochs, 3 seeds): local
  0.420, RigL 0.433, random 0.433: refuted by the document's own criterion
  (clearly worse than random regrowth).
* **E8 result and the E8b accounting.** Noise-aware tiles meet the 2 % band
  at every sigma <= 0.06 (two seeds; a third was dropped when the noisy
  forward ran at 15 s/step under CPU contention).  E8b counts A/D
  conversions two ways because the brief and the technical document differ:
  one conversion per 32-row tile partial sum ("tile boundaries only") makes
  the BPU projection *twice as expensive* as digital at 256 pJ per
  conversion; analog accumulation across the tiles of a layer with one
  conversion per output ("values stay analog inside tiles") makes it 3x
  cheaper, still 99 % conversion.  Both are reported; the energy claim rests
  entirely on the converter architecture and constant, not on the MACs
  (11 nJ/token).  Write energy is amortised from the measured writes per
  device per fine-tuning step (~4e-4) at 1.5 pulses per write.
* **E9 segment length.** Run with 1000-step segments instead of the planned
  2000: on this loaded CPU the frozen pass alone ran at 1.3 s/step
  (6000 steps = 2.2 h) and the local and feedback-alignment passes are
  slower, which would have pushed E9 past midnight.  The recovery criterion
  (tracking within 300 steps of a switch) is unchanged; the first attempt's
  frozen pass (logs/e9_firsttry.log) showed the Python-source segment at
  6.0 nats/char against 1.66 on Shakespeare, as expected.
* **E5 at 32x32 and 48x48 (Saturday night).** Programmed Burak-Fiete
  references on tiles certify as tori: 3/3 draws at 32x32 (H1 2.2-2.75 vs
  null ~0.5, H2 1.2-1.7 vs 0.13) and 1/1 at 48x48 (H1 5.3/5.2 vs 0.94, H2
  3.3 vs 0.28; 9 h single-threaded).  A fourth 32x32 draw, computed inside
  the self-organisation run, did not certify (78 near-equal H1 bars: the
  lattice hopped instead of sliding), so sliding is draw-dependent at the
  5 % mismatch default.  The self-organised 32x32 sheet formed no attractor
  at all (uniform activity, 99 release clusters, as the untrained control);
  the ring recipe does not transfer to a 1024-neuron sheet unchanged and is
  not tuned here.  BPU-1 tile B must be reassigned: a 5x6 grid cannot host a
  torus, and the smallest sheet that certified is 32x32.

## Substrate programme (2026-10-18): reference-first reporting, the write budget, and the reruns

**Reframe.**  The BPU is a substrate.  From this date every experiment compares the
SAME algorithm on (a) a float32 reference and (b) noisy BPU tiles, and success means
the tiles match the reference within a stated tolerance.  Our own local rules
(feedback alignment, direct feedback alignment, layer-local three-factor, growth)
are reported as additional arms, never as the headline.  `scripts/make_results_page.py`
puts a six-row "Substrate claims" scoreboard at the top (E9, E10, E8b, E3, E6, E7),
then a write-budget box (E4, E9, E10), then the remaining rows.  The weekend
results that were superseded are kept under `results/e9_v1`, `results/e10_v1`,
`results/e8b_v1` (git history has the scripts: `experiments/.e9_v1.py.bak` etc.).

### Write budget: accumulate-and-flush (`Tile.accumulate`, `TileLinear(write_mode="level")`)
* **Mechanism.**  A digital per-synapse accumulator `acc[i, j] += dW`.  A device is
  pulsed only when `|acc| >= one conductance level`, one level being
  `(G_max - G_min) / L` in conductance = `1 / L` in state units.  The flushed
  amount is a whole number of levels: the integer part of `|acc| / level` plus the
  remainder rounded stochastically (probability = remainder), so the expectation of
  what is written equals the accumulator.  The flushed amount, not the realised one,
  is subtracted (the digital side does not read the device back; the pulse-rounding
  error of the step law stays on the device).  Writes are counted only when a device
  is pulsed; `flush_rate` = write events / (accumulate calls x active synapses).
* **L from sigma_eff.**  Two levels are distinguishable when they differ by one
  effective noise standard deviation at full scale, so `L = round(1 / sigma_eff)`,
  clipped to [2, 256] (`DeviceParams.levels`).  Defaults: L = 21 for the numpy tile
  (sigma_eff 0.047), L = 25 for the torch tiles at sigma_eff 0.04.  One level is
  therefore 0.047 / 0.04 in state units, i.e. about one potentiation pulse near
  w = 0 and ~19 pulses near w = 1; `pulses_for` converts the flushed level count
  into pulses through the step law as before.
* **Accumulator semantics on the torch side.**  `TileOptimizer` keeps Adam's float
  moments and the per-synapse accumulator in float ("float accumulators"); the
  parameter the forward pass reads is always the realised device weight ("BPU
  devices as weights").  The legacy `write_mode="pulse"` (nearest whole pulse,
  residual kept) is unchanged and is what E8's results were produced with.
* **E4 with level writes** (10 seeds, 40 s development, no quota): 3/10 rings
  certify (1/10 controls) at 0.013 writes/device/s = 217 % of the 6e-3 budget,
  down from 0.111 (1850 %) with threshold writes at theta_w = 0.01, which certified
  5/10.  **With the 6e-3 quota on top** (720 s development, flushes ordered by
  accumulator size): 0/10, as with the quota on threshold writes.  Level-gated
  writes cut the write rate 8.5x and keep most of the self-organisation; the quota
  still starves the small depression writes that shape the kernel.  The budget is
  therefore met only by accepting a 10x longer development with a 10x slower cue,
  which was not run; the number to report is 217 % at the kernel's natural write
  rate.
* A whole-tile rewrite (`program`, the sheet's normalisation) clears the
  accumulator; a mask change clears it for the pruned synapses.

### Torch tile layer on CUDA
* The tile layer runs unchanged on the GPU (buffers follow `.to(device)`); the only
  CPU-side tensor was the position index of the nanoGPT.  Speed on this laptop's
  RTX 5070: 0.12 s per tiled training step vs 0.82 s on the CPU (batch 16 x 128).
* **The ideal model's attention is explicit softmax, not `scaled_dot_product_attention`**:
  the fused efficient-attention backward was found to be wrong on this GPU in another
  project (torch 2.11 cu128/cu130, see `~/blindinglight`), and T = 128 makes the
  explicit form cheap.  Tiled attention was always explicit.
* `--out` paths are now resolved against the repository root
  (`common._out_dir`), whatever the shell's directory.

### E9 (rewritten): six arms, same stream
* Arms: frozen; SGD-oracle on tiles (true backprop through the noisy reads, Adam in
  float, level-gated writes: the substrate ceiling); SGD on float32 (reference);
  feedback alignment; direct feedback alignment (`DFAContext`: the output error,
  d CE / d logits per token, is broadcast to every tile through its own fixed
  random matrix, the layer returns a zero input-gradient so nothing travels between
  layers; the head uses its own error); layer-local three-factor (every tile has a
  digital readout to the next-token target, minimises its own cross-entropy with a
  one-step gradient inside the tile: local error x presynaptic activity, nothing
  between layers; readouts learned digitally).  All gradient arms use AdamW at
  3e-4 with the same batches of the same stream (seed-paired).
* Criterion: SGD-oracle within 5 % of the float32 reference on the post-shift loss
  (mean over the last 200 steps of the Python segment, same step count) and writes
  within 6e-3 per device per second at the assumed 10 ms per step.  Secondary: the
  gap of FA / DFA / local to the oracle.  3 seeds; 1000-step segments as in the
  weekend run (2000-step segments are on the HPCC list).
* The weekend's Hebbian "local three-factor" rule (purely Hebbian eligibility,
  scalar modulator) is replaced by the layer-local-loss rule; the Hebbian rule
  degraded the model on every segment and has no analogue on the float32 side.

### E10 (rewritten): five algorithms x two substrates
* Arms: naive; EWC (diagonal empirical Fisher from 200 per-sample gradients after
  each task, penalty lambda / 2 sum F (theta - theta*)^2 on the tile weights,
  lambda = 100); experience replay (buffer = 1 % of the training set = 600
  samples shared equally by the tasks seen, half of every batch); FPL-gated
  consolidation (the weekend's fast/slow pair, gate = units above the median
  Feature Persistence Lifetime); FPL-gated + replay; joint training as the
  upper bound.  On float32 the fast/slow pair is two float tensors with the same
  decay and transfer rule.
* Ablations on tiles: FPL gate off (every synapse consolidates) and device
  retention tau_ret in {1e3, 1e4, 1e5, inf} s applied to the nonvolatile devices at
  10 ms of chip time per step; since the whole sequence is only ~40 s of chip
  time, every arm is also evaluated after a 1000 s rest (fast devices fully decayed,
  nonvolatile ones by exp(-1000 / tau_ret)), which is where tau_ret acts.
* Criterion: BPU-EWC and BPU-replay within 3 points of average accuracy of their
  float32 twins; forgetting reported per arm; writes against the budget.  3 seeds,
  run on the GPU.

### E8b (rewritten): measured digital control, 3 x 2 read-out table
* **Digital control = measured, not Horowitz.**  The same nanoGPT runs on this
  laptop's GPU with every Linear at INT8 (W8A8: per-channel int8 weights,
  per-token dynamic int8 activations, int32 accumulation through
  `torch._int_mm`; attention, LayerNorm, GELU, softmax in FP32 as on any INT8
  accelerator).  Energy = board power (nvidia-smi at 10 Hz) x time / tokens,
  reported with the idle power included and subtracted; FP32 on the same GPU the
  same way.  `torchao` is not installed, so the INT8 kernel is PyTorch's own int8
  GEMM; `torch.ao.quantization.quantize_dynamic` runs on the CPU only and its
  energy cannot be measured here (RAPL counters are root-only), so the CPU path is
  reported as throughput only.  A published H100 INT8 peak (1979 TOPS at 700 W) is
  included and labelled as published, not measured.  The measurement must be
  taken on an otherwise idle GPU (`--skip-train` after the training phase).
* **BPU read-out modes.**  (i) per-tile 8-bit ADC: one conversion per 32-row tile
  partial sum, added digitally; (ii) analog accumulation across the tiles of a
  layer, one conversion per output; (iii) spiking LIF read-out with no ADC: a LIF
  neuron per output integrates the current and its spike count (at most 2^b - 1
  per token, sign on a complementary pair) encodes the value.  Converter
  precisions 8-bit (256 pJ) and 4-bit (16 pJ).  Periphery: LIF 5 pJ per update,
  router 10 pJ per event, write driver 100 pJ per pulse, cache write 10 pJ.  LIF
  energy is counted per spike (one router event + one LIF update per spike; the
  integration itself is analog); the alternative of one update per time step of a
  2^b-step window would cost 2^b x 5 pJ per output and is stated in the JSON.
* **Accuracy of each mode** is measured in simulation, not assumed: the E8
  protocol at sigma = 0.04 (1500 ideal steps, 300 noise-aware steps) is re-run
  on the GPU, and the noise-aware model is evaluated with every read-out emulated
  (`ReadoutSpec`: converter ranges calibrated on one batch, values clip at the
  rails, attention score and value reads converted too).  The LIF spike rate is
  the measured mean count per output per token.
* **Scaling to 1T** by MACs proportional to parameters, per-layer conversions
  proportional to layers x d_model, per-tile conversions proportional to
  layers x d_model^2 / 32 (one conversion per 32-row block), for the standard
  dense shapes 12x768, 24x2048, 32x4096, 80x8192 and a dense 140x24576 (1.0T).

### E3 (replaced): moving cue instead of static drift
* A weak external bump (amplitude 0.1 of the tonic drive, width 1 rad) rotates
  at omega in {0.1, 0.5, 1.0} rad/s; the ring's bump must follow it; tracking
  error = mean |phi_bump - phi_cue| over 30 s.  Dynamic noise sweep
  {0.001, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08} plus 0 as the sign-test baseline,
  tau_H = 0, N = 128, 10 seeds, mismatch {0.02, 0.05, 0.10} plus a mismatch-free
  tile control and the ideal no-tile ring as the float32 reference.  Delta U is
  measured as before.  A second arm at cue amplitude 0.05 is run as well.
* **Why 0.1.**  Pilot (N = 128, 10 s, 2 seeds): the tracking error is dominated by
  the lag, which falls with the cue amplitude (at omega = 1 rad/s: 0.74 rad at
  A = 0.05, 0.36 at 0.1, 0.20 at 0.2, 0.12 at 0.4), and the mismatch adds to it only
  at low speed and weak cue (omega = 0.1: 0.065 -> 0.114 rad from mismatch 0 to
  0.1 at A = 0.05, 0.035 -> 0.063 at A = 0.1, 0.012 -> 0.021 at A = 0.4).  A = 0.1
  keeps the attractor dynamics (and their pinning) relevant while the ideal ring
  still tracks; a strong cue would make the test insensitive to the substrate.
* The float32 reference carries the dynamic noise as additive neuron noise only
  (there are no tile reads to perturb); the BPU arms split it half read, half neuron
  as before.

### E6: sign ties
* `np.sign` of an exact 0 read was mapped to +1 everywhere (cleanup iterations,
  the content read, the direct and attention reads), a bias toward the all-ones
  pattern.  `memory.signz` now breaks ties at random to +-1 from a fixed-seed
  generator (seed 20261018).  Rerun with 10 seeds (`--full`).

### E7: matched steps and the continued arm
* "fixed" was already matched in synapses AND training steps; the verdict now
  says so.  New arm `grown_continued`: after the growth phase the grown graph's
  structure is frozen and it trains for the fixed graph's trial count again
  (its final architecture then has had as many steps as the fixed one), then it
  is evaluated exactly like the other arms.  If it is still behind the fixed
  graph on the base tasks (nav by more than 0.05 or recall bits by more than
  0.02) the verdict logs a NEGATIVE.
* `--noise-free` runs the same script on tiles with zero d2d / c2c / read noise
  (pulse quantisation only) as the reference substrate (`results/e7_ref`).
* **E3 run budget (same evening).**  The first launch (T = 30 s, two cue amplitudes, 26
  arms) ran at ~35 min per arm on the loaded laptop: a profile showed 75 % of every run
  in the write-verify programming of 2 x 16k devices (8 verify reads per iteration, each a
  16k-Gaussian draw), repeated for every dynamic-noise level because the verify read noise
  was tied to sigma_dyn, so every noise level also got a slightly different hardware.  The
  programmed device states are now cached per (N, mismatch, seed) (verify at the device's
  nominal read noise 0.02) and reused across the sweep, which is what "one seed = one
  hardware across every noise level" was meant to be; the cached weights equal the directly
  programmed ones bit for bit.  The remaining cost is the two 16k-Gaussian read-noise draws
  per step, so T was cut from 30 s to 15 s and the second cue amplitude (0.05) dropped; the
  four T = 30 s arms that finished are kept in `results/e3_tracking/partial_T30_4arms.json`
  (all NOT U-shaped, tracking error 0.037-0.047 rad flat in sigma).  `--finalize` rebuilds
  the verdict and figure from a partial file.
* **E10 replay bug (same evening).**  The first full E10 run gave the replay arms 0.29
  average accuracy with 0.82 forgetting on BOTH substrates: the buffer drew one random
  permutation for the inputs and another for the labels, so half of every batch was
  mislabelled.  Fixed (one draw per task, `results/e10/results_replay_bug.json` keeps the
  bad run) and the replay arms were rerun with `--only-arms replay,fpl_replay`, which
  merges into the existing results.json; the other arms are untouched.  The incident is
  also why the page shows forgetting per arm next to the twin gap: a twin gap of -0.005
  between two broken arms would otherwise have passed the 3-point criterion.

### Outcomes (2026-10-18, laptop, 3 seeds unless stated)
* **E9**: float32 SGD post-shift 1.184; SGD on tiles with level-gated writes 1.483
  (+25 %) at 0.0107 writes/device/s = 178 % of budget; FA 2.70, DFA 2.89, layer-local
  2.70; frozen 5.98.  Not met.  Diagnostic: SGD on tiles with nearest-pulse writes
  1.226 (+1 %) at 8665 % of budget, so the device noise is free and the whole gap is
  the write gate, which buys a 50x lower write rate.  Both halves of the criterion pull
  against each other at lr 3e-4: fewer writes = slower learning.
* **E10**: twins within 3 points for EWC (0.947 / 0.943) and replay (0.942 / 0.935);
  met.  Naive 0.844 / 0.793 (the tiles forget more, 0.20 vs 0.14), FPL 0.840 / 0.828,
  joint 0.956 / 0.957.  FPL gate off is better than on (0.90 / 0.87); FPL arms lose
  13-16 points after a 1000 s rest (un-consolidated fast weights); tau_ret 1e3 s costs
  2-17 points after the rest, 1e4 s and above nothing.  Writes 7-24x the budget.
* **E8b**: GPU INT8 measured 256 uJ/token incl. idle (169 above idle), FP32 180 uJ
  (the int8 path is slower than FP32 for a model this small); H100 published peak 2.0 uJ.
  BPU: per-tile 8-bit 22.3 uJ, per-layer 8-bit 3.6 uJ, LIF 8-bit 11.6 uJ (55 spikes per
  output per token), all at -0.6 to -0.8 % loss (lossless); 4-bit cells 1.4 / 0.26 / 0.53
  uJ at +31 / +42 / +53 % loss with 99.9th-percentile converter ranges (+52 / +61 / +76 %
  with full-range converters).  4-bit read-out needs quantisation-aware fine-tuning,
  which was not run.
* **E3**: 0/9 U-shaped.  Tracking error is the cue lag (0.036 / 0.18 / 0.37 rad at
  omega 0.1 / 0.5 / 1), flat in dynamic noise, equal on tiles and float32 except at
  mismatch 0.1 and omega 0.1 (0.062 vs 0.036) where the pinning barrier (Delta U 0.41)
  shows and noise still does not help.
* **E6** (10 seeds, random ties): attention 20 / 24 / 24 / 28 / 28 / 28 pairs at
  sigma 0 ... 0.2, scaffold 0-4: the tie fix moved nothing outside seed noise.
* **E7**: grown 0.62 / fixed 0.84 nav at matched synapses and steps; grown continued
  0.80 nav, 0.90 recall (fixed 0.82): catches up; composition at chance everywhere;
  noise-free reference in results/e7_ref.
* **E4** level writes: 3/10 at 217 % of budget; + quota 0/10.
* **E7 reference run, take two.**  The first `--noise-free` run reproduced the BPU numbers to
  the last digit: Python 3.14 starts multiprocessing workers with forkserver, so the module
  global set in `main()` never reached them and the workers used the default noisy device.
  The device parameters now travel with each job's arguments (the first run is kept as
  `results/e7_ref_bug_was_noisy`, a useful determinism check: same seeds, same numbers).
* **E7 reference outcome.**  Noise-free tiles: nav grown 0.30 / fixed 0.75 /
  grown-continued 0.46, recall 0.84 / 0.85 / 0.89,
  Q 0.06; the verdict logs a NEGATIVE for the continued arm there.  The noisy tiles
  (0.62 / 0.84 / 0.80) are not worse than the noise-free reference; with 3 seeds and a
  seed-to-seed spread of 0.2 in navigation accuracy this is not a demonstrated benefit of
  noise, only the absence of a cost.  The 10-seed pair is job 4 on the HPCC list.

## Round two (2026-10-19): endurance budget, θ_flush, FoM-based converters, 10-seed E7

### Write budget redefined (`Tile.write_budget_report`, `torchtile.write_budget_report`)
* Three numbers replace the single per-second budget: **endurance** = total write events per
  device over the run against 1e6 cycles ("% of endurance"); **driver bandwidth** = write events
  per physical 32x32 tile per second against 1e4; the old 6e-3 writes/device/s is kept in the
  table as the **lifelong rate** for reference.  E9's criterion becomes "within 5 % of float32
  at any θ_flush with < 1e4 total writes per device"; E10 reports total writes per device per
  arm against the endurance; E4 reports certified fraction and total writes per device.
* **θ_flush ∈ {0.25, 0.5, 1.0} levels.**  A synapse is considered for a write when its
  accumulator holds θ_flush conductance levels; the flushed amount is still a whole number of
  levels with the remainder rounded stochastically, so below one level the device is pulsed
  with probability |acc| / level and otherwise waits.  Consequence measured in E4 (10 seeds,
  40 s): θ = 1.0 certifies 3/10 at 0.52 writes/device; θ = 0.5 certifies 0/10 at 3.9
  writes/device (bumps occupy 1.9/8 bins): a half-full accumulator is pulsed by a whole level
  half the time, so the smaller threshold makes the writes both more frequent and noisier.
  θ = 0.25 certifies 0/10 at 2.0e3 writes/device (0.2 % of endurance, flush rate 0.13) and
  1.0e5 writes per tile per second = 10× the 1e4 driver bandwidth: with a quarter-level
  threshold the STDP jitter keeps most accumulators above threshold and the stochastic
  rounding pulses them almost every step.  θ = 1 is the only setting that certifies at all
  and stays inside the driver bandwidth (8.2e-3 of it); the endurance (≤ 0.2 %) is never the
  binding constraint at these run lengths, the lifelong rate and the bandwidth are.

### E9 Pareto
* The four BPU gradient arms rerun at θ_flush 0.25 and 0.5 on the GPU, merged with the θ = 1
  run (same seeds, same stream; `--merge`), reported as post-shift loss vs total writes per
  device with the nearest-pulse diagnostic as the no-gate reference point.

### E8b converters
* `E_ADC = FoM_W × 2^b` with the Walden figure of merit FoM_W ∈ {10, 50} fJ per conversion
  step (Murmann ADC survey; 10 fJ is state of the art, 50 fJ conservative) and b ∈ {6, 8};
  the 256 pJ constant of the first version corresponded to FoM_W = 1 pJ, a hundred times
  worse than the survey, and the 4-bit cells are dropped (they cost 31–53 % in loss).  The 6-bit
  read-outs cost 1.4–1.9 % in loss (per-tile 1.35 %, per-layer 1.78 %, LIF 1.88 % at 13
  spikes per output per token), 8-bit are lossless (−0.6 to −0.8 %).  Table: per-tile /
  per-layer / LIF × FoM 10 / 50 × 6 / 8 bits; the LIF row has no converter so its FoM columns
  repeat.  Controls: the laptop GPU measurement (256 µJ/token incl. idle, 169 above idle),
  H100 INT8 at published peak (2.0 µJ/token for this model) and at 40 % utilisation
  (5.0 µJ).  Outcome: per-layer FoM-10 8-bit 71 nJ/token = 0.014× H100@40 % (6-bit 44 nJ at
  +1.8 %); per-tile FoM-10 8-bit 257 nJ; LIF 8-bit 11.6 µJ (spikes, not converters, are the
  LIF cost).  Scaling column "BPU / H100@40 %": 0.014 at 2.7M → 0.002 at 1T for the
  per-layer 8-bit cell, because conversions grow as layers × d_model and MACs as parameters
  (~ layers × d_model²), so the converter cost per MAC shrinks as 1/d_model.  The measurement
  block is reused from the idle-GPU run (`--reuse-measurement`).

### E10 headline
* Headline arms naive, EWC, replay, fpl_replay, joint; the fpl arm is diagnostic, not a
  consolidation policy (gate off beats gate on on both substrates) and stays in the notes.
  Total writes per device per arm: 1.8–4.4 over the whole sequence (≤ 0.0005 % of
  endurance).  Retention row τ_ret = 1e4 s only: no change at the end of the sequence or
  after a 1000 s rest for naive / EWC / replay, fpl_replay 0.930 → 0.768 after the rest (its
  fast weights decay).  Re-summarised from the existing runs (`--resummarize`), no rerun.

### E6
* `tests/test_memory.py::test_hopfield_k1` (N = 32, K = 1, σ = 0, pinv): exact recall from
  the clean cue and ≥ 95 % from cues with 20 % flipped bits.  It passed on the first run;
  nothing needed fixing.  The capacity sweep now includes K = 1 and 2 and was rerun (10 seeds).

### E3 cue-amplitude sweep (`results/e3_cue`)
* ω = 0.5 rad/s, cue ∈ {0.01, 0.02, 0.05, 0.1}, mismatch {0.05, 0.10}, σ as before, 10 seeds,
  float32 reference per cue.  A condition is "pinned" when the tile error at σ = 0 exceeds
  the float32 reference by more than 50 % (the lag is common to both; the excess is pinning);
  the U-test is applied only there.

### E7 at 10 seeds
* Tiles and noise-free reference, grown − fixed gap with a bootstrap 95 % CI over seeds for
  nav, recall and composite; "noise-assisted growth" is flagged if the tiles' nav gap exceeds
  the reference's at p < 0.05 (Welch one-sided t over per-seed gaps).  The 3-seed runs are
  kept as `results/e7_3seeds`, `results/e7_ref_3seeds`.
* **E3 cue-sweep outcome.**  No pinned regime: at ω = 0.5 rad/s the tracking error at σ = 0 is
  0.18 / 0.35 / 0.94 / 1.49 rad for cue 0.1 / 0.05 / 0.02 / 0.01 on the float32 reference and
  0.18 / 0.36 / 1.06–1.12 / 1.39–1.40 on tiles with mismatch 0.05–0.10: the loss of tracking
  at weak cues is the reference's too (the cue is too weak to drag the bump at this speed),
  and the mismatch adds at most 19 % (cue 0.02, mismatch 0.1), below the 1.5× pinned
  criterion.  Every curve is flat in dynamic noise (0/8 U-shaped), so the U-test had nothing
  to act on.  Finding the pinned regime needs ω below 0.5 (the ω = 0.1 run at cue 0.1 showed
  the only mismatch-dependent excess, +70 % at mismatch 0.1) with cue ≈ 0.02–0.05; queued.
* **E6 rerun outcome** (10 seeds, K from 1): every reader recalls K = 1 exactly at every σ up
  to 0.2 (the unit test's condition, on noisy tiles too); the scaffold readers hold 2–4 pairs
  at 90 % exact recall and attention 20 (σ = 0) to 28 (σ ≥ 0.08) — the capacity curve is
  unchanged by the tie fix, so the earlier scaffold ceiling is the 20 % flips, not a sign bias.
* **E7 10-seed outcome.**  Tiles: nav grown 0.42 / fixed 0.81 / continued 0.48, grown − fixed
  nav gap −0.38 (95 % CI −0.53..−0.23), continued −0.33 (−0.53..−0.13); reference (noise-free):
  0.45 / 0.69 / 0.58, gap −0.24 (−0.44..−0.04), continued −0.11 (−0.33..+0.14).  Recall and
  composite gaps are within ±0.05 of zero on both.  Welch one-sided p(tiles gap > reference
  gap) = 0.85 (Mann–Whitney 0.89): **no noise-assisted growth**; if anything the tiles' grown
  graph lags more.  The 3-seed "grown catches up when continued" (0.80 vs 0.84) does not
  survive 10 seeds (0.48 vs 0.81): the verdict logs the NEGATIVE on both substrates.
* **E9 Pareto outcome** (3 seeds, post-shift loss vs total writes per device; float32 1.184):
  SGD on tiles θ = 1: 1.483 (+25 %) at 0.32 writes/device; θ = 0.5: 1.426 (+21 %) at 3.5;
  θ = 0.25: 1.954 (+65 %) at 281 — the quarter-level threshold pulses whole levels on
  quarter-full accumulators and the stochastic overshoot wrecks the fine-tuning, as it wrecked
  E4.  FA / DFA / layer-local sit at +112 to +166 % at every θ.  **No level-gated arm meets
  the criterion** (within 5 % with < 1e4 writes/device).  The nearest-pulse diagnostic (no
  gate, every Adam step rounded to whole pulses) ends at 1.226 (+3.6 %) with 15.6 writes per
  device = 0.0016 % of the endurance, i.e. it meets both numbers.  Reading: under an
  endurance budget the accumulate-and-flush gate is a cost without a benefit — the device
  noise is free and the whole trainability gap is the gate; the gate only mattered under the
  old per-second lifelong rate.  The remaining real limit is the driver bandwidth
  (writes per tile per second), which the pulse-write run exceeds during the first steps
  after a switch; that number is in the JSON (`budget.writes_per_tile_per_s`) for every arm.

## Round three (2026-10-19): variability, used the way the brain uses it

### Four architecture changes (`bpusim/tile.py`, `device.py`, `torchtile.py`) and their biological analogues
1. **Write rule default = nearest_pulse.**  `Tile.accumulate` / `TileLinear(write_mode=
   "nearest_pulse")`: the requested change goes into a 16-bit signed per-synapse accumulator
   (full scale = one weight unit; `Tile.ACC_BITS`), the accumulator is flushed EVERY step as
   the nearest whole number of pulses per device, and the NOMINAL effect of those pulses (the
   noise-free, gain-free step law, `DeviceArray.nominal_state_after`) is subtracted, because
   the digital side has no read-back; the sub-pulse remainder and the quantisation error carry
   over.  The θ_flush gate and the stochastic rounding are gone from the default (the `level`
   rule is kept as an option; the round-two record stands: under the endurance budget the gate
   was a cost without a benefit).  The legacy torch `pulse` rule, which subtracted the realised
   (noisy) change, is kept for the E8 / E9-diagnostic record.  *Analogue:* plasticity is
   integrated over many sub-threshold events and expressed in discrete synaptic states (all-or-
   none LTP steps, Petersen et al. 1998; ~26 distinguishable spine sizes, Bartol et al. 2015):
   the accumulator is the biochemical integration, the pulse is the step.
2. **`tile.set_noise(sigma_r)` — the MODULATE primitive.**  A per-tile read-noise dial with
   range `MODULATE_RANGE = (0.005, 0.2)` (values outside are clipped and the clip is logged;
   0 = dial off for the reference arms); every call is a MODULATE event in
   `tile.primitive_log` / `OpCounter.modulates`.  *Analogue:* neuromodulatory control of neural
   variability — noradrenergic gain (locus coeruleus, Aston-Jones & Cohen 2005) and the
   dopamine-gated song variability of the songbird (LMAN → RA; Kao, Doupe & Brainard 2005):
   variability is a controlled quantity, not a nuisance.
3. **`tile.set_dropout(p)`.**  Every read drops each synapse independently with probability p
   (a per-read Bernoulli mask on the weight matrix, i.e. DropConnect on the crossbar).
   *Analogue:* probabilistic synaptic transmission — vesicle release fails on a large fraction
   of spikes (release probability 0.1–0.9, Allen & Stevens 1994; Branco & Staras 2009).
4. **Read-out default = layer-accumulate, 8-bit, FoM_W 10 fJ.**  `ReadoutSpec.default()` is
   attached to every device `TileLinear` (and to the attention cache reads) unless the caller
   passes `readout=None` or its own spec (E8b does); the converter range follows the largest
   |value| seen at each read site (an automatic-gain converter; `auto=True`, no calibration
   pass) and the quantiser is a straight-through estimator so gradients flow (the first
   version of the default killed every gradient: round / clamp have zero derivative, measured
   as "0 pulses, loss unchanged" in `test_optimizer_learns_through_tiles`).  `E_ADC = FoM_W ×
   2^b` is on the spec (`E_ADC_J`).  *Analogue:* dendritic analog integration followed by a
   single spike-initiation threshold at the axon hillock — the partial sums stay analog and are
   converted once per neuron, not once per branch.

Also new: `FloatTile`, the float32 REFERENCE substrate with the Tile interface (float weights,
ideal updates, the same multiplicative read noise and dropout injected on every read), and
`TileLinear(write_mode="float")`, its torch twin, so E11–E14 run the identical algorithm on both
substrates; `memory.ScaledTile / HopfieldAttention(float_ref=True)`.  Tests: 72 pass + the
pre-existing E7 route test.

### E11 regularisation (`experiments/e11_regularization.py`)
nanoGPT 2.7M trained FROM SCRATCH on 10 % of the Shakespeare training split (`--train-frac
0.1`, 100k characters: at 1500 steps × 4096 tokens the model sees the slice 60 times and
overfits, so a regulariser has something to do; on the full text the budget never reaches the
overfitting regime and σ* = 0 would be a statement about the budget, not about noise), read
noise σ_r ∈ {0, 0.02, 0.04, 0.06, 0.1} as the ONLY device noise (`split=(0, 0, 1)`, the
attention cache carries the same read noise and no write noise), dropout p ∈ {0.1, 0.3, 0.5} at
σ = 0, tiles (nearest-pulse writes, default read-out) and the float32 reference, 5 seeds.
Val loss and the Python-source loss are read as deployed (noise on) and clean (noise off);
shift gap = Python − val.  σ* = argmin of the seed-mean clean val loss with a bootstrap CI over
seeds (the bootstrap distribution over the σ grid is reported, so a CI of 0..0 means every
resample put the minimum at 0); the shift-gap slope vs σ with a bootstrap CI and the number of
seeds whose own slope is negative.
*Grid extended 2026-10-20:* the first sweep (σ ≤ 0.1) put σ* = 0.1 on the edge of the grid for
every seed on both substrates (val loss 3.37 → 2.28 on tiles, 3.58 → 3.01 float32, still falling),
i.e. the minimum was not bracketed, so the sweep was extended to the top of the dial (0.15, 0.2)
plus one beyond-dial diagnostic point (0.3, excluded from the σ* and slope tests) [2026-10-23: that point ran its Linear layers at 0.2, clipped by the dial, on both substrates; see "Correction 2026-10-23" at the end of this file]; the 80 runs
on disk were kept and only the new arms were run (`bpu-e11b`).

### E12 exploration (`experiments/e12_exploration.py`)
2-layer policy (8 → 32 → 4 bandit; 16 → 32 → 4 grid), hidden units `clip(u + 0.25, 0, 2)` (a
tonic drive so units do not die under negative modulators), greedy on one noisy read.
**Two renditions per context** (A, B) with independent read noise; per-unit noise sample =
deviation from the noise-free read (the periphery's estimate of the mean read; an averaged
read in hardware); eligibility Σ_t x_pre ξ_post over a rendition, presynaptic-normalised
(÷ |x_pre|²); update η (R_A − R_B)(e_A − e_B), η = 2.  *Why two renditions (measured):* the
single-rendition rule ΔW = η (R − R̄) x ξ with a running-mean baseline ended at chance on the
bandit after 6000 trials at every σ up to 1.0 and drove the weights to saturation; a linear
critic baseline (R − V(x)) also failed (0.0 accuracy on a 2-context version, i.e. the
anti-solution), because a greedy policy's chosen arm has a positively biased noise sample and
any baseline mismatch turns the bias into a drift; the two-rendition difference is exactly zero
when the renditions agree and needs no baseline.  Isolated tests (one context: P(correct) → 1.0;
hidden layer only / output layer only) are in the session log; the surviving design learns the
bandit to 0.6–0.7 within 7k pairs at σ = 0.1 on tiles.  *Physics:* multiplicative read noise
flips a greedy decision only when the margin is within ~σ of the scores (flip rate 1–10 % at
σ = 0.01–0.1, logged as `flip_rate`), and the rate falls as the learned margins grow — a
built-in annealing; the tiles explore MORE than the float32 reference because their d2d / c2c
and pulse quantisation add variability (flip rate 0.08 vs 0.001 at σ = 0.1 on the bandit).
Schedules: fixed {0.01, 0.02, 0.04, 0.1}, annealed 0.1 → 0.01, reward-gated (0.1 while the
20-trial mean reward is not above the 200-trial mean, else 0.01; each switch a MODULATE event).
Criteria: bandit 200-decision accuracy ≥ 0.7 (softmax REINFORCE reaches 0.91 on this task),
grid 50-episode mean steps ≤ 1.5× optimal.  Hypothesis tests: gated vs best fixed arm (paired
sign test over seeds on trials-to-criterion); log-log exponent of speed vs σ over σ ≤ 0.04 with
a bootstrap CI (σ² = the node-perturbation prediction for a fixed η).

### E13 stochastic resonance (`experiments/e13_stochastic_resonance.py`)
64 LIF neurons (τ_m 20 ms, θ = 1, dt 1 ms), one device each on a 1 × 64 tile programmed to 0.8,
drive 0.4 + 0.4 sin(2π 5 t) (crest 0.8 θ), σ_r ∈ {0, 0.005, …, 0.2} + beyond-dial diagnostic
levels {0.3, 0.5, 0.7, 1.0}, 20 s records, 10 seeds, tiles and the float32 reference.  SNR =
power of the population spike train at 5 Hz over the mean power in the surrounding 2 Hz band
(dB); per-neuron SNR also kept.  U-curve rule: interior maximum beating both ends of the dial
by ≥ 3 dB with paired sign tests (p < 0.05); σ* with a bootstrap CI.  *Physics written before
the run:* the membrane low-passes per-ms read noise by √(dt / 2τ_m) ≈ 0.16, so the dial's
maximum gives 0.025 of membrane noise against a 0.2 gap — the headline arm is expected to show
no spikes at all; a diagnostic grid drive ∈ {0.8, 0.9, 0.95} × τ_m ∈ {20, 5, 2} ms (tiles) maps
where the resonance falls into the dial and into the hypothesised band.

### E14 sampling (`experiments/e14_sampling.py`)
HopfieldAttention store of 8 random ±1 patterns, N = 64, β = 0.2; cue = α x_a + (1 − α) x_b for
α ∈ {0.5, 0.55, 0.6} (posteriors ≈ 0.5 / 0.78 / 0.93 on x_a); retrieved pattern = largest
overlap with the output; 400 trials per cue × 4 cue pairs, 5 seeds; "correct posterior" =
softmax_μ(β x_μ · cue), the Boltzmann posterior of the retrieval energy at temperature 1/β,
which is what the noise-free softmax computes; KL(posterior ‖ frequencies) with Laplace
smoothing; σ_r ∈ {0 … 0.2} + beyond-dial {0.3, 0.5, 0.7, 1.0}; tiles vs float32 reference.

### E3-final, E6, E9, E4, E5 reruns
* E3-final: `e3_tracking.py --mismatch 0.2,0.3 --cue-amp 0.05,0.02 --omegas 0.5,0.1 --seeds 10`
  (`results/e3_final`); pinned = σ-0 tile error > 1.5× the float32 reference; U-test only there;
  if nothing is pinned the verdict reads "no pinning up to 30 % mismatch; attractor robustness
  is the result".
* E6: K grid 1..16 (`COUNTS`), 10 seeds, the earlier grid (to 32 in steps of 4) kept in
  `results/e6_k32`.
* E9: nearest-pulse default, 5 seeds, arms frozen / sgd_float32 / sgd / fa / dfa / local; the
  new "loss vs locality" panel orders the BPU gradient arms backprop → FA → DFA → layer-local;
  the round-two level-gated Pareto run is kept in `results/e9_pareto`.
* E4: nearest-pulse, 20 seeds (`results/e4_np`): **13/20 certified** (2/20 controls), 4.06
  writes/device = 0.0004 % of endurance, 208 writes/tile/s = 2.1 % of the driver bandwidth,
  0.10 writes/device/s = 1693 % of the old lifelong rate; released bumps occupy 4.1/8 bins.
  Against the round-two level rule (θ = 1: 3/10 at 0.52 writes/device) the nearest-pulse rule
  certifies four times as often for eight times the writes, all of it inside the endurance and
  the bandwidth.
* E5 self-organisation with nearest-pulse writes, 5 seeds, 16×16 and 32×32 sheets
  (`results/e5_selforg_np`, `--procs 5`).

### Outcomes (2026-10-19 evening)
* **E14 sampling** (5 seeds): σ = 0 is deterministic (one pattern every time, KL 2.29 at α = 0.5).
  At exact ambiguity the dial samples the posterior: KL 2.29 → 0.38 / 0.08 / 0.02 at σ = 0.005 /
  0.01 / 0.02 and ≈ 0.02 from there to 1.0 on tiles (σ* 0.06, CI 0.03..0.2; float32 0.01 and
  σ* 0.1), with no rise even at σ = 1.0 (no spurious third pattern: 8 patterns in 64 dimensions
  are too far apart).  The asymmetric posteriors are NOT matched inside the dial: at α = 0.55
  (posterior 0.78 / 0.22) the retrieval stays deterministic up to σ = 0.2 (KL 0.746 flat) and
  needs σ ≈ 0.5–1.0 (KL 0.18 at 0.5, 0.015 at 1.0); α = 0.6 (0.93 / 0.07) likewise (KL 0.167
  flat, 0.016 at 1.0).  Reading: the score margin of an asymmetric cue is (2α − 1) N = 6.4 at
  α = 0.55 against a read-noise standard deviation of ≈ 4σ on the score, so flips need σ ≳ 1;
  the dial's range only samples decisions that are already within ~20 % of a tie.  Hypothesis
  "KL minimum at σ > 0" holds at α = 0.5 only; "one σ* for every α" fails.  Tiles and float32
  agree to within the seed noise at every point (the substrate is not the limit).
* **E12 exploration** (5 seeds, 10 000 pairs = 20 000 decisions): no hypothesis supported.
  Bandit on tiles: fixed σ = 0.1 reaches the 0.7 criterion in 3/5 seeds (median 7 042 decisions,
  final accuracy 0.61), σ = 0.04 in 3/5 late (0.60), σ ≤ 0.02 never (0.26 / 0.37); annealed
  4/5 (9 408, 0.63); reward-gated 3/5 (13 544, 0.61) — gated vs the best fixed arm 2/5 seeds
  faster, p = 1.  Float32 reference: σ = 0.1 4/5 (10 374), gated 2/5 (0.49).  Grid: at most 1/5
  seeds per arm reach 1.5× optimal on either substrate (final 12–15 steps against 4.5), i.e.
  the grid is not learned in the budget.  Speed exponent over σ ≤ 0.04: 0.07 (CI 0.00..0.17)
  bandit / tiles, 0.03 float32, 0.16 grid — nowhere near σ².  Flip rates (the actual exploration
  rate): 0.010 / 0.014 / 0.035 / 0.048 for σ 0.01 / 0.02 / 0.04 / 0.1 on tiles vs 0.006 / 0.012 /
  0.020 / 0.036 on float32: the tiles explore more than the reference (d2d / c2c and pulse
  quantisation add variability) and learn the bandit sooner.  Reading: with multiplicative
  read noise ≤ 0.2 the exploration rate is a few per cent of decisions and grows only
  linearly in σ, so the σ² law (which assumes the perturbation is the gradient probe) does not
  apply — the probe here is the rare decision flip — and a reward-gated dial that can only move
  between 0.01 and 0.1 has too little range to beat the fixed 0.1 arm.  The dial would need a
  top of order 1 (E14 finds the same number) to act as a real exploration knob.
* **E13 stochastic resonance** (10 seeds): the physics written before the run held.  Headline
  (drive 0.8× threshold, τ_m 20 ms): no spike at any dial setting (0–0.2) on tiles or float32;
  spikes appear at σ = 0.7 (0.04 Hz, SNR 17 dB) and 1.0 (0.46 Hz, 27.5 dB), and the SNR is still
  rising at 1.0 — the whole resonance sits beyond the dial.  The diagnostic grid is monotone in
  every cell within the dial (0/10 U-curves, 0 peaks in [0.02, 0.06]): the onset of firing moves
  into the dial as the membrane gets faster or the drive closer to threshold (τ_m 5 ms: 7 dB at
  σ = 0.2 for drive 0.8, 34 dB for 0.9; τ_m 2 ms: 32 / 44 / 48 dB at 0.2 for drives 0.8 / 0.9 /
  0.95), and the only hint of a descending branch is the fastest, strongest cell (0.95 × θ,
  2 ms: 50.2 dB at 0.5, 50.1 at 0.7, 50.2 at 1.0, saturating rather than falling; 0.9 × θ, 2 ms
  peaks at 0.7).  Tiles and the float32 reference coincide (0.8 / 20 ms: 17.0 vs 16.5 dB at 0.7,
  27.5 vs 27.6 at 1.0).  Reading: a per-read multiplicative noise of ≤ 20 % on a current that
  stays ≥ 20 % below threshold, low-passed by a 20 ms membrane, is an order of magnitude too
  small to cross; stochastic resonance of a LIF through a device needs membrane noise of order
  the gap, i.e. σ_r ≈ 1 at this read rate, or a drive within a few per cent of threshold.  The
  "noise band [0.02, 0.06]" is not anchored by this experiment; what the dial's range does do is
  set the firing onset (the SNR ascending branch) when the neuron is already near threshold.
* **E6 rerun, K grid 1..16** (10 seeds, 20 % flipped cues): scaffold readers hold 3 / 3 / 4 / 3 /
  4 / 5 pairs (pinv loop) at σ 0 / 0.02 / 0.04 / 0.08 / 0.12 / 0.2, delta rule the same, direct
  read 3–4; attention recalls every K up to the top of the grid (16) at every σ, so its capacity
  is ≥ 16 here (the earlier grid, kept in `results/e6_k32`, placed it at 20–28).  The finer grid
  changes nothing: K = 1 is exact everywhere, the scaffold ceiling is 3–5 and does not fall with
  device noise up to 0.2.
* **E3-final** (mismatch 0.2 / 0.3, cue 0.02 / 0.05, ω 0.5 / 0.1, 10 seeds): the pinned regime
  exists at 20–30 % mismatch — 5/8 mismatched conditions have a σ-0 tracking error above 1.5×
  the float32 reference (ω 0.1: 0.207 / 0.211 rad vs 0.068 at cue 0.05, 0.518 / 0.767 vs 0.153
  at cue 0.02 for mismatch 0.2 / 0.3; ω 0.5, cue 0.05, mismatch 0.3: 0.537 vs 0.346); pinning
  barrier ΔU 0.88 (mismatch 0.2) and 1.35 (0.3) against 0.001 mismatch-free.  In every pinned
  condition the error is FLAT in dynamic noise to three decimals across 0–0.08 (e.g. 0.207,
  0.207, …, 0.207), the interior minima are at the seed-noise level (4–6/10 seeds, p ≥ 0.75):
  **0/5 U-shaped**.  So the earlier "no pinned regime" was a matter of mismatch level, and the
  answer once pinning is there is that dynamic read / neuron noise up to 0.08 does not unpin a
  bump held by a barrier of order 1: the noise optimum of the brief is a negative at every
  mismatch tried, and the attractor's robustness (tiles within 10–30 % of float32 at 20 %
  mismatch for the moderate cue) stays the positive result.
* **E11 regularisation** (5 seeds, 110 runs, σ grid extended to the dial top + 0.3): the
  hypothesis holds on tiles and the minimum is bracketed.  Tiles, clean-read val loss: 3.371 (σ 0)
  → 3.005 / 2.635 / 2.420 / **2.277** (σ 0.1) / 2.294 / 2.364 (0.2) / 2.372 (0.3): **σ* = 0.1, CI
  0.1..0.1, per-seed argmin {0.1 ×3, 0.15 ×2}**, a 32 % drop from σ = 0 and a shallow basin —
  everything in 0.06–0.15 is within 6 % of the minimum, 0.2–0.3 within 4 %.  The float32 reference
  with the same read noise on float weights falls more slowly (3.580 → 3.008 at 0.1 → 2.471 at 0.2
  → 2.414 at 0.3): **σ* = 0.2 on the dial top** (CI 0.2..0.2), still improving beyond the dial, i.e.
  the float32 minimum sits above 0.2 [2026-10-23: the 0.3 values, here and for tiles, ran the Linear layers at 0.2 (dial clip); "above 0.2" rests on the attention-cache noise alone until the unclipped rerun reports].  The twin gap is therefore negative at every σ in the dial
  (tiles −6 % at σ 0, −26 % at 0.06, −4 % at 0.2, −2 % at 0.3): the tiles carry variability of
  their own even with the dial off (nearest-pulse quantisation, c2c on every read, 8-bit read-out),
  which shifts the whole curve to the left by roughly 0.1 of σ [HPCC 2026-10-23: the float32 bracket puts its
  optimum at σ* = 1.0, so the shift is about 0.9 of σ, a factor of ~10; see "HPCC runs"] [2026-10-23: WITHDRAWN, the bracket arms above 0.2 were clipped to 0.2 on the Linear layers; see "Correction 2026-10-23"] — the substrate is already part
  of the way to the optimum, and the dial finishes the job.  Train loss at σ = 0 is 0.58 (tiles) /
  0.44 (float32) against val 3.4 / 3.6, so σ = 0 is the overfitting regime the protocol asked for;
  at σ* train is 1.46 / 1.09.  Deployed (noise-on) val tracks clean val to within 2 % up to σ = 0.1
  and 4.5 % at 0.2: a model trained at σ runs at σ.  **Shift gap** (Python − Shakespeare val, clean):
  tiles 4.520 → 4.477 (0.1) → 4.150 (0.15) → **3.850 (0.2)**, slope −3.48 nats per unit σ, CI
  −3.84..−3.19, **5/5 seeds narrowing** — supported, but the narrowing is all above σ*, where the
  Python loss keeps falling (7.89 → 6.21) while val has turned up, so it is bought with in-
  distribution loss; float32 gap flat (4.37 → 4.36, slope +0.12, CI −0.77..+1.03) — on the
  reference the Python loss falls at the same rate as val.  **Dropout** (σ = 0): p* = 0.1 (CI
  0.1..0.1) with val 2.43 on both substrates (≈ σ 0.06), p = 0.3 3.10, p = 0.5 4.96 (clean read;
  deployed 2.32 / 2.46 / 2.58 — a net trained with dropout reads better WITH its dropout on, the
  clean read is a different, over-scaled network); substrate-independent to 1–4 %.  Reading: a
  per-read multiplicative noise of 10 % on every synapse is the best regulariser this model saw
  at this data budget, and it lives inside the dial; synapse failure at 10 % matches noise at 6 %;
  the two differ in deployment (noise-trained models read clean, dropout-trained models must
  keep the dropout on).  The round-three "noise band" for training is 0.06–0.15 on tiles.
* **E5 self-organisation, nearest-pulse writes** (5 seeds, 16×16 and 32×32, 12 000 development
  steps, 800 sample points / 350 for H2, 8 nulls; the unit ran 11.5 h on the CPU pool): **0/5
  self-organised sheets certify as a torus at either size**, 0/5 untrained controls, and the
  programmed Burak–Fiete reference certifies **2/5 at 16×16 and 2/5 at 32×32** under the same
  sampling (the failing reference seeds show 7–72 H1 bars and 8–91 H2 bars, i.e. the sampled
  manifold is noisy, not a different shape).  The trained sheets all read β0 ≈ 85–99 with no H1:
  the activity is fragmented into many isolated states rather than a sliding bump, the same
  failure mode as the weekend and round-two self-organisation runs (`results/e5`,
  `e5_selforg32`).  Reading: the write rule is not what stops the sheet from self-organising —
  nearest-pulse writes change nothing against the level rule here — and at this sample size
  the certifier's own pass rate on a programmed torus is 40 %, so "0/5 vs 2/5" is the honest
  comparison; the programmed module remains the only grid torus the simulator has produced
  (3/3 at 32×32 and 1/1 at 48×48 with the denser weekend protocol).  The self-organisation
  claim stays a negative; it is not blocked by the substrate.
* **E9 rerun, nearest-pulse default** (5 seeds, arms frozen / sgd_float32 / sgd / fa / dfa /
  local): post-shift loss (Python segment, last 200 steps) float32 SGD 1.224; sgd on tiles
  1.249 (+2.1 %, 18.1 writes/device = 0.0018 % of endurance); fa 2.404 (+96 %, 48 writes/device);
  dfa 2.885 (+136 %, 64.7); layer-local 2.683 (+119 %, 28).  Criterion met by sgd (within 5 %
  with < 1e4 writes/device), and the loss-vs-locality panel is monotone except that the local
  rule beats DFA: each step from backprop towards a local rule costs roughly a doubling of the
  post-shift loss at this budget, while the write cost stays four orders of magnitude inside
  the endurance for every arm.  With the gate gone the round-two Pareto question is closed: the
  ungated nearest-pulse writes are the default and the level-gated run is kept as history.
* **E4, nearest-pulse, 20 seeds**: numbers in the rerun list above (13/20 certified, 4.06
  writes/device); the round-two level rule is now the comparison arm, not the default.

## HPCC runs (2026-10-22/23, Arkansas HPCC "Pinnacle")

The MONDAY.md queue ran on Pinnacle (`scripts/hpcc/`, submit with `bash scripts/hpcc/submit.sh queue`): CentOS 7
compute nodes, so the env is conda-forge at glibc 2.17 with torch 2.5.1+cu124, transformers < 5 and datasets < 4
(`scripts/hpcc/build_env.sh`). GPU jobs ran on V100 32 GB. Every run is in its own results directory; the laptop runs
are untouched. Jobs ran 3–10x faster than the laptop-based estimates. Every log was read (no tracebacks).

Decisions taken while running:
* The E11 full-data run was split by substrate (`results/e11_full_tiles`, `results/e11_full_f32`) so the two halves
  could run on two GPUs. They are kept as two directories, each with its own summary, not merged.
* Both float32 grids put σ* on their top edge (0.7 on the small-data bracket, 0.2 on full data), so each was extended
  upward in the same results directory (finished arms are skipped): small data to 3.0 (`--sigmas
  0,0.1,0.2,0.3,0.4,0.5,0.7,1.0,1.4,2.0,3.0`), full data to 1.4. Extending a grid until the optimum is interior is not a
  threshold change: σ* is defined as the argmin over the grid, and an edge argmin is only a bound.
* E8 has no `--device` and runs on CPU only; MONDAY.md had listed it as a GPU job (corrected).
* [2026-10-23] Both extensions above 0.2 were clipped to 0.2 on the Linear layers by the dial; see "Correction 2026-10-23" below.

### Outcomes
* **E11, full data, tiles (`results/e11_full_tiles`, 5000 steps, 100 % of the text, 10 seeds): SUPPORTED.** σ* = 0.04
  on 10/10 seeds (bootstrap CI 0.04..0.04, excludes 0); val loss (clean read) 1.588 at σ = 0 → 1.503 at σ = 0.04.
  The shift gap narrows on 10/10 seeds, slope −12.52 nats per unit σ (CI −12.76..−12.31). The regulariser survives
  outside the small-data regime, with a smaller optimum than on 10 % of the text (0.04 vs 0.1), as expected when the
  model overfits less. Dropout p* = 0.1 (CI 0..0.1).
  [2026-10-23: unaffected by the dial clip: the σ* grid is 0–0.2 and contains no clipped point; only the excluded
  beyond-dial 0.3 diagnostic was clipped.]
* **E11, float32 brackets.** Small data: σ* = 1.0 (CI 1..1; per-seed [1.0, 0.7, 1.0, 1.0, 1.0]); full data: σ* = 0.4
  (CI 0.3..0.5). In both regimes the float32 optimum sits about 10x above the tile optimum (1.0 vs 0.1; 0.4 vs 0.04).
  The round-three wording "tiles reach the optimum at half the injected noise" was a lower bound and is replaced by
  "about an order of magnitude less injected noise". Mechanism (not tested here): the tile path already carries
  quantisation and write noise, so less injected read noise is needed to reach the same effective regularisation.
  **[2026-10-23: WITHDRAWN. Every arm of both brackets above 0.2 ran its Linear layers at 0.2 (dial clip); the optima
  1.0 and 0.4 and the ~10x are optima over attention-cache noise only. See "Correction 2026-10-23".]**
* **E3-final, 50 seeds (`results/e3_final50`): null, now a bound.** 0/8 mismatched conditions show a significant U in
  tracking error vs dynamic noise; 4/8 conditions are pinned (σ = 0 error > 1.5x the float32 reference) and 0 of those
  are U-shaped.
* **E5, 20 seeds at 32×32 (`results/e5_ref20`, `results/e5_selforg_np20`): null; the sheet is the limit.** The
  programmed DoG reference certifies 15/20 and 11/20 in two independent draws (26/40 = 65 %); self-organised sheets
  0/20; untrained control 0/20. MONDAY's rule ("if the reference stays near 40 % the certifier is the limit") is not met.
* **E12, 50 000 trials, 10 seeds (`results/e12_long`): partial.** Bandit on tiles: annealed reaches criterion fastest
  (median 6 427 trials, 10/10 seeds) against fixed 0.1 (16 217, 7/10) and gated (12 440, 9/10); gated vs fixed 0.1:
  5/10 seeds faster, p = 1, so the gated schedule is not supported. Speed ∝ σ^1.08 (CI 0.79..1.41) on tiles, σ^0.71
  (CI 0.44..1.02) on float32, where annealed (13 360) is no faster than fixed 0.1 (13 341). The grid task is still not
  learned by any arm at 5x the budget (≤ 10 % of seeds reach criterion).
* **E1, GPT-2 124M on WikiText-2, 1 seed (`results/e1_full`): null.** Fine-tuned perplexity is best with no read noise
  (23.38 at σ = 0; 24.42 at 0.04; 29.63 at 0.12); pretrained-at-noise collapses beyond σ = 0.06 (2 191 at 0.06).
  The expected accelerator-parity null: noise only costs on a pretrained model.
* **E9, 2000-step segments, 5 seeds (`results/e9_seg2000`, GPU, 2 h 38 min): criterion NOT met.** Post-shift loss
  (Python segment, last 200 steps): float32 SGD 0.940; sgd_bpu 1.034 (+10.0 %, 50.7 writes/device); fa_bpu +128.9 %;
  dfa_bpu +201.7 %; local_bpu +181.9 %. No nearest-pulse arm is within 5 % of float32. With the laptop's 1000-step
  segments sgd_bpu met the criterion (+2.1 %), so the round-three E9 "met" holds only at that segment length; the gap
  grows when each segment is trained longer. Reported as a negative beside the laptop result, not replacing it.
* E8 at full size (CPU, `results/e8_full`) is running.

## Correction 2026-10-23

**Correction 2026-10-23 (round four): the MODULATE dial clipped the float32 reference.** `TileLinear.set_noise` clips to the dial range [0.005, 0.2], and E11 set the float32 reference through it as well, so every E11 arm with σ > 0.2 ran its Linear layers (qkv, proj, fc, fc2, head) at σ_r = 0.2 and only the attention-cache read noise followed σ. Affected arms: every float32 arm above 0.2, i.e. `results/e11` (0.3, the beyond-dial point), `results/e11_f32_bracket` (0.3, 0.4, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0) and `results/e11_full_f32` (0.3, 0.4, 0.5, 0.7, 1.0, 1.4); and the tiles' beyond-dial 0.3 point (`results/e11`, `results/e11_full_tiles`), a diagnostic that was already excluded from σ* and the slope. **WITHDRAWN pending the unclipped reruns** (jobs 1366759 / 1366760, `results/e11_f32_unclipped`, `results/e11_full_f32_unclipped`): the float32 optima σ* = 1.0 (small data) and 0.4 (full data), the claim that the float32 optimum sits about 10x above the tile optimum ("an order of magnitude less injected noise"), the float32 shift-gap slopes over the extended grids (−0.16 to σ 3.0, −0.66 to 1.4) and every float32 or tiles value quoted at σ ≥ 0.3. **Unaffected:** every arm at σ ≤ 0.2 on both substrates. The tiles σ* = 0.1 (small data) and σ* = 0.04 (full data) stand: both were taken over the dial grid 0–0.2, which contains no clipped point; the 0.3 point was not in the grid, and the full-data bootstrap put all its mass on 0.04 (CI 0.04..0.04). The tiles shift-gap slopes, the twin gaps inside the dial and the dropout arms stand, and so does the round-three statement that the float32 minimum is not reached inside the dial (σ* = 0.2 on the dial top, small and full data). Code fixed the same day: the float32 reference is never clipped (`TileLinear.set_noise`, `FloatTile.set_noise`), device values beyond the dial are set directly as a diagnostic (`e11_regularization.set_all_noise`), `--legacy-dial-clip` reproduces the old behaviour for E11-cal, and `tests/test_torchtile.py::test_float_reference_noise_is_not_clipped_by_the_dial` checks both. No other experiment was affected: E12 stays inside the dial, E13 and E14 set beyond-dial values directly.
