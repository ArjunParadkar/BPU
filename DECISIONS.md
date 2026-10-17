# DECISIONS

Choices made while building `bpusim`, with the reason for each.  Where a
choice was forced by something measured during the build, the measurement is
given.  Numbers quoted here come from the diagnostic runs in this session;
the experiment results proper are in `results/<e>/results.json`.

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
