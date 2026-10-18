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
