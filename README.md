# bpusim

Simulator for the **BrAIn Processing Unit (BPU)**: a non-von Neumann
architecture on stochastic MoS2 memristor synapses.  The package implements
the equations of *The BPU from First Principles* layer by layer and the
experiment scripts reproduce E1-E7 of the paper.  Everything from E3 to E7
runs on a laptop CPU; E1 and E2 use PyTorch and accept `--device cuda`.

```
bpusim/                     library (NumPy; see module docstrings)
  device.py      L0  memristor: w in [0,1], saturating pulses, d2d/c2c/read noise, retention, dead mask, write-verify
  tile.py        L2  differential-pair crossbar with mask A; read(x), update(dW) -> pulses, program()
  neurons.py         LIF and rate neurons with additive noise
  plasticity.py  L1  STDP (window + online traces), eligibility trace, three-factor + surprise gate, homeostasis, Oja
  growth.py      L3  C_ij = <m x_i xi_j>, grow/prune within a budget, spawn
  attractor.py   L4  RingAttractor, GridModule, SelfOrganizingSheet (+ exploration cues)
  memory.py      L5  ScaffoldMemory (pseudo-inverse / delta rule, recall loop), HopfieldAttention
  modules.py     L8  Module, ModuleGraph (routing R, node-perturbation learning, route growth), layered_routing
  topology.py        ripser/gudhi persistent homology, circular-shift null, ring/torus certification
  metrics.py         SV, TC, FPL (WhiteBox), Newman modularity Q, mutual information
  noise.py           noise configurations and sweep helpers
experiments/   e1_gpt2_noise.py ... e7_modularity.py, common.py
tests/         pytest (54 tests)
scripts/       run_all.sh (E3-E7 end to end), make_figures.py (one PNG per experiment)
results/<e>/results.json, figures/<e>.png
DECISIONS.md   every modelling choice and the measurement that forced it
```

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt      # torch/transformers/datasets only needed for E1/E2
pip install -e .                      # or run with PYTHONPATH=.
pytest -q
```

## Run

```bash
scripts/run_all.sh            # E3-E7, 3 seeds, regenerates figures
scripts/run_all.sh --full     # 10 seeds
python experiments/e2_grow_vs_rigl.py --seeds 3            # MNIST (downloads 11 MB), --device cuda optional
python experiments/e1_gpt2_noise.py --smoke                # CPU pipeline check, no downloads
python experiments/e1_gpt2_noise.py --device cuda          # GPU host: GPT-2 124M + WikiText-2
python scripts/make_figures.py                             # figures from existing results.json
```

Every experiment writes `results/<e>/results.json` with a `verdict` block that
states in words what was found, including when the hypothesis was **not**
confirmed.

## Results (3 seeds unless stated; see `results/*/results.json`)

| Exp | What was measured | Outcome (3 seeds unless stated) | Verdict |
|---|---|---|---|
| E1 | GPT-2 124M perplexity through noisy tiles, with noise-aware fine-tuning | **not run here** (GPU host); `--smoke` pipeline check passes on CPU | pending |
| E2 | MNIST MLP at 10 % density: local grow rule vs RigL vs random regrowth, 5 epochs | test acc local 0.914 +- 0.003, RigL 0.923 +- 0.001, random 0.914 +- 0.002 | **success criterion met** (within a few percent of RigL, not worse than random); no advantage over random regrowth |
| E3 | 32-ring on a 32x32 tile, mismatch 0.05, path-integration RMSE vs dynamic noise, **10 seeds** | RMSE 0.38 (sigma 0) -> 0.33 (sigma 0.05, 7/10 seeds better) -> 0.41 (sigma 0.2); no-mismatch control 0.05 -> 0.34 monotonic | **not U-shaped** at p = 0.34; shallow dip inside the error bars |
| E4 | ring self-organised from a sparse random tile (STDP + growth + normalisation), beta1 = 1 vs circular-shift null | 2/3 seeds certified (H1 bars 0.26, 0.20 vs null 0.18, 0.16; third seed 0.10 vs 0.12), no extra β0 components in any seed; 0/3 untrained controls; released bumps cover 6/8 ring bins; 0.11 writes per device per second during development vs the 6e-3 endurance budget | **partial**: ring forms in most seeds, kernel still translation-variant; development is 18x over the write budget |
| E5 | grid self-organisation on 5x6 and 16x16 sheets, torus (1,2,1) vs null | 0/3 + 0/3 self-organised; **programmed DoG reference also 0/3 + 0/3** (5x6: 1-6 H1 bars, 0-1 H2; 16x16: 0-3 H1, 0-1 H2 and 12-39 extra β0 components, i.e. the lattice hops between discrete states instead of sliding); 0/3 controls | **not certified**; the certifier does not pass even the programmed module at these sizes |
| E6 | 32 +-1 pairs on 32x32 tiles, recall from 50 %-erased cues, sigma in {0 .. 0.2} | pseudo-inverse scaffold: >= 90 % exact recall only up to K = 4-8; delta-rule scaffold the same; **attention: 32/32 at every sigma** (0.98-1.00) | device noise is not the limit, the erasure is; attention wins by a wide margin |
| E7 | 8 generic modules, NAV + RECALL, grown module graph vs fixed layered graph at equal synapses | Q(grown) = 0.12 +- 0.01, rises from 0 as 12 routes grow; all modules carry > 0.1 bit about the task; nav accuracy grown 0.15 vs fixed 0.39, recall bits 0.70 vs 0.76; compositional transfer gap +0.01 (p = 1) | **null**: the graph grows and becomes weakly modular but learns the tasks worse than the fixed graph and composes no better |


## Honest summary

* **The device, tile, plasticity, growth, attractor, memory, topology and
  module layers all work and are unit-tested** (54 tests: device bounds and
  noise statistics, read error ~ sigma/sqrt(N), bump iff J1 > 2, exact
  pseudo-inverse recall, ripser certification of a synthetic circle and
  torus, node-perturbation sign recovery, route growth toward the output).
* **E3 does not show the predicted U.** With 10 seeds the mismatched ring's
  error dips from 0.38 to 0.33 rad at sigma = 0.05 and rises again, but 7/10
  paired seeds is p = 0.34.  Mismatch itself is very visible (0.38 vs 0.05 rad
  without it), so pinning is real; the noise-assisted escape is at best a
  small effect at this N and T.  `figures/e3_drift.png` has the curve.
* **E4 is a partial success.** Growing a ring from a sparse random tile
  required seven measured fixes (DECISIONS.md); the result certifies in 2 of
  3 seeds against 0 of 3 controls, with the learned kernel a clean cos-like
  profile, but the free bump still prefers parts of the ring.  Barcodes are
  in the figure and `results/e4/results.json`.
* **E5 is a negative result with a caveat.** No self-organised sheet
  certifies as a torus, and neither does the *programmed* grid module at 5x6
  or 16x16 under the same sampling and null, so the experiment as specified
  cannot distinguish a failed self-organisation from a certifier that needs
  larger sheets / denser sampling.  The 5x6 sheet is too small for a 2D
  attractor at all (the bump hops between columns).
* **E6 is clear.** Half-erased cues defeat the pseudo-inverse scaffold beyond
  ~8 pairs on a 32x32 tile regardless of device noise, while Hopfield
  attention through the same noisy tiles recalls all 32 patterns at every
  noise level up to sigma = 0.2.
* **E7 is null.** Routes grow where the node-perturbation statistic says
  they should (toward the output modules) and Q rises to ~0.12, but the
  local learner is too weak: the grown graph solves the tasks worse than a
  fixed layered graph with the same synapse budget, and neither composes.
* **E2** meets the technical document's success criterion (within a few
  percent of RigL, not worse than random) but shows no advantage over
  random regrowth.  **E1** is ready for a GPU host.
* **Write budget.** Self-organisation in E4 costs 0.11 writes per device per
  second, 18x the endurance budget of 6e-3 from the technical document; a
  developmental phase of this intensity has to be short or sparser.

The certifier follows the technical document: threshold at the 99th
percentile of the longest null feature, and a ring/torus certificate also
requires no extra persistent β0 component (fragmentation by pinning).  The
alignment of the code with the *Technical Document* and the *Patent-Style
Disclosure* is itemised at the end of `DECISIONS.md`.

Timings on this laptop (20-core CPU): E3 23 s, E4 48 s, E5 ~10 min
(16x16 tiles), E6 109 s, E7 100 s; `scripts/run_all.sh` ~15 min with 3
seeds when nothing else is running.  E2 ~1 min for 3 seeds x 5 epochs.

## Reference equations

The code implements exactly the equations listed in the task brief (device
step law and noise model, differential-pair tile, LIF/rate neurons, STDP /
eligibility / three-factor / homeostasis / Oja, covariance-driven growth with
budget and pruning, ring kernel `J0 + J1 cos d` with velocity shift
`-J1 sin d`, difference-of-Gaussians grid sheet, pseudo-inverse / delta-rule
heteroassociation with attractor cleanup, Hopfield attention
`X softmax(beta X^T xi)` through the noisy tile, persistent-homology
certification against a circular-shift null, and the module graph with
routing matrix, modularity and task information).  Where an equation left a
free choice, `DECISIONS.md` records what was chosen and why.
