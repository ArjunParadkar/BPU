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

RESULTS_TABLE

## Honest summary

HONEST_SUMMARY

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
