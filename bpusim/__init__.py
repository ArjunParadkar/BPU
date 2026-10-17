"""bpusim: simulator for the BrAIn Processing Unit (BPU).

Layers (see README.md):
  L0 device    -> bpusim.device      stochastic MoS2 memristor state w in [0,1]
  L2 tile      -> bpusim.tile        differential-pair crossbar with mask A
  neurons      -> bpusim.neurons     LIF and rate neurons
  L1 plasticity-> bpusim.plasticity  STDP, eligibility, three-factor, homeostasis, Oja
  L3 growth    -> bpusim.growth      noise-powered grow/prune/spawn
  L4 attractor -> bpusim.attractor   ring attractor and grid module
  L5 memory    -> bpusim.memory      scaffold heteroassociation, Hopfield attention
  L8 modules   -> bpusim.modules     module graph with routing matrix R
  verification -> bpusim.topology    persistent homology certification
                  bpusim.metrics     SV, TC, FPL, modularity Q, mutual information
  noise        -> bpusim.noise       noise configuration and sweep helpers
"""
__version__ = "0.1.0"
