#!/bin/bash
# Build the bpusim env on Pinnacle: conda-forge pinned to glibc 2.17 (CentOS 7 compute nodes), torch 2.5.1+cu124.
# Own Miniforge, because the site miniforge module points at a missing installer.
# Run it as a job, not on the login node (its process killer stops the conda solver):
#   sbatch --export=NONE -p cloud72 -q cloud -c4 -t 2:00:00 -o ~/bpusim_env_build.log scripts/hpcc/build_env.sh
set -e
MF=$HOME/miniforge3
if [ ! -x $MF/bin/conda ]; then
  curl -fsSL -o /tmp/mf_$USER.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/mf_$USER.sh -b -p $MF; rm -f /tmp/mf_$USER.sh
fi
export PATH=$MF/bin:$PATH
export CONDA_OVERRIDE_GLIBC=2.17 CONDA_PKGS_DIRS=$HOME/.conda/pkgs
ENV=$HOME/envs/bpusim
[ -d $ENV ] || conda create -y -p $ENV -c conda-forge --override-channels python=3.12 numpy scipy matplotlib scikit-learn networkx pytest ripser persim gudhi
$ENV/bin/pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# pyarrow via conda-forge: pip has no recent pyarrow wheel for glibc 2.17 and the source build fails
conda install -y -p $ENV -c conda-forge --override-channels pyarrow
# transformers 5 needs torch >= 2.6, whose wheels need glibc 2.28; 4.x runs on torch 2.5.1
$ENV/bin/pip install --no-cache-dir "transformers<5" "datasets<4"   # datasets 4+/5 need huggingface-hub >= 1, transformers 4 needs < 1
$ENV/bin/pip install --no-cache-dir -e $HOME/bpusim --no-deps
$ENV/bin/python -c "import torch,numpy,scipy,ripser,transformers;print('OK', torch.__version__, numpy.__version__, transformers.__version__)"
