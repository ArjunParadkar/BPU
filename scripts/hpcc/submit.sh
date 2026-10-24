#!/usr/bin/env bash
# MONDAY.md queue on Pinnacle (AHPCC, UArk).  Run from the repo root on the login node:
#   bash scripts/hpcc/submit.sh smoke     tests + --smoke runs (one CPU job, one V100 job), ~15 min
#   bash scripts/hpcc/submit.sh queue     the six MONDAY jobs
# Cluster facts this encodes (probed 2026-10-22): compute nodes are CentOS 7 (glibc 2.17) and need
# `sbatch --export=NONE`; QOS comp needs >= 32 cores (whole node), QOS gpu >= 24 cores; gpu72 = 1x V100 32 GB,
# agpu72 = 1x A100; the site sets OMP_NUM_THREADS=1, so pools and torch threads are set explicitly.
# The env is ~/envs/bpusim (scripts/hpcc/build_env.sh).  Logs: logs/hpcc/<name>-<jobid>.out.
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD; mkdir -p logs/hpcc
ENVPY=$HOME/envs/bpusim/bin
PRE="export PATH=$ENVPY:\$PATH PYTHONWARNINGS=ignore MPLBACKEND=Agg HF_HOME=\$HOME/.cache/huggingface OPENBLAS_NUM_THREADS=1; cd $ROOT/experiments"

# sub <name> cpu|gpu <time> [sbatch opts...] -- <command run from experiments/>
sub() {
  local name=$1 kind=$2 t=$3; shift 3; local opts=()
  while [ "$1" != "--" ]; do opts+=("$1"); shift; done; shift
  case $kind in
    cpu) opts+=(-p comp72 -q comp -c 32) ;;
    cpu1h) opts+=(-p comp01 -q comp -c 32) ;;
    gpu) opts+=(-p gpu72 -q gpu -c 24 --gres=gpu:1) ;;
    gpu6) opts+=(-p gpu06 -q gpu -c 24 --gres=gpu:1) ;;
    agpu6) opts+=(-p agpu06 -q gpu -c 24 --gres=gpu:1) ;;
    cpu6) opts+=(-p comp06 -q comp -c 32) ;;
  esac
  sbatch --parsable --export=NONE -N1 -n1 -t "$t" -J "bpu-$name" -o "logs/hpcc/$name-%j.out" "${opts[@]}" \
    --wrap "$PRE; echo \"== \$(hostname) \$(date)\"; $*; echo \"== done rc=\$? \$(date)\""
}

case "${1:-}" in
smoke)
  sub smoke-cpu cpu1h 0:50:00 -- "cd ..; OMP_NUM_THREADS=1 python -m pytest tests -q -x --deselect tests/test_modules.py::test_graph_grows_routes_toward_output; cd experiments; OMP_NUM_THREADS=1 python e12_exploration.py --smoke --procs 4 --out results/hpcc_smoke/e12; OMP_NUM_THREADS=1 python e3_tracking.py --seeds 2 --T 5 --procs 4 --mismatch 0.2 --cue-amp 0.05 --omegas 0.5 --out results/hpcc_smoke/e3; OMP_NUM_THREADS=1 python e5_grid_selforg.py --sheets 5x6 --seeds 1 --steps 2000 --points 200 --null 2 --out results/hpcc_smoke/e5"
  sub smoke-gpu gpu6 0:50:00 -- "nvidia-smi --query-gpu=name,memory.total --format=csv; python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'; python e11_regularization.py --smoke --device cuda --out results/hpcc_smoke/e11; python e1_gpt2_noise.py --smoke --out results/hpcc_smoke/e1"
  ;;
queue)
  # 1. E11 float32 bracket
  sub e11-f32-bracket gpu 24:00:00 -- "python e11_regularization.py --device cuda --threads 8 --substrates float32 --sigmas 0,0.1,0.2,0.3,0.4,0.5,0.7 --sigmas-beyond '' --seeds 5 --out results/e11_f32_bracket"
  # 2. E11 10 seeds full data, split by substrate (merge the two runs lists, then --resummarize)
  sub e11-full-tiles gpu 72:00:00 -- "python e11_regularization.py --device cuda --threads 8 --substrates tiles --seeds 10 --train-frac 1.0 --steps 5000 --out results/e11_full_tiles"
  sub e11-full-f32 gpu 72:00:00 -- "python e11_regularization.py --device cuda --threads 8 --substrates float32 --seeds 10 --train-frac 1.0 --steps 5000 --out results/e11_full_f32"
  # 3. E12 long budget (whole 32-core node, so 32 workers instead of the 16 in MONDAY)
  sub e12-long cpu 24:00:00 -- "OMP_NUM_THREADS=1 python e12_exploration.py --seeds 10 --trials 50000 --eta 2 --procs 32 --out results/e12_long"
  # 4. E3-final 50 seeds
  sub e3-final50 cpu 24:00:00 -- "OMP_NUM_THREADS=1 python e3_tracking.py --mismatch 0.2,0.3 --cue-amp 0.05,0.02 --omegas 0.5,0.1 --seeds 50 --procs 32 --out results/e3_final50"
  # 5. E5: programmed-reference check first, then self-organisation (starts after the reference job succeeds)
  ref=$(sub e5-ref20 cpu 12:00:00 -- "OMP_NUM_THREADS=1 python e5_grid_selforg.py --reference-only --sheets 32x32 --seeds 20 --points 800 --points-h2 350 --null 8 --procs 20 --out results/e5_ref20")
  echo "$ref"
  sub e5-selforg20 cpu 48:00:00 --dependency=afterok:$ref -- "OMP_NUM_THREADS=1 python e5_grid_selforg.py --sheets 32x32 --seeds 20 --points 800 --points-h2 350 --null 8 --steps 12000 --procs 20 --out results/e5_selforg_np20"
  # 6. E1 full (GPT-2 124M + WikiText-2 download on first run)
  sub e1-full gpu 24:00:00 -- "python e1_gpt2_noise.py --device cuda --steps 400 --batch 8 --block 512 --out results/e1_full"
  ;;
e11-bracket2)
  # 2026-10-23: the first bracket (to 0.7) put float32 sigma* on its top edge in 5/5 seeds; extend it (finished arms are skipped)
  sub e11-f32-bracket2 gpu 24:00:00 -- "python e11_regularization.py --device cuda --threads 8 --substrates float32 --sigmas 0,0.1,0.2,0.3,0.4,0.5,0.7,1.0,1.4,2.0,3.0 --sigmas-beyond '' --seeds 5 --out results/e11_f32_bracket"
  ;;
e11-full-bracket)
  # 2026-10-23: full-data float32 sigma* = 0.2 sat on the grid top in 10/10 seeds; extend the same runs upward
  sub e11-full-f32-bracket gpu 48:00:00 -- "python e11_regularization.py --device cuda --threads 8 --substrates float32 --sigmas 0,0.02,0.04,0.06,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1.0,1.4 --sigmas-beyond '' --seeds 10 --train-frac 1.0 --steps 5000 --out results/e11_full_f32"
  ;;
extras)
  # MONDAY.md "also useful if a node is free" (2026-10-23, GPUs idle after the main queue)
  sub e9-seg2000 gpu 48:00:00 -- "python e9_trainability.py --seeds 5 --seg 2000 --device cuda --threads 8 --out results/e9_seg2000"
  # E8 has no --device (CPU only; MONDAY.md said 1 GPU, corrected): a whole comp72 node
  sub e8-full cpu 72:00:00 -- "python e8_nanogpt_tiles.py --d 384 --layers 6 --heads 6 --steps 5000 --ft-steps 1000 --seeds 3 --threads 32 --out results/e8_full"
  ;;
e11cal)
  # MONDAY2 #1 (2026-10-23): E11-cal on a V100 (E11 kept no checkpoints, so the sigma = 0 arms are retrained; ~2 h), then the
  # unclipped float32 reruns above the dial top (the dial clipped every float32 arm > 0.2 to 0.2 on the Linear layers)
  cal=$(sub e11cal gpu6 5:50:00 -- "python e11cal_noise_calibration.py --device cuda --threads 8"); echo "e11cal $cal"
  u1=$(sub e11-unclip-small gpu6 5:50:00 --dependency=afterany:$cal -- "python e11_regularization.py --device cuda --threads 8 --substrates float32 --sigmas 0.3,0.4,0.5,0.7,1.0,1.4,2.0,3.0 --sigmas-beyond '' --dropouts '' --seeds 5 --float-unclipped --out results/e11_f32_unclipped"); echo "unclip-small $u1"
  u2=$(sub e11-unclip-full gpu6 5:50:00 --dependency=afterany:$u1 -- "python e11_regularization.py --device cuda --threads 8 --substrates float32 --sigmas 0.3,0.4,0.5,0.7,1.0,1.4 --sigmas-beyond '' --dropouts '' --seeds 10 --train-frac 1.0 --steps 5000 --float-unclipped --out results/e11_full_f32_unclipped; python e11cal_noise_calibration.py --analyze"); echo "unclip-full $u2"
  ;;
e1fix)
  # MONDAY2 #2 (2026-10-23): E1-fix + E16, two V100 lanes of 6 h, each followed by one resume job (finished arms are skipped)
  A="deploy,ft_float32_s0_n1200,ft_float32_s0.02_n1200,ft_float32_s0.04_n1200,ft_float32_s0.06_n1200,ft_float32_s0_n2000,ft_float32_s0.02_n2000,ft_float32_s0.04_n2000,ft_float32_s0.06_n2000,onchip_s0.04_n2000"
  B="ft_tiles_s0.04_n2000,ft_tiles_s0.04_n1200,ft_tiles_s0.02_n1200,ft_tiles_s0.06_n1200,ft_tiles_s0.02_n2000,ft_tiles_s0.06_n2000"
  a1=$(sub e1fix-a gpu6 5:55:00 -- "python e1fix_gpt2_deploy.py --device cuda --arms $A"); echo "e1fix-a $a1"
  a2=$(sub e1fix-a2 gpu6 5:55:00 --dependency=afterany:$a1 -- "python e1fix_gpt2_deploy.py --device cuda --arms $A"); echo "e1fix-a2 $a2"
  b1=$(sub e1fix-b gpu6 5:55:00 -- "python e1fix_gpt2_deploy.py --device cuda --arms $B"); echo "e1fix-b $b1"
  b2=$(sub e1fix-b2 gpu6 5:55:00 --dependency=afterany:$b1 -- "python e1fix_gpt2_deploy.py --device cuda --arms $B"); echo "e1fix-b2 $b2"
  ;;
*) echo "usage: bash scripts/hpcc/submit.sh smoke|queue|e11-bracket2|e11-full-bracket|extras|e11cal|e1fix|e15-data|e15-timing|e15"; exit 2 ;;
esac
squeue -u "$USER" -o "%.10i %.16j %.8P %.8T %.10M %R"
