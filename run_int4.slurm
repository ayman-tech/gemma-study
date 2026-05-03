#!/bin/bash
#SBATCH --job-name=gemma-int4
#SBATCH --output=logs/int4_%j.out      # stdout  (%j = job ID)
#SBATCH --error=logs/int4_%j.err       # stderr
#SBATCH --time=02:00:00                # max wall time (adjust as needed)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # CPU cores for data loading
#SBATCH --mem=16G                      # system RAM (increase if OOM)
#SBATCH --gres=gpu:1                   # request 1 GPU (INT4 requires CUDA)
#SBATCH --partition=gpu                # change to your cluster's GPU partition name

# --------------------------------------------------------------------------
# Environment setup
# --------------------------------------------------------------------------
mkdir -p logs

# Load modules — adjust to your HPC module system
# module load python/3.11
# module load cuda/12.1

# Activate your virtual environment (pick one):
# source ~/envs/gemma/bin/activate
# conda activate gemma

export PYTHONUTF8=1
export TOKENIZERS_PARALLELISM=false

# Hugging Face cache — point to a shared/fast filesystem on your HPC
export HF_HOME=${SCRATCH:-$HOME}/.cache/huggingface

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
echo "========================================"
echo "Job ID    : $SLURM_JOB_ID"
echo "Node      : $SLURMD_NODENAME"
echo "GPUs      : $CUDA_VISIBLE_DEVICES"
echo "Started   : $(date)"
echo "========================================"

# Override defaults via args if needed, e.g.:
# python train_int4_local.py --epochs 3 --num_rows 1000 --gpu
python train_int4_local.py --gpu

echo "Finished  : $(date)"
