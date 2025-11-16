#!/bin/bash
# SLURM script for training on 8x A10 GPUs

#SBATCH --job-name=transformer-lm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:a10:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

# Create logs directory
mkdir -p logs

# Activate environment (modify as needed)
# source activate your_env
# or
# module load python/3.9
# module load cuda/11.8

# Print GPU info
nvidia-smi

# Set environment variables
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

# Run training with torchrun
torchrun \
    --standalone \
    --nproc_per_node=8 \
    train_ddp.py \
    --config configs/default_config.yaml

echo "Training complete!"

