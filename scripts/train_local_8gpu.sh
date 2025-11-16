#!/bin/bash
# Local training script for 8 GPUs (without SLURM)

# Create necessary directories
mkdir -p logs checkpoints

# Set environment variables for better performance
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "Starting training on 8 GPUs..."
echo "GPU info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Run with torchrun
torchrun \
    --standalone \
    --nproc_per_node=8 \
    train_ddp.py \
    --config configs/default_config.yaml

echo "Training complete!"

