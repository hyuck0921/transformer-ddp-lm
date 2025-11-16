#!/bin/bash
# Local training script for Encoder-Decoder Transformer on 8 GPUs

# Create necessary directories
mkdir -p logs_enc_dec checkpoints_enc_dec

# Set environment variables for better performance
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

echo "Starting Encoder-Decoder Transformer training on 8 GPUs..."
echo "GPU info:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Run with torchrun
torchrun \
    --standalone \
    --nproc_per_node=8 \
    train_enc_dec_ddp.py \
    --config configs/enc_dec_config.yaml

echo "Training complete!"

