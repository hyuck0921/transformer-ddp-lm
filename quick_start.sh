#!/bin/bash
# Quick start script - prepares dataset and runs single GPU training

echo "============================================"
echo "Transformer Language Model - Quick Start"
echo "============================================"

# Step 1: Create directories
echo ""
echo "Step 1: Creating directories..."
mkdir -p data checkpoints logs

# Step 2: Prepare dataset
echo ""
echo "Step 2: Preparing toy dataset..."
python data/prepare_dataset.py --output data/toy_dataset.txt --repeat 100

# Step 3: Run single GPU training (for testing)
echo ""
echo "Step 3: Starting single GPU training..."
echo "(This is just a test. For 8 GPU training, use train_8gpu.sh)"
echo ""

python train_single.py --config configs/default_config.yaml

echo ""
echo "============================================"
echo "Quick start complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Check training logs in logs/"
echo "  2. View tensorboard: tensorboard --logdir logs/tensorboard"
echo "  3. For 8-GPU training: bash scripts/train_local_8gpu.sh"
echo "  4. For inference: python inference.py --checkpoint checkpoints/best_model.pt --interactive"
echo ""

