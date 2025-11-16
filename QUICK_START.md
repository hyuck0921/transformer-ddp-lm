# Quick Start Guide

## Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 8x A10 GPUs (or adjust configuration for available GPUs)

## Installation

1. Clone or navigate to this repository:
```bash
cd transformer-ddp-lm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Test your setup:
```bash
python test_setup.py
```

## Step-by-Step Tutorial

### 1. Prepare Dataset

Create a toy dataset:
```bash
python data/prepare_dataset.py --output data/toy_dataset.txt --repeat 100
```

This creates a small text corpus repeated 100 times for training.

**Optional**: Use your own text file:
```bash
python data/prepare_dataset.py --output data/toy_dataset.txt --custom-text /path/to/your/text.txt --repeat 50
```

### 2. Test with Single GPU

Before running on 8 GPUs, test with a single GPU:
```bash
python train_single.py --config configs/default_config.yaml
```

This will train for a few epochs to verify everything works.

### 3. Train with 8 GPUs (DDP)

#### Option A: Local Machine
```bash
bash scripts/train_local_8gpu.sh
```

Or manually:
```bash
torchrun --nproc_per_node=8 train_ddp.py --config configs/default_config.yaml
```

#### Option B: SLURM Cluster
```bash
sbatch scripts/train_8gpu.sh
```

### 4. Monitor Training

#### TensorBoard
```bash
tensorboard --logdir logs/tensorboard
```

Then open http://localhost:6006 in your browser.

#### Console Logs
Logs are saved in `logs/` directory.

### 5. Inference

After training, generate text:

#### Interactive Mode
```bash
python inference.py --checkpoint checkpoints/best_model.pt --interactive
```

#### Single Generation
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "The Transformer is a" \
    --max-length 500 \
    --temperature 0.8
```

#### Save to File
```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --prompt "Once upon a time" \
    --max-length 1000 \
    --output generated.txt
```

## Configuration

Edit `configs/default_config.yaml` to customize:

### Model Size
```yaml
model:
  dim: 256        # Increase for larger model
  depth: 6        # Number of transformer layers
  heads: 8        # Number of attention heads
  mlp_dim: 1024   # Feed-forward dimension
```

### Training
```yaml
training:
  batch_size: 32      # Per GPU
  num_epochs: 100
  learning_rate: 3.0e-4
  use_amp: true       # Mixed precision training
```

### Data
```yaml
data:
  dataset_path: "data/toy_dataset.txt"
  train_split: 0.9
  num_workers: 4
```

## Common Issues

### Out of Memory
- Reduce `batch_size` in config
- Reduce model size (`dim`, `depth`)
- Enable gradient checkpointing (add to code)

### Slow Training
- Increase `num_workers` for data loading
- Ensure `use_amp: true` for mixed precision
- Check GPU utilization: `watch -n 1 nvidia-smi`

### DDP Not Working
- Ensure all GPUs are visible: `echo $CUDA_VISIBLE_DEVICES`
- Check NCCL backend: `export NCCL_DEBUG=INFO`
- Verify network: Test with fewer GPUs first

## Expected Results

With default config on 8x A10 GPUs:
- **Training time**: ~5 minutes for 100 epochs (toy dataset)
- **Memory per GPU**: ~2-3 GB
- **Training speed**: ~10-15 sec/epoch
- **Validation perplexity**: Should decrease to ~3-5 after 100 epochs

## Advanced Usage

### Resume Training
```bash
torchrun --nproc_per_node=8 train_ddp.py \
    --config configs/default_config.yaml \
    --resume checkpoints/checkpoint_epoch_50.pt
```

### Change Number of GPUs
```bash
# For 4 GPUs
torchrun --nproc_per_node=4 train_ddp.py --config configs/default_config.yaml

# For 2 GPUs
torchrun --nproc_per_node=2 train_ddp.py --config configs/default_config.yaml
```

### Custom Dataset
1. Create a text file with your corpus
2. Update `dataset_path` in config
3. Run training as usual

## Project Structure

```
transformer-ddp-lm/
├── models/              # Model architecture
│   ├── transformer.py   # Transformer implementation
│   └── config.py        # Model configuration
├── data/                # Dataset utilities
│   ├── dataset.py       # PyTorch dataset
│   └── prepare_dataset.py
├── utils/               # Training utilities
│   ├── distributed.py   # DDP helpers
│   ├── trainer.py       # Training loop
│   └── metrics.py       # Evaluation metrics
├── configs/             # Configuration files
│   └── default_config.yaml
├── train_ddp.py         # Multi-GPU training
├── train_single.py      # Single GPU training
├── inference.py         # Text generation
└── test_setup.py        # Setup verification
```

## Tips for Best Results

1. **Start small**: Test with single GPU first
2. **Monitor training**: Use TensorBoard to watch loss curves
3. **Adjust temperature**: Lower (0.5-0.7) for more focused text, higher (0.9-1.2) for more creative
4. **Use top-k/top-p**: Balance between diversity and quality
5. **Train longer**: 100 epochs is quick, try 500-1000 for better results with small datasets

## Next Steps

1. Try different model sizes (tiny, small, medium, large)
2. Experiment with your own text corpus
3. Adjust generation parameters (temperature, top-k, top-p)
4. Modify architecture (add more layers, change attention mechanism)
5. Implement additional features (gradient checkpointing, FSDP for larger models)

## Getting Help

Check:
- README.md for detailed documentation
- test_setup.py output for system verification
- logs/ directory for training logs
- PyTorch DDP documentation: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

