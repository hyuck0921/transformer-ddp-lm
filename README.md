# Transformer Language Model with DDP Multi-GPU Training

A simple Transformer-based language model trained with PyTorch Distributed Data Parallel (DDP) on 8x A10 GPUs.

## Project Structure

```
transformer-ddp-lm/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── prepare_dataset.py          # Toy dataset preparation
├── models/
│   ├── transformer.py              # Transformer model from scratch
│   └── config.py                   # Model configuration
├── utils/
│   ├── distributed.py              # DDP utilities
│   ├── trainer.py                  # Training utilities
│   └── metrics.py                  # Evaluation metrics
├── configs/
│   └── default_config.yaml         # Training configuration
├── train_ddp.py                    # Multi-GPU training script (8 GPUs)
├── train_single.py                 # Single GPU training script
├── inference.py                    # Inference script
└── checkpoints/                    # Model checkpoints
└── logs/                           # Training logs
```

## Features

- **Transformer Language Model**: Character-level language model from scratch
- **Multi-GPU Training**: DDP with 8x A10 GPUs
- **Toy Dataset**: Small text corpus for quick experimentation
- **Gradient Accumulation**: Support for large effective batch sizes
- **Mixed Precision**: FP16/BF16 training with torch.amp
- **Checkpoint Management**: Automatic saving and resuming
- **Logging**: TensorBoard and console logging

## Requirements

- Python 3.8+
- PyTorch 2.0+
- 8x A10 GPUs (or adjust NUM_GPUS in config)
- CUDA 11.8+

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Dataset

```bash
python data/prepare_dataset.py
```

This creates a toy dataset from a sample text corpus.

### 2. Single GPU Training (for testing)

```bash
python train_single.py --config configs/default_config.yaml
```

### 3. Multi-GPU Training (8x A10)

```bash
torchrun --nproc_per_node=8 train_ddp.py --config configs/default_config.yaml
```

Or use the SLURM script:

```bash
sbatch scripts/train_8gpu.sh
```

### 4. Inference

```bash
python inference.py --checkpoint checkpoints/best_model.pt --prompt "Once upon a time"
```

## Configuration

Edit `configs/default_config.yaml` to customize:

- Model architecture (dim, depth, heads, etc.)
- Training hyperparameters (lr, batch_size, epochs, etc.)
- Dataset settings
- DDP settings

## Model Architecture

- **Type**: Decoder-only Transformer (GPT-like)
- **Default Config**:
  - Embedding dim: 256
  - Layers: 6
  - Attention heads: 8
  - FFN dim: 1024
  - Vocab size: Character-level (256 characters)
  - Context length: 512 tokens

## Training Details

- **Optimizer**: AdamW
- **Learning Rate**: 3e-4 with cosine decay
- **Batch Size**: 32 per GPU (256 effective with 8 GPUs)
- **Gradient Accumulation**: Configurable
- **Mixed Precision**: Automatic Mixed Precision (AMP)
- **Gradient Clipping**: 1.0

## Multi-GPU Training Strategy

This project uses PyTorch DDP (Distributed Data Parallel):

1. Each GPU holds a replica of the model
2. Data is split across GPUs
3. Gradients are synchronized across all GPUs
4. Model parameters are updated identically

### Why DDP?

- **Efficient**: Minimal communication overhead
- **Scalable**: Linear scaling up to 8 GPUs
- **Simple**: Easy to implement and debug

## Expected Performance

On 8x A10 GPUs with default config:
- Training speed: ~10-15 seconds per epoch (toy dataset)
- Memory per GPU: ~2-3 GB
- Total training time: ~5 minutes for 100 epochs

## Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/
```

### Logs

Training logs are saved to:
- Console output
- `logs/train_{timestamp}.log`
- TensorBoard events in `logs/tensorboard/`

## Tips

1. **Adjust batch size**: Modify `batch_size` in config based on GPU memory
2. **Gradient accumulation**: Use if you need larger effective batch size
3. **Mixed precision**: Enable for faster training (default: enabled)
4. **Checkpoint frequency**: Adjust `save_every` in config

## Troubleshooting

### Out of Memory (OOM)

- Reduce `batch_size` in config
- Enable gradient checkpointing
- Reduce model size (dim, depth, heads)

### Slow Training

- Check GPU utilization with `nvidia-smi`
- Increase `num_workers` for data loading
- Ensure AMP is enabled

### Hanging at Initialization

- Check that all GPUs are visible: `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`
- Verify network connectivity between GPUs
- Check NCCL backend

## Advanced Usage

### Resume Training

```bash
torchrun --nproc_per_node=8 train_ddp.py --config configs/default_config.yaml --resume checkpoints/last_checkpoint.pt
```

### Custom Dataset

Modify `data/prepare_dataset.py` to use your own text corpus.

### Model Modifications

Edit `models/transformer.py` to change architecture.

## License

MIT

## References

- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [Language Models are Unsupervised Multitask Learners (GPT-2)](https://openai.com/research/better-language-models)

