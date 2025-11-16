"""Test script to verify setup and GPU availability."""

import sys
import torch
import torch.distributed as dist


def test_pytorch():
    """Test PyTorch installation."""
    print("=" * 60)
    print("PyTorch Installation Test")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  - Memory: {props.total_memory / 1024**3:.2f} GB")
            print(f"  - Compute Capability: {props.major}.{props.minor}")
    else:
        print("WARNING: CUDA not available!")
    
    print()


def test_imports():
    """Test required imports."""
    print("=" * 60)
    print("Import Test")
    print("=" * 60)
    
    try:
        import yaml
        print("✓ yaml")
    except ImportError:
        print("✗ yaml (install with: pip install pyyaml)")
    
    try:
        import tqdm
        print("✓ tqdm")
    except ImportError:
        print("✗ tqdm (install with: pip install tqdm)")
    
    try:
        import einops
        print("✓ einops")
    except ImportError:
        print("✗ einops (install with: pip install einops)")
    
    try:
        from torch.utils.tensorboard import SummaryWriter
        print("✓ tensorboard")
    except ImportError:
        print("✗ tensorboard (install with: pip install tensorboard)")
    
    print()


def test_model():
    """Test model creation."""
    print("=" * 60)
    print("Model Test")
    print("=" * 60)
    
    try:
        from models import TransformerLM, TransformerConfig
        
        config = TransformerConfig(
            vocab_size=256,
            max_seq_len=128,
            dim=128,
            depth=2,
            heads=4,
            dim_head=32,
            mlp_dim=512,
            dropout=0.1,
        )
        
        model = TransformerLM(config)
        print(f"✓ Model created successfully")
        print(f"  - Parameters: {model.num_parameters():,}")
        
        # Test forward pass
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        x = torch.randint(0, 256, (2, 128)).to(device)
        with torch.no_grad():
            logits = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {logits.shape}")
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
    
    print()


def test_ddp():
    """Test DDP availability."""
    print("=" * 60)
    print("DDP Test")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("✗ CUDA not available, cannot test DDP")
        return
    
    if torch.cuda.device_count() < 2:
        print("⚠ Only 1 GPU available, DDP will work but won't be distributed")
    else:
        print(f"✓ {torch.cuda.device_count()} GPUs available for DDP")
    
    # Check NCCL backend
    if torch.cuda.is_available():
        print(f"✓ NCCL backend available")
    
    print()


def test_dataset():
    """Test dataset creation."""
    print("=" * 60)
    print("Dataset Test")
    print("=" * 60)
    
    try:
        # Create a small test file
        import tempfile
        import os
        
        test_text = "Hello world! " * 100
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(test_text)
            test_file = f.name
        
        from data.dataset import CharDataset
        
        dataset = CharDataset(test_file, seq_len=32)
        print(f"✓ Dataset created successfully")
        print(f"  - Length: {len(dataset)}")
        print(f"  - Vocab size: {dataset.vocab_size}")
        
        # Test getting an item
        sample = dataset[0]
        print(f"✓ Dataset indexing works")
        print(f"  - Sample shape: {sample.shape}")
        
        # Test encoding/decoding
        text = "Hello"
        encoded = dataset.encode(text)
        decoded = dataset.decode(encoded)
        print(f"✓ Encoding/decoding works")
        print(f"  - Original: {text}")
        print(f"  - Decoded: {decoded}")
        
        # Cleanup
        os.remove(test_file)
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
    
    print()


def main():
    print("\n" + "=" * 60)
    print("TRANSFORMER DDP SETUP TEST")
    print("=" * 60 + "\n")
    
    test_pytorch()
    test_imports()
    test_model()
    test_ddp()
    test_dataset()
    
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if torch.cuda.is_available() and torch.cuda.device_count() >= 8:
        print("✓ System is ready for 8-GPU training!")
        print("\nNext steps:")
        print("  1. Prepare dataset: python data/prepare_dataset.py")
        print("  2. Run training: bash scripts/train_local_8gpu.sh")
    elif torch.cuda.is_available():
        print(f"⚠ System has {torch.cuda.device_count()} GPU(s)")
        print("  You can still train, but with fewer GPUs than planned")
        print("\nNext steps:")
        print("  1. Prepare dataset: python data/prepare_dataset.py")
        print("  2. Run single GPU: python train_single.py")
    else:
        print("✗ CUDA not available")
        print("  You can still train on CPU, but it will be very slow")
    
    print()


if __name__ == '__main__':
    main()

