"""Multi-GPU Inference with Hugging Face Encoder-Decoder Models."""

import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path
from typing import List, Dict
import time


def setup_distributed():
    """Initialize distributed training."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def load_model_and_tokenizer(model_name: str, device: torch.device):
    """Load Hugging Face model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map=None  # We'll handle device placement manually
    )
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def batch_inference(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    num_beams: int = 4,
    device: torch.device = None
) -> List[str]:
    """Run batch inference."""
    # Tokenize
    inputs = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    # Decode
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return generated_texts


def main(args):
    # Setup
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    is_main = rank == 0
    
    if is_main:
        print("="*80)
        print(f"Multi-GPU Inference with Hugging Face Encoder-Decoder")
        print("="*80)
        print(f"Model: {args.model_name}")
        print(f"GPUs: {world_size}")
        print(f"Device: {device}")
        print("="*80)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)
    
    # Wrap with DDP if using multiple GPUs
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Load dataset
    if is_main:
        print(f"\nLoading dataset: {args.dataset_name}")
    
    if args.dataset_name == "cnn_dailymail":
        dataset = load_dataset("cnn_dailymail", "3.0.0", split=args.split)
        source_key = "article"
        target_key = "highlights"
    elif args.dataset_name == "xsum":
        dataset = load_dataset("xsum", split=args.split)
        source_key = "document"
        target_key = "summary"
    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")
    
    # Limit samples
    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    
    if is_main:
        print(f"Dataset size: {len(dataset)}")
        print(f"Batch size: {args.batch_size}")
        print(f"Samples per GPU: {len(dataset) // world_size}")
    
    # Split data across GPUs
    samples_per_gpu = len(dataset) // world_size
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu if rank < world_size - 1 else len(dataset)
    
    my_dataset = dataset.select(range(start_idx, end_idx))
    
    # Run inference
    results = []
    
    if is_main:
        pbar = tqdm(range(0, len(my_dataset), args.batch_size), desc="Inference")
    else:
        pbar = range(0, len(my_dataset), args.batch_size)
    
    start_time = time.time()
    
    for i in pbar:
        batch = my_dataset[i:i+args.batch_size]
        
        # Get source texts
        if isinstance(batch[source_key], list):
            source_texts = batch[source_key]
            target_texts = batch[target_key]
        else:
            source_texts = [batch[source_key]]
            target_texts = [batch[target_key]]
        
        # Generate
        generated = batch_inference(
            model,
            tokenizer,
            source_texts,
            max_length=args.max_length,
            num_beams=args.num_beams,
            device=device
        )
        
        # Store results
        for src, gen, tgt in zip(source_texts, generated, target_texts):
            results.append({
                "source": src,
                "generated": gen,
                "target": tgt
            })
    
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Save results
    if is_main:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"results_rank_{rank}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Inference complete!")
        print(f"Time: {inference_time:.2f}s")
        print(f"Samples: {len(results)}")
        print(f"Speed: {len(results)/inference_time:.2f} samples/sec")
        print(f"Results saved to: {output_file}")
        
        # Print examples
        print("\n" + "="*80)
        print("Example Results:")
        print("="*80)
        for i, result in enumerate(results[:3]):
            print(f"\nExample {i+1}:")
            print(f"Source: {result['source'][:200]}...")
            print(f"Generated: {result['generated']}")
            print(f"Target: {result['target']}")
            print("-"*80)
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    import os
    
    parser = argparse.ArgumentParser(description='Multi-GPU Inference with HF Encoder-Decoder')
    
    # Model
    parser.add_argument('--model-name', type=str, default='facebook/bart-base',
                        help='Hugging Face model name')
    
    # Dataset
    parser.add_argument('--dataset-name', type=str, default='cnn_dailymail',
                        choices=['cnn_dailymail', 'xsum'],
                        help='Dataset name')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples (0 for all)')
    
    # Inference
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--max-length', type=int, default=128,
                        help='Maximum generation length')
    parser.add_argument('--num-beams', type=int, default=4,
                        help='Number of beams for beam search')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='hf_inference_results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    main(args)

