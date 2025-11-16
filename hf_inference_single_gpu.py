"""Single GPU Inference with Hugging Face Encoder-Decoder Models."""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tqdm import tqdm
import json
from pathlib import Path
import time


def load_model_and_tokenizer(model_name: str, device: torch.device):
    """Load Hugging Face model and tokenizer."""
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
    )
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    print(f"  Parameters: {model.num_parameters():,}")
    
    return model, tokenizer


def batch_inference(model, tokenizer, texts, max_length, num_beams, device):
    """Run batch inference."""
    inputs = tokenizer(
        texts,
        max_length=1024,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    return generated_texts


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("Hugging Face Encoder-Decoder Inference")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print("="*80)
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)
    
    # Load dataset
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
    
    if args.num_samples > 0:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    
    # Run inference
    results = []
    start_time = time.time()
    
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Inference"):
        batch = dataset[i:i+args.batch_size]
        
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "results.json"
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HF Encoder-Decoder Inference')
    
    parser.add_argument('--model-name', type=str, default='facebook/bart-base',
                        help='Hugging Face model name')
    parser.add_argument('--dataset-name', type=str, default='cnn_dailymail',
                        choices=['cnn_dailymail', 'xsum'])
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--num-beams', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='hf_inference_results')
    
    args = parser.parse_args()
    
    main(args)

