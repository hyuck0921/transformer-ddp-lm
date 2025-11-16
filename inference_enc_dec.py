"""Inference script for Encoder-Decoder Transformer (Seq2Seq)."""

import argparse
import torch
from pathlib import Path

from models.transformer_enc_dec import EncoderDecoderTransformer
from models.config import TransformerConfig
from data.seq2seq_dataset import Seq2SeqDataset


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    model_config_dict = checkpoint['config']['model']
    model_config = TransformerConfig(
        vocab_size=model_config_dict['vocab_size'],
        max_seq_len=model_config_dict['max_seq_len'],
        dim=model_config_dict['dim'],
        depth=model_config_dict['depth'],
        heads=model_config_dict['heads'],
        dim_head=model_config_dict['dim_head'],
        mlp_dim=model_config_dict['mlp_dim'],
        dropout=model_config_dict['dropout'],
        use_rotary_emb=False,
    )
    
    # Create model
    model = EncoderDecoderTransformer(model_config).to(device)
    
    # Load state dict
    state_dict = checkpoint['model_state_dict']
    
    # Remove 'module.' prefix if present (from DDP)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  - Parameters: {model.num_parameters():,}")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    return model, checkpoint['config']


def translate(
    model: EncoderDecoderTransformer,
    dataset: Seq2SeqDataset,
    source_text: str,
    max_length: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device = None,
) -> str:
    """
    Translate/process source text to target.
    
    Args:
        model: Trained Encoder-Decoder model
        dataset: Dataset for encoding/decoding
        source_text: Input text
        max_length: Maximum generation length
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        device: Device to run on
    Returns:
        Generated target text
    """
    model.eval()
    
    # Encode source
    src = dataset.encode(source_text)
    src = torch.cat([src, torch.tensor([dataset.eos_token_id])])
    src = src.unsqueeze(0)  # Add batch dimension
    
    if device is not None:
        src = src.to(device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            src,
            max_length=max_length,
            bos_token_id=dataset.bos_token_id,
            eos_token_id=dataset.eos_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    
    # Decode
    generated_text = dataset.decode(generated[0])
    
    return generated_text


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model, config = load_model(args.checkpoint, device)
    
    # Create dataset for encoding/decoding
    print(f"\nCreating dataset for task: {config['data']['task']}")
    dataset = Seq2SeqDataset(
        num_samples=1,  # Just need it for encoding/decoding
        seq_len=config['model']['max_seq_len'],
        task=config['data']['task'],
        vocab_size=config['model']['vocab_size']
    )
    
    # Interactive mode
    if args.interactive:
        print("\n" + "="*80)
        print("INTERACTIVE MODE")
        print("="*80)
        print(f"Task: {config['data']['task']}")
        print("Enter source sequences. Type 'quit' or 'exit' to stop.")
        print("="*80 + "\n")
        
        while True:
            try:
                source = input("Source: ")
                
                if source.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not source.strip():
                    continue
                
                # Translate
                target = translate(
                    model=model,
                    dataset=dataset,
                    source_text=source,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=device,
                )
                
                print(f"Target: {target}\n")
                print("-"*80 + "\n")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    
    # Single translation mode
    else:
        if args.source is None:
            print("Error: --source required in non-interactive mode")
            return
        
        print(f"\nSource: {args.source}")
        print("="*80)
        
        # Translate
        target = translate(
            model=model,
            dataset=dataset,
            source_text=args.source,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        
        print(f"Target: {target}")
        
        # Save to file if specified
        if args.output is not None:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Source: {args.source}\n")
                f.write(f"Target: {target}\n")
            
            print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate/process with trained Encoder-Decoder model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--source', type=str, default=None,
                        help='Source sequence')
    parser.add_argument('--max-length', type=int, default=50,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=50,
                        help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Nucleus sampling threshold')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode')
    parser.add_argument('--output', type=str, default=None,
                        help='Save result to file')
    
    args = parser.parse_args()
    
    main(args)

