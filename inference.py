"""Inference script for text generation."""

import argparse
import torch
from pathlib import Path

from models import TransformerLM, TransformerConfig
from data.dataset import CharDataset


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
        use_rotary_emb=model_config_dict['use_rotary_emb'],
    )
    
    # Create model
    model = TransformerLM(model_config).to(device)
    
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


def generate_text(
    model: TransformerLM,
    dataset: CharDataset,
    prompt: str,
    max_length: int = 500,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    device: torch.device = None,
) -> str:
    """
    Generate text from prompt.
    
    Args:
        model: Trained model
        dataset: Dataset for encoding/decoding
        prompt: Initial text prompt
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        device: Device to run on
    Returns:
        Generated text
    """
    model.eval()
    
    # Encode prompt
    prompt_tensor = dataset.encode(prompt).unsqueeze(0)
    
    if device is not None:
        prompt_tensor = prompt_tensor.to(device)
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            prompt_tensor,
            max_length=max_length,
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
    
    # Load dataset (for encoding/decoding)
    print(f"\nLoading dataset from {config['data']['dataset_path']}...")
    dataset = CharDataset(
        config['data']['dataset_path'],
        seq_len=config['model']['max_seq_len']
    )
    
    # Interactive mode
    if args.interactive:
        print("\n" + "="*80)
        print("INTERACTIVE MODE")
        print("="*80)
        print("Enter prompts to generate text. Type 'quit' or 'exit' to stop.")
        print("="*80 + "\n")
        
        while True:
            try:
                prompt = input("Prompt: ")
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not prompt.strip():
                    continue
                
                # Generate
                generated_text = generate_text(
                    model=model,
                    dataset=dataset,
                    prompt=prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=device,
                )
                
                print(f"\nGenerated:\n{generated_text}\n")
                print("-"*80 + "\n")
            
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
    
    # Single generation mode
    else:
        if args.prompt is None:
            print("Error: --prompt required in non-interactive mode")
            return
        
        print(f"\nPrompt: {args.prompt}")
        print("="*80)
        
        # Generate
        generated_text = generate_text(
            model=model,
            dataset=dataset,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device,
        )
        
        print(f"\nGenerated:\n{generated_text}")
        
        # Save to file if specified
        if args.output is not None:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(generated_text)
            
            print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text with trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt for generation')
    parser.add_argument('--max-length', type=int, default=500,
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
                        help='Save generated text to file')
    
    args = parser.parse_args()
    
    main(args)

