"""Single GPU training script for testing."""

import argparse
import yaml
import torch
from pathlib import Path
from tqdm import tqdm

from models import TransformerLM, TransformerConfig
from data.dataset import get_dataloaders
from utils import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # Load config
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set seed
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Create model config
    model_config = TransformerConfig(
        vocab_size=config['model']['vocab_size'],
        max_seq_len=config['model']['max_seq_len'],
        dim=config['model']['dim'],
        depth=config['model']['depth'],
        heads=config['model']['heads'],
        dim_head=config['model']['dim_head'],
        mlp_dim=config['model']['mlp_dim'],
        dropout=config['model']['dropout'],
        use_rotary_emb=config['model']['use_rotary_emb'],
    )
    
    # Create model
    model = TransformerLM(model_config).to(device)
    
    num_params = model.num_parameters()
    print(f"Model parameters: {num_params:,}")
    
    # Load data
    train_loader, val_loader, dataset = get_dataloaders(
        text_path=config['data']['dataset_path'],
        batch_size=config['training']['batch_size'],
        seq_len=config['model']['max_seq_len'],
        train_split=config['data']['train_split'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        distributed=False,
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        betas=(config['training']['adam_beta1'], config['training']['adam_beta2']),
        eps=config['training']['adam_epsilon'],
        weight_decay=config['training']['weight_decay'],
    )
    
    # Create scheduler
    total_steps = len(train_loader) * config['training']['num_epochs']
    warmup_steps = config['training']['warmup_steps']
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create gradient scaler for mixed precision
    scaler = None
    if config['training']['use_amp'] and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device,
        scaler=scaler,
    )
    
    # Training loop
    print("Starting training...")
    
    for epoch in range(config['training']['num_epochs']):
        trainer.epoch = epoch
        
        # Training
        model.train()
        trainer.train_metrics.reset()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            trainer.train_metrics.update(**metrics)
            pbar.set_postfix(metrics)
            
            # Log
            if trainer.global_step % config['logging']['log_every'] == 0:
                trainer.log_metrics(metrics, prefix='train')
        
        # Validation
        if (epoch + 1) % config['evaluation']['eval_every'] == 0:
            val_metrics = trainer.validate(val_loader)
            print(f"Epoch {epoch+1} | Val Loss: {val_metrics['loss']:.4f} | Val PPL: {val_metrics['perplexity']:.2f}")
            trainer.log_metrics(val_metrics, prefix='val')
            
            # Generate samples
            if config['evaluation']['generate_samples']:
                model.eval()
                prompt = "The Transformer"
                prompt_tensor = dataset.encode(prompt).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    generated = model.generate(
                        prompt_tensor,
                        max_length=config['evaluation']['sample_length'],
                        temperature=config['generation']['temperature'],
                        top_k=config['generation']['top_k'],
                        top_p=config['generation']['top_p'],
                    )
                
                generated_text = dataset.decode(generated[0])
                print(f"\nGenerated sample:\n{generated_text}\n")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < trainer.best_val_loss
            if is_best:
                trainer.best_val_loss = val_metrics['loss']
            
            if (epoch + 1) % config['checkpoint']['save_every'] == 0 or is_best:
                checkpoint_path = Path(config['checkpoint']['save_dir']) / f'checkpoint_epoch_{epoch+1}.pt'
                trainer.save_checkpoint(str(checkpoint_path), is_best=is_best)
    
    # Final save
    final_path = Path(config['checkpoint']['save_dir']) / 'final_model.pt'
    trainer.save_checkpoint(str(final_path))
    print(f"Training complete! Final model saved to {final_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer LM (single GPU)')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    main(args)

