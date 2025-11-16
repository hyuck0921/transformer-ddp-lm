"""Multi-GPU training script with DDP (8x A10)."""

import os
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from tqdm import tqdm

from models import TransformerLM, TransformerConfig
from data.dataset import get_dataloaders
from utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    Trainer,
    DistributedLogger,
)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_lr_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Create learning rate scheduler."""
    total_steps = steps_per_epoch * config['training']['num_epochs']
    warmup_steps = config['training']['warmup_steps']
    
    if config['training']['lr_schedule'] == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        # Warmup + Cosine
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                min_lr = config['training']['learning_rate'] * config['training']['min_lr_ratio']
                return config['training']['min_lr_ratio'] + (1 - config['training']['min_lr_ratio']) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif config['training']['lr_schedule'] == 'linear':
        from torch.optim.lr_scheduler import LinearLR
        scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=config['training']['min_lr_ratio'], total_iters=total_steps)
    
    else:  # constant
        scheduler = None
    
    return scheduler


def main(args):
    # Load config
    config = load_config(args.config)
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed(backend=config['distributed']['backend'])
    
    # Set device
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(device)
    
    # Logger
    logger = DistributedLogger(verbose=True)
    logger.info(f"Rank {rank}/{world_size} | Device: {device}")
    
    # Set seed
    seed = config['seed'] + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
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
    
    if is_main_process():
        num_params = model.num_parameters()
        logger.info(f"Model parameters: {num_params:,}")
        logger.info(f"Model config: {model_config}")
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=config['distributed']['find_unused_parameters'],
        gradient_as_bucket_view=config['distributed']['gradient_as_bucket_view'],
    )
    
    # Load data
    train_loader, val_loader, dataset = get_dataloaders(
        text_path=config['data']['dataset_path'],
        batch_size=config['training']['batch_size'],
        seq_len=config['model']['max_seq_len'],
        train_split=config['data']['train_split'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        distributed=True,
        world_size=world_size,
        rank=rank,
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
    scheduler = get_lr_scheduler(optimizer, config, len(train_loader))
    
    # Create gradient scaler for mixed precision
    scaler = None
    if config['training']['use_amp']:
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
    
    # Resume from checkpoint if specified
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
    
    # Training loop
    logger.info("Starting training...")
    
    for epoch in range(trainer.epoch, config['training']['num_epochs']):
        trainer.epoch = epoch
        
        # Set epoch for sampler (important for distributed training)
        train_loader.sampler.set_epoch(epoch)
        
        # Training
        model.train()
        trainer.train_metrics.reset()
        
        if is_main_process():
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        else:
            pbar = train_loader
        
        for batch in pbar:
            metrics = trainer.train_step(batch)
            trainer.train_metrics.update(**metrics)
            
            # Log
            if trainer.global_step % config['logging']['log_every'] == 0:
                if is_main_process():
                    pbar.set_postfix(metrics)
                    trainer.log_metrics(metrics, prefix='train')
        
        # Validation
        if (epoch + 1) % config['evaluation']['eval_every'] == 0:
            val_metrics = trainer.validate(val_loader)
            
            if is_main_process():
                logger.info(f"Epoch {epoch+1} | Val Loss: {val_metrics['loss']:.4f} | Val PPL: {val_metrics['perplexity']:.2f}")
                trainer.log_metrics(val_metrics, prefix='val')
                
                # Generate samples
                if config['evaluation']['generate_samples']:
                    model.eval()
                    prompt = "The Transformer"
                    prompt_tensor = dataset.encode(prompt).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        generated = model.module.generate(
                            prompt_tensor,
                            max_length=config['evaluation']['sample_length'],
                            temperature=config['generation']['temperature'],
                            top_k=config['generation']['top_k'],
                            top_p=config['generation']['top_p'],
                        )
                    
                    generated_text = dataset.decode(generated[0])
                    logger.info(f"\nGenerated sample:\n{generated_text}\n")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < trainer.best_val_loss
            if is_best:
                trainer.best_val_loss = val_metrics['loss']
            
            if (epoch + 1) % config['checkpoint']['save_every'] == 0 or is_best:
                checkpoint_path = Path(config['checkpoint']['save_dir']) / f'checkpoint_epoch_{epoch+1}.pt'
                trainer.save_checkpoint(str(checkpoint_path), is_best=is_best)
    
    # Final save
    if is_main_process():
        final_path = Path(config['checkpoint']['save_dir']) / 'final_model.pt'
        trainer.save_checkpoint(str(final_path))
        logger.info(f"Training complete! Final model saved to {final_path}")
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer LM with DDP')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    main(args)

