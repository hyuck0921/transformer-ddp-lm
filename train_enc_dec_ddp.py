"""Multi-GPU training script for Encoder-Decoder Transformer with DDP."""

import os
import sys
import argparse
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from tqdm import tqdm

from models.transformer_enc_dec import EncoderDecoderTransformer
from models.config import TransformerConfig
from data.seq2seq_dataset import get_seq2seq_dataloaders
from utils import (
    setup_distributed,
    cleanup_distributed,
    is_main_process,
    DistributedLogger,
    reduce_tensor,
    get_world_size,
)
from utils.metrics import compute_accuracy, compute_perplexity, MetricsTracker


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_lr_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """Create learning rate scheduler."""
    total_steps = steps_per_epoch * config['training']['num_epochs']
    warmup_steps = config['training']['warmup_steps']
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            min_lr_ratio = config['training']['min_lr_ratio']
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_step(model, batch, optimizer, scaler, config, device, world_size):
    """Single training step."""
    model.train()
    
    src, tgt, src_mask, tgt_mask = batch
    src = src.to(device)
    tgt = tgt.to(device)
    src_mask = src_mask.to(device)
    tgt_mask = tgt_mask.to(device)
    
    # Target input: all except last token
    # Target output: all except first token (BOS)
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    tgt_mask_input = tgt_mask[:, :-1]
    
    # Forward pass with mixed precision
    if scaler is not None:
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if config['training']['amp_dtype'] == 'bfloat16' else torch.float16):
            logits = model(src, tgt_input, src_mask, tgt_mask_input)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
                ignore_index=0  # Ignore padding
            )
    else:
        logits = model(src, tgt_input, src_mask, tgt_mask_input)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=0
        )
    
    # Backward pass
    optimizer.zero_grad()
    if scaler is not None:
        scaler.scale(loss).backward()
        if config['training']['max_grad_norm'] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        if config['training']['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
        optimizer.step()
    
    # Compute metrics
    with torch.no_grad():
        accuracy = compute_accuracy(logits, tgt_output, ignore_index=0)
        perplexity = compute_perplexity(loss.item())
    
    # Reduce metrics across GPUs
    loss = reduce_tensor(loss, world_size).item()
    
    return {
        'loss': loss,
        'accuracy': accuracy,
        'perplexity': perplexity,
    }


@torch.no_grad()
def validate(model, dataloader, device, world_size):
    """Validate model."""
    model.eval()
    metrics = MetricsTracker()
    
    for batch in dataloader:
        src, tgt, src_mask, tgt_mask = batch
        src = src.to(device)
        tgt = tgt.to(device)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        tgt_mask_input = tgt_mask[:, :-1]
        
        logits = model(src, tgt_input, src_mask, tgt_mask_input)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
            ignore_index=0
        )
        accuracy = compute_accuracy(logits, tgt_output, ignore_index=0)
        
        loss = reduce_tensor(loss, world_size).item()
        
        metrics.update(loss=loss, accuracy=accuracy)
    
    val_loss = metrics.get('loss')
    val_acc = metrics.get('accuracy')
    val_ppl = compute_perplexity(val_loss)
    
    return {
        'loss': val_loss,
        'accuracy': val_acc,
        'perplexity': val_ppl,
    }


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
        use_rotary_emb=False,  # Encoder-Decoder uses sinusoidal positional encoding
    )
    
    # Create model
    model = EncoderDecoderTransformer(model_config).to(device)
    
    if is_main_process():
        num_params = model.num_parameters()
        logger.info(f"Model parameters: {num_params:,}")
        logger.info(f"Model type: Encoder-Decoder Transformer")
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=config['distributed']['find_unused_parameters'],
        gradient_as_bucket_view=config['distributed']['gradient_as_bucket_view'],
    )
    
    # Load data
    train_loader, val_loader, train_dataset, val_dataset = get_seq2seq_dataloaders(
        num_train=config['data']['num_train_samples'],
        num_val=config['data']['num_val_samples'],
        batch_size=config['training']['batch_size'],
        seq_len=config['model']['max_seq_len'],
        task=config['data']['task'],
        vocab_size=config['model']['vocab_size'],
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
    
    # Training loop
    logger.info("Starting training...")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        # Set epoch for sampler
        train_loader.sampler.set_epoch(epoch)
        
        # Training
        model.train()
        train_metrics = MetricsTracker()
        
        if is_main_process():
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        else:
            pbar = train_loader
        
        for batch in pbar:
            metrics = train_step(model, batch, optimizer, scaler, config, device, world_size)
            scheduler.step()
            train_metrics.update(**metrics)
            
            global_step += 1
            
            if is_main_process():
                pbar.set_postfix(metrics)
        
        # Validation
        if (epoch + 1) % config['evaluation']['eval_every'] == 0:
            val_metrics = validate(model, val_loader, device, world_size)
            
            if is_main_process():
                logger.info(f"Epoch {epoch+1} | Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val PPL: {val_metrics['perplexity']:.2f}")
                
                # Generate sample
                if config['evaluation']['generate_samples']:
                    model.eval()
                    # Take first validation sample
                    src, tgt, src_mask, tgt_mask = next(iter(val_loader))
                    src_sample = src[0:1].to(device)
                    src_mask_sample = src_mask[0:1].to(device)
                    
                    with torch.no_grad():
                        generated = model.module.generate(
                            src_sample,
                            max_length=config['model']['max_seq_len'],
                            bos_token_id=train_dataset.bos_token_id,
                            eos_token_id=train_dataset.eos_token_id,
                            temperature=config['generation']['temperature'],
                            top_k=config['generation']['top_k'],
                            top_p=config['generation']['top_p'],
                            src_mask=src_mask_sample,
                        )
                    
                    src_text = train_dataset.decode(src[0])
                    tgt_text = train_dataset.decode(tgt[0])
                    gen_text = train_dataset.decode(generated[0])
                    
                    logger.info(f"\nSample:")
                    logger.info(f"  Source: {src_text}")
                    logger.info(f"  Target: {tgt_text}")
                    logger.info(f"  Generated: {gen_text}\n")
            
            # Save checkpoint
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
            
            if is_main_process() and ((epoch + 1) % config['checkpoint']['save_every'] == 0 or is_best):
                checkpoint_dir = Path(config['checkpoint']['save_dir'])
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                checkpoint = {
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                
                if scaler is not None:
                    checkpoint['scaler_state_dict'] = scaler.state_dict()
                
                checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
                torch.save(checkpoint, checkpoint_path)
                
                if is_best:
                    best_path = checkpoint_dir / 'best_model.pt'
                    torch.save(checkpoint, best_path)
                    logger.info(f"Saved best model to {best_path}")
    
    # Final save
    if is_main_process():
        final_path = Path(config['checkpoint']['save_dir']) / 'final_model.pt'
        torch.save(checkpoint, final_path)
        logger.info(f"Training complete! Final model saved to {final_path}")
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Encoder-Decoder Transformer with DDP')
    parser.add_argument('--config', type=str, default='configs/enc_dec_config.yaml',
                        help='Path to config file')
    
    args = parser.parse_args()
    
    main(args)

