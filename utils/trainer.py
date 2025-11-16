"""Training utilities."""

import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from torch.utils.tensorboard import SummaryWriter

from .metrics import compute_loss, compute_accuracy, compute_perplexity, MetricsTracker
from .distributed import is_main_process, reduce_tensor, get_world_size


class Trainer:
    """Trainer for language model."""
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Dict[str, Any],
        device: torch.device,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.scaler = scaler
        
        self.world_size = get_world_size()
        self.is_main = is_main_process()
        
        # Metrics
        self.train_metrics = MetricsTracker()
        self.val_metrics = MetricsTracker()
        
        # Logging
        if self.is_main and config['logging']['use_tensorboard']:
            log_dir = Path(config['logging']['log_dir']) / 'tensorboard'
            log_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            batch: (batch_size, seq_len) input tokens
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        # Move to device
        batch = batch.to(self.device)
        
        # Prepare input and target
        # Input: all tokens except last
        # Target: all tokens except first
        x = batch[:, :-1]
        y = batch[:, 1:]
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.config['training']['amp_dtype'] == 'bfloat16' else torch.float16):
                logits = self.model(x)
                loss = compute_loss(logits, y)
        else:
            logits = self.model(x)
            loss = compute_loss(logits, y)
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config['training']['max_grad_norm'] > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['max_grad_norm']
            )
        
        # Optimizer step
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Compute metrics
        with torch.no_grad():
            accuracy = compute_accuracy(logits, y)
            perplexity = compute_perplexity(loss.item())
        
        # Reduce metrics across GPUs
        loss = reduce_tensor(loss, self.world_size).item()
        
        self.global_step += 1
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'perplexity': perplexity,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
    
    @torch.no_grad()
    def validate(self, dataloader) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Args:
            dataloader: Validation dataloader
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        self.val_metrics.reset()
        
        for batch in dataloader:
            batch = batch.to(self.device)
            
            x = batch[:, :-1]
            y = batch[:, 1:]
            
            logits = self.model(x)
            loss = compute_loss(logits, y)
            accuracy = compute_accuracy(logits, y)
            
            # Reduce metrics
            loss = reduce_tensor(loss, self.world_size).item()
            
            self.val_metrics.update(
                loss=loss,
                accuracy=accuracy,
            )
        
        val_loss = self.val_metrics.get('loss')
        val_acc = self.val_metrics.get('accuracy')
        val_ppl = compute_perplexity(val_loss)
        
        return {
            'loss': val_loss,
            'accuracy': val_acc,
            'perplexity': val_ppl,
        }
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = 'train'):
        """Log metrics to tensorboard and console."""
        if not self.is_main:
            return
        
        # TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f'{prefix}/{key}', value, self.global_step)
        
        # Console
        metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        print(f'[{prefix.upper()}] Step {self.global_step} | {metrics_str}')
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint."""
        if not self.is_main:
            return
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = Path(path).parent / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f'Saved best model to {best_path}')
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        if hasattr(self.model, 'module'):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f'Loaded checkpoint from {path} (epoch {self.epoch}, step {self.global_step})')

