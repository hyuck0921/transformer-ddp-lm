"""Evaluation metrics for language modeling."""

import torch
import torch.nn.functional as F
from typing import Optional


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss.
    
    Args:
        loss: Cross-entropy loss
    Returns:
        Perplexity
    """
    return torch.exp(torch.tensor(loss)).item()


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    """
    Compute token-level accuracy.
    
    Args:
        logits: (batch, seq_len, vocab_size) predictions
        targets: (batch, seq_len) ground truth
        ignore_index: Index to ignore in targets
    Returns:
        Accuracy as float
    """
    preds = logits.argmax(dim=-1)
    
    if ignore_index is not None:
        mask = targets != ignore_index
        correct = (preds == targets) & mask
        accuracy = correct.sum().item() / mask.sum().item()
    else:
        correct = preds == targets
        accuracy = correct.sum().item() / targets.numel()
    
    return accuracy


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: (batch, seq_len, vocab_size) predictions
        targets: (batch, seq_len) ground truth
        ignore_index: Index to ignore in targets
        reduction: 'mean', 'sum', or 'none'
    Returns:
        Loss tensor
    """
    batch_size, seq_len, vocab_size = logits.shape
    
    # Reshape for cross entropy
    logits = logits.reshape(-1, vocab_size)
    targets = targets.reshape(-1)
    
    loss = F.cross_entropy(
        logits,
        targets,
        ignore_index=ignore_index,
        reduction=reduction
    )
    
    return loss


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """Update with new value."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        return f'{self.avg:.4f}'


class MetricsTracker:
    """Track multiple metrics during training."""
    
    def __init__(self):
        self.metrics = {}
    
    def update(self, **kwargs):
        """Update metrics with keyword arguments."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = AverageMeter()
            self.metrics[key].update(value)
    
    def get(self, key: str) -> float:
        """Get average value of a metric."""
        if key not in self.metrics:
            return 0.0
        return self.metrics[key].avg
    
    def reset(self):
        """Reset all metrics."""
        for meter in self.metrics.values():
            meter.reset()
    
    def summary(self) -> dict:
        """Get summary of all metrics."""
        return {key: meter.avg for key, meter in self.metrics.items()}
    
    def __str__(self):
        return ' | '.join([f'{key}: {meter}' for key, meter in self.metrics.items()])

