"""Dataset for character-level language modeling."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


class CharDataset(Dataset):
    """Character-level language modeling dataset."""
    
    def __init__(self, text_path: str, seq_len: int = 512):
        """
        Args:
            text_path: Path to text file
            seq_len: Sequence length
        """
        self.seq_len = seq_len
        
        # Read text
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Character-level tokenization
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create mappings
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # Encode text
        self.data = torch.tensor([self.char_to_idx[ch] for ch in text], dtype=torch.long)
        
        print(f"Dataset loaded:")
        print(f"  - Text length: {len(text):,} characters")
        print(f"  - Vocabulary size: {self.vocab_size}")
        print(f"  - Sequence length: {seq_len}")
        print(f"  - Number of sequences: {len(self):,}")
    
    def __len__(self) -> int:
        """Number of sequences in the dataset."""
        return len(self.data) // self.seq_len
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Get a sequence.
        
        Args:
            idx: Index
        Returns:
            Sequence of tokens (seq_len + 1,) - includes target token
        """
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        
        # Get sequence
        sequence = self.data[start_idx:end_idx]
        
        # Pad if necessary
        if len(sequence) < self.seq_len + 1:
            pad_len = self.seq_len + 1 - len(sequence)
            sequence = torch.cat([sequence, torch.zeros(pad_len, dtype=torch.long)])
        
        return sequence
    
    def decode(self, indices: torch.Tensor) -> str:
        """
        Decode token indices to text.
        
        Args:
            indices: Token indices (any shape)
        Returns:
            Decoded text
        """
        if indices.dim() > 1:
            indices = indices.flatten()
        
        chars = [self.idx_to_char.get(idx.item(), '') for idx in indices]
        return ''.join(chars)
    
    def encode(self, text: str) -> torch.Tensor:
        """
        Encode text to token indices.
        
        Args:
            text: Text string
        Returns:
            Token indices
        """
        return torch.tensor([self.char_to_idx.get(ch, 0) for ch in text], dtype=torch.long)


def get_dataloaders(
    text_path: str,
    batch_size: int,
    seq_len: int = 512,
    train_split: float = 0.9,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
):
    """
    Create train and validation dataloaders.
    
    Args:
        text_path: Path to text file
        batch_size: Batch size per GPU
        seq_len: Sequence length
        train_split: Fraction of data for training
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        distributed: Whether using distributed training
        world_size: Number of GPUs
        rank: Current GPU rank
    Returns:
        train_loader, val_loader, dataset
    """
    # Load full dataset
    dataset = CharDataset(text_path, seq_len)
    
    # Split into train and validation
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"Split dataset:")
    print(f"  - Train: {len(train_dataset):,} sequences")
    print(f"  - Val: {len(val_dataset):,} sequences")
    
    # Create samplers for distributed training
    if distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42
        )
        val_sampler = torch.utils.data.DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=42
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    return train_loader, val_loader, dataset

