"""Seq2Seq dataset for Encoder-Decoder Transformer."""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple


class Seq2SeqDataset(Dataset):
    """
    Sequence-to-Sequence dataset.
    
    For demo purposes, creates synthetic tasks:
    - Reversal: Reverse the input sequence
    - Copy: Copy the input sequence
    - Addition: Add two numbers (e.g., "12+34" -> "46")
    """
    
    def __init__(
        self,
        num_samples: int = 10000,
        seq_len: int = 20,
        task: str = 'reversal',
        vocab_size: int = 256
    ):
        """
        Args:
            num_samples: Number of training samples
            seq_len: Maximum sequence length
            task: Task type ('reversal', 'copy', 'addition')
            vocab_size: Vocabulary size
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.task = task
        self.vocab_size = vocab_size
        
        # Special tokens
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.start_token_id = 3
        
        # Create vocabulary
        self.idx_to_char = {i: chr(i) for i in range(vocab_size)}
        self.char_to_idx = {chr(i): i for i in range(vocab_size)}
        
        # Special token mapping
        self.idx_to_char[self.pad_token_id] = '<PAD>'
        self.idx_to_char[self.bos_token_id] = '<BOS>'
        self.idx_to_char[self.eos_token_id] = '<EOS>'
        self.idx_to_char[self.start_token_id] = '<START>'
        
        print(f"Seq2Seq Dataset created:")
        print(f"  - Task: {task}")
        print(f"  - Samples: {num_samples}")
        print(f"  - Max sequence length: {seq_len}")
        print(f"  - Vocabulary size: {vocab_size}")
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a source-target pair.
        
        Returns:
            source: (src_len,) source sequence with EOS
            target: (tgt_len,) target sequence with BOS and EOS
        """
        if self.task == 'reversal':
            return self._generate_reversal()
        elif self.task == 'copy':
            return self._generate_copy()
        elif self.task == 'addition':
            return self._generate_addition()
        else:
            raise ValueError(f"Unknown task: {self.task}")
    
    def _generate_reversal(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a sequence reversal task."""
        # Random sequence length
        length = torch.randint(5, self.seq_len - 2, (1,)).item()
        
        # Random sequence (avoid special tokens)
        src_tokens = torch.randint(
            self.start_token_id + 1,
            self.vocab_size,
            (length,)
        )
        
        # Source: tokens + EOS
        src = torch.cat([src_tokens, torch.tensor([self.eos_token_id])])
        
        # Target: BOS + reversed tokens + EOS
        tgt_tokens = torch.flip(src_tokens, [0])
        tgt = torch.cat([
            torch.tensor([self.bos_token_id]),
            tgt_tokens,
            torch.tensor([self.eos_token_id])
        ])
        
        return src, tgt
    
    def _generate_copy(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a sequence copy task."""
        # Random sequence length
        length = torch.randint(5, self.seq_len - 2, (1,)).item()
        
        # Random sequence
        src_tokens = torch.randint(
            self.start_token_id + 1,
            self.vocab_size,
            (length,)
        )
        
        # Source: tokens + EOS
        src = torch.cat([src_tokens, torch.tensor([self.eos_token_id])])
        
        # Target: BOS + same tokens + EOS
        tgt = torch.cat([
            torch.tensor([self.bos_token_id]),
            src_tokens,
            torch.tensor([self.eos_token_id])
        ])
        
        return src, tgt
    
    def _generate_addition(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate an addition task (e.g., "12+34" -> "46")."""
        # Random two numbers
        num1 = torch.randint(1, 100, (1,)).item()
        num2 = torch.randint(1, 100, (1,)).item()
        result = num1 + num2
        
        # Convert to strings
        src_str = f"{num1}+{num2}"
        tgt_str = str(result)
        
        # Encode
        src = self.encode(src_str)
        src = torch.cat([src, torch.tensor([self.eos_token_id])])
        
        tgt_tokens = self.encode(tgt_str)
        tgt = torch.cat([
            torch.tensor([self.bos_token_id]),
            tgt_tokens,
            torch.tensor([self.eos_token_id])
        ])
        
        return src, tgt
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token indices."""
        indices = []
        for char in text:
            idx = ord(char)
            if idx < self.vocab_size:
                indices.append(idx)
        return torch.tensor(indices, dtype=torch.long)
    
    def decode(self, indices: torch.Tensor) -> str:
        """Decode token indices to text."""
        if indices.dim() > 1:
            indices = indices.flatten()
        
        chars = []
        for idx in indices:
            idx_val = idx.item()
            if idx_val == self.pad_token_id:
                continue
            elif idx_val == self.bos_token_id:
                chars.append('<BOS>')
            elif idx_val == self.eos_token_id:
                chars.append('<EOS>')
            elif idx_val < self.vocab_size:
                chars.append(chr(idx_val))
        
        return ''.join(chars)


def collate_fn(batch):
    """
    Collate function for batching variable-length sequences.
    
    Args:
        batch: List of (src, tgt) tuples
    Returns:
        src_batch: (batch, max_src_len) padded source sequences
        tgt_batch: (batch, max_tgt_len) padded target sequences
        src_mask: (batch, max_src_len) source padding mask (True for padding)
        tgt_mask: (batch, max_tgt_len) target padding mask
    """
    sources, targets = zip(*batch)
    
    # Get max lengths
    max_src_len = max(len(s) for s in sources)
    max_tgt_len = max(len(t) for t in targets)
    
    batch_size = len(sources)
    pad_token_id = 0
    
    # Initialize tensors
    src_batch = torch.full((batch_size, max_src_len), pad_token_id, dtype=torch.long)
    tgt_batch = torch.full((batch_size, max_tgt_len), pad_token_id, dtype=torch.long)
    
    src_mask = torch.ones(batch_size, max_src_len, dtype=torch.bool)
    tgt_mask = torch.ones(batch_size, max_tgt_len, dtype=torch.bool)
    
    # Fill tensors
    for i, (src, tgt) in enumerate(zip(sources, targets)):
        src_len = len(src)
        tgt_len = len(tgt)
        
        src_batch[i, :src_len] = src
        tgt_batch[i, :tgt_len] = tgt
        
        src_mask[i, :src_len] = False
        tgt_mask[i, :tgt_len] = False
    
    return src_batch, tgt_batch, src_mask, tgt_mask


def get_seq2seq_dataloaders(
    num_train: int = 10000,
    num_val: int = 1000,
    batch_size: int = 32,
    seq_len: int = 20,
    task: str = 'reversal',
    vocab_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
):
    """
    Create train and validation dataloaders for Seq2Seq.
    
    Args:
        num_train: Number of training samples
        num_val: Number of validation samples
        batch_size: Batch size per GPU
        seq_len: Maximum sequence length
        task: Task type ('reversal', 'copy', 'addition')
        vocab_size: Vocabulary size
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        distributed: Whether using distributed training
        world_size: Number of GPUs
        rank: Current GPU rank
    Returns:
        train_loader, val_loader, train_dataset, val_dataset
    """
    # Create datasets
    train_dataset = Seq2SeqDataset(
        num_samples=num_train,
        seq_len=seq_len,
        task=task,
        vocab_size=vocab_size
    )
    
    val_dataset = Seq2SeqDataset(
        num_samples=num_val,
        seq_len=seq_len,
        task=task,
        vocab_size=vocab_size
    )
    
    print(f"\nDataset split:")
    print(f"  - Train: {len(train_dataset):,} samples")
    print(f"  - Val: {len(val_dataset):,} samples")
    
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
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=False,
    )
    
    return train_loader, val_loader, train_dataset, val_dataset

