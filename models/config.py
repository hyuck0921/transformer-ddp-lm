"""Model configuration classes."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TransformerConfig:
    """Configuration for Transformer language model."""
    
    # Architecture
    vocab_size: int = 256
    max_seq_len: int = 512
    dim: int = 256
    depth: int = 6
    heads: int = 8
    dim_head: int = 32
    mlp_dim: int = 1024
    dropout: float = 0.1
    use_rotary_emb: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.dim % self.heads == 0, "dim must be divisible by heads"
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.max_seq_len > 0, "max_seq_len must be positive"
        assert self.depth > 0, "depth must be positive"
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"
    
    @property
    def num_parameters(self) -> int:
        """Estimate number of parameters."""
        # Embedding
        embedding_params = self.vocab_size * self.dim
        
        # Transformer blocks
        # Self-attention: Q, K, V, Out projections
        attn_params = 4 * (self.dim * self.dim)
        # Layer norms
        ln_params = 2 * self.dim
        # MLP: two linear layers
        mlp_params = 2 * (self.dim * self.mlp_dim)
        
        # Per block
        block_params = attn_params + mlp_params + 2 * ln_params
        total_block_params = block_params * self.depth
        
        # Output head
        output_params = self.dim * self.vocab_size
        
        # Final layer norm
        final_ln_params = self.dim
        
        total = embedding_params + total_block_params + output_params + final_ln_params
        return total
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'TransformerConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'dim': self.dim,
            'depth': self.depth,
            'heads': self.heads,
            'dim_head': self.dim_head,
            'mlp_dim': self.mlp_dim,
            'dropout': self.dropout,
            'use_rotary_emb': self.use_rotary_emb,
        }


def get_model_config(config_name: str = 'small') -> TransformerConfig:
    """Get predefined model configuration."""
    
    configs = {
        'tiny': TransformerConfig(
            vocab_size=256,
            max_seq_len=256,
            dim=128,
            depth=4,
            heads=4,
            dim_head=32,
            mlp_dim=512,
            dropout=0.1,
        ),
        'small': TransformerConfig(
            vocab_size=256,
            max_seq_len=512,
            dim=256,
            depth=6,
            heads=8,
            dim_head=32,
            mlp_dim=1024,
            dropout=0.1,
        ),
        'medium': TransformerConfig(
            vocab_size=256,
            max_seq_len=1024,
            dim=512,
            depth=8,
            heads=8,
            dim_head=64,
            mlp_dim=2048,
            dropout=0.1,
        ),
        'large': TransformerConfig(
            vocab_size=256,
            max_seq_len=1024,
            dim=768,
            depth=12,
            heads=12,
            dim_head=64,
            mlp_dim=3072,
            dropout=0.1,
        ),
    }
    
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(configs.keys())}")
    
    return configs[config_name]

