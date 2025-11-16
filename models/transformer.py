"""Transformer Language Model implementation from scratch."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from .config import TransformerConfig


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        
        # Cache
        self._cached_freqs = None
        self._cached_len = 0
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sin/cos for rotary embedding."""
        if seq_len <= self._cached_len and self._cached_freqs is not None:
            return self._cached_freqs[:seq_len]
        
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()
        
        self._cached_freqs = (cos, sin)
        self._cached_len = seq_len
        
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to queries and keys."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Attention(nn.Module):
    """Multi-head self-attention with optional rotary embeddings."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.heads = config.heads
        self.dim_head = config.dim_head
        self.scale = self.dim_head ** -0.5
        
        inner_dim = self.dim_head * self.heads
        
        self.to_qkv = nn.Linear(config.dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        
        if config.use_rotary_emb:
            self.rotary_emb = RotaryEmbedding(self.dim_head, config.max_seq_len)
        else:
            self.rotary_emb = None
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len, seq_len) or None for causal mask
        Returns:
            (batch, seq_len, dim)
        """
        batch, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        
        # Apply rotary embeddings
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(seq_len, x.device)
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        # Apply causal mask
        if mask is None:
            # Create causal mask
            mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).triu(1)
        
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.dim, config.mlp_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.dim),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Single Transformer block with pre-norm."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn = Attention(config)
        self.ff = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    """Transformer Language Model (decoder-only, GPT-like)."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        
        # Positional embeddings (learned, not rotary)
        if not config.use_rotary_emb:
            self.pos_emb = nn.Embedding(config.max_seq_len, config.dim)
        else:
            self.pos_emb = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.dim)
        
        # Output head
        self.to_logits = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie weights
        self.to_logits.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices
            mask: Optional attention mask
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        batch, seq_len = x.shape
        
        # Token embeddings
        x = self.token_emb(x)
        
        # Add positional embeddings if not using rotary
        if self.pos_emb is not None:
            positions = torch.arange(seq_len, device=x.device)
            x = x + self.pos_emb(positions)[None, :, :]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final norm and project to logits
        x = self.norm(x)
        logits = self.to_logits(x)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate text from a prompt.
        
        Args:
            prompt: (batch, seq_len) initial tokens
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
        Returns:
            (batch, seq_len + max_length) generated tokens
        """
        self.eval()
        
        for _ in range(max_length):
            # Get predictions for the last token
            # Take only last max_seq_len tokens if prompt is too long
            prompt_cond = prompt if prompt.shape[1] <= self.config.max_seq_len else prompt[:, -self.config.max_seq_len:]
            
            logits = self(prompt_cond)
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to prompt
            prompt = torch.cat([prompt, next_token], dim=1)
        
        return prompt
    
    def num_parameters(self, non_embedding: bool = False) -> int:
        """Count the number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_emb.weight.numel()
            if self.pos_emb is not None:
                n_params -= self.pos_emb.weight.numel()
        return n_params

