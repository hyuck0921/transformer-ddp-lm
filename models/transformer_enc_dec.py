"""Encoder-Decoder Transformer implementation (original Transformer architecture)."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange

from .config import TransformerConfig


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, dim: int, max_seq_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        
        pe = torch.zeros(max_seq_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.dim = config.dim
        self.heads = config.heads
        self.dim_head = config.dim_head
        self.scale = self.dim_head ** -0.5
        
        inner_dim = self.dim_head * self.heads
        
        self.to_q = nn.Linear(config.dim, inner_dim, bias=False)
        self.to_k = nn.Linear(config.dim, inner_dim, bias=False)
        self.to_v = nn.Linear(config.dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            q: Query (batch, seq_len_q, dim)
            k: Key (batch, seq_len_k, dim)
            v: Value (batch, seq_len_v, dim)
            mask: Attention mask (batch, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
        Returns:
            (batch, seq_len_q, dim)
        """
        batch_size = q.shape[0]
        
        # Project and split heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)
        
        # Scaled dot-product attention
        scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
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
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_dim, config.dim),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    """Single Transformer encoder layer."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: Padding mask (batch, seq_len, seq_len)
        Returns:
            (batch, seq_len, dim)
        """
        # Self-attention with residual connection
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class DecoderLayer(nn.Module):
    """Single Transformer decoder layer."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attn = MultiHeadAttention(config)
        self.cross_attn = MultiHeadAttention(config)
        self.ff = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.dim)
        self.norm2 = nn.LayerNorm(config.dim)
        self.norm3 = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Decoder input (batch, tgt_len, dim)
            encoder_out: Encoder output (batch, src_len, dim)
            self_attn_mask: Causal mask for decoder self-attention
            cross_attn_mask: Padding mask for cross-attention
        Returns:
            (batch, tgt_len, dim)
        """
        # Masked self-attention (causal)
        self_attn_out = self.self_attn(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(self_attn_out))
        
        # Cross-attention to encoder output
        cross_attn_out = self.cross_attn(x, encoder_out, encoder_out, cross_attn_mask)
        x = self.norm2(x + self.dropout(cross_attn_out))
        
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_enc = PositionalEncoding(config.dim, config.max_seq_len)
        
        self.layers = nn.ModuleList([
            EncoderLayer(config) for _ in range(config.depth)
        ])
        
        self.norm = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token indices
            mask: Padding mask (batch, seq_len)
        Returns:
            (batch, seq_len, dim) encoder output
        """
        # Token embeddings + positional encoding
        x = self.token_emb(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        
        # Create attention mask from padding mask
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            attn_mask = mask.unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = None
        
        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, attn_mask)
        
        x = self.norm(x)
        return x


class TransformerDecoder(nn.Module):
    """Transformer Decoder."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.pos_enc = PositionalEncoding(config.dim, config.max_seq_len)
        
        self.layers = nn.ModuleList([
            DecoderLayer(config) for _ in range(config.depth)
        ])
        
        self.norm = nn.LayerNorm(config.dim)
        self.dropout = nn.Dropout(config.dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, tgt_len) target token indices
            encoder_out: (batch, src_len, dim) encoder output
            tgt_mask: Target padding mask (batch, tgt_len)
            src_mask: Source padding mask (batch, src_len)
        Returns:
            (batch, tgt_len, dim) decoder output
        """
        batch_size, tgt_len = x.shape
        
        # Token embeddings + positional encoding
        x = self.token_emb(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        
        # Create causal mask for decoder self-attention
        causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=x.device, dtype=torch.bool),
            diagonal=1
        )
        
        # Combine with target padding mask if provided
        if tgt_mask is not None:
            tgt_padding_mask = tgt_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)
            self_attn_mask = causal_mask.unsqueeze(0) | tgt_padding_mask
        else:
            self_attn_mask = causal_mask.unsqueeze(0)
        
        # Create cross-attention mask from source padding
        if src_mask is not None:
            cross_attn_mask = src_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
        else:
            cross_attn_mask = None
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, encoder_out, self_attn_mask, cross_attn_mask)
        
        x = self.norm(x)
        return x


class EncoderDecoderTransformer(nn.Module):
    """Full Encoder-Decoder Transformer model (original Transformer architecture)."""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(config)
        
        # Output projection
        self.to_logits = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie weights between decoder embedding and output projection
        self.to_logits.weight = self.decoder.token_emb.weight
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: (batch, src_len) source token indices
            tgt: (batch, tgt_len) target token indices
            src_mask: Source padding mask (batch, src_len)
            tgt_mask: Target padding mask (batch, tgt_len)
        Returns:
            (batch, tgt_len, vocab_size) logits
        """
        # Encode source
        encoder_out = self.encoder(src, src_mask)
        
        # Decode to target
        decoder_out = self.decoder(tgt, encoder_out, tgt_mask, src_mask)
        
        # Project to vocabulary
        logits = self.to_logits(decoder_out)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        max_length: int = 100,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Greedy or sampling-based decoding.
        
        Args:
            src: (batch, src_len) source token indices
            max_length: Maximum generation length
            bos_token_id: Beginning of sequence token
            eos_token_id: End of sequence token
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            src_mask: Source padding mask
        Returns:
            (batch, gen_len) generated token indices
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device
        
        # Encode source once
        encoder_out = self.encoder(src, src_mask)
        
        # Start with BOS token
        tgt = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length):
            # Decode
            decoder_out = self.decoder(tgt, encoder_out, src_mask=src_mask)
            logits = self.to_logits(decoder_out[:, -1, :])  # (batch, vocab_size)
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # Append to sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Check for EOS
            finished = finished | (next_token.squeeze(-1) == eos_token_id)
            if finished.all():
                break
        
        return tgt
    
    def num_parameters(self, non_embedding: bool = False) -> int:
        """Count the number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.encoder.token_emb.weight.numel()
            n_params -= self.decoder.token_emb.weight.numel()
        return n_params

