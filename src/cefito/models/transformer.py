"""Minimal pre-norm transformer used by the predictor P_theta."""

import torch
import torch.nn.functional as F
from torch import nn


class Mlp(nn.Module):
    """Two-layer feed-forward block with a tanh-approximated GELU."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Attention(nn.Module):
    """Multi-head self-attention, optionally causal over the token sequence."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, causal: bool = True):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"hidden dim {dim} is not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.causal = causal
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, dim // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, heads, tokens, head_dim]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class TransformerBlock(nn.Module):
    """Pre-norm attention + MLP residual block.

    The layer norms are parameter free: the block is conditioned purely through
    the token sequence it is given, not through modulation.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, causal: bool = True):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, causal=causal)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x))
        return x + self.mlp(self.norm_mlp(x))


def sinusoidal_positional_embedding(num_tokens: int, dim: int) -> torch.Tensor:
    """Fixed 1-D sin/cos positional embeddings, ``[num_tokens, dim]``."""
    if dim % 2:
        raise ValueError(f"positional embedding dim must be even, got {dim}")
    positions = torch.arange(num_tokens, dtype=torch.float64)
    frequencies = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim / 2))
    angles = torch.outer(positions, frequencies)
    return torch.cat([angles.sin(), angles.cos()], dim=1).float()
