"""Small transformer primitives used by ElasticVideoTokenizer.

Kept minimal and self-contained (no imports from core.model) to avoid
coupling the tokenizer to the VAE's internals. RMSNorm, an MLP with
SiLU, and multi-head attention with optional additive mask.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root-mean-square layer norm — cheaper than LayerNorm."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x.to(self.scale.dtype) * self.scale)


class MLP(nn.Module):
    """Two-layer MLP with SiLU activation. Standard transformer FFN."""

    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        hidden = int(dim * mult)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.fc2(F.silu(self.fc1(x))))


class Attention(nn.Module):
    """Multi-head scaled dot-product attention.

    Supports an optional additive mask of shape (L_q, L_k) or broadcastable
    to the final attention-score tensor. Mask values are added before
    softmax, so use -inf (or a large negative) to block a position.
    """

    def __init__(self, dim, heads=6, dropout=0.0):
        super().__init__()
        assert dim % heads == 0, f"dim {dim} not divisible by heads {heads}"
        self.heads = heads
        self.d_head = dim // heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, mask=None):
        # x: (B, L, D)
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L, L)
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)
        out = attn @ v                               # (B, H, L, d_head)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.drop(self.proj(out))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: RMSNorm → Attn → + → RMSNorm → MLP → +."""

    def __init__(self, dim, heads, mlp_mult=4, dropout=0.0):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = Attention(dim, heads=heads, dropout=dropout)
        self.n2 = RMSNorm(dim)
        self.mlp = MLP(dim, mult=mlp_mult, dropout=dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.n1(x), mask=mask)
        x = x + self.mlp(self.n2(x))
        return x


class TransformerStack(nn.Module):
    """Sequence of TransformerBlocks. Accepts an optional shared mask."""

    def __init__(self, dim, depth, heads, mlp_mult=4, dropout=0.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_mult=mlp_mult, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, x, mask=None):
        for blk in self.blocks:
            x = blk(x, mask=mask)
        return x


class FactorizedPosEmb(nn.Module):
    """Learnable additive positional embedding factorized as pos_t + pos_h + pos_w.

    Allocated lazily on first call based on the grid shape it sees. The
    underlying tables are sized to a generous cap so grids up to that
    size work without reallocation; call `.reset(T, H, W, dim)` ahead of
    time if you want fixed allocation.
    """

    def __init__(self, dim, max_t=64, max_h=128, max_w=128):
        super().__init__()
        self.dim = dim
        self.max_t, self.max_h, self.max_w = max_t, max_h, max_w
        self.pos_t = nn.Parameter(torch.randn(max_t, dim) * 0.02)
        self.pos_h = nn.Parameter(torch.randn(max_h, dim) * 0.02)
        self.pos_w = nn.Parameter(torch.randn(max_w, dim) * 0.02)

    def forward(self, T, H, W, device=None, dtype=None):
        """Return (T*H*W, dim) additive embedding matrix."""
        if T > self.max_t or H > self.max_h or W > self.max_w:
            raise RuntimeError(
                f"FactorizedPosEmb: grid ({T},{H},{W}) exceeds caps "
                f"({self.max_t},{self.max_h},{self.max_w}). Increase max_* at init.")
        pt = self.pos_t[:T].view(T, 1, 1, self.dim)
        ph = self.pos_h[:H].view(1, H, 1, self.dim)
        pw = self.pos_w[:W].view(1, 1, W, self.dim)
        emb = (pt + ph + pw).reshape(T * H * W, self.dim)
        if device is not None and emb.device != device:
            emb = emb.to(device)
        if dtype is not None and emb.dtype != dtype:
            emb = emb.to(dtype)
        return emb
