"""PyTorch port of elastic/model.py from LargeWorldModel/ElasticTok.

Faithful 1:1 port of the ElasticTok model architecture. JAX/Flax
parallelism scaffolding (shard_map, ringattention, scan, remat, mesh
sharding constraints) is intentionally omitted — those are distributed
training plumbing, not part of the model behavior or training recipe.

What is preserved verbatim:
  - RMSNorm (pre-normalization, per-dim learned scale init to ones)
  - RoPE precompute + apply, theta-parameterized
  - Attention block: wq/wk/wv/wo dense projections, bias=False, normal
    init with `initializer_range`, RoPE on xq/xk, scaled dot-product,
    causal+segment+attention mask combination
  - MLP block: SwiGLU (w2(silu(w1(x)) * w3(x)))
  - TransformerBlock: pre-norm, x = x + attn(norm(x)); x = x + mlp(norm(x))
  - Encoder's is_kept_embed / is_masked_embed added to input based on
    encoding_mask (the ElasticTok mask-conditioning)
  - Pre-quant dense -> bottleneck (VAE or FSQ), post-quant dense
  - Decoder's final `tanh` activation on reconstruction
  - Masking-out latent at dropped positions: z = where(mask, z, 0)

Reference: https://github.com/LargeWorldModel/ElasticTok/blob/main/elastic/model.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.elastictok.bottleneck import get_bottleneck


# ---------------------------------------------------------------- Config

CONFIGS = {
    # `debug` and `200m` are the reference CONFIGS verbatim. The others
    # are additional size presets filling in the gap, all with the
    # same SwiGLU+RMSNorm+RoPE recipe.
    "debug": dict(
        hidden_size=256,
        intermediate_size=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_attention_heads=2,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
    ),
    "tiny": dict(
        hidden_size=384,
        intermediate_size=768,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_attention_heads=6,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
    ),
    "small": dict(
        hidden_size=512,
        intermediate_size=1024,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_attention_heads=8,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
    ),
    "base": dict(
        hidden_size=768,
        intermediate_size=1536,
        num_encoder_layers=8,
        num_decoder_layers=8,
        num_attention_heads=8,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
    ),
    "200m": dict(
        hidden_size=1024,
        intermediate_size=2048,
        num_encoder_layers=10,
        num_decoder_layers=10,
        num_attention_heads=8,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
    ),
    "large": dict(
        hidden_size=1280,
        intermediate_size=2560,
        num_encoder_layers=12,
        num_decoder_layers=12,
        num_attention_heads=10,
        initializer_range=0.02,
        rms_norm_eps=1e-5,
    ),
}


@dataclass
class ElasticTokConfig:
    """PyTorch port of ElasticTokConfig. Only the fields needed for the
    single-GPU model and training recipe — JAX sharding/scanning/remat
    fields omitted. Defaults match reference __init__.

    `mask_type`, `min_toks`, `max_toks`, `frames_per_block`,
    `lpips_loss_ratio`, `patch_size`, `bottleneck_type`, `fsq_quant_levels`,
    `vae_bottleneck_dim`, `max_sequence_length`, `theta`, `rms_norm_eps`,
    `initializer_range` — all preserved verbatim.
    """

    mask_type: str = "elastic"
    min_toks: int = 256
    max_toks: int = 2048
    frames_per_block: int = 1
    lpips_loss_ratio: float = 0.1

    patch_size: Tuple[int, int, int] = (1, 8, 8)
    # Input channels that the encoder's in_proj sees. Default 3 = RGB.
    # When Haar pre-compression is applied before patchify, set this to
    # 3 * 4**haar_levels (Haar quadruples channel count per level).
    in_channels: int = 3
    bottleneck_type: str = "fsq"
    fsq_quant_levels: Tuple[int, ...] = (8, 8, 8, 5, 5, 5)
    vae_bottleneck_dim: int = 8

    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_encoder_layers: int = 16
    num_decoder_layers: int = 16
    num_attention_heads: int = 32
    max_sequence_length: int = 4096
    theta: float = 10000.0

    rms_norm_eps: float = 1e-5
    initializer_range: float = 0.02

    @classmethod
    def load_config(cls, name_or_dict) -> "ElasticTokConfig":
        if isinstance(name_or_dict, str):
            if name_or_dict not in CONFIGS:
                raise ValueError(
                    f"Unknown config name {name_or_dict}; "
                    f"known: {list(CONFIGS)}")
            return cls(**CONFIGS[name_or_dict])
        return cls(**dict(name_or_dict))

    def update(self, updates: dict) -> None:
        for k, v in updates.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name)
                for f in self.__dataclass_fields__.values()}


# ---------------------------------------------------------------- RMSNorm

class RMSNorm(nn.Module):
    """Port of nn.RMSNorm. Learned `kernel` initialized to ones,
    normalization computed in fp32 and cast back."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # Ref name is 'kernel' but we keep conventional `weight` name in
        # PyTorch; init to ones per ref.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        out = (x32 * rms).to(in_dtype)
        return out * self.weight.to(in_dtype)


# ---------------------------------------------------------------- RoPE

def precompute_freqs_cis(dim: int, max_position_embedding: int,
                          theta: float = 10000.0,
                          dtype=torch.float32) -> torch.Tensor:
    """Port of precompute_freqs_cis. Returns complex tensor of shape
    (max_position_embedding, dim//2)."""
    freqs = 1.0 / (theta ** (
        torch.arange(0, dim, 2, dtype=dtype)[: (dim // 2)] / dim))
    t = torch.arange(max_position_embedding, dtype=dtype)
    freqs = torch.outer(t, freqs)
    # Complex representation: cos + i*sin
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return torch.complex(cos, sin)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                      freqs_cis: torch.Tensor) -> Tuple[torch.Tensor,
                                                         torch.Tensor]:
    """Port of apply_rotary_emb. xq/xk are (B, L, H, D). freqs_cis is
    (B, L, D//2) complex. Returns (xq_rot, xk_rot) in the same shape and
    dtype as inputs."""
    in_dtype = xq.dtype
    # Pair up last dim as (real, imag) and convert to complex.
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xq_c = torch.complex(xq_[..., 0], xq_[..., 1])
    xk_c = torch.complex(xk_[..., 0], xk_[..., 1])
    # Expand freqs_cis for head dim: (B, L, 1, D//2)
    freqs_cis = freqs_cis.reshape(
        *freqs_cis.shape[:2], 1, *freqs_cis.shape[2:])
    xq_out = xq_c * freqs_cis
    xk_out = xk_c * freqs_cis
    xq_out = torch.stack(
        [xq_out.real, xq_out.imag], dim=-1).reshape(*xq.shape)
    xk_out = torch.stack(
        [xk_out.real, xk_out.imag], dim=-1).reshape(*xk.shape)
    return xq_out.to(in_dtype), xk_out.to(in_dtype)


# ---------------------------------------------------------------- Attention

class Attention(nn.Module):
    """Port of Attention. Multi-head, RoPE, bias=False dense projections
    with normal(initializer_range) init. Causal block-masked attention
    over `max_toks`-sized blocks within the sequence (matches ref).
    """

    def __init__(self, config: ElasticTokConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        init = config.initializer_range

        self.wq = nn.Linear(self.embed_dim,
                             self.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.embed_dim,
                             self.num_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.embed_dim,
                             self.num_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        for m in (self.wq, self.wk, self.wv, self.wo):
            nn.init.normal_(m.weight, mean=0.0, std=init)

        freqs = precompute_freqs_cis(
            self.head_dim, config.max_sequence_length, theta=config.theta)
        self.register_buffer("freqs_cis", freqs, persistent=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-1], self.num_heads, self.head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape[:-2], self.embed_dim)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                segment_ids: torch.Tensor,
                position_ids: torch.Tensor) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        xq = self._split_heads(self.wq(hidden_states))
        xk = self._split_heads(self.wk(hidden_states))
        xv = self._split_heads(self.wv(hidden_states))

        # RoPE — gather per-position freqs (reference uses jnp.take).
        freqs = self.freqs_cis.to(hidden_states.device)[position_ids]  # (B, L, D//2)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs)

        # Build attention mask: attention_mask & segment_mask & causal_mask.
        # Ref builds all three as (1, 1, L, L)-broadcastable bool, ANDs them.
        att_mask = attention_mask[:, None, None, :]                       # (B,1,1,L)
        seg_mask = (segment_ids[:, :, None] == segment_ids[:, None, :]
                     )[:, None, :, :]                                     # (B,1,L,L)

        n_blocks = self.config.max_sequence_length // self.config.max_toks
        # causal_mask: (n_blocks, n_blocks) lower-triangular, repeat to (L, L)
        causal = torch.tril(
            torch.ones(n_blocks, n_blocks, dtype=torch.bool,
                       device=hidden_states.device))
        causal = causal.repeat_interleave(
            self.config.max_toks, dim=0).repeat_interleave(
                self.config.max_toks, dim=1)
        # Trim to actual L in case L < max_sequence_length
        causal = causal[:L, :L][None, None]                                # (1,1,L,L)

        mask = att_mask & seg_mask & causal                                # (B,1,L,L)

        # Scaled dot-product attention.
        # (B, H, L, D) layout
        xq_p = xq.transpose(1, 2)
        xk_p = xk.transpose(1, 2)
        xv_p = xv.transpose(1, 2)
        attn = F.scaled_dot_product_attention(
            xq_p, xk_p, xv_p, attn_mask=mask, dropout_p=0.0)
        attn = attn.transpose(1, 2).contiguous()                           # (B, L, H, D)

        attn = self._merge_heads(attn)
        return self.wo(attn)


# ---------------------------------------------------------------- MLP

class MLP(nn.Module):
    """Port: w2(silu(w1(x)) * w3(x)). SwiGLU with no bias."""

    def __init__(self, config: ElasticTokConfig):
        super().__init__()
        init = config.initializer_range
        self.w1 = nn.Linear(config.hidden_size,
                             config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size,
                             config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size,
                             config.intermediate_size, bias=False)
        for m in (self.w1, self.w2, self.w3):
            nn.init.normal_(m.weight, mean=0.0, std=init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------- Block

class TransformerBlock(nn.Module):
    """Pre-norm: x = x + attn(norm(x)); x = x + mlp(norm(x))."""

    def __init__(self, config: ElasticTokConfig):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = MLP(config)
        self.attention_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                segment_ids: torch.Tensor,
                position_ids: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(
            self.attention_norm(hidden_states),
            attention_mask, segment_ids, position_ids)
        hidden_states = hidden_states + attn_out
        ff_out = self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states + ff_out


class TransformerBlockCollection(nn.Module):
    """Sequence of TransformerBlocks, no scan/remat (those are JAX-only)."""

    def __init__(self, config: ElasticTokConfig, depth_attr: str):
        super().__init__()
        n = getattr(config, depth_attr)
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(n)])

    def forward(self, hidden_states, attention_mask,
                segment_ids, position_ids):
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states, attention_mask, segment_ids, position_ids)
        return hidden_states


# ---------------------------------------------------------------- Encoder / Decoder

class ElasticTokEncoder(nn.Module):
    """Encoder: in_proj -> + kept/masked embed -> blocks -> RMSNorm ->
    pre_quant -> bottleneck. Matches reference flow exactly, including
    variance_scaling init for is_kept/is_masked embeds."""

    def __init__(self, config: ElasticTokConfig):
        super().__init__()
        self.config = config
        init = config.initializer_range
        patch_dim = int(np.prod(config.patch_size) * config.in_channels)

        self.in_proj = nn.Linear(patch_dim, config.hidden_size, bias=False)
        nn.init.normal_(self.in_proj.weight, mean=0.0, std=init)

        # variance_scaling(1.0, "fan_in", "normal", out_axis=0) with
        # shape (1, hidden_size): fan_in = hidden_size (all axes except
        # out_axis=0), so std = sqrt(1/hidden_size). I originally ported
        # this as torch.randn (std=1.0), which is 16x too large for
        # hidden=256 — the embedding dominated the per-token vision
        # signal and the model couldn't learn per-patch variation.
        _embed_std = (1.0 / config.hidden_size) ** 0.5
        self.is_kept_embed = nn.Parameter(
            torch.randn(1, config.hidden_size) * _embed_std)
        self.is_masked_embed = nn.Parameter(
            torch.randn(1, config.hidden_size) * _embed_std)

        self.encoder_blocks = TransformerBlockCollection(
            config, "num_encoder_layers")
        self.ln_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.bottleneck = get_bottleneck(config)
        self.pre_quant = nn.Linear(
            config.hidden_size, self.bottleneck.proj_dim, bias=False)
        nn.init.normal_(self.pre_quant.weight, mean=0.0, std=init)

    def indexes_to_codes(self, z):
        return self.bottleneck.indexes_to_codes(z)

    def codes_to_indexes(self, z):
        return self.bottleneck.codes_to_indexes(z)

    def forward(self, vision: torch.Tensor,
                encoding_mask: torch.Tensor,
                attention_mask: torch.Tensor,
                segment_ids: torch.Tensor,
                position_ids: torch.Tensor,
                training: bool = True):
        x = self.in_proj(vision)
        # ref: input_embeds += where(encoding_mask[..., None], kept, masked)
        em = encoding_mask[..., None]
        x = x + torch.where(em, self.is_kept_embed, self.is_masked_embed)
        x = self.encoder_blocks(
            x, attention_mask, segment_ids, position_ids)
        x = self.ln_f(x)
        z = self.pre_quant(x)
        return self.bottleneck(
            z, encoding_mask, rng=(True if training else None))


class ElasticTokDecoder(nn.Module):
    """Decoder: post_quant -> blocks -> RMSNorm -> out_proj -> tanh."""

    def __init__(self, config: ElasticTokConfig):
        super().__init__()
        self.config = config
        init = config.initializer_range

        # Decoder takes the post-bottleneck z as input. For VAE that's
        # `bottleneck_dim`; for FSQ it's `len(levels)`. Both exposed via
        # `out_dim` on the bottleneck module.
        bottleneck = get_bottleneck(config)
        self.post_quant = nn.Linear(
            bottleneck.out_dim, config.hidden_size, bias=False)
        nn.init.normal_(self.post_quant.weight, mean=0.0, std=init)

        self.decoder_blocks = TransformerBlockCollection(
            config, "num_decoder_layers")
        self.ln_f = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        patch_dim = int(np.prod(config.patch_size) * config.in_channels)
        self.out_proj = nn.Linear(config.hidden_size, patch_dim, bias=False)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=init)

    def forward(self, z: torch.Tensor,
                encoding_mask: torch.Tensor,
                attention_mask: torch.Tensor,
                segment_ids: torch.Tensor,
                position_ids: torch.Tensor) -> torch.Tensor:
        h = self.post_quant(z)
        h = self.decoder_blocks(
            h, attention_mask, segment_ids, position_ids)
        h = self.ln_f(h)
        recon = self.out_proj(h)
        return torch.tanh(recon)


class ElasticTok(nn.Module):
    """Top-level wrapper. Matches reference __call__ / encode / decode /
    recon_with_mask behaviors."""

    def __init__(self, config: ElasticTokConfig):
        super().__init__()
        self.config = config
        self.encoder = ElasticTokEncoder(config)
        self.decoder = ElasticTokDecoder(config)

    def index_to_codes(self, z):
        return self.encoder.indexes_to_codes(z)

    def codes_to_indexes(self, z):
        return self.encoder.codes_to_indexes(z)

    def encode(self, vision, encoding_mask, attention_mask,
                segment_ids, position_ids, training: bool = True):
        z, stats = self.encoder(
            vision, encoding_mask, attention_mask,
            segment_ids, position_ids, training=training)
        # ref: z = jnp.where(encoding_mask[..., None], z, 0)
        z = torch.where(encoding_mask[..., None], z, torch.zeros_like(z))
        return z, stats

    def decode(self, z, encoding_mask, attention_mask,
                segment_ids, position_ids):
        return self.decoder(
            z, encoding_mask, attention_mask, segment_ids, position_ids)

    def forward(self, vision, encoding_mask, attention_mask,
                segment_ids, position_ids, training: bool = True,
                return_z: bool = False):
        assert vision.shape[1] <= self.config.max_sequence_length
        z, stats = self.encode(
            vision, encoding_mask, attention_mask,
            segment_ids, position_ids, training=training)
        recon = self.decoder(
            z, encoding_mask, attention_mask, segment_ids, position_ids)
        if return_z:
            return recon, stats, self.codes_to_indexes(z)
        return recon, stats
