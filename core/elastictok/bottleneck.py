"""PyTorch port of elastic/bottleneck.py from LargeWorldModel/ElasticTok.

Verbatim-faithful port of DiagonalGaussianDistribution + VAE + FSQ + get_bottleneck.
JAX primitives (jnp.*, jax.random, stop_gradient) are translated to their PyTorch
equivalents 1:1. Behavior matches the reference — especially:

- VAE KL computation formula and block aggregation.
- FSQ bound/round_ste/codes_to_indexes/indexes_to_codes including the
  exact `half_l * (1-eps)` bound, odd/even offset, and shift math.

Reference: https://github.com/LargeWorldModel/ElasticTok/blob/main/elastic/bottleneck.py
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class DiagonalGaussianDistribution:
    """Direct port. `parameters` is (..., 2*bottleneck_dim) concatenation of
    [mean, logvar] along the last axis (matches jnp.split(..., 2, axis=-1))."""

    def __init__(self, parameters: torch.Tensor):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=-1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self) -> torch.Tensor:
        # JAX `x = self.mean + self.std * jax.random.normal(...)` — use
        # torch.randn_like for equivalent behavior.
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, mask: torch.Tensor, block_size: int) -> torch.Tensor:
        # Reference (JAX):
        #   ndim = self.mean.ndim; assert ndim == 3
        #   B, L = self.mean.shape[:2]
        #   n_blocks = L // block_size
        #   kl = self.mean**2 + self.var - 1 - self.logvar
        #   mask = mask.reshape(B, n_blocks, block_size)
        #   kl = kl.reshape(B, n_blocks, block_size, kl.shape[-1])
        #   kl = 0.5 * jnp.sum(kl * mask[..., None], axis=(2, 3))
        #   kl = jnp.mean(kl, axis=1)
        #   return kl
        assert self.mean.ndim == 3, self.mean.shape
        B, L = self.mean.shape[:2]
        n_blocks = L // block_size
        kl = self.mean.pow(2) + self.var - 1.0 - self.logvar
        mask = mask.reshape(B, n_blocks, block_size)
        kl = kl.reshape(B, n_blocks, block_size, kl.shape[-1])
        kl = 0.5 * (kl * mask[..., None]).sum(dim=(2, 3))
        kl = kl.mean(dim=1)
        return kl

    def mode(self) -> torch.Tensor:
        return self.mean


class FSQ(nn.Module):
    """Direct port of the FSQ quantizer.

    In the reference FSQ is not an nn.Module (JAX doesn't need that), but
    in PyTorch we make it an nn.Module so codebook bookkeeping buffers live
    on the right device. No learned parameters.
    """

    def __init__(self, levels):
        super().__init__()
        self._levels = tuple(int(x) for x in levels)
        levels_np = np.asarray(self._levels, dtype=np.int64)
        basis_np = np.concatenate(
            ([1], np.cumprod(levels_np[:-1]))
        ).astype(np.uint32)
        self.register_buffer(
            "_levels_buf", torch.tensor(levels_np, dtype=torch.float32),
            persistent=False)
        self.register_buffer(
            "_basis_buf", torch.tensor(basis_np.astype(np.int64),
                                        dtype=torch.long),
            persistent=False)

    @property
    def n_codes(self) -> int:
        return int(np.prod(self._levels))

    @property
    def proj_dim(self) -> int:
        # Input dim to the bottleneck (size of `pre_quant` output).
        return len(self._levels)

    @property
    def out_dim(self) -> int:
        # Output dim of the bottleneck (size of `post_quant` input).
        return len(self._levels)

    def _round_ste(self, z: torch.Tensor) -> torch.Tensor:
        # JAX: `z + jax.lax.stop_gradient(jnp.round(z) - z)`
        z_hat = torch.round(z)
        return z + (z_hat - z).detach()

    def bound(self, z: torch.Tensor) -> torch.Tensor:
        # JAX formula:
        #   eps = 1e-3
        #   half_l = (levels - 1) * (1 - eps) / 2
        #   offset = where(levels % 2 == 1, 0.0, 0.5)
        #   shift = tan(offset / half_l)
        #   return tanh(z + shift) * half_l - offset
        eps = 1e-3
        levels = self._levels_buf.to(z.device, z.dtype)
        half_l = (levels - 1.0) * (1.0 - eps) / 2.0
        offset = torch.where(
            (levels.to(torch.long) % 2 == 1),
            torch.zeros_like(levels), torch.full_like(levels, 0.5))
        shift = torch.tan(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        z = self.bound(z)
        quantized = self._round_ste(z)
        half_width = (self._levels_buf.to(z.device) // 2).to(z.dtype)
        return quantized / half_width

    def forward(self, z: torch.Tensor, encoding_mask: torch.Tensor,
                rng=None):
        """Returns (quantized, stats dict). Matches reference __call__."""
        z_st = self.quantize(z)
        code_idxs = self.codes_to_indexes(z_st).reshape(-1)
        codes_one_hot = torch.nn.functional.one_hot(
            code_idxs.to(torch.long), num_classes=self.n_codes).to(z.dtype)
        avg_probs = codes_one_hot.mean(dim=0)
        perplexity = torch.exp(
            -(avg_probs * torch.log(avg_probs + 1e-10)).sum())
        usage = (codes_one_hot.sum(dim=0) >= 1).float().mean()
        return z_st, dict(
            perplexity=perplexity, codebook_usage=usage,
            # FSQ has no aux_loss (only VAE does); provide zero so the
            # training loop doesn't special-case.
            aux_loss=torch.zeros((), device=z.device, dtype=z.dtype))

    def _scale_and_shift(self, zhat_normalized: torch.Tensor) -> torch.Tensor:
        half_width = (self._levels_buf.to(zhat_normalized.device) // 2).to(
            zhat_normalized.dtype)
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: torch.Tensor) -> torch.Tensor:
        half_width = (self._levels_buf.to(zhat.device) // 2).to(zhat.dtype)
        return (zhat - half_width) / half_width

    def codes_to_indexes(self, zhat: torch.Tensor) -> torch.Tensor:
        assert zhat.shape[-1] == len(self._levels)
        zhat = self._scale_and_shift(zhat)
        basis = self._basis_buf.to(zhat.device).to(zhat.dtype)
        return (zhat * basis).sum(dim=-1).to(torch.long)

    def indexes_to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        indices = indices.unsqueeze(-1)
        basis = self._basis_buf.to(indices.device)
        levels = self._levels_buf.to(indices.device).to(torch.long)
        codes_non_centered = torch.fmod(
            torch.div(indices, basis, rounding_mode="floor"), levels)
        return self._scale_and_shift_inverse(codes_non_centered.float())


class VAE(nn.Module):
    """Direct port. No learned params — it's just a wrapper that interprets
    the pre-quant tensor as (mean, logvar) and computes KL aux_loss.

    NOTE: reference `bottleneck.py` hardcodes `posterior.kl(encoding_mask,
    2048)` with a `# TODO hardocded` comment. 2048 corresponds to the
    reference's `max_toks`. We plumb it through from the config so the
    code also works at other sequence lengths — same semantics as the
    intended-but-unimplemented parameterization.
    """

    def __init__(self, bottleneck_dim: int, block_size: int = 2048):
        super().__init__()
        self.bottleneck_dim = int(bottleneck_dim)
        self.block_size = int(block_size)

    @property
    def n_codes(self):
        raise NotImplementedError

    @property
    def proj_dim(self) -> int:
        # Input dim to the bottleneck: [mean; logvar] concat.
        return 2 * self.bottleneck_dim

    @property
    def out_dim(self) -> int:
        # Output dim after sample/mode: just the sample itself.
        return self.bottleneck_dim

    def forward(self, z: torch.Tensor, encoding_mask: torch.Tensor,
                rng=None):
        """Returns (sampled_or_mode, stats dict with 'aux_loss')."""
        posterior = DiagonalGaussianDistribution(z)
        if rng is None:
            sampled = posterior.mode()
        else:
            sampled = posterior.sample()
        kl = posterior.kl(encoding_mask, self.block_size)
        kl_loss = 1e-8 * kl.sum() / max(kl.shape[0], 1)
        return sampled, dict(aux_loss=kl_loss)

    def codes_to_indexes(self, z):
        return z

    def indexes_to_codes(self, z):
        return z


def get_bottleneck(config) -> nn.Module:
    """Factory matching reference `get_bottleneck(config)`."""
    if config.bottleneck_type == "fsq":
        return FSQ(config.fsq_quant_levels)
    elif config.bottleneck_type == "vae":
        # Plumb max_toks as block_size (replaces the hardcoded 2048 in ref).
        return VAE(config.vae_bottleneck_dim,
                    block_size=int(getattr(config, "max_toks", 2048)))
    else:
        raise ValueError(
            f"Unknown bottleneck_type: {config.bottleneck_type}")
