#!/usr/bin/env python3
"""Finite Scalar Quantization (FSQ).

Quantizes continuous latents into discrete tokens using per-dimension
rounding with straight-through estimator. No codebook, no learnable
parameters — codes are implicit from the quantized scalar values.

Reference: Mentzer et al., "Finite Scalar Quantization: VQ-VAE Made Simple"
https://arxiv.org/abs/2309.15505
"""

import torch
import torch.nn as nn
import math


class FSQ(nn.Module):
    """Finite Scalar Quantization layer.

    Bounds each dimension via tanh, rounds to nearest integer,
    and applies straight-through estimator for gradients.

    No learnable parameters.

    Args:
        levels: list of ints, quantization levels per dimension.
                e.g. [8,8,8,8,8,8] for 6 dims with 8 levels each,
                or [3,5,7,9,11] for heterogeneous levels.
    """

    def __init__(self, levels):
        super().__init__()
        self._levels = levels
        self.dim = len(levels)
        self.codebook_size = math.prod(levels)

        # Half-levels for bounding: (L-1)/2 per dimension
        half_levels = torch.tensor([(L - 1) / 2 for L in levels])
        self.register_buffer("_half_levels", half_levels)

        # Multipliers for index computation
        mults = []
        acc = 1
        for L in levels:
            mults.append(acc)
            acc *= L
        self.register_buffer("_mults", torch.tensor(mults, dtype=torch.long))

    @property
    def levels(self):
        return list(self._levels)

    @property
    def num_codes(self):
        return self.codebook_size

    def forward(self, z):
        """Quantize latent tensor with straight-through estimator.

        Args:
            z: (B, D, ...) continuous latent where D = self.dim.

        Returns:
            z_quant: same shape, quantized (gradients flow via STE).
            indices: (B, ...) integer indices in [0, codebook_size).
        """
        # Move channels to last dim for per-dimension ops
        # (B, D, ...) -> (B, ..., D)
        perm_fwd = [0] + list(range(2, z.ndim)) + [1]
        perm_back = [0, z.ndim - 1] + list(range(1, z.ndim - 1))
        z_last = z.permute(*perm_fwd)

        # Bound: tanh -> scale to [-(L-1)/2, (L-1)/2] per dim
        half = self._half_levels
        z_bounded = half * torch.tanh(z_last)

        # Round to nearest integer
        z_rounded = torch.round(z_bounded)

        # Straight-through estimator
        z_quant = z_bounded + (z_rounded - z_bounded).detach()

        # Compute flat indices
        dim_indices = (z_rounded + half).long()
        indices = (dim_indices * self._mults).sum(dim=-1)

        # Back to channel-first: (B, ..., D) -> (B, D, ...)
        z_quant = z_quant.permute(*perm_back).contiguous()

        return z_quant, indices

    def indices_to_codes(self, indices):
        """Convert integer indices back to quantized values.

        Args:
            indices: (B, ...) integer indices.

        Returns:
            (B, ..., D) quantized tensor with values in [-(L-1)/2, (L-1)/2].
        """
        codes = []
        remaining = indices
        for i, L in enumerate(self._levels):
            half = (L - 1) / 2
            codes.append((remaining % L).float() - half)
            remaining = remaining // L
        return torch.stack(codes, dim=-1)
