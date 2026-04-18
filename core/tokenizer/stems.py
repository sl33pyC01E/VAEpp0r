"""Pixel-domain stems for PixelVideoTokenizer.

A "stem" here turns a clip (N, T, 3, H, W) into a grid of patch tokens
(N, dim, T', H', W') and back. Interchangeable — try a plain Conv3d
patch stem, or a Haar-prefixed one that pre-compresses spatially (for
free, no params) before the conv learns what to do with the packed
channels.

These stems are distinct from the clip<->latent `StemChain` used by
ElasticVideoTokenizer. That one wraps a VAE; these are standalone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -- Haar helpers --------------------------------------------------------------
# Spatial-only, per-frame. One level = 2x spatial compression + 4x channel
# multiplier. Preserves temporal resolution (important for short T clips
# like T=4 where 3D-Haar would flatten T to 1).
#
# Copied from training/train_video3d.py so the tokenizer doesn't depend on
# the training-script module directly.

def _haar_down_2d(x):
    """(B, C, H, W) -> (B, 4C, H/2, W/2). Orthonormal 2D Haar wavelet."""
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return torch.cat([ll, lh, hl, hh], dim=1)


def _haar_up_2d(x):
    """(B, 4C, H, W) -> (B, C, 2H, 2W). Inverse of _haar_down_2d."""
    C = x.shape[1] // 4
    ll, lh, hl, hh = x[:, 0*C:1*C], x[:, 1*C:2*C], x[:, 2*C:3*C], x[:, 3*C:4*C]
    a = (ll + lh + hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll - lh - hl + hh) * 0.5
    B, Ch, H, W = a.shape
    out = torch.zeros(B, Ch, H * 2, W * 2, device=x.device, dtype=x.dtype)
    out[:, :, 0::2, 0::2] = a
    out[:, :, 0::2, 1::2] = b
    out[:, :, 1::2, 0::2] = c
    out[:, :, 1::2, 1::2] = d
    return out


def _haar_down_2d_n(x, n):
    for _ in range(n):
        x = _haar_down_2d(x)
    return x


def _haar_up_2d_n(x, n):
    for _ in range(n):
        x = _haar_up_2d(x)
    return x


# -- Patch stem ---------------------------------------------------------------

class PatchStem(nn.Module):
    """Pixel <-> patch-token grid with optional Haar spatial pre-compression.

    Forward pipeline for encode():
        clip (N, T, 3, H, W)
           -> per-frame 2D Haar `haar_levels` times
              (each level: 2x spatial, 4x channel; T untouched)
              -> Haar output (N, T, 3 * 4^haar_levels, H/2^haar, W/2^haar)
           -> pad T,H,W to multiples of (t_patch, s_patch, s_patch)
           -> permute to NCTHW for Conv3d
           -> Conv3d(in_ch_after_haar, dim, kernel=stride=(t_p, s_p, s_p))
           -> (N, dim, T', H', W')

    Inverse for decode():
        (N, dim, T', H', W')
           -> ConvTranspose3d back to (N, C_after_haar, T, H, W)
           -> per-frame inverse Haar `haar_levels` times
           -> clip (N, T, 3, H, W)

    Important: Haar is a lossless rearrangement (no params). It makes the
    transformer's job easier by presenting the high-frequency content as
    extra channels at a smaller spatial grid, rather than asking the conv
    stem to compress spatial detail on its own.
    """

    def __init__(self, in_channels=3, dim=384,
                 t_patch=2, s_patch=16, haar_levels=0):
        super().__init__()
        self.in_ch = int(in_channels)
        self.dim = int(dim)
        self.t_patch = int(t_patch)
        self.s_patch = int(s_patch)
        self.haar_levels = int(haar_levels)
        # Post-Haar channel count
        self.post_haar_ch = self.in_ch * (4 ** self.haar_levels)
        # Haar spatial compression factor
        self.haar_s_factor = 2 ** self.haar_levels

        # Conv3d patchify: kernel = stride = (t_patch, s_patch, s_patch)
        # operates on post-Haar spatial grid (NOT the original resolution).
        self.patchify = nn.Conv3d(
            self.post_haar_ch, dim,
            kernel_size=(self.t_patch, self.s_patch, self.s_patch),
            stride=(self.t_patch, self.s_patch, self.s_patch))

        self.unpatchify = nn.ConvTranspose3d(
            dim, self.post_haar_ch,
            kernel_size=(self.t_patch, self.s_patch, self.s_patch),
            stride=(self.t_patch, self.s_patch, self.s_patch))

    # ------------------------------------------------------------------
    # Useful queries for the caller
    # ------------------------------------------------------------------

    def total_s_downscale(self):
        """End-to-end spatial downscale: Haar * patch."""
        return self.haar_s_factor * self.s_patch

    def total_t_downscale(self):
        """End-to-end temporal downscale: patch only (Haar is 2D)."""
        return self.t_patch

    def grid_dims(self, T, H, W):
        """For a given clip (T, H, W) return the patched grid (T', H', W')
        that encode() will emit. Uses ceil-div to account for padding."""
        t_ds = self.total_t_downscale()
        s_ds = self.total_s_downscale()
        Tp = -(-T // t_ds)
        Hp = -(-H // s_ds)
        Wp = -(-W // s_ds)
        return Tp, Hp, Wp

    # ------------------------------------------------------------------
    # Haar helpers on video tensors
    # ------------------------------------------------------------------

    def _haar_encode_video(self, clip):
        """(N, T, C, H, W) -> (N, T, C*4^haar, H/2^haar, W/2^haar)."""
        if self.haar_levels == 0:
            return clip
        N, T, C, H, W = clip.shape
        # Flatten (N, T) so haar_down_2d can operate
        x = clip.reshape(N * T, C, H, W)
        x = _haar_down_2d_n(x, self.haar_levels)
        new_C = C * (4 ** self.haar_levels)
        new_H = H // self.haar_s_factor
        new_W = W // self.haar_s_factor
        return x.reshape(N, T, new_C, new_H, new_W)

    def _haar_decode_video(self, clip):
        """Inverse of _haar_encode_video."""
        if self.haar_levels == 0:
            return clip
        N, T, C, H, W = clip.shape
        x = clip.reshape(N * T, C, H, W)
        x = _haar_up_2d_n(x, self.haar_levels)
        base_C = C // (4 ** self.haar_levels)
        return x.reshape(N, T, base_C, H * self.haar_s_factor,
                         W * self.haar_s_factor)

    # ------------------------------------------------------------------
    # Pad / unpad to multiples of the patch grid
    # ------------------------------------------------------------------

    def _pad_to_patch(self, clip):
        """Pad (N,T,C,H,W) with replicate so T,H,W are multiples of the
        (t_patch, s_patch, s_patch) grid. Returns (padded_clip, orig_thw)."""
        _, T, _, H, W = clip.shape
        pad_t = (-T) % self.t_patch
        pad_h = (-H) % self.s_patch
        pad_w = (-W) % self.s_patch
        if pad_t == 0 and pad_h == 0 and pad_w == 0:
            return clip, (T, H, W)
        if pad_t:
            clip = torch.cat(
                [clip, clip[:, -1:].expand(-1, pad_t, -1, -1, -1)], dim=1)
        if pad_h or pad_w:
            clip = F.pad(clip, (0, pad_w, 0, pad_h), mode="replicate")
        return clip, (T, H, W)

    # ------------------------------------------------------------------
    # Stem-protocol methods
    # ------------------------------------------------------------------

    def encode(self, clip):
        """(N, T, 3, H, W) -> ((N, dim, T', H', W'), orig_thw).

        orig_thw is the ORIGINAL pre-Haar pre-pad clip shape, so
        decode() can crop the expanded output back to the user's
        input size even when Haar pre-compression was active.
        """
        assert clip.ndim == 5, f"expected NTCHW, got {tuple(clip.shape)}"
        _, T_orig, _, H_orig, W_orig = clip.shape
        orig_thw = (T_orig, H_orig, W_orig)
        # 1) Haar pre-compress (per frame, spatial only)
        x = self._haar_encode_video(clip)
        # 2) Pad to patch-grid multiples (post-Haar shape may be odd)
        x, _ = self._pad_to_patch(x)
        # 3) NTCHW -> NCTHW for Conv3d
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # 4) Patchify
        patches = self.patchify(x)                    # (N, dim, T', H', W')
        return patches, orig_thw

    def decode(self, patches, orig_thw):
        """(N, dim, T', H', W') -> (N, T, 3, H, W), cropped to orig_thw."""
        T, H, W = orig_thw
        x = self.unpatchify(patches)                   # (N, C_haar, Tp*t, Hp*s, Wp*s)
        x = x.permute(0, 2, 1, 3, 4).contiguous()      # (N, T_pad, C_haar, H_pad, W_pad)
        # Inverse Haar (spatial only)
        x = self._haar_decode_video(x)                 # (N, T_pad, 3, H_full, W_full)
        # Crop to original clip extent (remove pad)
        return x[:, :T, :, :H, :W]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def describe(self, T, H, W):
        Tp, Hp, Wp = self.grid_dims(T, H, W)
        s_total = self.total_s_downscale()
        t_total = self.total_t_downscale()
        haar_note = f"Haar {self.haar_levels}lvl ({self.haar_s_factor}x spatial, " \
                    f"{4**self.haar_levels}x channel) + " if self.haar_levels else ""
        return (f"PatchStem: {haar_note}"
                f"Conv3d patch(t={self.t_patch}, s={self.s_patch}) "
                f"-> grid ({Tp}, {Hp}, {Wp})  "
                f"total t_ds={t_total}  s_ds={s_total}")
