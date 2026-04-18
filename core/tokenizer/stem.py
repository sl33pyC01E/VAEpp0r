"""Stem adapters for ElasticVideoTokenizer.

A "stem" is any frozen, duck-typed object that exposes:
  - latent_channels   : int
  - t_downscale       : int
  - s_downscale       : int
  - encode_video(clip: (N,T,C,H,W)) -> latent (N,T',C_lat,H',W')
  - decode_video(latent, target_shape=(T,H,W)) -> clip (N,T,C,H,W)

`MiniVAE` and `MiniVAE3D` already satisfy this directly. `StemChain`
composes a VAE with a per-frame FlattenDeflatten bottleneck so the
tokenizer sees the post-flatten latent shape.
"""

import torch
import torch.nn as nn


class StemChain(nn.Module):
    """Compose a frozen VAE with a frozen per-frame flattener.

    Flattener is applied independently to each temporal slot of the
    VAE latent. This mirrors the per-frame pattern in
    experiments/flatten_video.py (reshape (B, T', C, H, W) to
    (B*T', C, H, W), apply 2D flatten, reshape back).

    The flattener's 1D output is reshaped back to its original spatial
    grid shape so the tokenizer can treat it as a (T', C_bn, H', W')
    grid with its usual positional embedding.

    Both the VAE and flattener are held frozen — no params are
    registered as trainable on this module.
    """

    def __init__(self, vae, flattener):
        super().__init__()
        # Stash via list so PyTorch doesn't register either as a submodule
        # (we want them frozen, outside the tokenizer's optimizer scope).
        self._vae_ref = [vae]
        self._flat_ref = [flattener]

        # Latent shape exposed to the tokenizer: same spatial grid as the
        # VAE's, but with bottleneck_channels instead of latent_channels.
        self.latent_channels = int(flattener.B_ch)
        self.t_downscale = int(vae.t_downscale)
        self.s_downscale = int(vae.s_downscale)
        # Cache the spatial grid the flattener was built for; we reshape
        # its 1D output back to (H, W) to give the tokenizer a 2D grid.
        self._flat_H = int(flattener.H)
        self._flat_W = int(flattener.W)

    @property
    def vae(self):
        return self._vae_ref[0]

    @property
    def flattener(self):
        return self._flat_ref[0]

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._vae_ref[0] = self._vae_ref[0].to(*args, **kwargs)
        self._flat_ref[0] = self._flat_ref[0].to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Stem protocol
    # ------------------------------------------------------------------

    def encode_video(self, clip):
        """(N, T, C, H, W) -> (N, T', B_ch, H', W')."""
        lat = self.vae.encode_video(clip)             # (N, T', C_lat, H', W')
        N, Tp, C, Hl, Wl = lat.shape
        lat_2d = lat.reshape(N * Tp, C, Hl, Wl)
        flat_1d = self.flattener.flatten(lat_2d)      # (N*T', B_ch, H*W)
        B_ch = flat_1d.shape[1]
        # Reshape 1D walk-ordered sequence back to (H', W'). This loses the
        # walk-order semantics of the flattener's 1D convs (kernel_size>1
        # would have used adjacency along the walk), but that's intentional
        # here — the tokenizer's positional embedding re-learns geometry
        # from the 2D layout. With flattener kernel_size=1 there's no
        # information loss (pure channel compression).
        flat_2d = flat_1d.reshape(N * Tp, B_ch, Hl, Wl)
        return flat_2d.reshape(N, Tp, B_ch, Hl, Wl)

    def decode_video(self, flat_latent, target_shape=None):
        """(N, T', B_ch, H', W') -> clip via deflatten then VAE decode."""
        N, Tp, B_ch, Hl, Wl = flat_latent.shape
        flat_2d = flat_latent.reshape(N * Tp, B_ch, Hl, Wl)
        # The flattener's deflatten expects (B, B_ch, H*W) — the 1D seq.
        flat_1d = flat_2d.reshape(N * Tp, B_ch, Hl * Wl)
        lat_2d = self.flattener.deflatten(flat_1d)    # (N*T', C_lat, H, W)
        lat = lat_2d.reshape(N, Tp, lat_2d.shape[1], Hl, Wl)
        return self.vae.decode_video(lat, target_shape=target_shape)
