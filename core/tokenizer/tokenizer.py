"""ElasticVideoTokenizer — 1D variable-length token encoder/decoder that
operates in MiniVAE3D latent space.

Architecture (all dims derived from the passed VAE instance):

    VAE latent grid (N, C_lat, T', H', W')
        │ flatten → (N, T'*H'*W', C_lat)
        │ in_proj (C_lat → dim) + FactorizedPosEmb(T', H', W')
        │ concat N_q learned queries
        ▼
    Transformer encoder stack
        │ take last N_q positions (query outputs)
        │ bottleneck_down (dim → d_bottleneck)
        ▼
    Tokens (N, N_q, d_bottleneck)   ← what the world model would consume
        │ (train-time: tail-drop to `keep`)
        │ bottleneck_up (d_bottleneck → dim)
        │ concat positional-only grid tokens
        ▼
    Transformer decoder stack
        │ take last T'*H'*W' positions
        │ out_proj (dim → C_lat)
        ▼
    Reconstructed latent grid (N, C_lat, T', H', W')

Trained with MSE in VAE latent space, with random per-batch `keep` so the
tokenizer is robust to any suffix-truncated budget at inference.
"""

import torch
import torch.nn as nn

from core.tokenizer.blocks import (
    RMSNorm, TransformerStack, FactorizedPosEmb)


class ElasticVideoTokenizer(nn.Module):
    """1D variable-length video tokenizer over a latent grid.

    Three use modes:
      1) With a stem (MiniVAE, MiniVAE3D, or StemChain): call forward(clip).
         The stem handles clip <-> latent conversion; tokenizer does
         latent <-> tokens.
      2) Pure latent-space: pass `stem=None` and explicit `C_lat` /
         `t_downscale` / `s_downscale`. Call forward_latent(z_latent)
         with pre-computed latents — no pixel conversion.
      3) Latent cache training: same as (2), with a data loader that
         yields pre-computed latent tensors from disk.

    Any object with .encode_video(clip), .decode_video(z, target_shape),
    .latent_channels, .t_downscale, .s_downscale counts as a stem.
    Duck-typed intentionally so MiniVAE, MiniVAE3D, and StemChain all
    work without inheritance.
    """

    def __init__(
        self,
        stem=None,                # duck-typed; see class docstring
        *,
        # Required when stem is None (or to override stem-derived values):
        C_lat=None,
        t_downscale=None,
        s_downscale=None,
        # Tokenizer arch:
        n_queries=128,
        dim=384,
        depth=6,
        heads=6,
        mlp_mult=4,
        d_bottleneck=8,
        min_keep=32,
        dropout=0.0,
        pos_emb_max_t=32,
        pos_emb_max_h=64,
        pos_emb_max_w=96,
        # Backward-compat: accept old `vae=` as alias for `stem=`
        vae=None,
    ):
        super().__init__()

        # Backward-compat: treat old vae kwarg as stem.
        if stem is None and vae is not None:
            stem = vae

        # Resolve latent shape specs. Stem values override explicit kwargs
        # when both are provided (prevents silent mismatch).
        if stem is not None:
            self.C_lat = int(getattr(stem, "latent_channels", C_lat))
            self.t_ds = int(getattr(stem, "t_downscale", t_downscale))
            self.s_ds = int(getattr(stem, "s_downscale", s_downscale))
        else:
            if C_lat is None or t_downscale is None or s_downscale is None:
                raise ValueError(
                    "ElasticVideoTokenizer with stem=None requires explicit "
                    "C_lat, t_downscale, and s_downscale kwargs.")
            self.C_lat = int(C_lat)
            self.t_ds = int(t_downscale)
            self.s_ds = int(s_downscale)

        # Stash the stem via a list so PyTorch doesn't register its weights
        # as trainable — we want it frozen. None when in pure-latent mode.
        self._stem_ref = [stem]

        self.n_queries = int(n_queries)
        self.dim = int(dim)
        self.d_bottleneck = int(d_bottleneck)
        self.min_keep = int(min_keep)
        assert self.min_keep <= self.n_queries

        # Latent-grid ↔ transformer-dim projections
        self.in_proj = nn.Linear(self.C_lat, dim)
        self.out_proj = nn.Linear(dim, self.C_lat)

        # Positional embedding on the latent grid
        self.pos_emb = FactorizedPosEmb(
            dim, max_t=pos_emb_max_t, max_h=pos_emb_max_h, max_w=pos_emb_max_w)

        # Learned query tokens + their own positional (index) embedding
        self.queries = nn.Parameter(torch.randn(self.n_queries, dim) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(self.n_queries, dim) * 0.02)

        # Encoder / decoder stacks
        self.enc = TransformerStack(
            dim, depth=depth, heads=heads, mlp_mult=mlp_mult, dropout=dropout)
        self.dec = TransformerStack(
            dim, depth=depth, heads=heads, mlp_mult=mlp_mult, dropout=dropout)

        # Bottleneck (continuous, small-dim — phase-2 will swap in FSQ here)
        self.bottleneck_down = nn.Linear(dim, d_bottleneck)
        self.bottleneck_up = nn.Linear(d_bottleneck, dim)

        # Final norm before out_proj
        self.out_norm = RMSNorm(dim)

    # ------------------------------------------------------------------
    # Frozen stem access
    # ------------------------------------------------------------------

    @property
    def stem(self):
        """The frozen stem (MiniVAE / MiniVAE3D / StemChain / None)."""
        return self._stem_ref[0]

    @property
    def vae(self):
        """Backward-compat alias for stem (read-only)."""
        return self._stem_ref[0]

    @property
    def has_stem(self):
        return self._stem_ref[0] is not None

    # Ensure the stem moves alongside the tokenizer when `.to(device)` is
    # called on the module tree.
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self._stem_ref[0] is not None:
            self._stem_ref[0] = self._stem_ref[0].to(*args, **kwargs)
        return self

    # ------------------------------------------------------------------
    # Encode / decode in LATENT space (no VAE involved)
    # ------------------------------------------------------------------

    def encode_latent(self, z_latent):
        """(N, T', C_lat, H', W') → (N, N_q, d_bottleneck).

        Uses MiniVAE3D's layout convention (time second, channels third).
        """
        assert z_latent.ndim == 5, f"expected (N,T,C,H,W), got {z_latent.shape}"
        N, T, C, H, W = z_latent.shape
        assert C == self.C_lat, (
            f"tokenizer was built for C_lat={self.C_lat}, got {C}")

        # Flatten grid: (N, T, C, H, W) -> (N, T*H*W, C)
        grid = z_latent.permute(0, 1, 3, 4, 2).reshape(N, T * H * W, C)
        grid = self.in_proj(grid)
        grid = grid + self.pos_emb(T, H, W, device=grid.device, dtype=grid.dtype)

        # Concat query tokens (with their own positional embedding)
        q = (self.queries + self.query_pos).unsqueeze(0).expand(N, -1, -1)
        x = torch.cat([grid, q], dim=1)

        # Full attention encoder. Block-causal masking across time is a
        # phase-2 refinement; for v1 we let queries attend to the full grid.
        x = self.enc(x)

        # Take query outputs
        q_out = x[:, -self.n_queries:, :]            # (N, N_q, dim)
        tokens = self.bottleneck_down(q_out)         # (N, N_q, d_bottleneck)
        return tokens

    def decode_tokens(self, tokens, T, H, W, keep=None):
        """(N, N_q, d_bottleneck) → (N, T, C_lat, H, W) (MiniVAE3D layout).

        If `keep` is an int in [1, N_q], positions at index ≥ keep are
        masked to zero before the up-projection — the ElasticTok tail-drop
        trick. The decoder still sees N_q query slots so it can learn to
        synthesize without the dropped queries.
        """
        N, Nq, Db = tokens.shape
        assert Nq == self.n_queries, (
            f"expected {self.n_queries} query tokens, got {Nq}")
        assert Db == self.d_bottleneck

        if keep is not None:
            keep = int(keep)
            if keep < self.n_queries:
                mask = torch.zeros(self.n_queries, device=tokens.device,
                                   dtype=tokens.dtype)
                mask[:keep] = 1.0
                tokens = tokens * mask.view(1, self.n_queries, 1)

        q = self.bottleneck_up(tokens)               # (N, N_q, dim)
        q = q + self.query_pos.unsqueeze(0)          # re-inject positions

        # Positional-only grid queries
        pos_grid = self.pos_emb(T, H, W, device=q.device, dtype=q.dtype)
        grid_in = pos_grid.unsqueeze(0).expand(N, -1, -1)

        x = torch.cat([q, grid_in], dim=1)
        x = self.dec(x)

        # Take grid outputs
        grid_out = x[:, self.n_queries:, :]          # (N, T*H*W, dim)
        grid_out = self.out_norm(grid_out)
        grid_out = self.out_proj(grid_out)           # (N, T*H*W, C_lat)
        # (N, T*H*W, C_lat) -> (N, T, H, W, C_lat) -> (N, T, C_lat, H, W)
        grid_out = grid_out.reshape(N, T, H, W, self.C_lat).permute(0, 1, 4, 2, 3)
        return grid_out

    # ------------------------------------------------------------------
    # Pure-latent pipeline — no stem needed.
    # ------------------------------------------------------------------

    def forward_latent(self, z_latent, keep=None):
        """(N, T', C_lat, H', W') -> dict, no stem involved.

        Use this when training on pre-computed latents (latent-cache mode)
        or when you've computed the stem output separately.
        """
        tokens = self.encode_latent(z_latent)
        Tp = z_latent.shape[1]
        Hp = z_latent.shape[3]
        Wp = z_latent.shape[4]
        z_latent_hat = self.decode_tokens(tokens, Tp, Hp, Wp, keep=keep)
        return dict(
            z_latent=z_latent,
            z_latent_hat=z_latent_hat,
            tokens=tokens,
            keep=keep if keep is not None else self.n_queries,
            clip_hat=None,
        )

    # ------------------------------------------------------------------
    # Full pipeline — clip -> frozen stem -> tokenize -> detokenize -> stem
    # ------------------------------------------------------------------

    def forward(self, clip, keep=None):
        """(N, T, 3, H, W) -> dict of intermediates.

        Requires a stem. Routes clip through stem.encode_video (no grad),
        through the tokenizer, and back through stem.decode_video (no grad).
        For pure-latent use without a stem, call forward_latent() instead.
        """
        if not self.has_stem:
            raise RuntimeError(
                "forward(clip) requires a stem. Pass one at construction "
                "or call forward_latent(z_latent) directly.")

        # Frozen stem encode (no grad)
        with torch.no_grad():
            z_latent = self.stem.encode_video(clip)  # (N, Tp, C, Hp, Wp)

        # Tokenizer
        tokens = self.encode_latent(z_latent)
        # Latent layout is (N, T', C, H', W')
        Tp, Hp, Wp = z_latent.shape[1], z_latent.shape[3], z_latent.shape[4]
        z_latent_hat = self.decode_tokens(tokens, Tp, Hp, Wp, keep=keep)

        # Frozen stem decode (no grad, for preview/eval only — not in loss)
        with torch.no_grad():
            _, T, _, H, W = clip.shape
            clip_hat = self.stem.decode_video(
                z_latent_hat.detach(), target_shape=(T, H, W))

        return dict(
            z_latent=z_latent,
            z_latent_hat=z_latent_hat,
            tokens=tokens,
            keep=keep if keep is not None else self.n_queries,
            clip_hat=clip_hat,
        )

    # ------------------------------------------------------------------
    # Preview helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reconstruct(self, clip, keeps=(32, 64, 128)):
        """Return a dict {keep: clip_hat} rendered at multiple budgets.
        Requires a stem (pixel reconstruction isn't possible without one).
        For latent-space eval, use reconstruct_latent()."""
        if not self.has_stem:
            raise RuntimeError(
                "reconstruct(clip) requires a stem. "
                "Use reconstruct_latent(z_latent) for pure-latent mode.")
        out = {}
        z_latent = self.stem.encode_video(clip)
        tokens = self.encode_latent(z_latent)
        Tp, Hp, Wp = z_latent.shape[1], z_latent.shape[3], z_latent.shape[4]
        _, T, _, H, W = clip.shape
        for k in keeps:
            k_eff = min(int(k), self.n_queries)
            z_hat = self.decode_tokens(tokens, Tp, Hp, Wp, keep=k_eff)
            out[k_eff] = self.stem.decode_video(z_hat, target_shape=(T, H, W))
        return out

    @torch.no_grad()
    def reconstruct_latent(self, z_latent, keeps=(32, 64, 128)):
        """Return a dict {keep: z_latent_hat} at multiple budgets.
        Pure latent-space — no stem needed. Useful when training on a
        latent cache with no way to render pixels."""
        out = {}
        tokens = self.encode_latent(z_latent)
        Tp, Hp, Wp = z_latent.shape[1], z_latent.shape[3], z_latent.shape[4]
        for k in keeps:
            k_eff = min(int(k), self.n_queries)
            out[k_eff] = self.decode_tokens(tokens, Tp, Hp, Wp, keep=k_eff)
        return out
