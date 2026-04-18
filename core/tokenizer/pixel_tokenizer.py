"""Standalone pixel-domain video tokenizer (TiTok/ElasticTok style).

No VAE. The whole thing is:

    clip
      -> PatchStem.encode      (Haar + conv patchify)
      -> flatten + pos_emb
      -> concat N_q learned queries
      -> TransformerStack
      -> keep query outputs
      -> bottleneck_down       (dim -> d_bottleneck)
                |
           compressed tokens (N, N_q, d_bottleneck)
                |
      -> bottleneck_up         (d_bottleneck -> dim)
      -> concat positional grid tokens
      -> TransformerStack
      -> keep grid outputs
      -> PatchStem.decode      (conv unpatchify + inverse Haar)
    clip_hat

The PatchStem is swappable — see core/tokenizer/stems.py. Training-time
tail-drop (`keep < N_q`) is the ElasticTok variable-budget trick.
"""

import torch
import torch.nn as nn

from core.tokenizer.blocks import (
    RMSNorm, TransformerStack, FactorizedPosEmb)
from core.tokenizer.stems import PatchStem


class PixelVideoTokenizer(nn.Module):
    """A TiTok/ElasticTok-style video tokenizer, pixels to pixels.

    All pixel<->grid conversion happens in the PatchStem. This class
    handles the transformer core + queries + bottleneck only.
    """

    def __init__(
        self,
        # Clip geometry (for pos-emb cap sizing and forward pad)
        T=4, H=368, W=640,
        in_channels=3,
        # Stem
        t_patch=2, s_patch=16, haar_levels=0,
        # Tokenizer arch
        n_queries=128,
        dim=384,
        depth=6,
        heads=6,
        mlp_mult=4,
        d_bottleneck=8,
        min_keep=32,
        dropout=0.0,
        # Bottleneck type: "vae" = Gaussian posterior + KL (ElasticTok);
        #                  "linear" = plain deterministic projection.
        bottleneck_type="vae",
        # Pos-emb caps (must be >= patched grid dims; inferred if None)
        pos_emb_max_t=None,
        pos_emb_max_h=None,
        pos_emb_max_w=None,
    ):
        super().__init__()
        self.T = int(T)
        self.H = int(H)
        self.W = int(W)
        self.in_ch = int(in_channels)
        self.n_queries = int(n_queries)
        self.dim = int(dim)
        self.d_bottleneck = int(d_bottleneck)
        self.min_keep = int(min_keep)
        assert self.min_keep <= self.n_queries

        # Stem — owns patchify/unpatchify + optional Haar.
        self.stem = PatchStem(
            in_channels=self.in_ch, dim=dim,
            t_patch=t_patch, s_patch=s_patch, haar_levels=haar_levels)

        # Infer pos-emb caps from configured (T,H,W) at the patched grid.
        Tp_ref, Hp_ref, Wp_ref = self.stem.grid_dims(self.T, self.H, self.W)
        if pos_emb_max_t is None:
            pos_emb_max_t = max(4, Tp_ref + 2)
        if pos_emb_max_h is None:
            pos_emb_max_h = max(12, Hp_ref + 4)
        if pos_emb_max_w is None:
            pos_emb_max_w = max(16, Wp_ref + 4)

        self.pos_emb = FactorizedPosEmb(
            dim, max_t=pos_emb_max_t,
            max_h=pos_emb_max_h, max_w=pos_emb_max_w)

        # Init scale matches TiTok (bytedance/1d-tokenizer):
        #   scale = width ** -0.5  e.g. dim=384 -> 0.051
        # vs the old 0.02 blanket, this gives query/mask embeddings
        # enough magnitude to actually influence attention early.
        _init_scale = dim ** -0.5

        # Learned query tokens + their own positional embedding
        self.queries = nn.Parameter(
            torch.randn(self.n_queries, dim) * _init_scale)
        self.query_pos = nn.Parameter(
            torch.randn(self.n_queries, dim) * _init_scale)

        # TiTok decoder mask token (bytedance/1d-tokenizer:
        # modeling/modules/blocks.py, TiTokDecoder.__init__):
        #     self.mask_token = nn.Parameter(scale * torch.randn(1, 1, W))
        # A single learnable vector broadcast to every grid position in
        # the decoder. Without it, grid positions at decode input are
        # pure pos_emb noise at init and the decoder produces the same
        # blob for every input.
        self.mask_token = nn.Parameter(
            torch.randn(1, 1, dim) * _init_scale)

        # ElasticTok encoder mask-conditioning (arXiv 2410.08368 §3):
        # "The encoder is additionally conditioned on the token mask
        # (represented as a binary mask) by replacing 0's and 1's with
        # different learned embedding vectors."
        # Two learned embeddings — [0]=dropped, [1]=kept — added to
        # query tokens so the encoder knows which slots will survive
        # tail-drop and can pack important info into early slots.
        self.keep_mask_emb = nn.Parameter(
            torch.randn(2, dim) * _init_scale)

        # Encoder / decoder transformer stacks
        self.enc = TransformerStack(
            dim, depth=depth, heads=heads, mlp_mult=mlp_mult, dropout=dropout)
        self.dec = TransformerStack(
            dim, depth=depth, heads=heads, mlp_mult=mlp_mult, dropout=dropout)

        # Bottleneck — ElasticTok's primary variant is "vae": a Gaussian
        # posterior N(mean, exp(logvar)) with reparameterization and a
        # small KL-to-prior penalty (aux_loss, w=1e-8 in run_train.sh).
        # This regularizes the latent so unused dims don't collapse and
        # forces information to spread across the bottleneck budget.
        # The "linear" variant is the plain deterministic projection.
        self.bottleneck_type = str(bottleneck_type)
        if self.bottleneck_type == "vae":
            # 2*d_bottleneck: first half mean, second half logvar.
            self.bottleneck_down = nn.Linear(dim, 2 * d_bottleneck)
        else:
            self.bottleneck_down = nn.Linear(dim, d_bottleneck)
        self.bottleneck_up = nn.Linear(d_bottleneck, dim)

        self.out_norm = RMSNorm(dim)

    # ------------------------------------------------------------------
    # Token <-> token-grid transformer core
    # ------------------------------------------------------------------

    def _encode_core(self, patches, keep=None, sample=True):
        """(N, dim, Tp, Hp, Wp) -> (tokens, aux_dict).

        `keep`: ElasticTok encoder conditioning — first `keep` query
        slots tagged as "kept", rest as "dropped".
        `sample`: for VAE bottleneck, whether to reparameterize-sample
        (True, training) or take the posterior mean (False, eval).

        Returns:
            tokens: (N, N_q, d_bottleneck)
            aux:    dict with 'kl' if vae bottleneck, else 'kl' = 0
        """
        N, D, Tp, Hp, Wp = patches.shape
        seq = patches.reshape(N, D, Tp * Hp * Wp).transpose(1, 2)
        seq = seq + self.pos_emb(Tp, Hp, Wp,
                                  device=seq.device, dtype=seq.dtype)
        q = (self.queries + self.query_pos).unsqueeze(0).expand(N, -1, -1)

        # ElasticTok encoder conditioning.
        k_eff = self.n_queries if keep is None else \
            max(0, min(int(keep), self.n_queries))
        cond = torch.empty(self.n_queries, self.dim,
                           device=q.device, dtype=q.dtype)
        cond[:k_eff] = self.keep_mask_emb[1].to(q.dtype)     # kept
        cond[k_eff:] = self.keep_mask_emb[0].to(q.dtype)     # dropped
        q = q + cond.unsqueeze(0)

        x = torch.cat([seq, q], dim=1)
        x = self.enc(x)
        q_out = x[:, -self.n_queries:, :]
        proj = self.bottleneck_down(q_out)

        # VAE bottleneck: split into mean/logvar, reparameterize, KL.
        # Formula matches elastic/bottleneck.py DiagonalGaussianDistribution:
        #     kl = mean^2 + exp(logvar) - 1 - logvar
        # Scale is per-batch-mean (averaged over N_q and d).
        if self.bottleneck_type == "vae":
            mean, logvar = proj.chunk(2, dim=-1)
            logvar = logvar.clamp(-30.0, 20.0)  # numerical stability
            if sample and self.training:
                std = (0.5 * logvar).exp()
                tokens = mean + std * torch.randn_like(mean)
            else:
                tokens = mean
            kl = (mean.pow(2) + logvar.exp() - 1.0 - logvar).mean()
            aux = {"kl": kl}
        else:
            tokens = proj
            aux = {"kl": torch.zeros((), device=tokens.device,
                                      dtype=tokens.dtype)}
        return tokens, aux

    def _decode_core(self, tokens, Tp, Hp, Wp, keep=None):
        """(N, N_q, d_bottleneck) -> (N, dim, Tp, Hp, Wp)."""
        N = tokens.shape[0]
        if keep is not None:
            keep = int(keep)
            if keep < self.n_queries:
                mask = torch.zeros(self.n_queries, device=tokens.device,
                                   dtype=tokens.dtype)
                mask[:keep] = 1.0
                tokens = tokens * mask.view(1, self.n_queries, 1)

        q = self.bottleneck_up(tokens)
        q = q + self.query_pos.unsqueeze(0)

        # TiTok decoder input for grid positions: learned mask_token
        # broadcast across every grid slot, plus the factorized pos_emb.
        # Ref: bytedance/1d-tokenizer blocks.py TiTokDecoder.forward:
        #     mask_tokens = self.mask_token.repeat(B, grid^2, 1)
        #     x = torch.cat([cls, mask_tokens + pos_emb, z_q], dim=1)
        grid_len = Tp * Hp * Wp
        pos_grid = self.pos_emb(Tp, Hp, Wp, device=q.device, dtype=q.dtype)
        grid_in = (self.mask_token.expand(N, grid_len, -1).to(q.dtype)
                   + pos_grid.unsqueeze(0))

        x = torch.cat([q, grid_in], dim=1)
        x = self.dec(x)

        grid_out = x[:, self.n_queries:, :]
        grid_out = self.out_norm(grid_out)
        # (N, S, dim) -> (N, dim, Tp, Hp, Wp)
        return grid_out.transpose(1, 2).reshape(N, self.dim, Tp, Hp, Wp)

    # ------------------------------------------------------------------
    # Public encode / decode (pixel-in, pixel-out)
    # ------------------------------------------------------------------

    def encode(self, clip, keep=None, sample=True):
        """(N, T, 3, H, W) -> (tokens, (Tp, Hp, Wp), orig_thw, aux)."""
        patches, orig_thw = self.stem.encode(clip)
        tokens, aux = self._encode_core(patches, keep=keep, sample=sample)
        _, _, Tp, Hp, Wp = patches.shape
        return tokens, (Tp, Hp, Wp), orig_thw, aux

    def decode(self, tokens, Tp, Hp, Wp, orig_thw, keep=None):
        """(N, N_q, d_bottleneck) -> (N, T, 3, H, W)."""
        patches = self._decode_core(tokens, Tp, Hp, Wp, keep=keep)
        return self.stem.decode(patches, orig_thw)

    # ------------------------------------------------------------------
    # Full forward + batch preview helper
    # ------------------------------------------------------------------

    def forward(self, clip, keep=None):
        """(N, T, 3, H, W) -> dict of intermediates incl. aux (KL)."""
        tokens, (Tp, Hp, Wp), orig_thw, aux = self.encode(
            clip, keep=keep, sample=True)
        recon = self.decode(tokens, Tp, Hp, Wp, orig_thw, keep=keep)
        return dict(
            tokens=tokens,
            recon=recon,
            keep=keep if keep is not None else self.n_queries,
            grid=(Tp, Hp, Wp),
            aux=aux,
        )

    @torch.no_grad()
    def reconstruct(self, clip, keeps=(32, 64, 128)):
        """Return {keep: clip_hat} for side-by-side preview. Uses
        posterior mean (sample=False) for determinism."""
        out = {}
        for k in keeps:
            k_eff = min(int(k), self.n_queries)
            tokens, (Tp, Hp, Wp), orig_thw, _ = self.encode(
                clip, keep=k_eff, sample=False)
            out[k_eff] = self.decode(
                tokens, Tp, Hp, Wp, orig_thw, keep=k_eff)
        return out

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def config_summary(self):
        lines = [
            f"  clip:     T={self.T}  HxW={self.H}x{self.W}  "
            f"in_ch={self.in_ch}",
            "  stem:     " + self.stem.describe(self.T, self.H, self.W),
            f"  queries:  N_q={self.n_queries}  "
            f"d_bottleneck={self.d_bottleneck}  min_keep={self.min_keep}",
            f"  xformer:  dim={self.dim}  "
            f"depth={len(self.enc.blocks)}  (encoder and decoder each)",
        ]
        return "\n".join(lines)
