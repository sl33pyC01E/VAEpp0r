"""Elastic 1D video tokenizer — trained on top of a frozen MiniVAE3D stem.

Turns a (T', H', W') continuous latent grid from the VAE encoder into a
compact 1D sequence of N_q learned query tokens, quantized (optionally)
through a small continuous bottleneck. A decoder inverts this back to a
latent grid of the same shape, which the frozen VAE decoder can render
to pixels.

The tokenizer is trained independently of the VAE — VAE encode/decode
run with no_grad during tokenizer training, so the VAE's weights don't
move.
"""

from core.tokenizer.tokenizer import ElasticVideoTokenizer
from core.tokenizer.stem import StemChain
from core.tokenizer.pixel_tokenizer import PixelVideoTokenizer

__all__ = ["ElasticVideoTokenizer", "StemChain", "PixelVideoTokenizer"]
