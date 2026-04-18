"""PyTorch port of LargeWorldModel/ElasticTok (model + bottleneck only).

Training glue lives in `training/train_elastictok.py`, GUI lives in
`gui/elastictok_tabs.py`. This module is just the model classes.
"""

from core.elastictok.model import ElasticTokConfig, ElasticTok, CONFIGS
from core.elastictok.bottleneck import (
    DiagonalGaussianDistribution, FSQ, VAE, get_bottleneck,
)

__all__ = [
    "ElasticTokConfig", "ElasticTok", "CONFIGS",
    "DiagonalGaussianDistribution", "FSQ", "VAE", "get_bottleneck",
]
