"""Tabular data generators for Genesis."""

from genesis.generators.tabular.base import BaseTabularGenerator
from genesis.generators.tabular.ctgan import CTGANGenerator
from genesis.generators.tabular.gaussian_copula import GaussianCopulaGenerator
from genesis.generators.tabular.tvae import TVAEGenerator

# Register built-in generators with plugin system
from genesis.plugins import register_generator

register_generator(
    "ctgan",
    description="CTGAN (Conditional Tabular GAN) for tabular data generation",
    tags=["tabular", "deep-learning", "gan"],
)(CTGANGenerator)

register_generator(
    "tvae",
    description="TVAE (Tabular VAE) for tabular data generation",
    tags=["tabular", "deep-learning", "vae"],
)(TVAEGenerator)

register_generator(
    "gaussian_copula",
    description="Gaussian Copula for statistical tabular data generation",
    tags=["tabular", "statistical", "fast"],
)(GaussianCopulaGenerator)

__all__ = [
    "BaseTabularGenerator",
    "CTGANGenerator",
    "TVAEGenerator",
    "GaussianCopulaGenerator",
]
