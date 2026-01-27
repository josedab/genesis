"""Synthetic data generators for Genesis."""

from genesis.generators.auto import get_method_recommendations, select_generator
from genesis.generators.tabular import (
    BaseTabularGenerator,
    CTGANGenerator,
    GaussianCopulaGenerator,
    TVAEGenerator,
)
from genesis.generators.text import (
    BaseTextGenerator,
    HuggingFaceBackend,
    LLMTextGenerator,
    OpenAIBackend,
)
from genesis.generators.timeseries import (
    BaseTimeSeriesGenerator,
    StatisticalTimeSeriesGenerator,
    TimeGANGenerator,
)

__all__ = [
    # Auto selection
    "select_generator",
    "get_method_recommendations",
    # Tabular
    "BaseTabularGenerator",
    "CTGANGenerator",
    "TVAEGenerator",
    "GaussianCopulaGenerator",
    # Time series
    "BaseTimeSeriesGenerator",
    "TimeGANGenerator",
    "StatisticalTimeSeriesGenerator",
    # Text
    "BaseTextGenerator",
    "LLMTextGenerator",
    "OpenAIBackend",
    "HuggingFaceBackend",
]
