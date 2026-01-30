"""Auto-selection logic for choosing the best generator."""

from typing import Optional

import pandas as pd

from genesis.core.base import BaseGenerator
from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.types import DataSchema, GeneratorMethod


def select_generator(
    data: pd.DataFrame,
    method: GeneratorMethod,
    config: Optional[GeneratorConfig] = None,
    privacy: Optional[PrivacyConfig] = None,
    schema: Optional[DataSchema] = None,
) -> BaseGenerator:
    """Select and instantiate the appropriate generator.

    Args:
        data: Training data
        method: Generation method (AUTO will select based on data)
        config: Generator configuration
        privacy: Privacy configuration
        schema: Pre-computed data schema

    Returns:
        Instantiated generator
    """
    from genesis.generators.tabular import (
        CTGANGenerator,
        GaussianCopulaGenerator,
        TVAEGenerator,
    )

    if method == GeneratorMethod.AUTO:
        method = _auto_select_method(data, schema)

    if method == GeneratorMethod.CTGAN:
        return CTGANGenerator(config=config, privacy=privacy)
    elif method == GeneratorMethod.TVAE:
        return TVAEGenerator(config=config, privacy=privacy)
    elif method == GeneratorMethod.GAUSSIAN_COPULA:
        return GaussianCopulaGenerator(config=config, privacy=privacy)
    elif method == GeneratorMethod.TIMEGAN:
        from genesis.generators.timeseries import TimeGANGenerator

        return TimeGANGenerator(config=config, privacy=privacy)
    elif method == GeneratorMethod.STATISTICAL:
        from genesis.generators.timeseries import StatisticalTimeSeriesGenerator

        return StatisticalTimeSeriesGenerator(config=config, privacy=privacy)
    elif method == GeneratorMethod.LLM:
        from genesis.generators.text import LLMTextGenerator

        return LLMTextGenerator(config=config, privacy=privacy)
    else:
        # Default to CTGAN
        return CTGANGenerator(config=config, privacy=privacy)


def _auto_select_method(
    data: pd.DataFrame,
    schema: Optional[DataSchema] = None,
) -> GeneratorMethod:
    """Automatically select the best generation method based on data characteristics.

    Args:
        data: Training data
        schema: Pre-computed data schema

    Returns:
        Selected GeneratorMethod
    """
    n_samples = len(data)
    n_columns = len(data.columns)

    # Count column types
    numeric_cols = data.select_dtypes(include=["number"]).columns
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns

    n_numeric = len(numeric_cols)
    n_categorical = len(categorical_cols)

    # Heuristics for method selection

    # For very small datasets, use Gaussian Copula (faster, no training)
    if n_samples < 500:
        return GeneratorMethod.GAUSSIAN_COPULA

    # For datasets with mostly categorical columns, CTGAN handles discrete better
    categorical_ratio = n_categorical / n_columns if n_columns > 0 else 0
    if categorical_ratio > 0.7:
        return GeneratorMethod.CTGAN

    # For datasets with mostly numeric columns, TVAE often works better
    if n_numeric / n_columns > 0.8:
        return GeneratorMethod.TVAE

    # Check for high cardinality categoricals
    high_cardinality = False
    for col in categorical_cols:
        if data[col].nunique() > 50:
            high_cardinality = True
            break

    # CTGAN handles high cardinality better
    if high_cardinality:
        return GeneratorMethod.CTGAN

    # For medium-sized datasets with mixed types, CTGAN is generally robust
    if n_samples < 10000:
        return GeneratorMethod.CTGAN

    # For large datasets, Gaussian Copula is faster
    if n_samples > 100000:
        return GeneratorMethod.GAUSSIAN_COPULA

    # Default to CTGAN
    return GeneratorMethod.CTGAN


def get_method_recommendations(data: pd.DataFrame) -> dict:
    """Get recommendations for generation methods based on data.

    Args:
        data: Input data

    Returns:
        Dictionary with method recommendations and reasons
    """
    n_samples = len(data)
    numeric_cols = len(data.select_dtypes(include=["number"]).columns)
    categorical_cols = len(data.select_dtypes(include=["object", "category"]).columns)

    recommendations = {}

    # CTGAN
    ctgan_score = 0.5
    ctgan_reasons = []

    if categorical_cols > numeric_cols:
        ctgan_score += 0.2
        ctgan_reasons.append("Good for categorical-heavy data")
    if n_samples >= 1000:
        ctgan_score += 0.1
        ctgan_reasons.append("Sufficient data for GAN training")

    recommendations["ctgan"] = {
        "score": min(ctgan_score, 1.0),
        "reasons": ctgan_reasons,
    }

    # TVAE
    tvae_score = 0.5
    tvae_reasons = []

    if numeric_cols > categorical_cols:
        tvae_score += 0.2
        tvae_reasons.append("Good for numeric-heavy data")
    if n_samples >= 500:
        tvae_score += 0.1
        tvae_reasons.append("Sufficient data for VAE training")

    recommendations["tvae"] = {
        "score": min(tvae_score, 1.0),
        "reasons": tvae_reasons,
    }

    # Gaussian Copula
    gc_score = 0.4
    gc_reasons = []

    if n_samples < 1000:
        gc_score += 0.3
        gc_reasons.append("Fast for small datasets")
    if n_samples > 50000:
        gc_score += 0.2
        gc_reasons.append("Efficient for large datasets")
    gc_reasons.append("No deep learning required")

    recommendations["gaussian_copula"] = {
        "score": min(gc_score, 1.0),
        "reasons": gc_reasons,
    }

    return recommendations
