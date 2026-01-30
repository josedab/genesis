"""AutoML Synthesis - Automatic method selection for synthetic data generation.

This module provides intelligent automatic selection of the best synthetic data
generation method based on dataset characteristics using meta-learning.

Features:
    - Meta-feature extraction from datasets
    - Rule-based method selection with confidence scores
    - Automatic hyperparameter configuration
    - Support for speed vs quality trade-offs

Example:
    Basic usage with the convenience function::

        from genesis import auto_synthesize

        # One-line automatic synthesis
        synthetic_df = auto_synthesize(df, n_samples=1000)

    Using the class for more control::

        from genesis.automl import AutoMLSynthesizer

        automl = AutoMLSynthesizer(prefer_quality=True)
        automl.fit(df)
        synthetic = automl.generate(1000)

        print(f"Selected method: {automl.selected_method}")
        print(f"Confidence: {automl.selection_confidence:.1%}")

Classes:
    MetaFeatureExtractor: Extracts meta-features from datasets
    MethodSelector: Selects optimal generation method
    AutoMLSynthesizer: End-to-end automatic synthesizer

Functions:
    auto_synthesize: One-line convenience function for automatic synthesis
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


class GenerationMethod(str, Enum):
    """Available generation methods for synthetic data.

    Attributes:
        GAUSSIAN_COPULA: Fast statistical method, good for highly correlated data.
        CTGAN: Deep learning GAN, best for complex mixed-type data.
        TVAE: Variational autoencoder, good balance of speed and quality.
        COPULA_GAN: Combines copulas with GAN training.
        FAST_ML: Lightweight ML-based method for quick generation.
    """

    GAUSSIAN_COPULA = "gaussian_copula"
    CTGAN = "ctgan"
    TVAE = "tvae"
    COPULA_GAN = "copulagan"
    FAST_ML = "fast_ml"


@dataclass
class DatasetMetaFeatures:
    """Meta-features extracted from a dataset for method selection.

    These features characterize the dataset and are used by the MethodSelector
    to choose the optimal generation method.

    Attributes:
        n_rows: Number of rows in the dataset.
        n_columns: Total number of columns.
        n_numeric: Number of numeric columns.
        n_categorical: Number of categorical columns.
        n_datetime: Number of datetime columns.
        numeric_mean_skewness: Average skewness of numeric columns.
        numeric_mean_kurtosis: Average kurtosis of numeric columns.
        numeric_outlier_ratio: Fraction of values that are outliers.
        numeric_missing_ratio: Fraction of missing numeric values.
        categorical_mean_cardinality: Average unique values per category.
        categorical_max_cardinality: Maximum unique values in any category.
        categorical_imbalance_ratio: Class imbalance measure.
        mean_correlation: Average pairwise correlation.
        max_correlation: Maximum pairwise correlation.
        correlation_clusters: Number of correlated column clusters.
        multimodal_columns: Columns with multimodal distributions.
        highly_skewed_columns: Columns with high skewness.
        estimated_complexity: Overall complexity score (0-1).
    """

    # Basic statistics
    n_rows: int = 0
    n_columns: int = 0
    n_numeric: int = 0
    n_categorical: int = 0
    n_datetime: int = 0

    # Numeric features
    numeric_mean_skewness: float = 0.0
    numeric_mean_kurtosis: float = 0.0
    numeric_outlier_ratio: float = 0.0
    numeric_missing_ratio: float = 0.0

    # Categorical features
    categorical_mean_cardinality: float = 0.0
    categorical_max_cardinality: int = 0
    categorical_imbalance_ratio: float = 0.0

    # Correlation features
    mean_correlation: float = 0.0
    max_correlation: float = 0.0
    correlation_clusters: int = 0

    # Distribution features
    multimodal_columns: int = 0
    highly_skewed_columns: int = 0

    # Complexity indicators
    estimated_complexity: float = 0.0

    def to_vector(self) -> np.ndarray:
        """Convert meta-features to a numeric vector for model input.

        Returns:
            numpy.ndarray: Feature vector with log-transformed size features.
        """
        return np.array(
            [
                np.log1p(self.n_rows),
                np.log1p(self.n_columns),
                self.n_numeric / max(self.n_columns, 1),
                self.n_categorical / max(self.n_columns, 1),
                self.numeric_mean_skewness,
                self.numeric_mean_kurtosis,
                self.numeric_outlier_ratio,
                self.numeric_missing_ratio,
                np.log1p(self.categorical_mean_cardinality),
                np.log1p(self.categorical_max_cardinality),
                self.categorical_imbalance_ratio,
                self.mean_correlation,
                self.max_correlation,
                self.multimodal_columns / max(self.n_numeric, 1),
                self.highly_skewed_columns / max(self.n_numeric, 1),
                self.estimated_complexity,
            ]
        )


@dataclass
class MethodRecommendation:
    """A recommendation for a specific generation method.

    Attributes:
        method: The recommended GenerationMethod.
        confidence: Confidence score from 0 to 1.
        reasons: List of reasons supporting this recommendation.
        estimated_quality: Expected quality score (0-1).
        estimated_time_factor: Relative time multiplier (1.0 = baseline).

    Example:
        >>> rec = MethodRecommendation(
        ...     method=GenerationMethod.CTGAN,
        ...     confidence=0.85,
        ...     reasons=["High cardinality categories", "Complex correlations"]
        ... )
    """

    method: GenerationMethod
    confidence: float
    reasons: List[str] = field(default_factory=list)
    estimated_quality: float = 0.0
    estimated_time_factor: float = 1.0


@dataclass
class AutoMLResult:
    """Result of the AutoML method selection process.

    Attributes:
        recommended_method: The best method for this dataset.
        confidence: Confidence in the recommendation (0-1).
        all_recommendations: All methods ranked by suitability.
        meta_features: Extracted dataset characteristics.
        analysis_time_seconds: Time taken for analysis.

    Example:
        >>> result = selector.select(features)
        >>> print(f"Use {result.recommended_method} with {result.confidence:.0%} confidence")
    """

    recommended_method: GenerationMethod
    confidence: float
    all_recommendations: List[MethodRecommendation]
    meta_features: DatasetMetaFeatures
    analysis_time_seconds: float = 0.0


class MetaFeatureExtractor:
    """Extract meta-features from datasets for intelligent method selection.

    This class analyzes a dataset and extracts characteristics that help
    determine the optimal generation method. Features include data types,
    correlations, distributions, and complexity indicators.

    Attributes:
        sample_size: Maximum rows to analyze (sampling for large datasets).

    Example:
        >>> extractor = MetaFeatureExtractor(sample_size=5000)
        >>> features = extractor.extract(df)
        >>> print(f"Dataset has {features.n_rows} rows, {features.n_columns} columns")
        >>> print(f"Complexity: {features.estimated_complexity:.2f}")
    """

    def __init__(self, sample_size: int = 10000):
        """Initialize the meta-feature extractor.

        Args:
            sample_size: Maximum rows to sample for analysis. Larger values
        """
        self.sample_size = sample_size

    def extract(self, data: pd.DataFrame) -> DatasetMetaFeatures:
        """Extract meta-features from a dataset.

        Args:
            data: Input DataFrame

        Returns:
            DatasetMetaFeatures with computed characteristics
        """
        # Sample if too large
        if len(data) > self.sample_size:
            data = data.sample(n=self.sample_size, random_state=42)

        features = DatasetMetaFeatures()

        # Basic counts
        features.n_rows = len(data)
        features.n_columns = len(data.columns)

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=["object", "category"]).columns.tolist()
        datetime_cols = data.select_dtypes(include=["datetime64"]).columns.tolist()

        features.n_numeric = len(numeric_cols)
        features.n_categorical = len(categorical_cols)
        features.n_datetime = len(datetime_cols)

        # Numeric features
        if numeric_cols:
            features.numeric_mean_skewness = self._compute_mean_skewness(data[numeric_cols])
            features.numeric_mean_kurtosis = self._compute_mean_kurtosis(data[numeric_cols])
            features.numeric_outlier_ratio = self._compute_outlier_ratio(data[numeric_cols])
            features.numeric_missing_ratio = data[numeric_cols].isnull().mean().mean()
            features.multimodal_columns = self._count_multimodal(data[numeric_cols])
            features.highly_skewed_columns = self._count_highly_skewed(data[numeric_cols])

        # Categorical features
        if categorical_cols:
            cardinalities = [data[col].nunique() for col in categorical_cols]
            features.categorical_mean_cardinality = np.mean(cardinalities)
            features.categorical_max_cardinality = max(cardinalities)
            features.categorical_imbalance_ratio = self._compute_imbalance_ratio(
                data[categorical_cols]
            )

        # Correlation features
        if len(numeric_cols) > 1:
            corr_matrix = data[numeric_cols].corr().abs()
            np.fill_diagonal(corr_matrix.values, 0)
            features.mean_correlation = corr_matrix.mean().mean()
            features.max_correlation = corr_matrix.max().max()
            features.correlation_clusters = self._estimate_correlation_clusters(corr_matrix)

        # Complexity estimate
        features.estimated_complexity = self._estimate_complexity(features)

        return features

    def _compute_mean_skewness(self, df: pd.DataFrame) -> float:
        """Compute mean skewness across numeric columns."""
        skewnesses = []
        for col in df.columns:
            try:
                skew = df[col].dropna().skew()
                if not np.isnan(skew):
                    skewnesses.append(abs(skew))
            except Exception:
                pass
        return np.mean(skewnesses) if skewnesses else 0.0

    def _compute_mean_kurtosis(self, df: pd.DataFrame) -> float:
        """Compute mean kurtosis across numeric columns."""
        kurtoses = []
        for col in df.columns:
            try:
                kurt = df[col].dropna().kurtosis()
                if not np.isnan(kurt):
                    kurtoses.append(abs(kurt))
            except Exception:
                pass
        return np.mean(kurtoses) if kurtoses else 0.0

    def _compute_outlier_ratio(self, df: pd.DataFrame) -> float:
        """Compute ratio of outliers using IQR method."""
        outlier_counts = []
        for col in df.columns:
            try:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                if iqr > 0:
                    outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).mean()
                    outlier_counts.append(outliers)
            except Exception:
                pass
        return np.mean(outlier_counts) if outlier_counts else 0.0

    def _count_multimodal(self, df: pd.DataFrame) -> int:
        """Count columns with multimodal distributions."""
        multimodal = 0
        for col in df.columns:
            try:
                values = df[col].dropna().values
                if len(values) > 100:
                    # Simple heuristic: check for multiple peaks in histogram
                    hist, _ = np.histogram(values, bins=20)
                    peaks = np.sum((hist[1:-1] > hist[:-2]) & (hist[1:-1] > hist[2:]))
                    if peaks > 1:
                        multimodal += 1
            except Exception:
                pass
        return multimodal

    def _count_highly_skewed(self, df: pd.DataFrame) -> int:
        """Count highly skewed columns (|skew| > 2)."""
        highly_skewed = 0
        for col in df.columns:
            try:
                skew = df[col].dropna().skew()
                if abs(skew) > 2:
                    highly_skewed += 1
            except Exception:
                pass
        return highly_skewed

    def _compute_imbalance_ratio(self, df: pd.DataFrame) -> float:
        """Compute average class imbalance ratio for categorical columns."""
        ratios = []
        for col in df.columns:
            try:
                value_counts = df[col].value_counts(normalize=True)
                if len(value_counts) > 1:
                    ratio = value_counts.iloc[0] / value_counts.iloc[-1]
                    ratios.append(min(ratio, 100))  # Cap extreme ratios
            except Exception:
                pass
        return np.mean(ratios) if ratios else 1.0

    def _estimate_correlation_clusters(self, corr_matrix: pd.DataFrame) -> int:
        """Estimate number of correlation clusters."""
        # Simple heuristic: count groups of highly correlated variables
        threshold = 0.7
        n_cols = len(corr_matrix)
        if n_cols < 2:
            return 0

        visited = set()
        clusters = 0

        for i in range(n_cols):
            if i not in visited:
                # BFS to find connected components
                queue = [i]
                visited.add(i)
                while queue:
                    current = queue.pop(0)
                    for j in range(n_cols):
                        if j not in visited and corr_matrix.iloc[current, j] > threshold:
                            visited.add(j)
                            queue.append(j)
                clusters += 1

        return clusters

    def _estimate_complexity(self, features: DatasetMetaFeatures) -> float:
        """Estimate overall data complexity score (0-1)."""
        complexity = 0.0

        # Size complexity
        if features.n_rows > 100000:
            complexity += 0.15
        elif features.n_rows > 10000:
            complexity += 0.1

        # Column complexity
        if features.n_columns > 50:
            complexity += 0.15
        elif features.n_columns > 20:
            complexity += 0.1

        # Distribution complexity
        if features.numeric_mean_skewness > 2:
            complexity += 0.15
        if features.multimodal_columns > 0:
            complexity += 0.1 * min(features.multimodal_columns / max(features.n_numeric, 1), 1)

        # Cardinality complexity
        if features.categorical_max_cardinality > 100:
            complexity += 0.15
        elif features.categorical_max_cardinality > 20:
            complexity += 0.1

        # Correlation complexity
        if features.max_correlation > 0.9:
            complexity += 0.1

        # Imbalance complexity
        if features.categorical_imbalance_ratio > 10:
            complexity += 0.1

        return min(complexity, 1.0)


class MethodSelector:
    """Select the best generation method based on meta-features."""

    # Method characteristics for rule-based selection
    METHOD_PROFILES = {
        GenerationMethod.GAUSSIAN_COPULA: {
            "max_complexity": 0.4,
            "handles_high_cardinality": False,
            "handles_multimodal": False,
            "speed": "fast",
            "quality_ceiling": 0.85,
        },
        GenerationMethod.FAST_ML: {
            "max_complexity": 0.3,
            "handles_high_cardinality": True,
            "handles_multimodal": False,
            "speed": "very_fast",
            "quality_ceiling": 0.75,
        },
        GenerationMethod.CTGAN: {
            "max_complexity": 0.9,
            "handles_high_cardinality": True,
            "handles_multimodal": True,
            "speed": "slow",
            "quality_ceiling": 0.95,
        },
        GenerationMethod.TVAE: {
            "max_complexity": 0.8,
            "handles_high_cardinality": True,
            "handles_multimodal": True,
            "speed": "medium",
            "quality_ceiling": 0.92,
        },
        GenerationMethod.COPULA_GAN: {
            "max_complexity": 0.85,
            "handles_high_cardinality": True,
            "handles_multimodal": True,
            "speed": "slow",
            "quality_ceiling": 0.93,
        },
    }

    def __init__(self, prefer_speed: bool = False, prefer_quality: bool = True):
        """Initialize method selector.

        Args:
            prefer_speed: Prefer faster methods when quality is similar
            prefer_quality: Prefer higher quality methods
        """
        self.prefer_speed = prefer_speed
        self.prefer_quality = prefer_quality

    def select(self, features: DatasetMetaFeatures) -> AutoMLResult:
        """Select the best method based on meta-features.

        Args:
            features: Extracted meta-features

        Returns:
            AutoMLResult with recommendation and analysis
        """
        import time

        start_time = time.time()

        recommendations = []

        for method in GenerationMethod:
            recommendation = self._evaluate_method(method, features)
            recommendations.append(recommendation)

        # Sort by confidence (descending)
        recommendations.sort(key=lambda r: r.confidence, reverse=True)

        # Apply preference adjustments
        if self.prefer_speed and not self.prefer_quality:
            # Boost fast methods
            for rec in recommendations:
                profile = self.METHOD_PROFILES[rec.method]
                if profile["speed"] == "very_fast":
                    rec.confidence *= 1.2
                elif profile["speed"] == "fast":
                    rec.confidence *= 1.1
            recommendations.sort(key=lambda r: r.confidence, reverse=True)

        best = recommendations[0]

        return AutoMLResult(
            recommended_method=best.method,
            confidence=min(best.confidence, 1.0),
            all_recommendations=recommendations,
            meta_features=features,
            analysis_time_seconds=time.time() - start_time,
        )

    def _evaluate_method(
        self, method: GenerationMethod, features: DatasetMetaFeatures
    ) -> MethodRecommendation:
        """Evaluate a specific method for the given features."""
        profile = self.METHOD_PROFILES[method]
        confidence = 0.5  # Base confidence
        reasons = []

        # Complexity match
        if features.estimated_complexity <= profile["max_complexity"]:
            confidence += 0.2
            reasons.append(
                f"Complexity ({features.estimated_complexity:.2f}) within method capability"
            )
        else:
            confidence -= 0.2
            reasons.append(
                f"Complexity ({features.estimated_complexity:.2f}) exceeds optimal range"
            )

        # High cardinality handling
        if features.categorical_max_cardinality > 50:
            if profile["handles_high_cardinality"]:
                confidence += 0.15
                reasons.append("Handles high cardinality categories well")
            else:
                confidence -= 0.2
                reasons.append("May struggle with high cardinality categories")

        # Multimodal handling
        if features.multimodal_columns > 0:
            if profile["handles_multimodal"]:
                confidence += 0.15
                reasons.append("Handles multimodal distributions")
            else:
                confidence -= 0.15
                reasons.append("May not capture multimodal distributions")

        # Size considerations
        if features.n_rows < 1000:
            if method == GenerationMethod.GAUSSIAN_COPULA:
                confidence += 0.1
                reasons.append("Good for small datasets")
            elif method in [GenerationMethod.CTGAN, GenerationMethod.TVAE]:
                confidence -= 0.1
                reasons.append("Deep learning methods need more data")
        elif features.n_rows > 50000:
            if method == GenerationMethod.CTGAN:
                confidence += 0.1
                reasons.append("Scales well with large datasets")

        # Skewness handling
        if features.highly_skewed_columns > features.n_numeric * 0.3:
            if method in [GenerationMethod.CTGAN, GenerationMethod.TVAE]:
                confidence += 0.1
                reasons.append("Handles skewed distributions")
            elif method == GenerationMethod.GAUSSIAN_COPULA:
                confidence -= 0.1
                reasons.append("Gaussian assumption may not hold for skewed data")

        # Correlation complexity
        if features.max_correlation > 0.8:
            if method in [GenerationMethod.CTGAN, GenerationMethod.COPULA_GAN]:
                confidence += 0.1
                reasons.append("Captures complex correlations")

        # Speed adjustment
        time_factors = {
            "very_fast": 0.5,
            "fast": 1.0,
            "medium": 3.0,
            "slow": 10.0,
        }

        return MethodRecommendation(
            method=method,
            confidence=max(0.1, min(confidence, 1.0)),
            reasons=reasons,
            estimated_quality=profile["quality_ceiling"] * confidence,
            estimated_time_factor=time_factors[profile["speed"]],
        )


class AutoMLSynthesizer:
    """Automatic synthetic data generation with intelligent method selection."""

    def __init__(
        self,
        prefer_speed: bool = False,
        prefer_quality: bool = True,
        fallback_method: GenerationMethod = GenerationMethod.GAUSSIAN_COPULA,
    ):
        """Initialize AutoML synthesizer.

        Args:
            prefer_speed: Prefer faster methods
            prefer_quality: Prefer higher quality methods
            fallback_method: Method to use if selection fails
        """
        self.extractor = MetaFeatureExtractor()
        self.selector = MethodSelector(prefer_speed=prefer_speed, prefer_quality=prefer_quality)
        self.fallback_method = fallback_method

        self._automl_result: Optional[AutoMLResult] = None
        self._generator = None
        self._fitted = False

    @property
    def selected_method(self) -> Optional[GenerationMethod]:
        """Get the auto-selected method."""
        if self._automl_result:
            return self._automl_result.recommended_method
        return None

    @property
    def selection_confidence(self) -> float:
        """Get confidence in the selection."""
        if self._automl_result:
            return self._automl_result.confidence
        return 0.0

    @property
    def meta_features(self) -> Optional[DatasetMetaFeatures]:
        """Get extracted meta-features."""
        if self._automl_result:
            return self._automl_result.meta_features
        return None

    def analyze(self, data: pd.DataFrame) -> AutoMLResult:
        """Analyze data and select best method without fitting.

        Args:
            data: Input DataFrame

        Returns:
            AutoMLResult with recommendation
        """
        features = self.extractor.extract(data)
        self._automl_result = self.selector.select(features)
        return self._automl_result

    def fit(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
        **kwargs,
    ) -> "AutoMLSynthesizer":
        """Analyze data, select method, and fit generator.

        Args:
            data: Training data
            discrete_columns: Categorical column names
            **kwargs: Additional arguments passed to generator

        Returns:
            Self for chaining
        """
        # Analyze and select method
        self._automl_result = self.analyze(data)
        method = self._automl_result.recommended_method

        # Create and fit generator
        self._generator = self._create_generator(method, **kwargs)

        try:
            if discrete_columns:
                self._generator.fit(data, discrete_columns=discrete_columns)
            else:
                # Auto-detect discrete columns
                discrete_columns = data.select_dtypes(
                    include=["object", "category", "bool"]
                ).columns.tolist()
                self._generator.fit(data, discrete_columns=discrete_columns)

            self._fitted = True
        except Exception as e:
            # Fallback on error
            print(
                f"Warning: {method.value} failed ({e}), falling back to {self.fallback_method.value}"
            )
            self._generator = self._create_generator(self.fallback_method, **kwargs)
            if discrete_columns:
                self._generator.fit(data, discrete_columns=discrete_columns)
            else:
                discrete_columns = data.select_dtypes(
                    include=["object", "category", "bool"]
                ).columns.tolist()
                self._generator.fit(data, discrete_columns=discrete_columns)
            self._fitted = True

        return self

    def generate(self, n_samples: int) -> pd.DataFrame:
        """Generate synthetic samples.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame with synthetic data
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before generate()")

        return self._generator.generate(n_samples)

    def _create_generator(self, method: GenerationMethod, **kwargs):
        """Create generator instance for the specified method using plugin registry."""
        from genesis.plugins import get_generator

        # Map generation methods to plugin names
        method_to_plugin = {
            GenerationMethod.GAUSSIAN_COPULA: "gaussian_copula",
            GenerationMethod.FAST_ML: "gaussian_copula",  # Alias for fast statistical method
            GenerationMethod.CTGAN: "ctgan",
            GenerationMethod.TVAE: "tvae",
            GenerationMethod.COPULA_GAN: "ctgan",  # Use CTGAN as base
        }

        plugin_name = method_to_plugin.get(method, "gaussian_copula")
        generator_class = get_generator(plugin_name)

        if generator_class is None:
            # Fallback to direct import if plugin not found
            import warnings

            from genesis.generators.tabular import GaussianCopulaGenerator

            warnings.warn(
                f"Generator '{plugin_name}' not found in plugin registry, "
                "using GaussianCopulaGenerator fallback"
            )
            generator_class = GaussianCopulaGenerator

        return generator_class(**kwargs)

    def get_selection_report(self) -> str:
        """Get human-readable report of method selection."""
        if not self._automl_result:
            return "No analysis performed yet. Call analyze() or fit() first."

        result = self._automl_result
        lines = [
            "=" * 60,
            "AutoML Synthesis Report",
            "=" * 60,
            "",
            f"Recommended Method: {result.recommended_method.value}",
            f"Confidence: {result.confidence:.1%}",
            f"Analysis Time: {result.analysis_time_seconds:.3f}s",
            "",
            "Dataset Characteristics:",
            f"  - Rows: {result.meta_features.n_rows:,}",
            f"  - Columns: {result.meta_features.n_columns}",
            f"  - Numeric: {result.meta_features.n_numeric}",
            f"  - Categorical: {result.meta_features.n_categorical}",
            f"  - Complexity: {result.meta_features.estimated_complexity:.2f}",
            "",
            "Method Rankings:",
        ]

        for i, rec in enumerate(result.all_recommendations[:5], 1):
            lines.append(f"  {i}. {rec.method.value}: {rec.confidence:.1%}")
            for reason in rec.reasons[:2]:
                lines.append(f"     - {reason}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


def auto_synthesize(
    data: pd.DataFrame,
    n_samples: Optional[int] = None,
    discrete_columns: Optional[List[str]] = None,
    prefer_speed: bool = False,
    prefer_quality: bool = True,
    verbose: bool = True,
    **kwargs,
) -> Tuple[pd.DataFrame, AutoMLResult]:
    """Convenience function for automatic synthetic data generation.

    Args:
        data: Training data
        n_samples: Number of samples to generate (default: same as input)
        discrete_columns: Categorical column names
        prefer_speed: Prefer faster methods
        prefer_quality: Prefer higher quality methods
        verbose: Print selection report
        **kwargs: Additional arguments for generator

    Returns:
        Tuple of (synthetic_data, automl_result)
    """
    synthesizer = AutoMLSynthesizer(
        prefer_speed=prefer_speed,
        prefer_quality=prefer_quality,
    )

    synthesizer.fit(data, discrete_columns=discrete_columns, **kwargs)

    if verbose:
        print(synthesizer.get_selection_report())

    n_samples = n_samples or len(data)
    synthetic = synthesizer.generate(n_samples)

    return synthetic, synthesizer._automl_result
