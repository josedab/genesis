"""Tests for AutoML Synthesis module."""

import numpy as np
import pandas as pd
import pytest

from genesis.automl import (
    AutoMLResult,
    AutoMLSynthesizer,
    GenerationMethod,
    MetaFeatureExtractor,
    MethodSelector,
    auto_synthesize,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 500),
            "income": np.random.normal(50000, 15000, 500),
            "category": np.random.choice(["A", "B", "C"], 500),
            "is_active": np.random.choice([True, False], 500),
        }
    )


class TestMetaFeatureExtractor:
    """Tests for MetaFeatureExtractor."""

    def test_extract_basic_features(self, sample_data: pd.DataFrame) -> None:
        """Test basic feature extraction."""
        extractor = MetaFeatureExtractor()
        features = extractor.extract(sample_data)

        assert features.n_rows == 500
        assert features.n_columns == 4
        assert features.n_numeric == 2
        assert features.n_categorical >= 1

    def test_extract_numeric_stats(self, sample_data: pd.DataFrame) -> None:
        """Test numeric statistics extraction."""
        extractor = MetaFeatureExtractor()
        features = extractor.extract(sample_data)

        assert features.numeric_mean_skewness >= 0
        assert features.numeric_outlier_ratio >= 0
        assert features.numeric_outlier_ratio <= 1

    def test_to_vector(self, sample_data: pd.DataFrame) -> None:
        """Test conversion to feature vector."""
        extractor = MetaFeatureExtractor()
        features = extractor.extract(sample_data)

        vector = features.to_vector()
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 16


class TestMethodSelector:
    """Tests for MethodSelector."""

    def test_select_returns_recommendation(self, sample_data: pd.DataFrame) -> None:
        """Test that selector returns valid recommendation."""
        extractor = MetaFeatureExtractor()
        features = extractor.extract(sample_data)

        selector = MethodSelector()
        result = selector.select(features)

        assert isinstance(result, AutoMLResult)
        assert result.recommended_method in GenerationMethod
        assert 0 <= result.confidence <= 1

    def test_all_methods_evaluated(self, sample_data: pd.DataFrame) -> None:
        """Test that all methods are evaluated."""
        extractor = MetaFeatureExtractor()
        features = extractor.extract(sample_data)

        selector = MethodSelector()
        result = selector.select(features)

        methods = {r.method for r in result.all_recommendations}
        assert len(methods) == len(GenerationMethod)

    def test_prefer_speed(self, sample_data: pd.DataFrame) -> None:
        """Test speed preference."""
        extractor = MetaFeatureExtractor()
        features = extractor.extract(sample_data)

        selector = MethodSelector(prefer_speed=True, prefer_quality=False)
        result = selector.select(features)

        # With speed preference, confidence should be adjusted
        # Find fast method recommendations
        fast_recs = [
            r
            for r in result.all_recommendations
            if r.method in [GenerationMethod.GAUSSIAN_COPULA, GenerationMethod.FAST_ML]
        ]

        # Fast methods should exist in recommendations
        assert len(fast_recs) > 0


class TestAutoMLSynthesizer:
    """Tests for AutoMLSynthesizer."""

    def test_analyze(self, sample_data: pd.DataFrame) -> None:
        """Test analysis without fitting."""
        synthesizer = AutoMLSynthesizer()
        result = synthesizer.analyze(sample_data)

        assert result.recommended_method is not None
        assert result.meta_features is not None

    def test_fit_and_generate(self, sample_data: pd.DataFrame) -> None:
        """Test full fit and generate cycle."""
        synthesizer = AutoMLSynthesizer()
        synthesizer.fit(sample_data)

        synthetic = synthesizer.generate(100)

        assert len(synthetic) == 100
        assert list(synthetic.columns) == list(sample_data.columns)

    def test_selection_report(self, sample_data: pd.DataFrame) -> None:
        """Test selection report generation."""
        synthesizer = AutoMLSynthesizer()
        synthesizer.fit(sample_data)

        report = synthesizer.get_selection_report()

        assert "AutoML" in report
        assert "Recommended Method" in report


class TestAutoSynthesize:
    """Tests for auto_synthesize convenience function."""

    def test_returns_data_and_result(self, sample_data: pd.DataFrame) -> None:
        """Test that function returns both data and result."""
        synthetic, result = auto_synthesize(
            sample_data,
            n_samples=50,
            verbose=False,
        )

        assert len(synthetic) == 50
        assert isinstance(result, AutoMLResult)
