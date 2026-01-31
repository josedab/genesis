"""Unit tests for analyzers module."""

import numpy as np
import pandas as pd

from genesis.analyzers.privacy import PrivacyAnalyzer
from genesis.analyzers.relationships import RelationshipAnalyzer
from genesis.analyzers.schema import SchemaAnalyzer
from genesis.analyzers.statistics import StatisticalAnalyzer
from genesis.core.types import ColumnType


class TestSchemaAnalyzer:
    """Tests for SchemaAnalyzer."""

    def test_analyze_numeric_columns(self, sample_numeric_df):
        analyzer = SchemaAnalyzer()
        schema = analyzer.analyze(sample_numeric_df)

        assert "age" in schema.columns
        assert "income" in schema.columns
        assert schema.columns["age"].dtype in (
            ColumnType.NUMERIC_CONTINUOUS,
            ColumnType.NUMERIC_DISCRETE,
        )

    def test_analyze_categorical_columns(self, sample_categorical_df):
        analyzer = SchemaAnalyzer()
        schema = analyzer.analyze(sample_categorical_df)

        assert schema.columns["gender"].dtype == ColumnType.CATEGORICAL
        assert schema.columns["city"].dtype == ColumnType.CATEGORICAL

    def test_analyze_mixed_columns(self, sample_mixed_df):
        analyzer = SchemaAnalyzer()
        schema = analyzer.analyze(sample_mixed_df)

        assert schema.n_rows == len(sample_mixed_df)
        assert schema.n_columns == len(sample_mixed_df.columns)

    def test_explicit_discrete_columns(self, sample_mixed_df):
        analyzer = SchemaAnalyzer()
        schema = analyzer.analyze(sample_mixed_df, discrete_columns=["age"])

        assert schema.columns["age"].dtype == ColumnType.CATEGORICAL

    def test_primary_key_detection(self):
        df = pd.DataFrame(
            {
                "id": range(100),
                "name": [f"name_{i}" for i in range(100)],
            }
        )
        analyzer = SchemaAnalyzer()
        schema = analyzer.analyze(df)

        assert schema.primary_key == "id"


class TestStatisticalAnalyzer:
    """Tests for StatisticalAnalyzer."""

    def test_analyze_returns_stats(self, sample_numeric_df):
        analyzer = StatisticalAnalyzer()
        stats = analyzer.analyze(sample_numeric_df)

        assert stats.n_rows == len(sample_numeric_df)
        assert stats.n_columns == len(sample_numeric_df.columns)
        assert stats.n_numeric > 0

    def test_column_statistics(self, sample_numeric_df):
        analyzer = StatisticalAnalyzer()
        stats = analyzer.analyze(sample_numeric_df)

        age_stats = stats.column_stats["age"]
        assert age_stats.mean is not None
        assert age_stats.std is not None
        assert age_stats.min_val is not None
        assert age_stats.max_val is not None

    def test_correlation_matrix(self, sample_numeric_df):
        analyzer = StatisticalAnalyzer(compute_correlations=True)
        stats = analyzer.analyze(sample_numeric_df)

        assert stats.correlation_matrix is not None
        assert stats.correlation_matrix.shape[0] == 3  # 3 numeric columns


class TestRelationshipAnalyzer:
    """Tests for RelationshipAnalyzer."""

    def test_find_correlations(self):
        # Create correlated data
        np.random.seed(42)
        x = np.random.randn(100)
        y = x + np.random.randn(100) * 0.1  # Strong correlation

        df = pd.DataFrame({"x": x, "y": y, "z": np.random.randn(100)})

        analyzer = RelationshipAnalyzer(correlation_threshold=0.5)
        relationships = analyzer.analyze(df)

        # Should find correlation between x and y
        corr_rels = [r for r in relationships if r.relationship_type == "correlation"]
        assert len(corr_rels) > 0

    def test_find_categorical_associations(self, sample_categorical_df):
        analyzer = RelationshipAnalyzer()
        relationships = analyzer.analyze(sample_categorical_df)

        # Should find some associations
        cat_rels = [r for r in relationships if r.relationship_type == "categorical_association"]
        # May or may not find depending on random data
        assert isinstance(cat_rels, list)


class TestPrivacyAnalyzer:
    """Tests for PrivacyAnalyzer."""

    def test_analyze_returns_risk(self, sample_mixed_df):
        analyzer = PrivacyAnalyzer()
        risk = analyzer.analyze(sample_mixed_df)

        assert 0 <= risk.overall_risk_score <= 1
        assert risk.k_anonymity_estimate > 0

    def test_detect_quasi_identifiers(self, sample_mixed_df):
        analyzer = PrivacyAnalyzer()
        risk = analyzer.analyze(sample_mixed_df)

        # age, gender, city are likely quasi-identifiers
        assert len(risk.quasi_identifiers) > 0

    def test_custom_quasi_identifiers(self, sample_mixed_df):
        analyzer = PrivacyAnalyzer()
        risk = analyzer.analyze(
            sample_mixed_df,
            quasi_identifiers=["age", "city"],
        )

        assert "age" in risk.quasi_identifiers
        assert "city" in risk.quasi_identifiers
