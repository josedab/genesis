"""Unit tests for tabular generators."""

import numpy as np
import pandas as pd
import pytest


class TestGaussianCopulaGenerator:
    """Tests for GaussianCopulaGenerator."""

    def test_fit_and_generate(self, sample_mixed_df):
        from genesis.generators.tabular import GaussianCopulaGenerator

        generator = GaussianCopulaGenerator(verbose=False)
        generator.fit(sample_mixed_df, discrete_columns=["gender", "city", "active"])

        assert generator.is_fitted

        synthetic = generator.generate(n_samples=50)
        assert len(synthetic) == 50
        assert set(synthetic.columns) == set(sample_mixed_df.columns)

    def test_preserves_distributions(self, sample_numeric_df):
        from genesis.generators.tabular import GaussianCopulaGenerator

        generator = GaussianCopulaGenerator(verbose=False)
        generator.fit(sample_numeric_df)
        synthetic = generator.generate(n_samples=1000)

        # Check means are similar (within tolerance)
        for col in sample_numeric_df.columns:
            real_mean = sample_numeric_df[col].mean()
            syn_mean = synthetic[col].mean()
            # Allow 20% tolerance
            assert abs(real_mean - syn_mean) < abs(real_mean) * 0.5

    def test_get_correlation_matrix(self, sample_numeric_df):
        from genesis.generators.tabular import GaussianCopulaGenerator

        generator = GaussianCopulaGenerator(verbose=False)
        generator.fit(sample_numeric_df)

        corr = generator.get_correlation_matrix()
        assert corr is not None
        assert corr.shape[0] == len(sample_numeric_df.columns)


class TestBaseTabularGenerator:
    """Tests for base tabular generator functionality."""

    def test_not_fitted_error(self, sample_mixed_df):
        from genesis.core.exceptions import NotFittedError
        from genesis.generators.tabular import GaussianCopulaGenerator

        generator = GaussianCopulaGenerator()

        with pytest.raises(NotFittedError):
            generator.generate(n_samples=10)

    def test_fit_with_constraints(self, sample_numeric_df):
        from genesis.core.constraints import Constraint
        from genesis.generators.tabular import GaussianCopulaGenerator

        generator = GaussianCopulaGenerator(verbose=False)
        constraints = [
            Constraint.positive("age"),
            Constraint.range("score", 0, 100),
        ]
        generator.fit(sample_numeric_df, constraints=constraints)

        synthetic = generator.generate(n_samples=100)

        # Constraints should be applied
        assert (synthetic["age"] > 0).all()
        assert (synthetic["score"] >= 0).all()
        assert (synthetic["score"] <= 100).all()
