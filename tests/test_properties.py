"""Property-based tests using Hypothesis.

These tests verify invariants that should hold for any valid input,
helping discover edge cases that unit tests might miss.
"""

import numpy as np
import pandas as pd
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st


# Custom strategies for valid data generation
@st.composite
def numeric_dataframes(draw, min_rows=10, max_rows=100, min_cols=1, max_cols=5):
    """Generate DataFrames with numeric columns."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))

    columns = {}
    for i in range(n_cols):
        col_name = f"col_{i}"
        # Generate values without NaN/Inf
        values = draw(
            st.lists(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        columns[col_name] = values

    return pd.DataFrame(columns)


@st.composite
def mixed_dataframes(draw, min_rows=10, max_rows=50, n_numeric=2, n_categorical=2):
    """Generate DataFrames with mixed column types."""
    n_rows = draw(st.integers(min_value=min_rows, max_value=max_rows))

    columns = {}

    # Numeric columns
    for i in range(n_numeric):
        values = draw(
            st.lists(
                st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        columns[f"numeric_{i}"] = values

    # Categorical columns
    for i in range(n_categorical):
        n_categories = draw(st.integers(min_value=2, max_value=5))
        categories = [f"cat_{j}" for j in range(n_categories)]
        values = draw(
            st.lists(
                st.sampled_from(categories),
                min_size=n_rows,
                max_size=n_rows,
            )
        )
        columns[f"categorical_{i}"] = values

    return pd.DataFrame(columns)


class TestGaussianCopulaProperties:
    """Property-based tests for Gaussian Copula generator."""

    @given(data=numeric_dataframes(min_rows=20, max_rows=50, min_cols=2, max_cols=4))
    @settings(max_examples=10, deadline=60000, suppress_health_check=[HealthCheck.too_slow])
    def test_output_shape_matches_request(self, data):
        """Generated data should have the requested number of rows and same columns."""
        from genesis.generators.tabular import GaussianCopulaGenerator

        assume(len(data) >= 10)  # Need minimum data
        assume(data.std().min() > 0)  # Need some variance

        generator = GaussianCopulaGenerator(verbose=False)
        generator.fit(data)

        n_samples = 20
        synthetic = generator.generate(n_samples=n_samples)

        assert len(synthetic) == n_samples
        assert set(synthetic.columns) == set(data.columns)

    @given(data=numeric_dataframes(min_rows=30, max_rows=50, min_cols=2, max_cols=3))
    @settings(max_examples=5, deadline=60000, suppress_health_check=[HealthCheck.too_slow])
    def test_no_nan_in_output(self, data):
        """Generated data should not contain NaN values."""
        from genesis.generators.tabular import GaussianCopulaGenerator

        assume(len(data) >= 10)
        assume(data.std().min() > 0)

        generator = GaussianCopulaGenerator(verbose=False)
        generator.fit(data)

        synthetic = generator.generate(n_samples=20)

        assert not synthetic.isna().any().any()

    @given(data=mixed_dataframes(min_rows=30, max_rows=50))
    @settings(max_examples=5, deadline=60000, suppress_health_check=[HealthCheck.too_slow])
    def test_categorical_values_preserved(self, data):
        """Categorical columns should only contain known categories."""
        from genesis.generators.tabular import GaussianCopulaGenerator

        assume(len(data) >= 10)

        cat_cols = [c for c in data.columns if c.startswith("categorical_")]

        generator = GaussianCopulaGenerator(verbose=False)
        generator.fit(data, discrete_columns=cat_cols)

        synthetic = generator.generate(n_samples=30)

        for col in cat_cols:
            real_categories = set(data[col].unique())
            synthetic_categories = set(synthetic[col].unique())
            assert synthetic_categories.issubset(real_categories)


class TestConstraintProperties:
    """Property-based tests for constraint system."""

    @given(
        values=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=5, max_size=20
        ),
        min_val=st.floats(min_value=-50, max_value=0),
        max_val=st.floats(min_value=1, max_value=50),
    )
    def test_range_constraint_enforced(self, values, min_val, max_val):
        """Range constraint should clip values to valid range."""
        from genesis.core.constraints import Constraint

        assume(min_val < max_val)

        df = pd.DataFrame({"value": values})
        constraint = Constraint.range("value", min_val, max_val)

        result = constraint.transform(df)

        assert result["value"].min() >= min_val
        assert result["value"].max() <= max_val

    @given(
        values=st.lists(
            st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=5, max_size=20
        )
    )
    def test_positive_constraint_enforced(self, values):
        """Positive constraint should make all values positive."""
        from genesis.core.constraints import Constraint

        df = pd.DataFrame({"value": values})
        constraint = Constraint.positive("value")

        result = constraint.transform(df)

        assert (result["value"] > 0).all()


class TestTransformerProperties:
    """Property-based tests for data transformers."""

    @given(data=numeric_dataframes(min_rows=20, max_rows=50, min_cols=2, max_cols=3))
    @settings(max_examples=10, deadline=30000, suppress_health_check=[HealthCheck.too_slow])
    def test_transformer_roundtrip(self, data):
        """Transform + inverse_transform should approximately recover original."""
        from genesis.utils.transformers import DataTransformer

        assume(len(data) >= 10)
        assume(data.std().min() > 0)

        transformer = DataTransformer()
        transformer.fit(data)

        transformed = transformer.transform(data)
        recovered = transformer.inverse_transform(transformed)

        # Should have same shape and columns
        assert recovered.shape == data.shape
        assert set(recovered.columns) == set(data.columns)

        # Values should be close (allowing for mode-specific normalization approximation)
        for col in data.columns:
            # Check correlation is high (values are related)
            corr = np.corrcoef(data[col].values, recovered[col].values)[0, 1]
            assert corr > 0.8 or np.isclose(data[col].std(), 0)


class TestEvaluationProperties:
    """Property-based tests for evaluation metrics."""

    @given(
        n=st.integers(min_value=50, max_value=100),
        mean=st.floats(min_value=-10, max_value=10),
        std=st.floats(min_value=0.1, max_value=5),
    )
    @settings(max_examples=10)
    def test_identical_data_high_fidelity(self, n, mean, std):
        """Identical real and synthetic data should have perfect fidelity."""
        from genesis.evaluation.statistical import kolmogorov_smirnov_test

        data = pd.Series(np.random.normal(mean, std, n))

        result = kolmogorov_smirnov_test(data, data)

        assert result["score"] == 1.0  # Identical data = perfect score

    @given(
        n=st.integers(min_value=50, max_value=100),
        shift=st.floats(min_value=5, max_value=20),
    )
    @settings(max_examples=10)
    def test_different_data_lower_fidelity(self, n, shift):
        """Very different data should have lower fidelity score."""
        from genesis.evaluation.statistical import kolmogorov_smirnov_test

        real = pd.Series(np.random.normal(0, 1, n))
        synthetic = pd.Series(np.random.normal(shift, 1, n))

        result = kolmogorov_smirnov_test(real, synthetic)

        # Large shift should result in low score
        assert result["score"] < 0.5
