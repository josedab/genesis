"""Unit tests for utility functions."""

import numpy as np
import pandas as pd
import pytest

from genesis.core.exceptions import ValidationError
from genesis.utils.preprocessing import (
    balance_classes,
    clip_outliers,
    detect_outliers,
    handle_missing_values,
    normalize_column_names,
)
from genesis.utils.transformers import (
    CategoricalTransformer,
    DataTransformer,
    NumericalTransformer,
)
from genesis.utils.validation import (
    validate_dataframe,
    validate_positive,
    validate_probability,
    validate_range,
)


class TestPreprocessing:
    """Tests for preprocessing utilities."""

    def test_handle_missing_values(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, np.nan, 4],
                "b": ["x", np.nan, "y", "z"],
            }
        )

        result, fill_values = handle_missing_values(df)
        assert result.isna().sum().sum() == 0
        assert "a" in fill_values

    def test_detect_outliers_iqr(self):
        df = pd.DataFrame(
            {
                "a": list(range(100)) + [1000],  # 1000 is outlier
            }
        )

        outliers = detect_outliers(df, method="iqr", columns=["a"])
        assert outliers["a"][-1]  # Last value should be marked as outlier

    def test_clip_outliers(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 100, 5],
            }
        )

        result = clip_outliers(df, lower_percentile=5, upper_percentile=95)
        assert result["a"].max() < 100

    def test_normalize_column_names(self):
        df = pd.DataFrame(
            {
                "Column Name": [1],
                "another-column": [2],
                "123start": [3],
            }
        )

        result, mapping = normalize_column_names(df)
        assert all(col.islower() or col.startswith("col_") for col in result.columns)

    def test_balance_classes_oversample(self):
        df = pd.DataFrame(
            {
                "feature": range(110),
                "target": ["A"] * 100 + ["B"] * 10,
            }
        )

        result = balance_classes(df, "target", strategy="oversample")
        counts = result["target"].value_counts()
        assert counts["A"] == counts["B"]


class TestTransformers:
    """Tests for data transformers."""

    def test_numerical_transformer(self):
        data = pd.Series(np.random.randn(100))
        transformer = NumericalTransformer()

        transformer.fit(data)
        transformed = transformer.transform(data)

        # Should output multi-dimensional (value + modes)
        assert transformed.ndim == 2

        # Inverse should approximately recover original
        recovered = transformer.inverse_transform(transformed)
        assert len(recovered) == len(data)

    def test_categorical_transformer(self):
        data = pd.Series(["A", "B", "C", "A", "B"])
        transformer = CategoricalTransformer()

        transformer.fit(data)
        transformed = transformer.transform(data)

        # Should be one-hot encoded
        assert transformed.shape[1] == 3

        recovered = transformer.inverse_transform(transformed)
        assert list(recovered) == list(data)

    def test_data_transformer(self, sample_mixed_df):
        transformer = DataTransformer()
        transformer.fit(sample_mixed_df, discrete_columns=["gender", "city", "active"])

        transformed = transformer.transform(sample_mixed_df)
        assert transformed.shape[0] == len(sample_mixed_df)

        recovered = transformer.inverse_transform(transformed)
        assert set(recovered.columns) == set(sample_mixed_df.columns)


class TestValidation:
    """Tests for validation utilities."""

    def test_validate_dataframe_valid(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = validate_dataframe(df)
        assert isinstance(result, pd.DataFrame)

    def test_validate_dataframe_none(self):
        with pytest.raises(ValidationError):
            validate_dataframe(None)

    def test_validate_dataframe_empty(self):
        with pytest.raises(ValidationError):
            validate_dataframe(pd.DataFrame())

    def test_validate_positive(self):
        assert validate_positive(5) == 5.0
        with pytest.raises(ValidationError):
            validate_positive(-1)

    def test_validate_range(self):
        assert validate_range(5, 0, 10) == 5.0
        with pytest.raises(ValidationError):
            validate_range(15, 0, 10)

    def test_validate_probability(self):
        assert validate_probability(0.5) == 0.5
        with pytest.raises(ValidationError):
            validate_probability(1.5)
