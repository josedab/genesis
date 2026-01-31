"""Unit tests for core module."""

import pandas as pd

from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.constraints import (
    Constraint,
    ConstraintSet,
)
from genesis.core.types import ColumnMetadata, ColumnType, DataSchema


class TestGeneratorConfig:
    """Tests for GeneratorConfig."""

    def test_default_values(self):
        config = GeneratorConfig()
        assert config.epochs == 300
        assert config.batch_size == 500
        assert config.verbose is True

    def test_custom_values(self):
        config = GeneratorConfig(epochs=100, batch_size=256, verbose=False)
        assert config.epochs == 100
        assert config.batch_size == 256
        assert config.verbose is False

    def test_to_dict(self):
        config = GeneratorConfig(epochs=50)
        d = config.to_dict()
        assert d["epochs"] == 50
        assert "batch_size" in d


class TestPrivacyConfig:
    """Tests for PrivacyConfig."""

    def test_default_values(self):
        config = PrivacyConfig()
        assert config.enable_differential_privacy is False
        assert config.epsilon == 1.0

    def test_privacy_level_presets(self):
        config = PrivacyConfig(privacy_level="high")
        assert config.enable_differential_privacy is True
        assert config.suppress_rare_categories is True

    def test_to_dict(self):
        config = PrivacyConfig(epsilon=0.5)
        d = config.to_dict()
        assert d["epsilon"] == 0.5


class TestConstraints:
    """Tests for constraints."""

    def test_positive_constraint_validate(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        constraint = Constraint.positive("a")
        assert constraint.validate(df) == True

        df_negative = pd.DataFrame({"a": [-1, 2, 3]})
        assert constraint.validate(df_negative) == False

    def test_positive_constraint_transform(self):
        df = pd.DataFrame({"a": [-1, -2, 3, 4]})
        constraint = Constraint.positive("a")
        result = constraint.transform(df)
        assert (result["a"] > 0).all()

    def test_range_constraint_validate(self):
        df = pd.DataFrame({"age": [25, 30, 35]})
        constraint = Constraint.range("age", 0, 120)
        assert constraint.validate(df) == True

        df_invalid = pd.DataFrame({"age": [25, 150]})
        assert constraint.validate(df_invalid) == False

    def test_range_constraint_transform(self):
        df = pd.DataFrame({"age": [25, 150, -5]})
        constraint = Constraint.range("age", 0, 100)
        result = constraint.transform(df)
        assert result["age"].max() <= 100
        assert result["age"].min() >= 0

    def test_unique_constraint_validate(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4]})
        constraint = Constraint.unique("id")
        assert constraint.validate(df) is True

        df_duplicates = pd.DataFrame({"id": [1, 2, 2, 3]})
        assert constraint.validate(df_duplicates) is False

    def test_unique_constraint_transform(self):
        df = pd.DataFrame({"id": [1, 2, 2, 3]})
        constraint = Constraint.unique("id")
        result = constraint.transform(df)
        assert len(result["id"]) == len(result["id"].unique())

    def test_constraint_set(self):
        constraints = ConstraintSet(
            [
                Constraint.positive("a"),
                Constraint.range("a", 0, 100),
            ]
        )

        df = pd.DataFrame({"a": [1, 50, 99]})
        results = constraints.validate(df)
        assert all(results.values())

        df_invalid = pd.DataFrame({"a": [-1, 50, 150]})
        results = constraints.validate(df_invalid)
        assert not all(results.values())

    def test_constraint_set_transform(self):
        constraints = ConstraintSet(
            [
                Constraint.positive("a"),
                Constraint.range("a", 0, 100),
            ]
        )

        df = pd.DataFrame({"a": [-10, 50, 150]})
        result = constraints.transform(df)
        assert (result["a"] > 0).all()
        assert result["a"].max() <= 100


class TestDataSchema:
    """Tests for DataSchema."""

    def test_get_column_names(self):
        schema = DataSchema()
        schema.columns["age"] = ColumnMetadata(
            name="age",
            dtype=ColumnType.NUMERIC_CONTINUOUS,
        )
        schema.columns["gender"] = ColumnMetadata(
            name="gender",
            dtype=ColumnType.CATEGORICAL,
        )

        all_cols = schema.get_column_names()
        assert len(all_cols) == 2

        numeric_cols = schema.get_column_names(ColumnType.NUMERIC_CONTINUOUS)
        assert numeric_cols == ["age"]

    def test_get_numeric_columns(self):
        schema = DataSchema()
        schema.columns["age"] = ColumnMetadata(
            name="age",
            dtype=ColumnType.NUMERIC_CONTINUOUS,
        )
        schema.columns["count"] = ColumnMetadata(
            name="count",
            dtype=ColumnType.NUMERIC_DISCRETE,
        )
        schema.columns["name"] = ColumnMetadata(
            name="name",
            dtype=ColumnType.CATEGORICAL,
        )

        numeric = schema.get_numeric_columns()
        assert "age" in numeric
        assert "count" in numeric
        assert "name" not in numeric

    def test_to_dict(self):
        schema = DataSchema(n_rows=100, n_columns=3)
        d = schema.to_dict()
        assert d["n_rows"] == 100
        assert d["n_columns"] == 3
