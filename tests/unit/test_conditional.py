"""Tests for conditional generation module."""

import numpy as np
import pandas as pd
import pytest

from genesis.generators.conditional import (
    Condition,
    ConditionalSampler,
    ConditionSet,
    Operator,
    ScenarioGenerator,
    Upsampler,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, 1000),
            "income": np.random.normal(50000, 20000, 1000),
            "country": np.random.choice(["US", "UK", "CA", "FR"], 1000),
            "is_fraud": np.random.choice([True, False], 1000, p=[0.1, 0.9]),
        }
    )


class TestCondition:
    """Tests for Condition class."""

    def test_eq_operator(self, sample_data: pd.DataFrame) -> None:
        """Test equality operator."""
        cond = Condition("country", Operator.EQ, "US")
        mask = cond.evaluate(sample_data)
        assert mask.sum() > 0
        assert all(sample_data.loc[mask, "country"] == "US")

    def test_gt_operator(self, sample_data: pd.DataFrame) -> None:
        """Test greater than operator."""
        cond = Condition("age", Operator.GT, 50)
        mask = cond.evaluate(sample_data)
        assert all(sample_data.loc[mask, "age"] > 50)

    def test_ge_operator(self, sample_data: pd.DataFrame) -> None:
        """Test greater than or equal operator."""
        cond = Condition("age", Operator.GE, 50)
        mask = cond.evaluate(sample_data)
        assert all(sample_data.loc[mask, "age"] >= 50)

    def test_lt_operator(self, sample_data: pd.DataFrame) -> None:
        """Test less than operator."""
        cond = Condition("income", Operator.LT, 40000)
        mask = cond.evaluate(sample_data)
        assert all(sample_data.loc[mask, "income"] < 40000)

    def test_le_operator(self, sample_data: pd.DataFrame) -> None:
        """Test less than or equal operator."""
        cond = Condition("income", Operator.LE, 40000)
        mask = cond.evaluate(sample_data)
        assert all(sample_data.loc[mask, "income"] <= 40000)

    def test_in_operator(self, sample_data: pd.DataFrame) -> None:
        """Test IN operator."""
        cond = Condition("country", Operator.IN, ["US", "UK"])
        mask = cond.evaluate(sample_data)
        assert all(sample_data.loc[mask, "country"].isin(["US", "UK"]))

    def test_not_in_operator(self, sample_data: pd.DataFrame) -> None:
        """Test NOT IN operator."""
        cond = Condition("country", Operator.NOT_IN, ["US", "UK"])
        mask = cond.evaluate(sample_data)
        assert all(~sample_data.loc[mask, "country"].isin(["US", "UK"]))

    def test_between_operator(self, sample_data: pd.DataFrame) -> None:
        """Test BETWEEN operator."""
        cond = Condition("age", Operator.BETWEEN, (30, 50))
        mask = cond.evaluate(sample_data)
        ages = sample_data.loc[mask, "age"]
        assert all((ages >= 30) & (ages <= 50))

    def test_between_requires_tuple(self) -> None:
        """Test that BETWEEN operator requires tuple."""
        with pytest.raises(Exception):
            Condition("age", Operator.BETWEEN, 30)

    def test_to_dict_from_dict(self) -> None:
        """Test serialization round trip."""
        cond = Condition("age", Operator.GE, 18)
        d = cond.to_dict()
        restored = Condition.from_dict(d)
        assert restored.column == cond.column
        assert restored.operator == cond.operator
        assert restored.value == cond.value


class TestConditionSet:
    """Tests for ConditionSet class."""

    def test_multiple_conditions(self, sample_data: pd.DataFrame) -> None:
        """Test multiple conditions combined with AND."""
        cond_set = ConditionSet(
            [
                Condition("age", Operator.GE, 30),
                Condition("country", Operator.EQ, "US"),
            ]
        )
        mask = cond_set.evaluate(sample_data)
        matching = sample_data[mask]
        assert all(matching["age"] >= 30)
        assert all(matching["country"] == "US")

    def test_from_dict_simple(self, sample_data: pd.DataFrame) -> None:
        """Test creating ConditionSet from simple dict."""
        cond_set = ConditionSet.from_dict({"country": "US"})
        mask = cond_set.evaluate(sample_data)
        assert all(sample_data.loc[mask, "country"] == "US")

    def test_from_dict_with_operators(self, sample_data: pd.DataFrame) -> None:
        """Test creating ConditionSet from dict with operators."""
        cond_set = ConditionSet.from_dict(
            {
                "age": (">=", 30),
                "income": ("<", 60000),
            }
        )
        mask = cond_set.evaluate(sample_data)
        matching = sample_data[mask]
        assert all(matching["age"] >= 30)
        assert all(matching["income"] < 60000)

    def test_from_dict_between(self, sample_data: pd.DataFrame) -> None:
        """Test BETWEEN operator via from_dict."""
        cond_set = ConditionSet.from_dict(
            {
                "age": ("between", (25, 45)),
            }
        )
        mask = cond_set.evaluate(sample_data)
        ages = sample_data.loc[mask, "age"]
        assert all((ages >= 25) & (ages <= 45))

    def test_from_dict_in_operator(self, sample_data: pd.DataFrame) -> None:
        """Test IN operator via from_dict."""
        cond_set = ConditionSet.from_dict(
            {
                "country": ("in", ["US", "CA"]),
            }
        )
        mask = cond_set.evaluate(sample_data)
        assert all(sample_data.loc[mask, "country"].isin(["US", "CA"]))

    def test_empty_conditions(self, sample_data: pd.DataFrame) -> None:
        """Test empty condition set returns all True."""
        cond_set = ConditionSet([])
        mask = cond_set.evaluate(sample_data)
        assert mask.all()


class TestConditionalSampler:
    """Tests for ConditionalSampler class."""

    def test_sample_with_conditions(self) -> None:
        """Test sampling with conditions."""
        np.random.seed(42)

        def generator_fn(n: int) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "value": np.random.randint(0, 100, n),
                    "category": np.random.choice(["A", "B", "C"], n),
                }
            )

        sampler = ConditionalSampler(max_trials=50, batch_size=500)
        result = sampler.sample(
            generator_fn=generator_fn,
            n_samples=100,
            conditions={"category": "A"},
        )

        assert len(result) == 100
        assert all(result["category"] == "A")

    def test_sample_with_multiple_conditions(self) -> None:
        """Test sampling with multiple conditions."""
        np.random.seed(42)

        def generator_fn(n: int) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "value": np.random.randint(0, 100, n),
                    "category": np.random.choice(["A", "B", "C"], n),
                }
            )

        sampler = ConditionalSampler(max_trials=100, batch_size=1000)
        result = sampler.sample(
            generator_fn=generator_fn,
            n_samples=50,
            conditions={
                "category": "A",
                "value": (">=", 50),
            },
        )

        assert len(result) == 50
        assert all(result["category"] == "A")
        assert all(result["value"] >= 50)

    def test_estimate_feasibility(self) -> None:
        """Test feasibility estimation."""
        np.random.seed(42)

        def generator_fn(n: int) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "category": np.random.choice(["A", "B", "C"], n, p=[0.1, 0.3, 0.6]),
                }
            )

        sampler = ConditionalSampler()
        feasibility = sampler.estimate_feasibility(
            generator_fn=generator_fn,
            conditions={"category": "A"},
            sample_size=1000,
        )

        assert feasibility["feasible"]
        assert 0.05 < feasibility["acceptance_rate"] < 0.2  # ~10%


class TestUpsampler:
    """Tests for Upsampler class."""

    def test_fit_analyzes_distribution(self, sample_data: pd.DataFrame) -> None:
        """Test that fit analyzes class distribution."""

        # Mock generator
        class MockGenerator:
            pass

        upsampler = Upsampler(MockGenerator(), "is_fraud")
        upsampler.fit(sample_data)

        assert upsampler._class_distribution is not None
        assert True in upsampler._class_distribution or False in upsampler._class_distribution


class TestScenarioGenerator:
    """Tests for ScenarioGenerator class."""

    def test_generate_scenarios(self) -> None:
        """Test generating multiple scenarios."""
        np.random.seed(42)

        class MockGenerator:
            def generate(self, n: int) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "fraud": np.random.choice([True, False], n, p=[0.3, 0.7]),
                        "amount": np.random.exponential(1000, n),
                    }
                )

        scenario_gen = ScenarioGenerator(MockGenerator())
        scenarios = [
            {"fraud": True},
            {"fraud": False},
        ]

        result = scenario_gen.generate_scenarios(
            scenarios=scenarios,
            samples_per_scenario=50,
        )

        assert "scenario_id" in result.columns
        assert len(result[result["scenario_id"] == 0]) <= 50
        assert len(result[result["scenario_id"] == 1]) <= 50


class TestConditionEdgeCases:
    """Edge case tests."""

    def test_missing_column_raises_error(self, sample_data: pd.DataFrame) -> None:
        """Test that missing column raises error."""
        cond = Condition("nonexistent", Operator.EQ, "value")
        with pytest.raises(Exception):
            cond.evaluate(sample_data)

    def test_ne_operator(self, sample_data: pd.DataFrame) -> None:
        """Test not equal operator."""
        cond = Condition("country", Operator.NE, "US")
        mask = cond.evaluate(sample_data)
        assert all(sample_data.loc[mask, "country"] != "US")

    def test_like_operator(self) -> None:
        """Test LIKE operator with regex."""
        df = pd.DataFrame({"name": ["John Smith", "Jane Doe", "John Doe"]})
        cond = Condition("name", Operator.LIKE, "^John")
        mask = cond.evaluate(df)
        assert mask.sum() == 2


class TestConditionBuilder:
    """Tests for ConditionBuilder fluent API."""

    def test_simple_eq_condition(self) -> None:
        """Test building simple equality condition."""
        from genesis.generators.conditional import ConditionBuilder

        conditions = ConditionBuilder().where("country").eq("US").build()

        assert len(conditions) == 1
        assert conditions.conditions[0].column == "country"
        assert conditions.conditions[0].operator == Operator.EQ
        assert conditions.conditions[0].value == "US"

    def test_chained_conditions(self, sample_data: pd.DataFrame) -> None:
        """Test building multiple chained conditions."""
        from genesis.generators.conditional import ConditionBuilder

        conditions = (
            ConditionBuilder()
            .where("age")
            .gte(30)
            .where("income")
            .between(40000, 80000)
            .where("country")
            .in_(["US", "UK"])
            .build()
        )

        assert len(conditions) == 3

        # Verify evaluation works
        mask = conditions.evaluate(sample_data)
        filtered = sample_data[mask]

        assert all(filtered["age"] >= 30)
        assert all((filtered["income"] >= 40000) & (filtered["income"] <= 80000))
        assert all(filtered["country"].isin(["US", "UK"]))

    def test_all_operators(self) -> None:
        """Test all available operators in builder."""
        from genesis.generators.conditional import ConditionBuilder

        builder = ConditionBuilder()

        # Test each operator
        builder.where("a").eq(1)
        builder.where("b").ne(2)
        builder.where("c").gt(3)
        builder.where("d").gte(4)
        builder.where("e").lt(5)
        builder.where("f").lte(6)
        builder.where("g").in_([7, 8])
        builder.where("h").not_in([9, 10])
        builder.where("i").between(11, 12)
        builder.where("j").like("pattern")

        conditions = builder.build()
        assert len(conditions) == 10

    def test_builder_requires_where_first(self) -> None:
        """Test that builder requires where() before condition."""
        from genesis.core.exceptions import ValidationError
        from genesis.generators.conditional import ConditionBuilder

        builder = ConditionBuilder()
        with pytest.raises(ValidationError):
            builder.eq("value")


class TestGuidedConditionalSampler:
    """Tests for GuidedConditionalSampler."""

    @pytest.fixture
    def fitted_sampler(self, sample_data: pd.DataFrame):
        """Create fitted sampler."""
        from genesis.generators.conditional import GuidedConditionalSampler

        sampler = GuidedConditionalSampler(strategy="iterative_refinement")
        sampler.fit(sample_data)
        return sampler

    def test_fit_extracts_statistics(self, sample_data: pd.DataFrame) -> None:
        """Test that fit extracts column statistics."""
        from genesis.generators.conditional import GuidedConditionalSampler

        sampler = GuidedConditionalSampler()
        sampler.fit(sample_data)

        assert "age" in sampler._column_stats
        assert "income" in sampler._column_stats
        assert "country" in sampler._column_stats

        # Check numeric stats
        age_stats = sampler._column_stats["age"]
        assert "mean" in age_stats
        assert "std" in age_stats
        assert "quantiles" in age_stats

        # Check categorical stats
        country_stats = sampler._column_stats["country"]
        assert "value_counts" in country_stats

    def test_sample_with_iterative_strategy(self, fitted_sampler) -> None:
        """Test sampling with iterative refinement strategy."""
        np.random.seed(42)

        def generator_fn(n: int) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "age": np.random.randint(18, 80, n),
                    "income": np.random.normal(50000, 20000, n),
                    "country": np.random.choice(["US", "UK", "CA", "FR"], n),
                }
            )

        conditions = {"age": (">=", 50), "country": "US"}
        result = fitted_sampler.sample(
            generator_fn=generator_fn,
            n_samples=100,
            conditions=conditions,
        )

        # May not get all 100 due to condition rarity, but should get some
        assert len(result) > 0
        assert all(result["age"] >= 50)
        assert all(result["country"] == "US")

    def test_importance_sampling_strategy(self, sample_data: pd.DataFrame) -> None:
        """Test importance sampling strategy."""
        from genesis.generators.conditional import GuidedConditionalSampler

        sampler = GuidedConditionalSampler(strategy="importance_sampling")
        sampler.fit(sample_data)

        np.random.seed(42)

        def generator_fn(n: int) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "age": np.random.randint(18, 80, n),
                    "income": np.random.normal(50000, 20000, n),
                    "country": np.random.choice(["US", "UK", "CA", "FR"], n),
                }
            )

        conditions = {"age": (">=", 60)}
        result = sampler.sample(
            generator_fn=generator_fn,
            n_samples=50,
            conditions=conditions,
        )

        assert len(result) > 0
        assert all(result["age"] >= 60)
