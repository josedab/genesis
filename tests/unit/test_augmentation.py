"""Tests for Synthetic Data Augmentation module."""

import numpy as np
import pandas as pd
import pytest

from genesis.augmentation import (
    AugmentationPlanner,
    AugmentationResult,
    AugmentationStrategy,
    ImbalanceAnalyzer,
    SyntheticAugmenter,
    analyze_imbalance,
)


@pytest.fixture
def imbalanced_data() -> pd.DataFrame:
    """Create imbalanced dataset."""
    np.random.seed(42)

    # Majority class: 900 samples
    majority = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 900),
            "feature2": np.random.normal(0, 1, 900),
            "target": ["A"] * 900,
        }
    )

    # Minority class: 100 samples
    minority = pd.DataFrame(
        {
            "feature1": np.random.normal(2, 1, 100),
            "feature2": np.random.normal(2, 1, 100),
            "target": ["B"] * 100,
        }
    )

    return pd.concat([majority, minority], ignore_index=True)


class TestImbalanceAnalyzer:
    """Tests for ImbalanceAnalyzer."""

    def test_analyze_distribution(self, imbalanced_data: pd.DataFrame) -> None:
        """Test distribution analysis."""
        analyzer = ImbalanceAnalyzer()
        distribution = analyzer.analyze(imbalanced_data, "target")

        assert "A" in distribution
        assert "B" in distribution
        assert distribution["A"].count == 900
        assert distribution["B"].count == 100

    def test_detect_minority_class(self, imbalanced_data: pd.DataFrame) -> None:
        """Test minority class detection."""
        analyzer = ImbalanceAnalyzer()

        minorities = analyzer.get_minority_classes(imbalanced_data, "target")

        assert "B" in minorities
        assert "A" not in minorities

    def test_imbalance_ratio(self, imbalanced_data: pd.DataFrame) -> None:
        """Test imbalance ratio calculation."""
        analyzer = ImbalanceAnalyzer()

        ratio = analyzer.get_imbalance_ratio(imbalanced_data, "target")

        assert ratio == 9.0  # 900/100


class TestAugmentationPlanner:
    """Tests for AugmentationPlanner."""

    def test_plan_oversample(self, imbalanced_data: pd.DataFrame) -> None:
        """Test oversampling plan."""
        planner = AugmentationPlanner(target_ratio=1.0)

        plan = planner.plan(
            imbalanced_data,
            "target",
            strategy=AugmentationStrategy.OVERSAMPLE,
        )

        assert plan.planned_distribution["A"] == 900  # Unchanged
        assert plan.planned_distribution["B"] >= 100  # Increased

    def test_plan_undersample(self, imbalanced_data: pd.DataFrame) -> None:
        """Test undersampling plan."""
        planner = AugmentationPlanner(target_ratio=1.0)

        plan = planner.plan(
            imbalanced_data,
            "target",
            strategy=AugmentationStrategy.UNDERSAMPLE,
        )

        # Majority should be reduced
        assert plan.planned_distribution["A"] <= 900

    def test_plan_summary(self, imbalanced_data: pd.DataFrame) -> None:
        """Test plan summary generation."""
        planner = AugmentationPlanner()
        plan = planner.plan(imbalanced_data, "target")

        summary = plan.summary()

        assert "Augmentation Plan" in summary
        assert "target" in summary


class TestSyntheticAugmenter:
    """Tests for SyntheticAugmenter."""

    def test_fit(self, imbalanced_data: pd.DataFrame) -> None:
        """Test fitting augmenter."""
        augmenter = SyntheticAugmenter(method="gaussian_copula")
        augmenter.fit(imbalanced_data, "target")

        assert augmenter._generator is not None

    def test_augment_oversample(self, imbalanced_data: pd.DataFrame) -> None:
        """Test oversampling augmentation."""
        augmenter = SyntheticAugmenter(
            method="gaussian_copula",
            target_ratio=0.5,
        )
        augmenter.fit(imbalanced_data, "target")

        result = augmenter.augment(strategy=AugmentationStrategy.OVERSAMPLE)

        assert isinstance(result, AugmentationResult)
        assert len(result.augmented_data) >= len(imbalanced_data)


class TestAnalyzeImbalance:
    """Tests for analyze_imbalance function."""

    def test_returns_analysis(self, imbalanced_data: pd.DataFrame) -> None:
        """Test imbalance analysis."""
        analysis = analyze_imbalance(imbalanced_data, "target")

        assert "imbalance_ratio" in analysis
        assert "minority_classes" in analysis
        assert "distribution" in analysis
        assert analysis["imbalance_ratio"] == 9.0
