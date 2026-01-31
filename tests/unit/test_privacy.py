"""Unit tests for privacy module."""

import numpy as np
import pandas as pd

from genesis.privacy.anonymity import (
    check_k_anonymity,
    check_l_diversity,
    enforce_k_anonymity,
    suppress_rare_categories,
)
from genesis.privacy.differential import (
    DPAccountant,
    add_gaussian_noise,
    add_laplace_noise,
    clip_gradients,
)


class TestDPAccountant:
    """Tests for DPAccountant."""

    def test_initial_budget(self):
        accountant = DPAccountant(epsilon=1.0, delta=1e-5)
        assert accountant.remaining_budget == 1.0
        assert accountant.spent_budget == 0.0

    def test_spend_budget(self):
        accountant = DPAccountant(epsilon=1.0)
        success = accountant.spend(0.3, "query1")

        assert success
        assert accountant.remaining_budget == 0.7
        assert accountant.spent_budget == 0.3

    def test_budget_exhaustion(self):
        accountant = DPAccountant(epsilon=1.0)
        accountant.spend(0.8, "query1")

        success = accountant.spend(0.5, "query2")  # Would exceed budget
        assert not success
        assert accountant.spent_budget == 0.8

    def test_noise_scale(self):
        accountant = DPAccountant()
        scale = accountant.get_noise_scale(sensitivity=1.0, epsilon=0.5)
        assert scale == 2.0  # 1.0 / 0.5


class TestNoiseMechanisms:
    """Tests for noise mechanisms."""

    def test_laplace_noise(self):
        data = np.zeros((100, 10))
        noisy = add_laplace_noise(data, sensitivity=1.0, epsilon=1.0)

        # Noisy data should be different from original
        assert not np.allclose(data, noisy)

        # Mean should be close to 0
        assert abs(noisy.mean()) < 1.0

    def test_gaussian_noise(self):
        data = np.zeros((100, 10))
        noisy = add_gaussian_noise(data, sensitivity=1.0, epsilon=1.0, delta=1e-5)

        assert not np.allclose(data, noisy)
        assert abs(noisy.mean()) < 1.0

    def test_clip_gradients(self):
        gradients = np.array([[10.0, 0.0], [0.0, 5.0], [3.0, 4.0]])
        clipped = clip_gradients(gradients, max_norm=5.0)

        # Check norms are at most max_norm
        norms = np.linalg.norm(clipped, axis=1)
        assert all(norms <= 5.0 + 1e-6)


class TestKAnonymity:
    """Tests for k-anonymity."""

    def test_check_k_anonymity_satisfied(self):
        # Create data with groups of at least 5
        df = pd.DataFrame(
            {
                "age": [25] * 10 + [30] * 10 + [35] * 10,
                "gender": ["M"] * 15 + ["F"] * 15,
            }
        )

        result = check_k_anonymity(df, ["age", "gender"], k=5)
        assert result["satisfies_k"]
        assert result["achieved_k"] >= 5

    def test_check_k_anonymity_violated(self):
        df = pd.DataFrame(
            {
                "age": [25, 26, 27, 28, 29],  # All unique
                "gender": ["M", "F", "M", "F", "M"],
            }
        )

        result = check_k_anonymity(df, ["age"], k=2)
        assert not result["satisfies_k"]
        assert result["achieved_k"] == 1

    def test_enforce_k_anonymity(self):
        # Create data with some groups having fewer than k=5 members
        # Groups with counts: age 20-24 have 1 each (5 unique), age 25 has 10
        df = pd.DataFrame(
            {
                "age": [20, 21, 22, 23, 24] + [25] * 10,  # 5 unique ages + 10 with age=25
                "name": [f"name_{i}" for i in range(15)],
            }
        )

        result, stats = enforce_k_anonymity(df, ["age"], k=5, method="suppress")

        # After suppression, only age=25 group (10 members) should remain
        # Check that remaining data satisfies k-anonymity
        check_result = check_k_anonymity(result, ["age"], k=5)
        assert check_result["satisfies_k"]


class TestLDiversity:
    """Tests for l-diversity."""

    def test_check_l_diversity_satisfied(self):
        df = pd.DataFrame(
            {
                "age": [25] * 10 + [30] * 10,
                "disease": ["A", "B", "C", "D", "E"] * 2 + ["F", "G", "H", "I", "J"] * 2,
            }
        )

        result = check_l_diversity(df, ["age"], "disease", l_value=3)
        assert result["satisfies_l"]

    def test_check_l_diversity_violated(self):
        df = pd.DataFrame(
            {
                "age": [25] * 10,
                "disease": ["A"] * 10,  # No diversity
            }
        )

        result = check_l_diversity(df, ["age"], "disease", l_value=2)
        assert not result["satisfies_l"]


class TestRareSuppression:
    """Tests for rare category suppression."""

    def test_suppress_rare_categories(self):
        df = pd.DataFrame(
            {
                "category": ["A"] * 90 + ["B"] * 5 + ["C"] * 3 + ["D"] * 2,
            }
        )

        result, counts = suppress_rare_categories(df, threshold=0.05)

        # Rare categories should be replaced
        unique_cats = result["category"].unique()
        assert "D" not in unique_cats or "OTHER" in unique_cats
        assert counts.get("category", 0) > 0
