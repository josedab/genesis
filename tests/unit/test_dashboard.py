"""Tests for quality dashboard."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genesis.dashboard import (
    QualityDashboard,
    create_dashboard,
)


@pytest.fixture
def real_data() -> pd.DataFrame:
    """Create real data for testing."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "numeric1": np.random.randn(500),
            "numeric2": np.random.randn(500) * 2 + 1,
            "category": np.random.choice(["A", "B", "C"], 500),
        }
    )


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """Create synthetic data for testing."""
    np.random.seed(123)
    return pd.DataFrame(
        {
            "numeric1": np.random.randn(500) * 1.1,  # Slightly different
            "numeric2": np.random.randn(500) * 2 + 0.9,
            "category": np.random.choice(["A", "B", "C"], 500, p=[0.4, 0.35, 0.25]),
        }
    )


class TestQualityDashboard:
    """Tests for QualityDashboard."""

    def test_init(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> None:
        """Test dashboard initialization."""
        dashboard = QualityDashboard(real_data, synthetic_data)

        assert dashboard.real_data is not None
        assert dashboard.synthetic_data is not None
        assert dashboard.title == "Genesis Quality Dashboard"

    def test_compute_metrics(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> None:
        """Test computing quality metrics."""
        dashboard = QualityDashboard(real_data, synthetic_data)
        metrics = dashboard.compute_metrics()

        assert "overall_score" in metrics
        assert "statistical_fidelity" in metrics or "fidelity_score" in metrics
        assert isinstance(metrics["overall_score"], (int, float))

    def test_generate_html_report(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> None:
        """Test HTML report generation."""
        dashboard = QualityDashboard(real_data, synthetic_data)
        html = dashboard.generate_html_report()

        assert isinstance(html, str)
        assert len(html) > 1000
        assert "<html" in html.lower()
        assert "Genesis" in html
        assert "Quality" in html

    def test_save_report(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> None:
        """Test saving HTML report to file."""
        dashboard = QualityDashboard(real_data, synthetic_data)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            dashboard.save_report(f.name)

            # Verify file was created and has content
            assert Path(f.name).exists()
            content = Path(f.name).read_text()
            assert len(content) > 1000

    def test_generate_plotly_figures(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
    ) -> None:
        """Test Plotly figure generation."""
        dashboard = QualityDashboard(real_data, synthetic_data)

        try:
            figures = dashboard.generate_plotly_figures()

            assert isinstance(figures, dict)
            # Should have distribution, correlation, and quality gauge
            assert len(figures) >= 1

            # Check that figures have expected keys
            expected_keys = {"distributions", "correlation", "quality_gauge"}
            assert len(expected_keys & set(figures.keys())) > 0

        except ImportError:
            pytest.skip("Plotly not installed")


class TestCreateDashboard:
    """Tests for create_dashboard convenience function."""

    def test_returns_html(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> None:
        """Test that create_dashboard returns HTML when no path given."""
        html = create_dashboard(real_data, synthetic_data)

        assert isinstance(html, str)
        assert "<html" in html.lower()

    def test_saves_to_path(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> None:
        """Test saving to output path."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            result = create_dashboard(real_data, synthetic_data, output_path=f.name)

            assert result is None
            assert Path(f.name).exists()


class TestDashboardEdgeCases:
    """Tests for edge cases in dashboard."""

    def test_empty_numeric_columns(self) -> None:
        """Test with no numeric columns."""
        real = pd.DataFrame({"cat1": ["A", "B"] * 50, "cat2": ["X", "Y"] * 50})
        synth = pd.DataFrame({"cat1": ["A", "B"] * 50, "cat2": ["X", "Y"] * 50})

        dashboard = QualityDashboard(real, synth)
        html = dashboard.generate_html_report()

        assert isinstance(html, str)

    def test_single_column(self) -> None:
        """Test with single column."""
        np.random.seed(42)
        real = pd.DataFrame({"value": np.random.randn(100)})
        synth = pd.DataFrame({"value": np.random.randn(100)})

        dashboard = QualityDashboard(real, synth)
        html = dashboard.generate_html_report()

        assert isinstance(html, str)

    def test_mismatched_columns_handled(self) -> None:
        """Test that mismatched columns are handled gracefully or raise appropriately."""
        real = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        synth = pd.DataFrame({"a": [1, 2, 3], "c": [7, 8, 9]})  # Different column

        dashboard = QualityDashboard(real, synth)
        # Either handles gracefully or raises a clear error
        try:
            html = dashboard.generate_html_report()
            assert isinstance(html, str)
        except (ValueError, KeyError, AttributeError):
            # It's acceptable to raise if columns don't match
            pass
