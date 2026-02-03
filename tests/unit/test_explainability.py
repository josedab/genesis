"""Tests for explainable generation."""

import numpy as np
import pandas as pd
import pytest

from genesis.explainability import (
    Attribution,
    AttributionTracker,
    FeatureExplanation,
    FeatureImportanceCalculator,
    LineageTracker,
    RecordExplanation,
    compute_data_hash,
)


@pytest.fixture
def sample_source_data() -> pd.DataFrame:
    """Create sample source data."""
    np.random.seed(42)
    return pd.DataFrame({
        "id": [f"src_{i}" for i in range(100)],
        "age": np.random.randint(18, 80, 100),
        "income": np.random.uniform(20000, 150000, 100),
        "score": np.random.uniform(0, 1, 100),
    })


@pytest.fixture
def sample_synthetic_data() -> pd.DataFrame:
    """Create sample synthetic data."""
    np.random.seed(43)
    return pd.DataFrame({
        "_synthetic_id": [f"syn_{i:06d}" for i in range(50)],
        "age": np.random.randint(18, 80, 50),
        "income": np.random.uniform(20000, 150000, 50),
        "score": np.random.uniform(0, 1, 50),
    })


class TestAttribution:
    """Tests for Attribution dataclass."""

    def test_top_sources(self) -> None:
        """Test getting top influential sources."""
        attribution = Attribution(
            synthetic_id="syn_001",
            source_attributions=[
                {"source_id": "src_1", "influence": 0.1},
                {"source_id": "src_2", "influence": 0.5},
                {"source_id": "src_3", "influence": 0.3},
                {"source_id": "src_4", "influence": 0.1},
            ],
        )

        top = attribution.top_sources(2)

        assert len(top) == 2
        assert top[0]["source_id"] == "src_2"
        assert top[1]["source_id"] == "src_3"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        attribution = Attribution(
            synthetic_id="syn_001",
            source_attributions=[{"source_id": "src_1", "influence": 0.5}],
            method="nearest_neighbor",
        )

        result = attribution.to_dict()

        assert result["synthetic_id"] == "syn_001"
        assert result["method"] == "nearest_neighbor"
        assert len(result["sources"]) == 1


class TestFeatureExplanation:
    """Tests for FeatureExplanation dataclass."""

    def test_basic_explanation(self) -> None:
        """Test basic feature explanation."""
        explanation = FeatureExplanation(
            feature_name="age",
            generated_value=35,
            generation_method="statistical_sampling",
            influences=[
                {"source_id": "src_1", "source_value": 34, "influence": 0.6},
                {"source_id": "src_2", "source_value": 36, "influence": 0.4},
            ],
            confidence=0.85,
            explanation_text="Interpolated from nearby records",
        )

        assert explanation.feature_name == "age"
        assert explanation.confidence == 0.85
        assert len(explanation.influences) == 2


class TestRecordExplanation:
    """Tests for RecordExplanation dataclass."""

    def test_summary_generation(self) -> None:
        """Test summary text generation."""
        attribution = Attribution(
            synthetic_id="syn_001",
            source_attributions=[
                {"source_id": "src_1", "influence": 0.6},
                {"source_id": "src_2", "influence": 0.4},
            ],
        )

        explanation = RecordExplanation(
            record_id="syn_001",
            record_data={"age": 35, "income": 50000},
            feature_explanations={
                "age": FeatureExplanation(
                    feature_name="age",
                    generated_value=35,
                    generation_method="sampling",
                    influences=[],
                )
            },
            overall_attribution=attribution,
        )

        summary = explanation.summary

        assert "syn_001" in summary
        assert "src_1" in summary
        assert "60" in summary or "0.6" in summary  # Influence percentage


class TestAttributionTracker:
    """Tests for AttributionTracker."""

    def test_track_source_data(self, sample_source_data: pd.DataFrame) -> None:
        """Test tracking source data."""
        tracker = AttributionTracker()
        tracker.track_source_data(sample_source_data, "id")

        assert tracker._source_data is not None
        assert len(tracker._source_index) == 100

    def test_record_attribution(self, sample_source_data: pd.DataFrame) -> None:
        """Test recording attribution."""
        tracker = AttributionTracker()
        tracker.track_source_data(sample_source_data, "id")

        attribution = tracker.record_attribution(
            synthetic_id="syn_001",
            source_ids=["src_0", "src_1", "src_2"],
            influences=[0.5, 0.3, 0.2],
            method="knn",
        )

        assert attribution.synthetic_id == "syn_001"
        assert len(attribution.source_attributions) == 3
        assert attribution.method == "knn"

    def test_get_attribution(self, sample_source_data: pd.DataFrame) -> None:
        """Test retrieving attribution."""
        tracker = AttributionTracker()
        tracker.track_source_data(sample_source_data, "id")

        tracker.record_attribution(
            synthetic_id="syn_001",
            source_ids=["src_0"],
            influences=[1.0],
        )

        attribution = tracker.get_attribution("syn_001")
        assert attribution is not None
        assert attribution.synthetic_id == "syn_001"

    def test_get_nonexistent_attribution(self) -> None:
        """Test retrieving nonexistent attribution."""
        tracker = AttributionTracker()
        result = tracker.get_attribution("nonexistent")
        assert result is None

    def test_get_source_influence(self, sample_source_data: pd.DataFrame) -> None:
        """Test getting influence of a source record."""
        tracker = AttributionTracker()
        tracker.track_source_data(sample_source_data, "id")

        tracker.record_attribution("syn_001", ["src_0"], [0.5])
        tracker.record_attribution("syn_002", ["src_0", "src_1"], [0.3, 0.7])

        influence = tracker.get_source_influence("src_0")

        assert "syn_001" in influence
        assert "syn_002" in influence
        assert influence["syn_001"] == 0.5
        assert influence["syn_002"] == 0.3

    def test_compute_attribution_matrix(
        self, sample_source_data: pd.DataFrame
    ) -> None:
        """Test computing attribution matrix."""
        tracker = AttributionTracker()
        tracker.track_source_data(sample_source_data, "id")

        tracker.record_attribution("syn_001", ["src_0", "src_1"], [0.6, 0.4])
        tracker.record_attribution("syn_002", ["src_1", "src_2"], [0.5, 0.5])

        matrix = tracker.compute_attribution_matrix()

        assert matrix.shape[0] == 2  # 2 synthetic records
        assert matrix.shape[1] == 3  # 3 unique source records


class TestFeatureImportanceCalculator:
    """Tests for FeatureImportanceCalculator."""

    def test_fit_correlation(
        self,
        sample_source_data: pd.DataFrame,
        sample_synthetic_data: pd.DataFrame,
    ) -> None:
        """Test fitting with correlation method."""
        calculator = FeatureImportanceCalculator(method="correlation")
        calculator.fit(sample_source_data, sample_synthetic_data)

        importance = calculator.get_importance()

        assert "age" in importance
        assert "income" in importance
        assert "score" in importance

        # Should sum to approximately 1
        total = sum(importance.values())
        assert 0.99 < total < 1.01

    def test_get_top_features(
        self,
        sample_source_data: pd.DataFrame,
        sample_synthetic_data: pd.DataFrame,
    ) -> None:
        """Test getting top features."""
        calculator = FeatureImportanceCalculator()
        calculator.fit(sample_source_data, sample_synthetic_data)

        top = calculator.get_top_features(2)

        assert len(top) == 2
        assert isinstance(top[0], tuple)
        assert isinstance(top[0][0], str)  # Feature name
        assert isinstance(top[0][1], float)  # Importance score


class TestLineageTracker:
    """Tests for LineageTracker."""

    def test_record_generation(self) -> None:
        """Test recording generation run."""
        tracker = LineageTracker()

        record = tracker.record_generation(
            run_id="run_001",
            source_hash="abc123",
            generator_type="ctgan",
            generator_config={"epochs": 100},
            output_count=10000,
        )

        assert record["run_id"] == "run_001"
        assert record["source_hash"] == "abc123"
        assert record["output_count"] == 10000

    def test_get_lineage(self) -> None:
        """Test retrieving lineage record."""
        tracker = LineageTracker()

        tracker.record_generation(
            run_id="run_001",
            source_hash="abc123",
            generator_type="ctgan",
            generator_config={},
            output_count=10000,
        )

        lineage = tracker.get_lineage("run_001")
        assert lineage is not None
        assert lineage["run_id"] == "run_001"

    def test_get_all_lineage(self) -> None:
        """Test getting all lineage records."""
        tracker = LineageTracker()

        tracker.record_generation("run_001", "hash1", "ctgan", {}, 1000)
        tracker.record_generation("run_002", "hash2", "tvae", {}, 2000)

        all_lineage = tracker.get_all_lineage()
        assert len(all_lineage) == 2

    def test_to_dataframe(self) -> None:
        """Test converting lineage to DataFrame."""
        tracker = LineageTracker()

        tracker.record_generation("run_001", "hash1", "ctgan", {}, 1000)
        tracker.record_generation("run_002", "hash2", "tvae", {}, 2000)

        df = tracker.to_dataframe()

        assert len(df) == 2
        assert "run_id" in df.columns
        assert "output_count" in df.columns


class TestComputeDataHash:
    """Tests for compute_data_hash function."""

    def test_hash_is_consistent(self, sample_source_data: pd.DataFrame) -> None:
        """Test that same data produces same hash."""
        hash1 = compute_data_hash(sample_source_data)
        hash2 = compute_data_hash(sample_source_data)

        assert hash1 == hash2

    def test_hash_is_different_for_different_data(
        self,
        sample_source_data: pd.DataFrame,
        sample_synthetic_data: pd.DataFrame,
    ) -> None:
        """Test that different data produces different hash."""
        hash1 = compute_data_hash(sample_source_data)
        hash2 = compute_data_hash(sample_synthetic_data)

        assert hash1 != hash2

    def test_hash_length(self, sample_source_data: pd.DataFrame) -> None:
        """Test hash length."""
        hash_value = compute_data_hash(sample_source_data)

        assert len(hash_value) == 16  # Truncated to 16 chars
