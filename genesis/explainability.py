"""Explainable Synthetic Data Generation for Genesis.

This module provides explainability features for synthetic data generation,
helping users understand how synthetic records are created and which real
records influenced their generation.

Features:
- Attribution scoring: Which real records influenced a synthetic record
- Feature importance: Which features matter most in generation
- Generation explanations: Human-readable explanations
- Lineage tracking: Full provenance of synthetic records

Example:
    >>> from genesis.explainability import ExplainableGenerator, AttributionTracker
    >>>
    >>> tracker = AttributionTracker()
    >>> generator = ExplainableGenerator(base_generator, tracker)
    >>>
    >>> synthetic_data = generator.generate(1000)
    >>> explanation = generator.explain_record(synthetic_data.iloc[0])
    >>> print(explanation.summary)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from genesis.core.base import BaseGenerator
from genesis.core.exceptions import GenesisError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Attribution:
    """Attribution of a synthetic record to source records."""

    synthetic_id: str
    source_attributions: List[Dict[str, Any]]
    total_influence: float = 1.0
    method: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def top_sources(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get top N influential source records."""
        sorted_sources = sorted(
            self.source_attributions,
            key=lambda x: x.get("influence", 0),
            reverse=True,
        )
        return sorted_sources[:n]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "synthetic_id": self.synthetic_id,
            "sources": self.source_attributions,
            "total_influence": self.total_influence,
            "method": self.method,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class FeatureExplanation:
    """Explanation for a single feature."""

    feature_name: str
    generated_value: Any
    generation_method: str
    influences: List[Dict[str, Any]]
    confidence: float = 1.0
    explanation_text: str = ""


@dataclass
class RecordExplanation:
    """Full explanation for a synthetic record."""

    record_id: str
    record_data: Dict[str, Any]
    feature_explanations: Dict[str, FeatureExplanation]
    overall_attribution: Attribution
    generation_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Synthetic Record Explanation (ID: {self.record_id})",
            "=" * 50,
            "",
            "Top Influencing Source Records:",
        ]

        for source in self.overall_attribution.top_sources(3):
            lines.append(
                f"  - Record {source.get('source_id', 'N/A')}: "
                f"{source.get('influence', 0):.2%} influence"
            )

        lines.extend([
            "",
            "Feature-level Explanations:",
        ])

        for feat_name, explanation in self.feature_explanations.items():
            lines.append(
                f"  {feat_name}: {explanation.explanation_text or explanation.generation_method}"
            )

        return "\n".join(lines)


class AttributionTracker:
    """Tracks attribution between synthetic and source records.

    Example:
        >>> tracker = AttributionTracker()
        >>> tracker.track_source_data(original_df, "id")
        >>>
        >>> # During generation
        >>> tracker.record_attribution(
        ...     synthetic_id="syn_001",
        ...     source_ids=["src_005", "src_012"],
        ...     influences=[0.6, 0.4]
        ... )
    """

    def __init__(self) -> None:
        self._source_data: Optional[pd.DataFrame] = None
        self._source_id_column: str = "id"
        self._attributions: Dict[str, Attribution] = {}
        self._source_index: Dict[str, int] = {}

    def track_source_data(
        self,
        data: pd.DataFrame,
        id_column: str = "id",
    ) -> None:
        """Track source data for attribution.

        Args:
            data: Source DataFrame
            id_column: Column with unique identifiers
        """
        self._source_data = data.copy()
        self._source_id_column = id_column

        # Build index for fast lookup
        self._source_index = {
            str(row[id_column]): idx
            for idx, row in data.iterrows()
        }

        logger.info(f"Tracking {len(data)} source records for attribution")

    def record_attribution(
        self,
        synthetic_id: str,
        source_ids: List[str],
        influences: List[float],
        method: str = "learned",
    ) -> Attribution:
        """Record attribution for a synthetic record.

        Args:
            synthetic_id: ID of synthetic record
            source_ids: IDs of influencing source records
            influences: Influence scores (should sum to ~1)
            method: Attribution method used

        Returns:
            Created Attribution object
        """
        if len(source_ids) != len(influences):
            raise ValueError("source_ids and influences must have same length")

        source_attributions = []
        for src_id, influence in zip(source_ids, influences):
            attr = {
                "source_id": src_id,
                "influence": influence,
            }

            # Add source record data if available
            if self._source_data is not None and src_id in self._source_index:
                idx = self._source_index[src_id]
                attr["source_data"] = self._source_data.iloc[idx].to_dict()

            source_attributions.append(attr)

        attribution = Attribution(
            synthetic_id=synthetic_id,
            source_attributions=source_attributions,
            total_influence=sum(influences),
            method=method,
        )

        self._attributions[synthetic_id] = attribution
        return attribution

    def get_attribution(self, synthetic_id: str) -> Optional[Attribution]:
        """Get attribution for a synthetic record."""
        return self._attributions.get(synthetic_id)

    def get_source_influence(self, source_id: str) -> Dict[str, float]:
        """Get all synthetic records influenced by a source record.

        Args:
            source_id: Source record ID

        Returns:
            Dictionary mapping synthetic IDs to influence scores
        """
        result = {}

        for syn_id, attr in self._attributions.items():
            for src_attr in attr.source_attributions:
                if src_attr["source_id"] == source_id:
                    result[syn_id] = src_attr["influence"]
                    break

        return result

    def compute_attribution_matrix(self) -> pd.DataFrame:
        """Compute full attribution matrix.

        Returns:
            DataFrame with synthetic records as rows, source records as columns
        """
        if not self._attributions:
            return pd.DataFrame()

        # Get all unique source IDs
        source_ids = set()
        for attr in self._attributions.values():
            for src_attr in attr.source_attributions:
                source_ids.add(src_attr["source_id"])

        source_ids = sorted(source_ids)
        synthetic_ids = list(self._attributions.keys())

        # Build matrix
        matrix = np.zeros((len(synthetic_ids), len(source_ids)))
        source_idx_map = {s: i for i, s in enumerate(source_ids)}

        for i, syn_id in enumerate(synthetic_ids):
            attr = self._attributions[syn_id]
            for src_attr in attr.source_attributions:
                j = source_idx_map[src_attr["source_id"]]
                matrix[i, j] = src_attr["influence"]

        return pd.DataFrame(
            matrix,
            index=synthetic_ids,
            columns=source_ids,
        )


class FeatureImportanceCalculator:
    """Calculates feature importance for generation.

    Example:
        >>> calculator = FeatureImportanceCalculator()
        >>> calculator.fit(source_data, synthetic_data)
        >>> importance = calculator.get_importance()
    """

    def __init__(self, method: str = "correlation") -> None:
        """Initialize calculator.

        Args:
            method: Calculation method ('correlation', 'mutual_info', 'permutation')
        """
        self.method = method
        self._importance: Dict[str, float] = {}

    def fit(
        self,
        source_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
    ) -> "FeatureImportanceCalculator":
        """Calculate feature importance.

        Args:
            source_data: Original data
            synthetic_data: Generated data

        Returns:
            Self for chaining
        """
        if self.method == "correlation":
            self._importance = self._correlation_importance(source_data, synthetic_data)
        elif self.method == "mutual_info":
            self._importance = self._mutual_info_importance(source_data, synthetic_data)
        else:
            self._importance = self._correlation_importance(source_data, synthetic_data)

        return self

    def _correlation_importance(
        self,
        source: pd.DataFrame,
        synthetic: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate importance based on correlation preservation."""
        importance = {}

        numeric_cols = source.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in synthetic.columns]

        if len(common_cols) < 2:
            return {c: 1.0 / len(common_cols) for c in common_cols}

        # Calculate correlation matrices
        source_corr = source[common_cols].corr()
        synthetic_corr = synthetic[common_cols].corr()

        # Importance = how well correlations are preserved for each feature
        for col in common_cols:
            source_col_corrs = source_corr[col].drop(col)
            synth_col_corrs = synthetic_corr[col].drop(col)

            # Calculate correlation difference
            diff = np.abs(source_col_corrs - synth_col_corrs).mean()
            importance[col] = 1.0 - min(diff, 1.0)

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def _mutual_info_importance(
        self,
        source: pd.DataFrame,
        synthetic: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate importance based on mutual information."""
        try:
            from sklearn.feature_selection import mutual_info_regression
        except ImportError:
            return self._correlation_importance(source, synthetic)

        importance = {}
        numeric_cols = source.select_dtypes(include=[np.number]).columns
        common_cols = [c for c in numeric_cols if c in synthetic.columns]

        for col in common_cols:
            other_cols = [c for c in common_cols if c != col]
            if not other_cols:
                importance[col] = 1.0
                continue

            # MI between this column and others
            mi_source = mutual_info_regression(
                source[other_cols],
                source[col],
                random_state=42,
            ).mean()

            mi_synth = mutual_info_regression(
                synthetic[other_cols],
                synthetic[col],
                random_state=42,
            ).mean()

            # Higher importance if MI is preserved
            importance[col] = 1.0 - min(abs(mi_source - mi_synth), 1.0)

        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}

        return importance

    def get_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return dict(self._importance)

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N most important features."""
        sorted_features = sorted(
            self._importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_features[:n]


class ExplainableGenerator:
    """Generator with built-in explainability.

    Example:
        >>> tracker = AttributionTracker()
        >>> generator = ExplainableGenerator(
        ...     base_generator=my_generator,
        ...     tracker=tracker
        ... )
        >>> data = generator.generate(1000)
        >>> explanation = generator.explain_record(data.iloc[0])
    """

    def __init__(
        self,
        base_generator: BaseGenerator,
        tracker: Optional[AttributionTracker] = None,
        k_neighbors: int = 5,
        track_all: bool = True,
    ) -> None:
        """Initialize explainable generator.

        Args:
            base_generator: Underlying generator
            tracker: Attribution tracker
            k_neighbors: Number of neighbors for attribution
            track_all: Track attribution for all records
        """
        self.base_generator = base_generator
        self.tracker = tracker or AttributionTracker()
        self.k_neighbors = k_neighbors
        self.track_all = track_all

        self._source_data: Optional[pd.DataFrame] = None
        self._synthetic_data: Optional[pd.DataFrame] = None
        self._feature_importance: Optional[FeatureImportanceCalculator] = None

    def fit(self, data: pd.DataFrame, id_column: str = "id") -> "ExplainableGenerator":
        """Fit the generator and track source data.

        Args:
            data: Source data
            id_column: ID column for attribution

        Returns:
            Self for chaining
        """
        self._source_data = data.copy()

        # Ensure ID column exists
        if id_column not in data.columns:
            data = data.copy()
            data[id_column] = range(len(data))

        self.tracker.track_source_data(data, id_column)
        self.base_generator.fit(data)

        return self

    def generate(
        self,
        n_rows: int,
        compute_attribution: bool = True,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data with attribution tracking.

        Args:
            n_rows: Number of rows to generate
            compute_attribution: Whether to compute attribution
            random_state: Random seed

        Returns:
            Generated DataFrame
        """
        synthetic = self.base_generator.generate(n_rows)
        self._synthetic_data = synthetic.copy()

        # Add synthetic IDs
        synthetic["_synthetic_id"] = [f"syn_{i:06d}" for i in range(len(synthetic))]

        if compute_attribution and self._source_data is not None:
            self._compute_attributions(synthetic)

        # Calculate feature importance
        if self._source_data is not None:
            self._feature_importance = FeatureImportanceCalculator()
            self._feature_importance.fit(self._source_data, synthetic)

        return synthetic

    def _compute_attributions(self, synthetic: pd.DataFrame) -> None:
        """Compute attribution for all synthetic records."""
        if self._source_data is None:
            return

        # Use nearest neighbor approach for attribution
        try:
            from sklearn.neighbors import NearestNeighbors

            # Prepare data
            numeric_cols = self._source_data.select_dtypes(include=[np.number]).columns
            common_cols = [c for c in numeric_cols if c in synthetic.columns]

            if not common_cols:
                return

            source_matrix = self._source_data[common_cols].fillna(0).values
            synth_matrix = synthetic[common_cols].fillna(0).values

            # Fit nearest neighbors
            nn = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(self._source_data)))
            nn.fit(source_matrix)

            # Find neighbors for each synthetic record
            distances, indices = nn.kneighbors(synth_matrix)

            for i, (dists, idxs) in enumerate(zip(distances, indices)):
                syn_id = synthetic.iloc[i]["_synthetic_id"]

                # Convert distances to influences (inverse)
                if np.all(dists == 0):
                    influences = np.ones(len(dists)) / len(dists)
                else:
                    inv_dists = 1.0 / (dists + 1e-6)
                    influences = inv_dists / inv_dists.sum()

                # Get source IDs
                id_col = self.tracker._source_id_column
                source_ids = [
                    str(self._source_data.iloc[idx][id_col])
                    for idx in idxs
                ]

                self.tracker.record_attribution(
                    synthetic_id=syn_id,
                    source_ids=source_ids,
                    influences=list(influences),
                    method="nearest_neighbor",
                )

        except ImportError:
            logger.warning("sklearn not available for attribution computation")

    def explain_record(
        self,
        record: Union[pd.Series, Dict[str, Any], str],
    ) -> RecordExplanation:
        """Generate explanation for a synthetic record.

        Args:
            record: Record to explain (Series, dict, or synthetic_id)

        Returns:
            RecordExplanation object
        """
        # Get record data and ID
        if isinstance(record, str):
            syn_id = record
            if self._synthetic_data is not None:
                mask = self._synthetic_data["_synthetic_id"] == syn_id
                if mask.any():
                    record_data = self._synthetic_data[mask].iloc[0].to_dict()
                else:
                    record_data = {}
            else:
                record_data = {}
        elif isinstance(record, pd.Series):
            syn_id = record.get("_synthetic_id", "unknown")
            record_data = record.to_dict()
        else:
            syn_id = record.get("_synthetic_id", "unknown")
            record_data = dict(record)

        # Get attribution
        attribution = self.tracker.get_attribution(syn_id)
        if attribution is None:
            attribution = Attribution(
                synthetic_id=syn_id,
                source_attributions=[],
                method="unknown",
            )

        # Generate feature explanations
        feature_explanations = self._generate_feature_explanations(
            record_data, attribution
        )

        return RecordExplanation(
            record_id=syn_id,
            record_data=record_data,
            feature_explanations=feature_explanations,
            overall_attribution=attribution,
        )

    def _generate_feature_explanations(
        self,
        record_data: Dict[str, Any],
        attribution: Attribution,
    ) -> Dict[str, FeatureExplanation]:
        """Generate explanations for each feature."""
        explanations = {}

        # Get feature importance
        importance = {}
        if self._feature_importance:
            importance = self._feature_importance.get_importance()

        for feature, value in record_data.items():
            if feature.startswith("_"):
                continue

            # Collect influences from source records
            influences = []
            for src_attr in attribution.source_attributions:
                src_data = src_attr.get("source_data", {})
                if feature in src_data:
                    influences.append({
                        "source_id": src_attr["source_id"],
                        "source_value": src_data[feature],
                        "influence": src_attr["influence"],
                    })

            # Generate explanation text
            explanation_text = self._generate_explanation_text(
                feature, value, influences, importance.get(feature, 0)
            )

            explanations[feature] = FeatureExplanation(
                feature_name=feature,
                generated_value=value,
                generation_method="statistical_sampling",
                influences=influences,
                confidence=importance.get(feature, 0.5),
                explanation_text=explanation_text,
            )

        return explanations

    def _generate_explanation_text(
        self,
        feature: str,
        value: Any,
        influences: List[Dict[str, Any]],
        importance: float,
    ) -> str:
        """Generate human-readable explanation for a feature."""
        if not influences:
            return f"Generated independently (importance: {importance:.1%})"

        top_influences = sorted(influences, key=lambda x: x["influence"], reverse=True)[:3]

        if len(top_influences) == 1:
            src = top_influences[0]
            return (
                f"Value {value} influenced by record {src['source_id']} "
                f"(value: {src['source_value']}, influence: {src['influence']:.1%})"
            )
        else:
            source_vals = [str(i["source_value"]) for i in top_influences]
            return (
                f"Value {value} interpolated from sources with values: "
                f"{', '.join(source_vals[:3])}"
            )

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self._feature_importance:
            return self._feature_importance.get_importance()
        return {}


class LineageTracker:
    """Tracks complete lineage/provenance of synthetic data.

    Example:
        >>> tracker = LineageTracker()
        >>> tracker.record_generation(
        ...     run_id="run_001",
        ...     source_hash="abc123",
        ...     generator_config={...},
        ...     output_count=10000
        ... )
    """

    def __init__(self) -> None:
        self._lineage_records: List[Dict[str, Any]] = []

    def record_generation(
        self,
        run_id: str,
        source_hash: str,
        generator_type: str,
        generator_config: Dict[str, Any],
        output_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a generation run.

        Args:
            run_id: Unique run identifier
            source_hash: Hash of source data
            generator_type: Type of generator used
            generator_config: Generator configuration
            output_count: Number of records generated
            metadata: Additional metadata

        Returns:
            Lineage record
        """
        record = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "source_hash": source_hash,
            "generator_type": generator_type,
            "generator_config": generator_config,
            "output_count": output_count,
            "metadata": metadata or {},
        }

        self._lineage_records.append(record)
        return record

    def get_lineage(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get lineage record for a run."""
        for record in self._lineage_records:
            if record["run_id"] == run_id:
                return record
        return None

    def get_all_lineage(self) -> List[Dict[str, Any]]:
        """Get all lineage records."""
        return list(self._lineage_records)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert lineage to DataFrame."""
        return pd.DataFrame(self._lineage_records)


def compute_data_hash(data: pd.DataFrame) -> str:
    """Compute hash of DataFrame for lineage tracking."""
    import hashlib

    # Convert to bytes and hash
    data_bytes = data.to_csv(index=False).encode()
    return hashlib.sha256(data_bytes).hexdigest()[:16]
