"""Anomaly synthesis for generating realistic outliers and edge cases.

This module provides tools for generating synthetic anomalies:
- Fraud transactions
- Security intrusions
- Equipment failures
- Rare medical conditions

Example:
    >>> from genesis.anomaly import AnomalyGenerator, AnomalyType
    >>>
    >>> # Generate fraud-like transactions
    >>> gen = AnomalyGenerator(normal_data=transactions)
    >>> frauds = gen.generate(
    ...     n_samples=100,
    ...     anomaly_type=AnomalyType.STATISTICAL,
    ...     severity=0.8,
    ... )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class AnomalyType(Enum):
    """Types of anomalies to generate."""

    STATISTICAL = "statistical"  # Outliers based on distribution
    POINT = "point"  # Individual point anomalies
    CONTEXTUAL = "contextual"  # Anomalies in context
    COLLECTIVE = "collective"  # Groups of related anomalies
    PATTERN = "pattern"  # Pattern-based anomalies
    ADVERSARIAL = "adversarial"  # Adversarial examples


class AnomalyProfile(Enum):
    """Pre-defined anomaly profiles for common use cases."""

    FRAUD = "fraud"  # Financial fraud patterns
    INTRUSION = "intrusion"  # Network intrusion patterns
    EQUIPMENT_FAILURE = "equipment"  # Industrial equipment failures
    MEDICAL = "medical"  # Rare medical conditions
    CUSTOM = "custom"  # User-defined patterns


@dataclass
class AnomalyConfig:
    """Configuration for anomaly generation."""

    severity: float = 0.5  # 0.0-1.0, how extreme the anomaly
    rarity: float = 0.1  # Target proportion of anomalies
    preserve_structure: bool = True  # Maintain some realistic structure
    columns: Optional[List[str]] = None  # Columns to modify (None = all)
    seed: Optional[int] = None

    # Profile-specific settings
    profile_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyResult:
    """Result of anomaly generation."""

    data: pd.DataFrame
    n_anomalies: int
    anomaly_indices: List[int]
    anomaly_scores: List[float]
    modifications: Dict[int, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_anomalies": self.n_anomalies,
            "anomaly_indices": self.anomaly_indices,
            "anomaly_scores": self.anomaly_scores,
        }


class AnomalyGenerator:
    """Generator for synthetic anomalies and outliers.

    Creates realistic anomalies by modifying normal data patterns
    in controllable ways.
    """

    def __init__(
        self,
        normal_data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
    ):
        """Initialize anomaly generator.

        Args:
            normal_data: Reference normal data
            discrete_columns: Categorical columns
        """
        self.normal_data = normal_data
        self.discrete_columns = discrete_columns or []

        # Infer discrete columns
        if not self.discrete_columns:
            self.discrete_columns = self._infer_discrete()

        # Compute statistics for anomaly generation
        self._stats = self._compute_stats()

    def generate(
        self,
        n_samples: int,
        anomaly_type: AnomalyType = AnomalyType.STATISTICAL,
        config: Optional[AnomalyConfig] = None,
    ) -> AnomalyResult:
        """Generate anomalous samples.

        Args:
            n_samples: Number of anomalies to generate
            anomaly_type: Type of anomaly to generate
            config: Anomaly configuration

        Returns:
            AnomalyResult with generated anomalies
        """
        config = config or AnomalyConfig()

        if config.seed is not None:
            np.random.seed(config.seed)

        logger.info(f"Generating {n_samples} {anomaly_type.value} anomalies")

        if anomaly_type == AnomalyType.STATISTICAL:
            return self._generate_statistical(n_samples, config)
        elif anomaly_type == AnomalyType.POINT:
            return self._generate_point(n_samples, config)
        elif anomaly_type == AnomalyType.CONTEXTUAL:
            return self._generate_contextual(n_samples, config)
        elif anomaly_type == AnomalyType.COLLECTIVE:
            return self._generate_collective(n_samples, config)
        elif anomaly_type == AnomalyType.PATTERN:
            return self._generate_pattern(n_samples, config)
        elif anomaly_type == AnomalyType.ADVERSARIAL:
            return self._generate_adversarial(n_samples, config)
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    def generate_from_profile(
        self,
        n_samples: int,
        profile: AnomalyProfile,
        severity: float = 0.5,
    ) -> AnomalyResult:
        """Generate anomalies based on a predefined profile.

        Args:
            n_samples: Number of anomalies to generate
            profile: Anomaly profile to use
            severity: How extreme the anomalies should be (0-1)

        Returns:
            AnomalyResult with generated anomalies
        """
        config = AnomalyConfig(severity=severity)

        if profile == AnomalyProfile.FRAUD:
            return self._generate_fraud_pattern(n_samples, config)
        elif profile == AnomalyProfile.INTRUSION:
            return self._generate_intrusion_pattern(n_samples, config)
        elif profile == AnomalyProfile.EQUIPMENT_FAILURE:
            return self._generate_equipment_pattern(n_samples, config)
        elif profile == AnomalyProfile.MEDICAL:
            return self._generate_medical_pattern(n_samples, config)
        else:
            return self.generate(n_samples, AnomalyType.STATISTICAL, config)

    def inject_anomalies(
        self,
        data: pd.DataFrame,
        anomaly_rate: float = 0.05,
        anomaly_type: AnomalyType = AnomalyType.STATISTICAL,
        config: Optional[AnomalyConfig] = None,
    ) -> Tuple[pd.DataFrame, List[int]]:
        """Inject anomalies into existing data.

        Args:
            data: Data to inject anomalies into
            anomaly_rate: Proportion of data to make anomalous
            anomaly_type: Type of anomalies to inject
            config: Anomaly configuration

        Returns:
            Tuple of (modified data, anomaly indices)
        """
        n_anomalies = int(len(data) * anomaly_rate)

        # Generate anomalies
        result = self.generate(n_anomalies, anomaly_type, config)

        # Select random indices to replace
        indices = np.random.choice(len(data), n_anomalies, replace=False)

        # Create modified data
        modified = data.copy()
        for i, idx in enumerate(indices):
            if i < len(result.data):
                modified.iloc[idx] = result.data.iloc[i]

        return modified, list(indices)

    def _infer_discrete(self) -> List[str]:
        """Infer discrete columns."""
        discrete = []
        for col in self.normal_data.columns:
            if not pd.api.types.is_numeric_dtype(self.normal_data[col]):
                discrete.append(col)
            elif self.normal_data[col].nunique() < 20:
                discrete.append(col)
        return discrete

    def _compute_stats(self) -> Dict[str, Any]:
        """Compute statistics for anomaly generation."""
        stats = {}

        for col in self.normal_data.columns:
            col_data = self.normal_data[col]

            if pd.api.types.is_numeric_dtype(col_data):
                stats[col] = {
                    "type": "numeric",
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()),
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "q1": float(col_data.quantile(0.25)),
                    "q3": float(col_data.quantile(0.75)),
                    "iqr": float(col_data.quantile(0.75) - col_data.quantile(0.25)),
                }
            else:
                value_counts = col_data.value_counts(normalize=True)
                stats[col] = {
                    "type": "categorical",
                    "categories": list(value_counts.index),
                    "frequencies": list(value_counts.values),
                    "n_unique": len(value_counts),
                }

        return stats

    def _generate_statistical(
        self,
        n_samples: int,
        config: AnomalyConfig,
    ) -> AnomalyResult:
        """Generate statistical outliers."""
        # Sample base rows
        base_indices = np.random.choice(len(self.normal_data), n_samples, replace=True)
        anomalies = self.normal_data.iloc[base_indices].copy().reset_index(drop=True)

        modifications = {}
        anomaly_scores = []

        columns = config.columns or self.normal_data.columns.tolist()

        for idx in range(n_samples):
            row_mods = {}
            row_score = 0.0

            # Modify numeric columns
            for col in columns:
                if col not in self._stats:
                    continue

                stat = self._stats[col]

                if stat["type"] == "numeric":
                    # Generate outlier value
                    original = anomalies.loc[idx, col]

                    # Determine direction and magnitude
                    direction = np.random.choice([-1, 1])
                    magnitude = 2 + config.severity * 4  # 2-6 std deviations

                    if stat["std"] > 0:
                        new_value = stat["mean"] + direction * magnitude * stat["std"]

                        # Optionally preserve some structure
                        if config.preserve_structure and np.random.random() > config.severity:
                            # Keep value in reasonable range
                            new_value = np.clip(
                                new_value, stat["min"] - stat["iqr"], stat["max"] + stat["iqr"]
                            )

                        # Cast column to float to avoid dtype incompatibility
                        if anomalies[col].dtype != float:
                            anomalies[col] = anomalies[col].astype(float)
                        anomalies.loc[idx, col] = new_value
                        row_mods[col] = {"original": original, "new": new_value}
                        row_score += abs(new_value - stat["mean"]) / stat["std"]

                elif stat["type"] == "categorical" and np.random.random() < config.severity:
                    # Rare category or invalid category
                    if len(stat["categories"]) > 1:
                        # Pick rare category
                        rare_idx = np.random.randint(
                            max(1, int(len(stat["categories"]) * 0.8)), len(stat["categories"])
                        )
                        new_value = stat["categories"][rare_idx]
                        original = anomalies.loc[idx, col]
                        anomalies.loc[idx, col] = new_value
                        row_mods[col] = {"original": original, "new": new_value}
                        row_score += 1.0

            modifications[idx] = row_mods
            anomaly_scores.append(row_score / max(len(columns), 1))

        return AnomalyResult(
            data=anomalies,
            n_anomalies=n_samples,
            anomaly_indices=list(range(n_samples)),
            anomaly_scores=anomaly_scores,
            modifications=modifications,
        )

    def _generate_point(
        self,
        n_samples: int,
        config: AnomalyConfig,
    ) -> AnomalyResult:
        """Generate point anomalies (single extreme values)."""
        # Sample base rows
        base_indices = np.random.choice(len(self.normal_data), n_samples, replace=True)
        anomalies = self.normal_data.iloc[base_indices].copy().reset_index(drop=True)

        modifications = {}
        anomaly_scores = []

        numeric_cols = [
            c for c in self.normal_data.columns if self._stats.get(c, {}).get("type") == "numeric"
        ]

        for idx in range(n_samples):
            row_mods = {}

            # Pick 1-2 columns to modify
            n_cols = np.random.randint(1, min(3, len(numeric_cols) + 1))
            cols_to_modify = np.random.choice(numeric_cols, n_cols, replace=False)

            for col in cols_to_modify:
                stat = self._stats[col]
                original = anomalies.loc[idx, col]

                # Extreme value in one direction
                direction = np.random.choice([-1, 1])
                magnitude = 3 + config.severity * 5  # 3-8 std deviations

                if stat["std"] > 0:
                    new_value = stat["mean"] + direction * magnitude * stat["std"]
                    anomalies.loc[idx, col] = new_value
                    row_mods[col] = {"original": original, "new": new_value}

            modifications[idx] = row_mods
            score = sum(
                abs(anomalies.loc[idx, c] - self._stats[c]["mean"]) / self._stats[c]["std"]
                for c in cols_to_modify
                if self._stats[c]["std"] > 0
            ) / len(cols_to_modify)
            anomaly_scores.append(score)

        return AnomalyResult(
            data=anomalies,
            n_anomalies=n_samples,
            anomaly_indices=list(range(n_samples)),
            anomaly_scores=anomaly_scores,
            modifications=modifications,
        )

    def _generate_contextual(
        self,
        n_samples: int,
        config: AnomalyConfig,
    ) -> AnomalyResult:
        """Generate contextual anomalies (normal values in wrong context)."""
        # Sample base rows
        base_indices = np.random.choice(len(self.normal_data), n_samples, replace=True)
        anomalies = self.normal_data.iloc[base_indices].copy().reset_index(drop=True)

        modifications = {}
        anomaly_scores = []

        numeric_cols = [
            c for c in self.normal_data.columns if self._stats.get(c, {}).get("type") == "numeric"
        ]

        for idx in range(n_samples):
            row_mods = {}

            # Swap values between columns to break correlations
            if len(numeric_cols) >= 2:
                cols = np.random.choice(numeric_cols, 2, replace=False)
                original_values = [anomalies.loc[idx, c] for c in cols]

                # Swap with values from different rows
                donor_idx = np.random.randint(len(self.normal_data))
                for i, col in enumerate(cols):
                    new_value = self.normal_data.iloc[donor_idx][cols[1 - i]]
                    anomalies.loc[idx, col] = new_value
                    row_mods[col] = {"original": original_values[i], "new": new_value}

            modifications[idx] = row_mods
            anomaly_scores.append(config.severity)

        return AnomalyResult(
            data=anomalies,
            n_anomalies=n_samples,
            anomaly_indices=list(range(n_samples)),
            anomaly_scores=anomaly_scores,
            modifications=modifications,
        )

    def _generate_collective(
        self,
        n_samples: int,
        config: AnomalyConfig,
    ) -> AnomalyResult:
        """Generate collective anomalies (groups of related anomalies)."""
        # Generate in clusters
        n_clusters = max(1, n_samples // 5)
        samples_per_cluster = n_samples // n_clusters

        all_anomalies = []
        all_modifications = {}
        all_scores = []

        for _cluster in range(n_clusters):
            # Sample a base row for the cluster
            base_idx = np.random.randint(len(self.normal_data))
            base_row = self.normal_data.iloc[base_idx].copy()

            # Create cluster-specific modifications
            cluster_mods = {}
            numeric_cols = [
                c
                for c in self.normal_data.columns
                if self._stats.get(c, {}).get("type") == "numeric"
            ]

            for col in numeric_cols:
                stat = self._stats[col]
                if np.random.random() < config.severity:
                    # Cluster shift
                    shift = np.random.normal(0, stat["std"] * config.severity * 2)
                    cluster_mods[col] = shift

            # Generate cluster members
            for _i in range(samples_per_cluster):
                row = base_row.copy()
                row_mods = {}

                for col, shift in cluster_mods.items():
                    original = row[col]
                    # Add noise within cluster
                    noise = np.random.normal(0, self._stats[col]["std"] * 0.2)
                    new_value = original + shift + noise
                    row[col] = new_value
                    row_mods[col] = {"original": original, "new": new_value}

                all_anomalies.append(row)
                idx = len(all_anomalies) - 1
                all_modifications[idx] = row_mods
                all_scores.append(config.severity)

        # Convert to DataFrame
        anomalies = pd.DataFrame(all_anomalies[:n_samples]).reset_index(drop=True)

        return AnomalyResult(
            data=anomalies,
            n_anomalies=n_samples,
            anomaly_indices=list(range(n_samples)),
            anomaly_scores=all_scores[:n_samples],
            modifications={k: v for k, v in all_modifications.items() if k < n_samples},
        )

    def _generate_pattern(
        self,
        n_samples: int,
        config: AnomalyConfig,
    ) -> AnomalyResult:
        """Generate pattern-based anomalies."""
        # Use statistical as base but with consistent patterns
        return self._generate_statistical(n_samples, config)

    def _generate_adversarial(
        self,
        n_samples: int,
        config: AnomalyConfig,
    ) -> AnomalyResult:
        """Generate adversarial examples (subtle but impactful changes)."""
        # Sample base rows
        base_indices = np.random.choice(len(self.normal_data), n_samples, replace=True)
        anomalies = self.normal_data.iloc[base_indices].copy().reset_index(drop=True)

        modifications = {}
        anomaly_scores = []

        numeric_cols = [
            c for c in self.normal_data.columns if self._stats.get(c, {}).get("type") == "numeric"
        ]

        for idx in range(n_samples):
            row_mods = {}

            # Small perturbations to many columns
            for col in numeric_cols:
                stat = self._stats[col]
                original = anomalies.loc[idx, col]

                # Small but consistent perturbation
                perturbation = stat["std"] * config.severity * 0.5 * np.random.choice([-1, 1])
                new_value = original + perturbation

                anomalies.loc[idx, col] = new_value
                row_mods[col] = {"original": original, "new": new_value, "delta": perturbation}

            modifications[idx] = row_mods
            anomaly_scores.append(config.severity * 0.5)  # Lower score for subtle changes

        return AnomalyResult(
            data=anomalies,
            n_anomalies=n_samples,
            anomaly_indices=list(range(n_samples)),
            anomaly_scores=anomaly_scores,
            modifications=modifications,
        )

    def _generate_fraud_pattern(
        self,
        n_samples: int,
        config: AnomalyConfig,
    ) -> AnomalyResult:
        """Generate fraud-like patterns."""
        # Typical fraud patterns: unusual amounts, times, locations
        config.profile_settings = {
            "amount_multiplier": 5 + config.severity * 15,  # 5-20x normal
            "velocity_increase": True,  # Multiple transactions quickly
        }
        return self._generate_statistical(n_samples, config)

    def _generate_intrusion_pattern(
        self,
        n_samples: int,
        config: AnomalyConfig,
    ) -> AnomalyResult:
        """Generate network intrusion patterns."""
        # Typical intrusion patterns: unusual ports, packet sizes, frequencies
        config.profile_settings = {
            "unusual_ports": True,
            "burst_traffic": True,
        }
        return self._generate_collective(n_samples, config)

    def _generate_equipment_pattern(
        self,
        n_samples: int,
        config: AnomalyConfig,
    ) -> AnomalyResult:
        """Generate equipment failure patterns."""
        # Typical failure patterns: gradual degradation or sudden spikes
        config.profile_settings = {
            "degradation": True,
            "spike_probability": 0.3,
        }
        return self._generate_statistical(n_samples, config)

    def _generate_medical_pattern(
        self,
        n_samples: int,
        config: AnomalyConfig,
    ) -> AnomalyResult:
        """Generate rare medical condition patterns."""
        # Rare combinations of symptoms/values
        config.profile_settings = {
            "rare_combinations": True,
        }
        return self._generate_contextual(n_samples, config)


class BalancedDatasetGenerator:
    """Generate balanced datasets with controlled anomaly ratios.

    Example:
        >>> gen = BalancedDatasetGenerator(
        ...     normal_data=normal_df,
        ...     anomaly_ratio=0.1,
        ... )
        >>> balanced_data, labels = gen.generate(10000)
    """

    def __init__(
        self,
        normal_data: pd.DataFrame,
        anomaly_ratio: float = 0.1,
        discrete_columns: Optional[List[str]] = None,
    ):
        """Initialize balanced dataset generator.

        Args:
            normal_data: Reference normal data
            anomaly_ratio: Target ratio of anomalies (0-1)
            discrete_columns: Categorical columns
        """
        self.normal_data = normal_data
        self.anomaly_ratio = anomaly_ratio
        self.anomaly_generator = AnomalyGenerator(normal_data, discrete_columns)

    def generate(
        self,
        n_samples: int,
        anomaly_types: Optional[List[AnomalyType]] = None,
        config: Optional[AnomalyConfig] = None,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate balanced dataset with normal and anomalous samples.

        Args:
            n_samples: Total number of samples
            anomaly_types: Types of anomalies to include (default: all)
            config: Anomaly configuration

        Returns:
            Tuple of (data, labels) where labels 0=normal, 1=anomaly
        """
        n_anomalies = int(n_samples * self.anomaly_ratio)
        n_normal = n_samples - n_anomalies

        # Sample normal data
        normal_indices = np.random.choice(len(self.normal_data), n_normal, replace=True)
        normal_samples = self.normal_data.iloc[normal_indices].copy()
        normal_labels = np.zeros(n_normal)

        # Generate anomalies
        anomaly_types = anomaly_types or [AnomalyType.STATISTICAL]
        anomalies_per_type = n_anomalies // len(anomaly_types)

        anomaly_samples = []
        for atype in anomaly_types:
            result = self.anomaly_generator.generate(anomalies_per_type, atype, config)
            anomaly_samples.append(result.data)

        anomalies = pd.concat(anomaly_samples, ignore_index=True)
        anomaly_labels = np.ones(len(anomalies))

        # Combine and shuffle
        combined = pd.concat([normal_samples, anomalies], ignore_index=True)
        labels = np.concatenate([normal_labels, anomaly_labels])

        # Shuffle
        shuffle_idx = np.random.permutation(len(combined))
        combined = combined.iloc[shuffle_idx].reset_index(drop=True)
        labels = labels[shuffle_idx]

        return combined, labels


__all__ = [
    "AnomalyGenerator",
    "AnomalyType",
    "AnomalyProfile",
    "AnomalyConfig",
    "AnomalyResult",
    "BalancedDatasetGenerator",
]
