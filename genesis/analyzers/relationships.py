"""Relationship detection between columns and tables."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ColumnRelationship:
    """Represents a relationship between two columns."""

    column1: str
    column2: str
    relationship_type: str  # 'correlation', 'functional', 'categorical_association'
    strength: float  # 0 to 1
    direction: Optional[str] = None  # 'positive', 'negative', None for non-directional
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column1": self.column1,
            "column2": self.column2,
            "relationship_type": self.relationship_type,
            "strength": self.strength,
            "direction": self.direction,
            "details": self.details,
        }


@dataclass
class ForeignKeyCandidate:
    """Represents a potential foreign key relationship between tables."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str
    confidence: float  # 0 to 1
    coverage: float  # Fraction of child values found in parent
    cardinality: str  # '1:1', '1:N', 'N:M'

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "child_table": self.child_table,
            "child_column": self.child_column,
            "parent_table": self.parent_table,
            "parent_column": self.parent_column,
            "confidence": self.confidence,
            "coverage": self.coverage,
            "cardinality": self.cardinality,
        }


class RelationshipAnalyzer:
    """Analyzer for detecting relationships between columns."""

    def __init__(
        self,
        correlation_threshold: float = 0.3,
        functional_dependency_threshold: float = 0.95,
        categorical_association_threshold: float = 0.1,
    ) -> None:
        """Initialize the relationship analyzer.

        Args:
            correlation_threshold: Min correlation to report
            functional_dependency_threshold: Min ratio for functional dependency
            categorical_association_threshold: Min Cramér's V for association
        """
        self.correlation_threshold = correlation_threshold
        self.functional_dependency_threshold = functional_dependency_threshold
        self.categorical_association_threshold = categorical_association_threshold

    def analyze(self, data: pd.DataFrame) -> List[ColumnRelationship]:
        """Analyze all relationships in a DataFrame.

        Args:
            data: DataFrame to analyze

        Returns:
            List of detected relationships
        """
        relationships = []

        # Get column types
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

        # Numeric correlations
        relationships.extend(self._find_numeric_correlations(data, numeric_cols))

        # Categorical associations
        relationships.extend(self._find_categorical_associations(data, categorical_cols))

        # Functional dependencies
        relationships.extend(self._find_functional_dependencies(data))

        # Mixed relationships (numeric-categorical)
        relationships.extend(self._find_mixed_relationships(data, numeric_cols, categorical_cols))

        return relationships

    def _find_numeric_correlations(
        self,
        data: pd.DataFrame,
        numeric_cols: List[str],
    ) -> List[ColumnRelationship]:
        """Find correlations between numeric columns."""
        relationships = []

        if len(numeric_cols) < 2:
            return relationships

        corr_matrix = data[numeric_cols].corr()

        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i + 1 :]:
                corr = corr_matrix.loc[col1, col2]

                if pd.isna(corr):
                    continue

                if abs(corr) >= self.correlation_threshold:
                    relationships.append(
                        ColumnRelationship(
                            column1=col1,
                            column2=col2,
                            relationship_type="correlation",
                            strength=abs(corr),
                            direction="positive" if corr > 0 else "negative",
                            details={"correlation_coefficient": corr},
                        )
                    )

        return relationships

    def _find_categorical_associations(
        self,
        data: pd.DataFrame,
        categorical_cols: List[str],
    ) -> List[ColumnRelationship]:
        """Find associations between categorical columns using Cramér's V."""
        relationships = []

        if len(categorical_cols) < 2:
            return relationships

        for i, col1 in enumerate(categorical_cols):
            for col2 in categorical_cols[i + 1 :]:
                cramers_v = self._compute_cramers_v(data[col1], data[col2])

                if cramers_v >= self.categorical_association_threshold:
                    relationships.append(
                        ColumnRelationship(
                            column1=col1,
                            column2=col2,
                            relationship_type="categorical_association",
                            strength=cramers_v,
                            details={"cramers_v": cramers_v},
                        )
                    )

        return relationships

    def _compute_cramers_v(self, col1: pd.Series, col2: pd.Series) -> float:
        """Compute Cramér's V statistic for association between categorical variables."""
        # Create contingency table
        contingency = pd.crosstab(col1, col2)

        if contingency.size == 0:
            return 0.0

        # Chi-squared test
        chi2 = stats.chi2_contingency(contingency)[0]
        n = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1

        if n == 0 or min_dim == 0:
            return 0.0

        return np.sqrt(chi2 / (n * min_dim))

    def _find_functional_dependencies(
        self,
        data: pd.DataFrame,
    ) -> List[ColumnRelationship]:
        """Find functional dependencies (column A determines column B)."""
        relationships = []

        for col1 in data.columns:
            for col2 in data.columns:
                if col1 == col2:
                    continue

                # Check if col1 functionally determines col2
                ratio = self._check_functional_dependency(data, col1, col2)

                if ratio >= self.functional_dependency_threshold:
                    relationships.append(
                        ColumnRelationship(
                            column1=col1,
                            column2=col2,
                            relationship_type="functional_dependency",
                            strength=ratio,
                            details={
                                "determinant": col1,
                                "dependent": col2,
                                "determination_ratio": ratio,
                            },
                        )
                    )

        return relationships

    def _check_functional_dependency(
        self,
        data: pd.DataFrame,
        determinant: str,
        dependent: str,
    ) -> float:
        """Check if determinant column functionally determines dependent column.

        Returns the ratio of unique determinant values that map to exactly one
        dependent value.
        """
        # Group by determinant and count unique dependent values
        grouped = data.groupby(determinant)[dependent].nunique()

        # Count how many determinant values map to exactly one dependent value
        n_functional = (grouped == 1).sum()
        n_total = len(grouped)

        if n_total == 0:
            return 0.0

        return n_functional / n_total

    def _find_mixed_relationships(
        self,
        data: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> List[ColumnRelationship]:
        """Find relationships between numeric and categorical columns."""
        relationships = []

        for num_col in numeric_cols:
            for cat_col in categorical_cols:
                # Use point-biserial correlation or ANOVA-based measure
                strength = self._compute_numeric_categorical_association(
                    data[num_col], data[cat_col]
                )

                if strength >= self.correlation_threshold:
                    relationships.append(
                        ColumnRelationship(
                            column1=num_col,
                            column2=cat_col,
                            relationship_type="numeric_categorical_association",
                            strength=strength,
                            details={"eta_squared": strength},
                        )
                    )

        return relationships

    def _compute_numeric_categorical_association(
        self,
        numeric_col: pd.Series,
        categorical_col: pd.Series,
    ) -> float:
        """Compute association strength between numeric and categorical columns using eta-squared."""
        # Group numeric values by category
        groups = []
        for category in categorical_col.dropna().unique():
            mask = categorical_col == category
            group_values = numeric_col[mask].dropna()
            if len(group_values) > 0:
                groups.append(group_values.values)

        if len(groups) < 2:
            return 0.0

        # Perform one-way ANOVA
        try:
            f_stat, p_value = stats.f_oneway(*groups)

            if np.isnan(f_stat):
                return 0.0

            # Compute eta-squared (effect size)
            all_values = np.concatenate(groups)
            grand_mean = np.mean(all_values)

            ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
            ss_total = np.sum((all_values - grand_mean) ** 2)

            if ss_total == 0:
                return 0.0

            eta_squared = ss_between / ss_total
            return eta_squared

        except Exception:
            return 0.0


class MultiTableRelationshipAnalyzer:
    """Analyzer for detecting relationships between multiple tables."""

    def __init__(
        self,
        min_coverage: float = 0.8,
        min_confidence: float = 0.7,
    ) -> None:
        """Initialize the multi-table analyzer.

        Args:
            min_coverage: Minimum coverage for FK candidate
            min_confidence: Minimum confidence for FK candidate
        """
        self.min_coverage = min_coverage
        self.min_confidence = min_confidence

    def find_foreign_keys(
        self,
        tables: Dict[str, pd.DataFrame],
    ) -> List[ForeignKeyCandidate]:
        """Find potential foreign key relationships between tables.

        Args:
            tables: Dictionary mapping table names to DataFrames

        Returns:
            List of foreign key candidates
        """
        candidates = []
        table_names = list(tables.keys())

        for child_name in table_names:
            child_table = tables[child_name]

            for parent_name in table_names:
                if child_name == parent_name:
                    continue

                parent_table = tables[parent_name]

                # Check each column pair
                for child_col in child_table.columns:
                    for parent_col in parent_table.columns:
                        candidate = self._check_fk_candidate(
                            child_name,
                            child_table[child_col],
                            parent_name,
                            parent_table[parent_col],
                        )

                        if candidate is not None:
                            candidates.append(candidate)

        return candidates

    def _check_fk_candidate(
        self,
        child_table_name: str,
        child_col: pd.Series,
        parent_table_name: str,
        parent_col: pd.Series,
    ) -> Optional[ForeignKeyCandidate]:
        """Check if a column pair could be a foreign key relationship."""
        child_values = set(child_col.dropna().unique())
        parent_values = set(parent_col.dropna().unique())

        if len(child_values) == 0 or len(parent_values) == 0:
            return None

        # Check coverage (how many child values exist in parent)
        covered = child_values & parent_values
        coverage = len(covered) / len(child_values)

        if coverage < self.min_coverage:
            return None

        # Check if parent values are unique (primary key candidate)
        parent_unique_ratio = len(parent_values) / len(parent_col.dropna())

        # Compute confidence based on coverage and uniqueness
        confidence = coverage * parent_unique_ratio

        if confidence < self.min_confidence:
            return None

        # Determine cardinality
        cardinality = self._determine_cardinality(child_col, parent_col)

        return ForeignKeyCandidate(
            child_table=child_table_name,
            child_column=child_col.name,
            parent_table=parent_table_name,
            parent_column=parent_col.name,
            confidence=confidence,
            coverage=coverage,
            cardinality=cardinality,
        )

    def _determine_cardinality(
        self,
        child_col: pd.Series,
        parent_col: pd.Series,
    ) -> str:
        """Determine the cardinality of the relationship."""
        child_unique = child_col.nunique()
        child_total = len(child_col.dropna())
        parent_unique = parent_col.nunique()

        # If child values are unique, it's 1:1
        if child_unique == child_total:
            return "1:1"

        # If multiple child rows per parent value, it's 1:N
        if child_unique < parent_unique:
            return "1:N"

        return "N:M"


def detect_relationships(
    data: pd.DataFrame,
    correlation_threshold: float = 0.3,
) -> List[ColumnRelationship]:
    """Convenience function to detect relationships in a DataFrame.

    Args:
        data: DataFrame to analyze
        correlation_threshold: Minimum correlation to report

    Returns:
        List of detected relationships
    """
    analyzer = RelationshipAnalyzer(correlation_threshold=correlation_threshold)
    return analyzer.analyze(data)
