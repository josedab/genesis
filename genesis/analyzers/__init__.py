"""Data analysis module for Genesis."""

from genesis.analyzers.privacy import (
    PrivacyAnalyzer,
    PrivacyRiskAssessment,
    assess_privacy_risk,
)
from genesis.analyzers.relationships import (
    ColumnRelationship,
    ForeignKeyCandidate,
    MultiTableRelationshipAnalyzer,
    RelationshipAnalyzer,
    detect_relationships,
)
from genesis.analyzers.schema import (
    SchemaAnalyzer,
    infer_schema,
)
from genesis.analyzers.statistics import (
    DatasetStats,
    StatisticalAnalyzer,
    UnivariateStats,
    compare_correlation_matrices,
    compute_correlation_matrix,
)

__all__ = [
    # Schema
    "SchemaAnalyzer",
    "infer_schema",
    # Statistics
    "StatisticalAnalyzer",
    "UnivariateStats",
    "DatasetStats",
    "compute_correlation_matrix",
    "compare_correlation_matrices",
    # Relationships
    "RelationshipAnalyzer",
    "MultiTableRelationshipAnalyzer",
    "ColumnRelationship",
    "ForeignKeyCandidate",
    "detect_relationships",
    # Privacy
    "PrivacyAnalyzer",
    "PrivacyRiskAssessment",
    "assess_privacy_risk",
]
