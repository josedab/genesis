"""Multi-table synthesis module for Genesis."""

from genesis.multitable.constraints import (
    compute_cardinality_distribution,
    enforce_referential_integrity,
    sample_child_counts,
    validate_referential_integrity,
)
from genesis.multitable.generator import MultiTableGenerator
from genesis.multitable.schema import ForeignKey, RelationalSchema, TableSchema

__all__ = [
    # Schema
    "RelationalSchema",
    "TableSchema",
    "ForeignKey",
    # Generator
    "MultiTableGenerator",
    # Constraints
    "validate_referential_integrity",
    "enforce_referential_integrity",
    "compute_cardinality_distribution",
    "sample_child_counts",
]
