"""Multi-table synthetic data generator."""

import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from genesis.core.base import BaseGenerator
from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.types import FittingResult, ProgressCallback
from genesis.generators.tabular import CTGANGenerator, GaussianCopulaGenerator
from genesis.multitable.constraints import (
    compute_cardinality_distribution,
    sample_child_counts,
    validate_referential_integrity,
)
from genesis.multitable.schema import ForeignKey, RelationalSchema
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class MultiTableGenerator(BaseGenerator):
    """Generator for multi-table relational data.

    Generates synthetic data while preserving referential integrity
    and cross-table correlations.

    Example:
        >>> tables = {"users": users_df, "orders": orders_df}
        >>> schema = RelationalSchema.from_dataframes(tables, foreign_keys=[
        ...     {"child_table": "orders", "child_column": "user_id",
        ...      "parent_table": "users", "parent_column": "id"}
        ... ])
        >>> generator = MultiTableGenerator()
        >>> generator.fit(tables, schema)
        >>> synthetic_tables = generator.generate(n_samples={"users": 100, "orders": 500})
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        privacy: Optional[PrivacyConfig] = None,
        table_generator: str = "gaussian_copula",
        verbose: bool = True,
    ) -> None:
        """Initialize multi-table generator.

        Args:
            config: Generator configuration
            privacy: Privacy configuration
            table_generator: Generator to use for individual tables
            verbose: Whether to print progress
        """
        super().__init__(config, privacy)
        self.table_generator_type = table_generator
        self.verbose = verbose

        self._schema: Optional[RelationalSchema] = None
        self._table_generators: Dict[str, BaseGenerator] = {}
        self._cardinality_distributions: Dict[str, Dict[str, Any]] = {}
        self._original_tables: Dict[str, pd.DataFrame] = {}

    def _fit_impl(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> FittingResult:
        """Not used - use fit_tables instead."""
        raise NotImplementedError("Use fit_tables() for multi-table generation")

    def fit_tables(
        self,
        tables: Dict[str, pd.DataFrame],
        schema: RelationalSchema,
        discrete_columns: Optional[Dict[str, List[str]]] = None,
    ) -> "MultiTableGenerator":
        """Fit generator to multiple tables.

        Args:
            tables: Dictionary of table names to DataFrames
            schema: Relational schema
            discrete_columns: Discrete columns per table

        Returns:
            Self for method chaining
        """
        start_time = time.time()

        self._schema = schema
        self._original_tables = {name: df.copy() for name, df in tables.items()}
        discrete_columns = discrete_columns or {}

        if self.verbose:
            logger.info(f"Fitting multi-table generator on {len(tables)} tables")

        # Validate integrity
        violations = validate_referential_integrity(tables, schema)
        if violations:
            logger.warning(f"Referential integrity violations found: {violations}")

        # Get topological order (parents before children)
        order = schema.get_topological_order()

        # Fit generators in topological order
        for table_name in order:
            if table_name not in tables:
                logger.warning(f"Table '{table_name}' not found in data, skipping")
                continue

            if self.verbose:
                logger.info(f"Fitting generator for table '{table_name}'")

            df = tables[table_name]
            table_discrete = discrete_columns.get(table_name, [])

            # Create and fit generator
            if self.table_generator_type == "ctgan":
                gen = CTGANGenerator(config=self.config, privacy=self.privacy)
            else:
                gen = GaussianCopulaGenerator(config=self.config, privacy=self.privacy)

            gen.fit(df, discrete_columns=table_discrete)
            self._table_generators[table_name] = gen

            # Compute cardinality distributions for foreign keys
            for fk in schema.tables[table_name].foreign_keys:
                if fk.parent_table in tables:
                    key = (
                        f"{fk.child_table}.{fk.child_column}->{fk.parent_table}.{fk.parent_column}"
                    )
                    self._cardinality_distributions[key] = compute_cardinality_distribution(
                        tables[fk.parent_table],
                        df,
                        fk,
                    )

        self._is_fitted = True

        if self.verbose:
            fitting_time = time.time() - start_time
            logger.info(f"Multi-table fitting completed in {fitting_time:.2f}s")

        return self

    def _generate_impl(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> pd.DataFrame:
        """Not used - use generate_tables instead."""
        raise NotImplementedError("Use generate_tables() for multi-table generation")

    def generate_tables(
        self,
        n_samples: Optional[Dict[str, int]] = None,
        scale_factor: float = 1.0,
    ) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data for all tables.

        Args:
            n_samples: Number of samples per table (optional)
            scale_factor: Scale factor relative to original sizes

        Returns:
            Dictionary of table names to synthetic DataFrames
        """
        if not self._is_fitted or self._schema is None:
            raise RuntimeError("Generator not fitted. Call fit_tables() first.")

        synthetic_tables: Dict[str, pd.DataFrame] = {}

        # Determine sample sizes
        if n_samples is None:
            n_samples = {}
            for name, df in self._original_tables.items():
                n_samples[name] = int(len(df) * scale_factor)

        # Generate in topological order
        order = self._schema.get_topological_order()

        for table_name in order:
            if table_name not in self._table_generators:
                continue

            n = n_samples.get(table_name, 100)

            if self.verbose:
                logger.info(f"Generating {n} rows for table '{table_name}'")

            # Generate base synthetic data
            gen = self._table_generators[table_name]
            synthetic_df = gen.generate(n)

            # Handle foreign keys
            table_schema = self._schema.tables[table_name]
            for fk in table_schema.foreign_keys:
                if fk.parent_table in synthetic_tables:
                    synthetic_df = self._enforce_fk(
                        synthetic_df,
                        synthetic_tables[fk.parent_table],
                        fk,
                    )

            synthetic_tables[table_name] = synthetic_df

        return synthetic_tables

    def _enforce_fk(
        self,
        child_df: pd.DataFrame,
        parent_df: pd.DataFrame,
        fk: ForeignKey,
    ) -> pd.DataFrame:
        """Enforce foreign key constraint."""
        if fk.child_column not in child_df.columns:
            return child_df

        if fk.parent_column not in parent_df.columns:
            return child_df

        # Get valid parent values
        parent_values = parent_df[fk.parent_column].dropna().unique()

        if len(parent_values) == 0:
            return child_df

        # Sample parent values for child records
        result = child_df.copy()
        n_children = len(result)

        # Use cardinality distribution if available
        key = f"{fk.child_table}.{fk.child_column}->{fk.parent_table}.{fk.parent_column}"
        if key in self._cardinality_distributions:
            cardinality = self._cardinality_distributions[key]
            # Sample child counts per parent
            child_counts = sample_child_counts(cardinality, len(parent_values))

            # Distribute children to parents
            parent_assignments = []
            for parent_val, count in zip(parent_values, child_counts):
                parent_assignments.extend([parent_val] * count)

            # Pad or truncate
            if len(parent_assignments) < n_children:
                extra = np.random.choice(parent_values, n_children - len(parent_assignments))
                parent_assignments.extend(extra)
            elif len(parent_assignments) > n_children:
                parent_assignments = parent_assignments[:n_children]

            np.random.shuffle(parent_assignments)
            result[fk.child_column] = parent_assignments
        else:
            # Simple random assignment
            result[fk.child_column] = np.random.choice(parent_values, n_children)

        return result
