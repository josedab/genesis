"""Multi-table relational schema handling."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


@dataclass
class ForeignKey:
    """Represents a foreign key relationship."""

    child_table: str
    child_column: str
    parent_table: str
    parent_column: str


@dataclass
class TableSchema:
    """Schema for a single table."""

    name: str
    columns: List[str]
    primary_key: Optional[str] = None
    foreign_keys: List[ForeignKey] = field(default_factory=list)


@dataclass
class RelationalSchema:
    """Schema for a relational database."""

    tables: Dict[str, TableSchema] = field(default_factory=dict)

    def add_table(
        self,
        name: str,
        columns: List[str],
        primary_key: Optional[str] = None,
    ) -> None:
        """Add a table to the schema."""
        self.tables[name] = TableSchema(
            name=name,
            columns=columns,
            primary_key=primary_key,
        )

    def add_foreign_key(
        self,
        child_table: str,
        child_column: str,
        parent_table: str,
        parent_column: str,
    ) -> None:
        """Add a foreign key relationship."""
        if child_table not in self.tables:
            raise ValueError(f"Table '{child_table}' not found")
        if parent_table not in self.tables:
            raise ValueError(f"Table '{parent_table}' not found")

        fk = ForeignKey(child_table, child_column, parent_table, parent_column)
        self.tables[child_table].foreign_keys.append(fk)

    def get_parent_tables(self, table_name: str) -> List[str]:
        """Get parent tables for a given table."""
        if table_name not in self.tables:
            return []

        parents = []
        for fk in self.tables[table_name].foreign_keys:
            if fk.parent_table not in parents:
                parents.append(fk.parent_table)
        return parents

    def get_child_tables(self, table_name: str) -> List[str]:
        """Get child tables for a given table."""
        children = []
        for table in self.tables.values():
            for fk in table.foreign_keys:
                if fk.parent_table == table_name and table.name not in children:
                    children.append(table.name)
        return children

    def get_topological_order(self) -> List[str]:
        """Get tables in topological order (parents before children)."""
        visited: Set[str] = set()
        order: List[str] = []

        def visit(table_name: str) -> None:
            if table_name in visited:
                return
            visited.add(table_name)

            # Visit parents first
            for parent in self.get_parent_tables(table_name):
                visit(parent)

            order.append(table_name)

        for table_name in self.tables:
            visit(table_name)

        return order

    def to_dict(self) -> Dict[str, Any]:
        """Convert schema to dictionary."""
        return {
            "tables": {
                name: {
                    "columns": table.columns,
                    "primary_key": table.primary_key,
                    "foreign_keys": [
                        {
                            "child_column": fk.child_column,
                            "parent_table": fk.parent_table,
                            "parent_column": fk.parent_column,
                        }
                        for fk in table.foreign_keys
                    ],
                }
                for name, table in self.tables.items()
            }
        }

    @classmethod
    def from_dataframes(
        cls,
        tables: Dict[str, pd.DataFrame],
        foreign_keys: Optional[List[Dict[str, str]]] = None,
    ) -> "RelationalSchema":
        """Infer schema from DataFrames.

        Args:
            tables: Dictionary of table names to DataFrames
            foreign_keys: List of FK definitions with keys:
                - child_table, child_column, parent_table, parent_column

        Returns:
            RelationalSchema instance
        """
        schema = cls()

        for name, df in tables.items():
            # Detect primary key
            pk = None
            for col in df.columns:
                if df[col].nunique() == len(df) and df[col].notna().all():
                    pk = col
                    break

            schema.add_table(name, list(df.columns), pk)

        # Add foreign keys
        if foreign_keys:
            for fk_def in foreign_keys:
                schema.add_foreign_key(
                    fk_def["child_table"],
                    fk_def["child_column"],
                    fk_def["parent_table"],
                    fk_def["parent_column"],
                )

        return schema

    @classmethod
    def discover(
        cls,
        tables: Dict[str, pd.DataFrame],
        detect_foreign_keys: bool = True,
        fk_threshold: float = 0.95,
    ) -> "RelationalSchema":
        """Auto-discover schema including foreign keys from DataFrames.

        Uses heuristics to detect primary keys and foreign key relationships
        based on column names, data types, and value overlap.

        Args:
            tables: Dictionary of table names to DataFrames
            detect_foreign_keys: Whether to auto-detect FKs
            fk_threshold: Minimum overlap ratio for FK detection

        Returns:
            RelationalSchema with discovered relationships
        """
        schema = cls()
        pk_columns: Dict[str, Tuple[str, Set]] = {}  # table -> (pk_col, pk_values)

        # First pass: detect primary keys and add tables
        for name, df in tables.items():
            pk = cls._detect_primary_key(df, name)
            schema.add_table(name, list(df.columns), pk)

            if pk:
                pk_columns[name] = (pk, set(df[pk].dropna().unique()))

        # Second pass: detect foreign keys
        if detect_foreign_keys:
            detected_fks = cls._detect_foreign_keys(tables, pk_columns, fk_threshold)
            for fk_def in detected_fks:
                try:
                    schema.add_foreign_key(**fk_def)
                except ValueError:
                    pass  # Skip invalid FKs

        return schema

    @classmethod
    def _detect_primary_key(cls, df: pd.DataFrame, table_name: str) -> Optional[str]:
        """Detect primary key column using heuristics."""
        candidates = []

        for col in df.columns:
            col_lower = col.lower()

            # Check uniqueness and non-null
            is_unique = df[col].nunique() == len(df)
            is_complete = df[col].notna().all()

            if not (is_unique and is_complete):
                continue

            # Score based on naming conventions
            score = 0
            if col_lower == "id":
                score = 100
            elif col_lower == f"{table_name.rstrip('s')}_id":
                score = 90
            elif col_lower.endswith("_id") or col_lower.endswith("id"):
                score = 80
            elif "key" in col_lower or "pk" in col_lower:
                score = 70
            else:
                score = 50  # Unique but no naming hint

            candidates.append((col, score))

        if candidates:
            candidates.sort(key=lambda x: -x[1])
            return candidates[0][0]

        return None

    @classmethod
    def _detect_foreign_keys(
        cls,
        tables: Dict[str, pd.DataFrame],
        pk_columns: Dict[str, Tuple[str, Set]],
        threshold: float,
    ) -> List[Dict[str, str]]:
        """Detect foreign key relationships."""
        detected = []

        for child_name, child_df in tables.items():
            for col in child_df.columns:
                col_lower = col.lower()

                # Skip if this is the table's own PK
                if child_name in pk_columns and pk_columns[child_name][0] == col:
                    continue

                # Check naming patterns that suggest FK
                potential_parents = []

                # Pattern: table_id -> table
                if col_lower.endswith("_id"):
                    base_name = col_lower[:-3]
                    for parent_name in tables.keys():
                        parent_lower = parent_name.lower()
                        if (
                            base_name == parent_lower
                            or base_name == parent_lower.rstrip("s")
                            or base_name + "s" == parent_lower
                        ):
                            potential_parents.append(parent_name)

                # Pattern: exact match to another table's column name
                for parent_name, (pk_col, pk_values) in pk_columns.items():
                    if parent_name == child_name:
                        continue

                    # Check value overlap
                    child_values = set(child_df[col].dropna().unique())
                    if not child_values:
                        continue

                    overlap = len(child_values & pk_values) / len(child_values)

                    if overlap >= threshold:
                        detected.append(
                            {
                                "child_table": child_name,
                                "child_column": col,
                                "parent_table": parent_name,
                                "parent_column": pk_col,
                            }
                        )
                        break  # Only one FK per column

        return detected
