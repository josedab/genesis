"""Visual schema editor backend and schema representation.

This module provides the backend for a visual schema editor:
- Schema definition and serialization
- REST API endpoints for schema management
- Export to Genesis configuration

Example:
    >>> from genesis.schema_editor import SchemaDefinition, ColumnDefinition
    >>>
    >>> schema = SchemaDefinition("customers")
    >>> schema.add_column(ColumnDefinition(
    ...     name="age",
    ...     dtype="integer",
    ...     constraints={"min": 0, "max": 120},
    ... ))
    >>>
    >>> config = schema.to_generator_config()
"""

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class ColumnDataType(Enum):
    """Supported column data types."""

    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"
    CATEGORY = "category"
    TEXT = "text"
    EMAIL = "email"
    PHONE = "phone"
    UUID = "uuid"


class ConstraintType(Enum):
    """Types of constraints."""

    RANGE = "range"
    POSITIVE = "positive"
    UNIQUE = "unique"
    NOT_NULL = "not_null"
    REGEX = "regex"
    IN_LIST = "in_list"
    FOREIGN_KEY = "foreign_key"
    CUSTOM = "custom"


@dataclass
class ConstraintDefinition:
    """Definition of a column constraint."""

    constraint_type: ConstraintType
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.constraint_type.value,
            "parameters": self.parameters,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintDefinition":
        return cls(
            constraint_type=ConstraintType(data["type"]),
            parameters=data.get("parameters", {}),
        )

    def to_genesis_constraint(self) -> Any:
        """Convert to Genesis constraint object."""
        from genesis.core.constraints import (
            PositiveConstraint,
            RangeConstraint,
            UniqueConstraint,
        )

        if self.constraint_type == ConstraintType.POSITIVE:
            return PositiveConstraint(self.parameters.get("column", ""))
        elif self.constraint_type == ConstraintType.RANGE:
            return RangeConstraint(
                column=self.parameters.get("column", ""),
                min_value=self.parameters.get("min"),
                max_value=self.parameters.get("max"),
            )
        elif self.constraint_type == ConstraintType.UNIQUE:
            return UniqueConstraint(self.parameters.get("column", ""))

        return None


@dataclass
class ColumnDefinition:
    """Definition of a column in the schema."""

    name: str
    dtype: ColumnDataType
    nullable: bool = True
    unique: bool = False
    description: str = ""
    constraints: List[ConstraintDefinition] = field(default_factory=list)
    categories: Optional[List[Any]] = None
    distribution: Optional[str] = None
    distribution_params: Dict[str, Any] = field(default_factory=dict)

    # For foreign keys
    is_foreign_key: bool = False
    references_table: Optional[str] = None
    references_column: Optional[str] = None

    # Visual editor metadata
    position_x: int = 0
    position_y: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype.value,
            "nullable": self.nullable,
            "unique": self.unique,
            "description": self.description,
            "constraints": [c.to_dict() for c in self.constraints],
            "categories": self.categories,
            "distribution": self.distribution,
            "distribution_params": self.distribution_params,
            "is_foreign_key": self.is_foreign_key,
            "references_table": self.references_table,
            "references_column": self.references_column,
            "position_x": self.position_x,
            "position_y": self.position_y,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnDefinition":
        constraints = [ConstraintDefinition.from_dict(c) for c in data.get("constraints", [])]
        return cls(
            name=data["name"],
            dtype=ColumnDataType(data["dtype"]),
            nullable=data.get("nullable", True),
            unique=data.get("unique", False),
            description=data.get("description", ""),
            constraints=constraints,
            categories=data.get("categories"),
            distribution=data.get("distribution"),
            distribution_params=data.get("distribution_params", {}),
            is_foreign_key=data.get("is_foreign_key", False),
            references_table=data.get("references_table"),
            references_column=data.get("references_column"),
            position_x=data.get("position_x", 0),
            position_y=data.get("position_y", 0),
        )


@dataclass
class RelationshipDefinition:
    """Definition of a relationship between tables."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str
    relationship_type: str = "many-to-one"  # one-to-one, one-to-many, many-to-one

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_table": self.source_table,
            "source_column": self.source_column,
            "target_table": self.target_table,
            "target_column": self.target_column,
            "relationship_type": self.relationship_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RelationshipDefinition":
        return cls(
            source_table=data["source_table"],
            source_column=data["source_column"],
            target_table=data["target_table"],
            target_column=data["target_column"],
            relationship_type=data.get("relationship_type", "many-to-one"),
        )


@dataclass
class SchemaDefinition:
    """Complete schema definition for a table."""

    name: str
    columns: List[ColumnDefinition] = field(default_factory=list)
    description: str = ""
    primary_key: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    # Visual editor metadata
    position_x: int = 0
    position_y: int = 0
    color: str = "#3498db"

    def add_column(self, column: ColumnDefinition) -> None:
        """Add a column to the schema."""
        self.columns.append(column)

    def remove_column(self, name: str) -> bool:
        """Remove a column by name."""
        for i, col in enumerate(self.columns):
            if col.name == name:
                del self.columns[i]
                return True
        return False

    def get_column(self, name: str) -> Optional[ColumnDefinition]:
        """Get a column by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def get_discrete_columns(self) -> List[str]:
        """Get list of discrete/categorical columns."""
        discrete = []
        for col in self.columns:
            if col.dtype in (ColumnDataType.CATEGORY, ColumnDataType.BOOLEAN):
                discrete.append(col.name)
            elif col.categories is not None:
                discrete.append(col.name)
        return discrete

    def get_constraints(self) -> List[Any]:
        """Get all constraints as Genesis constraint objects."""
        constraints = []
        for col in self.columns:
            for c in col.constraints:
                c.parameters["column"] = col.name
                genesis_constraint = c.to_genesis_constraint()
                if genesis_constraint:
                    constraints.append(genesis_constraint)
        return constraints

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "columns": [c.to_dict() for c in self.columns],
            "description": self.description,
            "primary_key": self.primary_key,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "position_x": self.position_x,
            "position_y": self.position_y,
            "color": self.color,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaDefinition":
        columns = [ColumnDefinition.from_dict(c) for c in data.get("columns", [])]
        return cls(
            name=data["name"],
            columns=columns,
            description=data.get("description", ""),
            primary_key=data.get("primary_key"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            position_x=data.get("position_x", 0),
            position_y=data.get("position_y", 0),
            color=data.get("color", "#3498db"),
        )

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        """Save schema to file."""
        path = Path(path)
        path.write_text(self.to_json())
        logger.info(f"Schema saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "SchemaDefinition":
        """Load schema from file."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str = "inferred_schema",
    ) -> "SchemaDefinition":
        """Infer schema from a DataFrame."""
        schema = cls(name=name)

        for col_name in df.columns:
            col_data = df[col_name]

            # Infer dtype
            if pd.api.types.is_integer_dtype(col_data):
                dtype = ColumnDataType.INTEGER
            elif pd.api.types.is_float_dtype(col_data):
                dtype = ColumnDataType.FLOAT
            elif pd.api.types.is_bool_dtype(col_data):
                dtype = ColumnDataType.BOOLEAN
            elif pd.api.types.is_datetime64_any_dtype(col_data):
                dtype = ColumnDataType.DATETIME
            elif col_data.nunique() < 20:
                dtype = ColumnDataType.CATEGORY
            else:
                dtype = ColumnDataType.STRING

            # Get categories for categorical columns
            categories = None
            if dtype == ColumnDataType.CATEGORY:
                categories = list(col_data.unique())

            # Create column definition
            column = ColumnDefinition(
                name=col_name,
                dtype=dtype,
                nullable=col_data.isna().any(),
                unique=col_data.nunique() == len(col_data),
                categories=categories,
            )

            # Add range constraint for numeric columns
            if dtype in (ColumnDataType.INTEGER, ColumnDataType.FLOAT):
                column.constraints.append(
                    ConstraintDefinition(
                        constraint_type=ConstraintType.RANGE,
                        parameters={
                            "min": float(col_data.min()),
                            "max": float(col_data.max()),
                        },
                    )
                )

            schema.add_column(column)

        return schema

    def to_generator_config(self) -> Dict[str, Any]:
        """Export to Genesis generator configuration."""
        return {
            "discrete_columns": self.get_discrete_columns(),
            "constraints": self.get_constraints(),
            "schema": self.to_dict(),
        }

    def to_python_code(self) -> str:
        """Export as Python code for Genesis."""
        lines = [
            "from genesis import SyntheticGenerator, Constraint",
            "",
            "# Define constraints",
            "constraints = [",
        ]

        for col in self.columns:
            for c in col.constraints:
                if c.constraint_type == ConstraintType.RANGE:
                    lines.append(
                        f"    Constraint.range('{col.name}', "
                        f"{c.parameters.get('min')}, {c.parameters.get('max')}),"
                    )
                elif c.constraint_type == ConstraintType.POSITIVE:
                    lines.append(f"    Constraint.positive('{col.name}'),")
                elif c.constraint_type == ConstraintType.UNIQUE:
                    lines.append(f"    Constraint.unique('{col.name}'),")

        lines.append("]")
        lines.append("")
        lines.append("# Discrete columns")
        lines.append(f"discrete_columns = {self.get_discrete_columns()}")
        lines.append("")
        lines.append("# Create generator")
        lines.append("generator = SyntheticGenerator(method='auto')")
        lines.append("generator.fit(")
        lines.append("    data=your_data,")
        lines.append("    discrete_columns=discrete_columns,")
        lines.append("    constraints=constraints,")
        lines.append(")")
        lines.append("")
        lines.append("# Generate synthetic data")
        lines.append("synthetic = generator.generate(n_samples=1000)")

        return "\n".join(lines)


@dataclass
class ProjectDefinition:
    """Multi-table project definition."""

    id: str
    name: str
    tables: List[SchemaDefinition] = field(default_factory=list)
    relationships: List[RelationshipDefinition] = field(default_factory=list)
    description: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def add_table(self, table: SchemaDefinition) -> None:
        """Add a table to the project."""
        self.tables.append(table)

    def remove_table(self, name: str) -> bool:
        """Remove a table by name."""
        for i, table in enumerate(self.tables):
            if table.name == name:
                del self.tables[i]
                return True
        return False

    def get_table(self, name: str) -> Optional[SchemaDefinition]:
        """Get a table by name."""
        for table in self.tables:
            if table.name == name:
                return table
        return None

    def add_relationship(self, relationship: RelationshipDefinition) -> None:
        """Add a relationship between tables."""
        self.relationships.append(relationship)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "tables": [t.to_dict() for t in self.tables],
            "relationships": [r.to_dict() for r in self.relationships],
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectDefinition":
        tables = [SchemaDefinition.from_dict(t) for t in data.get("tables", [])]
        relationships = [RelationshipDefinition.from_dict(r) for r in data.get("relationships", [])]
        return cls(
            id=data["id"],
            name=data["name"],
            tables=tables,
            relationships=relationships,
            description=data.get("description", ""),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
        )

    def to_json(self, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        """Save project to file."""
        path = Path(path)
        path.write_text(self.to_json())
        logger.info(f"Project saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ProjectDefinition":
        """Load project from file."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_dict(data)


class SchemaEditorAPI:
    """Backend API for visual schema editor.

    This can be used to build REST endpoints or as a direct Python API.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize API.

        Args:
            storage_path: Path for storing projects (default: memory-only)
        """
        self.storage_path = storage_path
        self._projects: Dict[str, ProjectDefinition] = {}

    def create_project(
        self,
        name: str,
        description: str = "",
    ) -> ProjectDefinition:
        """Create a new project."""
        from datetime import datetime

        project_id = str(uuid.uuid4())[:8]
        project = ProjectDefinition(
            id=project_id,
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
        )

        self._projects[project_id] = project
        logger.info(f"Created project: {name} ({project_id})")
        return project

    def get_project(self, project_id: str) -> Optional[ProjectDefinition]:
        """Get project by ID."""
        return self._projects.get(project_id)

    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects."""
        return [
            {"id": p.id, "name": p.name, "n_tables": len(p.tables)} for p in self._projects.values()
        ]

    def delete_project(self, project_id: str) -> bool:
        """Delete a project."""
        if project_id in self._projects:
            del self._projects[project_id]
            return True
        return False

    def add_table(
        self,
        project_id: str,
        table: SchemaDefinition,
    ) -> bool:
        """Add table to project."""
        project = self._projects.get(project_id)
        if project:
            project.add_table(table)
            return True
        return False

    def update_table(
        self,
        project_id: str,
        table_name: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update table definition."""
        project = self._projects.get(project_id)
        if project:
            table = project.get_table(table_name)
            if table:
                for key, value in updates.items():
                    if hasattr(table, key):
                        setattr(table, key, value)
                return True
        return False

    def add_column(
        self,
        project_id: str,
        table_name: str,
        column: ColumnDefinition,
    ) -> bool:
        """Add column to table."""
        project = self._projects.get(project_id)
        if project:
            table = project.get_table(table_name)
            if table:
                table.add_column(column)
                return True
        return False

    def add_relationship(
        self,
        project_id: str,
        source_table: str,
        source_column: str,
        target_table: str,
        target_column: str,
    ) -> bool:
        """Add relationship between tables."""
        project = self._projects.get(project_id)
        if project:
            relationship = RelationshipDefinition(
                source_table=source_table,
                source_column=source_column,
                target_table=target_table,
                target_column=target_column,
            )
            project.add_relationship(relationship)
            return True
        return False

    def infer_schema(
        self,
        project_id: str,
        table_name: str,
        data: pd.DataFrame,
    ) -> Optional[SchemaDefinition]:
        """Infer schema from uploaded data."""
        schema = SchemaDefinition.from_dataframe(data, table_name)

        project = self._projects.get(project_id)
        if project:
            project.add_table(schema)

        return schema

    def generate_preview(
        self,
        project_id: str,
        table_name: str,
        n_samples: int = 100,
    ) -> Optional[pd.DataFrame]:
        """Generate preview synthetic data."""
        project = self._projects.get(project_id)
        if not project:
            return None

        table = project.get_table(table_name)
        if not table:
            return None

        # Simple preview generation based on schema
        data = {}
        import numpy as np

        for col in table.columns:
            if col.dtype == ColumnDataType.INTEGER:
                params = col.distribution_params
                data[col.name] = np.random.randint(
                    params.get("min", 0),
                    params.get("max", 100),
                    n_samples,
                )
            elif col.dtype == ColumnDataType.FLOAT:
                params = col.distribution_params
                data[col.name] = np.random.normal(
                    params.get("mean", 0),
                    params.get("std", 1),
                    n_samples,
                )
            elif col.dtype == ColumnDataType.CATEGORY and col.categories:
                data[col.name] = np.random.choice(col.categories, n_samples)
            elif col.dtype == ColumnDataType.BOOLEAN:
                data[col.name] = np.random.choice([True, False], n_samples)
            else:
                data[col.name] = [f"{col.name}_{i}" for i in range(n_samples)]

        return pd.DataFrame(data)

    def export_python(self, project_id: str) -> str:
        """Export project as Python code."""
        project = self._projects.get(project_id)
        if not project:
            return ""

        lines = [f"# Generated from project: {project.name}", ""]

        for table in project.tables:
            lines.append(f"# Table: {table.name}")
            lines.append(table.to_python_code())
            lines.append("")

        return "\n".join(lines)

    def export_yaml(self, project_id: str) -> str:
        """Export project as YAML configuration."""
        import yaml

        project = self._projects.get(project_id)
        if not project:
            return ""

        return yaml.dump(project.to_dict(), default_flow_style=False)


__all__ = [
    # Schema definitions
    "SchemaDefinition",
    "ColumnDefinition",
    "ConstraintDefinition",
    "RelationshipDefinition",
    "ProjectDefinition",
    # Types
    "ColumnDataType",
    "ConstraintType",
    # API
    "SchemaEditorAPI",
]
