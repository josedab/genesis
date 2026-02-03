"""Schema definitions for test data generation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


class DataType(Enum):
    """Supported data types for test generation."""

    # Basic types
    STRING = "string"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    DATE = "date"
    DATETIME = "datetime"
    TIME = "time"

    # Semantic types
    EMAIL = "email"
    PHONE = "phone"
    NAME = "name"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    ADDRESS = "address"
    CITY = "city"
    COUNTRY = "country"
    ZIP_CODE = "zip_code"
    UUID = "uuid"
    URL = "url"
    IP_ADDRESS = "ip_address"

    # Financial
    CREDIT_CARD = "credit_card"
    CURRENCY = "currency"
    PRICE = "price"
    ACCOUNT_NUMBER = "account_number"

    # Identifiers
    SSN = "ssn"
    USERNAME = "username"
    PASSWORD = "password"

    # Text
    TEXT = "text"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"
    WORD = "word"

    # Categorical
    CATEGORY = "category"
    ENUM = "enum"

    # Special
    FOREIGN_KEY = "foreign_key"
    PRIMARY_KEY = "primary_key"
    SEQUENCE = "sequence"
    CONSTANT = "constant"
    CUSTOM = "custom"


@dataclass
class ColumnSpec:
    """Specification for a single column.

    Attributes:
        name: Column name
        dtype: Data type
        nullable: Whether nulls are allowed
        unique: Whether values must be unique
        min_value: Minimum value (for numeric types)
        max_value: Maximum value (for numeric types)
        min_length: Minimum length (for string types)
        max_length: Maximum length (for string types)
        pattern: Regex pattern (for string types)
        choices: List of allowed values (for categorical types)
        distribution: Distribution type ('uniform', 'normal', 'exponential')
        null_probability: Probability of null values
        custom_generator: Custom generation function
        foreign_key_ref: Reference for foreign key (table.column)
    """

    name: str
    dtype: DataType
    nullable: bool = False
    unique: bool = False
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    choices: Optional[List[Any]] = None
    distribution: str = "uniform"
    null_probability: float = 0.0
    custom_generator: Optional[Callable[[], Any]] = None
    foreign_key_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "dtype": self.dtype.value,
            "nullable": self.nullable,
            "unique": self.unique,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "pattern": self.pattern,
            "choices": self.choices,
            "distribution": self.distribution,
            "null_probability": self.null_probability,
            "foreign_key_ref": self.foreign_key_ref,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ColumnSpec":
        """Create from dictionary."""
        dtype = data.get("dtype", "string")
        if isinstance(dtype, str):
            dtype = DataType(dtype)

        return cls(
            name=data["name"],
            dtype=dtype,
            nullable=data.get("nullable", False),
            unique=data.get("unique", False),
            min_value=data.get("min_value"),
            max_value=data.get("max_value"),
            min_length=data.get("min_length"),
            max_length=data.get("max_length"),
            pattern=data.get("pattern"),
            choices=data.get("choices"),
            distribution=data.get("distribution", "uniform"),
            null_probability=data.get("null_probability", 0.0),
            foreign_key_ref=data.get("foreign_key_ref"),
        )


def parse_column_spec(spec_string: str) -> Tuple[DataType, Dict[str, Any]]:
    """Parse a column specification string.

    Supports formats like:
    - "string" -> string type, no constraints
    - "int:0-100" -> integer from 0 to 100
    - "float:0.0-1.0" -> float from 0.0 to 1.0
    - "string:5-20" -> string with length 5-20
    - "enum:a,b,c" -> enum with choices a, b, c
    - "email" -> email type
    - "string?" -> nullable string
    - "int!:0-100" -> unique integer 0-100

    Args:
        spec_string: Specification string

    Returns:
        Tuple of (DataType, constraint_dict)

    Example:
        >>> dtype, constraints = parse_column_spec("int:18-65")
        >>> dtype == DataType.INT
        True
        >>> constraints == {"min_value": 18, "max_value": 65}
        True
    """
    constraints: Dict[str, Any] = {}

    # Check for nullable marker
    if spec_string.endswith("?"):
        constraints["nullable"] = True
        spec_string = spec_string[:-1]

    # Check for unique marker
    if "!" in spec_string:
        constraints["unique"] = True
        spec_string = spec_string.replace("!", "")

    # Split type and constraints
    parts = spec_string.split(":", 1)
    type_str = parts[0].lower().strip()

    # Parse type
    try:
        dtype = DataType(type_str)
    except ValueError:
        # Default to string for unknown types
        dtype = DataType.STRING

    # Parse constraints if present
    if len(parts) > 1:
        constraint_str = parts[1].strip()

        # Range format: min-max
        range_match = re.match(r"(-?\d+\.?\d*)-(-?\d+\.?\d*)", constraint_str)
        if range_match:
            if dtype in (DataType.INT, DataType.SEQUENCE, DataType.PRIMARY_KEY):
                constraints["min_value"] = int(range_match.group(1))
                constraints["max_value"] = int(range_match.group(2))
            elif dtype in (DataType.FLOAT, DataType.PRICE, DataType.CURRENCY):
                constraints["min_value"] = float(range_match.group(1))
                constraints["max_value"] = float(range_match.group(2))
            elif dtype == DataType.STRING:
                constraints["min_length"] = int(range_match.group(1))
                constraints["max_length"] = int(range_match.group(2))

        # Enum format: choice1,choice2,choice3
        elif "," in constraint_str:
            constraints["choices"] = [c.strip() for c in constraint_str.split(",")]
            if dtype == DataType.STRING:
                dtype = DataType.ENUM

        # Single value (max length for strings, max value for numbers)
        elif constraint_str.isdigit():
            if dtype == DataType.STRING:
                constraints["max_length"] = int(constraint_str)
            else:
                constraints["max_value"] = int(constraint_str)

    return dtype, constraints


@dataclass
class SchemaDefinition:
    """Definition of a test data schema.

    Example:
        >>> schema = SchemaDefinition(
        ...     name="customers",
        ...     columns=[
        ...         ColumnSpec("id", DataType.PRIMARY_KEY),
        ...         ColumnSpec("name", DataType.NAME),
        ...         ColumnSpec("email", DataType.EMAIL, unique=True),
        ...         ColumnSpec("age", DataType.INT, min_value=18, max_value=100),
        ...     ]
        ... )
    """

    name: str
    columns: List[ColumnSpec] = field(default_factory=list)
    description: Optional[str] = None
    primary_key: Optional[str] = None
    foreign_keys: Dict[str, str] = field(default_factory=dict)

    def add_column(
        self,
        name: str,
        dtype: Union[str, DataType],
        **kwargs: Any,
    ) -> "SchemaDefinition":
        """Add a column to the schema.

        Args:
            name: Column name
            dtype: Data type (string or DataType enum)
            **kwargs: Additional column specifications

        Returns:
            Self for chaining
        """
        if isinstance(dtype, str):
            dtype_enum, constraints = parse_column_spec(dtype)
            kwargs = {**constraints, **kwargs}
        else:
            dtype_enum = dtype

        spec = ColumnSpec(name=name, dtype=dtype_enum, **kwargs)
        self.columns.append(spec)
        return self

    def get_column(self, name: str) -> Optional[ColumnSpec]:
        """Get a column by name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SchemaDefinition":
        """Create schema from dictionary specification.

        Supports shorthand format:
        {
            "name": {"dtype": "string", "min_length": 2},
            "age": "int:18-100",
            "email": "email!",  # unique
        }
        """
        name = data.get("_name", data.get("name", "schema"))
        description = data.get("_description", data.get("description"))

        columns = []
        for col_name, col_spec in data.items():
            if col_name.startswith("_"):
                continue

            if isinstance(col_spec, str):
                dtype, constraints = parse_column_spec(col_spec)
                columns.append(ColumnSpec(name=col_name, dtype=dtype, **constraints))
            elif isinstance(col_spec, dict):
                dtype_str = col_spec.get("dtype", "string")
                if isinstance(dtype_str, str):
                    dtype = DataType(dtype_str)
                else:
                    dtype = dtype_str
                columns.append(
                    ColumnSpec(
                        name=col_name,
                        dtype=dtype,
                        nullable=col_spec.get("nullable", False),
                        unique=col_spec.get("unique", False),
                        min_value=col_spec.get("min_value"),
                        max_value=col_spec.get("max_value"),
                        min_length=col_spec.get("min_length"),
                        max_length=col_spec.get("max_length"),
                        pattern=col_spec.get("pattern"),
                        choices=col_spec.get("choices"),
                    )
                )

        return cls(name=name, columns=columns, description=description)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "columns": [col.to_dict() for col in self.columns],
            "primary_key": self.primary_key,
            "foreign_keys": self.foreign_keys,
        }


# Common schema presets
class SchemaPresets:
    """Pre-defined schemas for common data patterns."""

    @staticmethod
    def user() -> SchemaDefinition:
        """Standard user schema."""
        return SchemaDefinition(
            name="users",
            columns=[
                ColumnSpec("id", DataType.PRIMARY_KEY),
                ColumnSpec("username", DataType.USERNAME, unique=True, min_length=3, max_length=20),
                ColumnSpec("email", DataType.EMAIL, unique=True),
                ColumnSpec("first_name", DataType.FIRST_NAME),
                ColumnSpec("last_name", DataType.LAST_NAME),
                ColumnSpec("created_at", DataType.DATETIME),
                ColumnSpec("is_active", DataType.BOOL),
            ],
            primary_key="id",
        )

    @staticmethod
    def customer() -> SchemaDefinition:
        """Customer schema for e-commerce."""
        return SchemaDefinition(
            name="customers",
            columns=[
                ColumnSpec("customer_id", DataType.PRIMARY_KEY),
                ColumnSpec("name", DataType.NAME),
                ColumnSpec("email", DataType.EMAIL, unique=True),
                ColumnSpec("phone", DataType.PHONE),
                ColumnSpec("address", DataType.ADDRESS),
                ColumnSpec("city", DataType.CITY),
                ColumnSpec("country", DataType.COUNTRY),
                ColumnSpec("zip_code", DataType.ZIP_CODE),
                ColumnSpec("created_at", DataType.DATETIME),
            ],
            primary_key="customer_id",
        )

    @staticmethod
    def order() -> SchemaDefinition:
        """Order schema for e-commerce."""
        return SchemaDefinition(
            name="orders",
            columns=[
                ColumnSpec("order_id", DataType.PRIMARY_KEY),
                ColumnSpec("customer_id", DataType.FOREIGN_KEY, foreign_key_ref="customers.customer_id"),
                ColumnSpec("order_date", DataType.DATETIME),
                ColumnSpec("total_amount", DataType.PRICE, min_value=0),
                ColumnSpec("status", DataType.ENUM, choices=["pending", "shipped", "delivered", "cancelled"]),
            ],
            primary_key="order_id",
            foreign_keys={"customer_id": "customers.customer_id"},
        )

    @staticmethod
    def transaction() -> SchemaDefinition:
        """Financial transaction schema."""
        return SchemaDefinition(
            name="transactions",
            columns=[
                ColumnSpec("transaction_id", DataType.UUID),
                ColumnSpec("account_id", DataType.ACCOUNT_NUMBER),
                ColumnSpec("amount", DataType.CURRENCY, min_value=-10000, max_value=10000),
                ColumnSpec("transaction_type", DataType.ENUM, choices=["debit", "credit", "transfer"]),
                ColumnSpec("timestamp", DataType.DATETIME),
                ColumnSpec("description", DataType.SENTENCE),
            ],
            primary_key="transaction_id",
        )

    @staticmethod
    def patient() -> SchemaDefinition:
        """Healthcare patient schema."""
        return SchemaDefinition(
            name="patients",
            columns=[
                ColumnSpec("patient_id", DataType.PRIMARY_KEY),
                ColumnSpec("first_name", DataType.FIRST_NAME),
                ColumnSpec("last_name", DataType.LAST_NAME),
                ColumnSpec("date_of_birth", DataType.DATE),
                ColumnSpec("gender", DataType.ENUM, choices=["M", "F", "O"]),
                ColumnSpec("blood_type", DataType.ENUM, choices=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]),
                ColumnSpec("email", DataType.EMAIL),
                ColumnSpec("phone", DataType.PHONE),
            ],
            primary_key="patient_id",
        )
