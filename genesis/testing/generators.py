"""Test data generators based on column specifications."""

from __future__ import annotations

import hashlib
import random
import string
import uuid
from datetime import date, datetime, time, timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from genesis.testing.schemas import ColumnSpec, DataType, SchemaDefinition
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class ColumnGenerator:
    """Generator for individual column values based on specifications.

    This class generates realistic test data for a single column
    based on its type and constraints.
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self._random = random.Random(seed)

        # Pre-defined data pools
        self._first_names = [
            "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
            "Linda", "William", "Elizabeth", "David", "Barbara", "Richard", "Susan",
            "Joseph", "Jessica", "Thomas", "Sarah", "Charles", "Karen", "Emma",
            "Olivia", "Ava", "Isabella", "Sophia", "Mia", "Charlotte", "Amelia",
            "Liam", "Noah", "Oliver", "Elijah", "Lucas", "Mason", "Logan", "Alexander",
        ]
        self._last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
            "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
        ]
        self._cities = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
            "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
            "Fort Worth", "Columbus", "Charlotte", "Seattle", "Denver", "Boston",
            "London", "Paris", "Tokyo", "Berlin", "Sydney", "Toronto", "Mumbai",
        ]
        self._countries = [
            "United States", "United Kingdom", "Canada", "Germany", "France",
            "Japan", "Australia", "India", "Brazil", "Mexico", "Italy", "Spain",
        ]
        self._domains = ["gmail.com", "yahoo.com", "outlook.com", "example.com", "company.org"]
        self._street_types = ["Street", "Avenue", "Road", "Boulevard", "Lane", "Drive", "Way"]
        self._words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
            "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "labore",
        ]

    def generate(
        self,
        spec: ColumnSpec,
        n_samples: int,
        existing_values: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Generate values for a column.

        Args:
            spec: Column specification
            n_samples: Number of values to generate
            existing_values: Existing values (for foreign keys)

        Returns:
            List of generated values
        """
        # Handle nullable
        null_mask = None
        if spec.nullable and spec.null_probability > 0:
            null_mask = self._rng.random(n_samples) < spec.null_probability

        # Generate based on type
        generator_method = self._get_generator_method(spec.dtype)
        values = generator_method(spec, n_samples, existing_values)

        # Apply null mask
        if null_mask is not None:
            values = [None if is_null else v for v, is_null in zip(values, null_mask)]

        # Handle uniqueness
        if spec.unique:
            values = self._ensure_unique(values, spec, n_samples)

        return values

    def _get_generator_method(self, dtype: DataType) -> Callable:
        """Get the appropriate generator method for a data type."""
        generators = {
            # Basic types
            DataType.STRING: self._generate_string,
            DataType.INT: self._generate_int,
            DataType.FLOAT: self._generate_float,
            DataType.BOOL: self._generate_bool,
            DataType.DATE: self._generate_date,
            DataType.DATETIME: self._generate_datetime,
            DataType.TIME: self._generate_time,
            # Semantic types
            DataType.EMAIL: self._generate_email,
            DataType.PHONE: self._generate_phone,
            DataType.NAME: self._generate_name,
            DataType.FIRST_NAME: self._generate_first_name,
            DataType.LAST_NAME: self._generate_last_name,
            DataType.ADDRESS: self._generate_address,
            DataType.CITY: self._generate_city,
            DataType.COUNTRY: self._generate_country,
            DataType.ZIP_CODE: self._generate_zip_code,
            DataType.UUID: self._generate_uuid,
            DataType.URL: self._generate_url,
            DataType.IP_ADDRESS: self._generate_ip_address,
            # Financial
            DataType.CREDIT_CARD: self._generate_credit_card,
            DataType.CURRENCY: self._generate_currency,
            DataType.PRICE: self._generate_price,
            DataType.ACCOUNT_NUMBER: self._generate_account_number,
            # Identifiers
            DataType.SSN: self._generate_ssn,
            DataType.USERNAME: self._generate_username,
            DataType.PASSWORD: self._generate_password,
            # Text
            DataType.TEXT: self._generate_text,
            DataType.PARAGRAPH: self._generate_paragraph,
            DataType.SENTENCE: self._generate_sentence,
            DataType.WORD: self._generate_word,
            # Categorical
            DataType.CATEGORY: self._generate_category,
            DataType.ENUM: self._generate_enum,
            # Special
            DataType.PRIMARY_KEY: self._generate_primary_key,
            DataType.SEQUENCE: self._generate_sequence,
            DataType.FOREIGN_KEY: self._generate_foreign_key,
            DataType.CONSTANT: self._generate_constant,
        }
        return generators.get(dtype, self._generate_string)

    def _ensure_unique(
        self,
        values: List[Any],
        spec: ColumnSpec,
        n_samples: int,
    ) -> List[Any]:
        """Ensure all values are unique."""
        unique_values = list(set(v for v in values if v is not None))
        null_count = sum(1 for v in values if v is None)

        # Generate more values if needed
        max_attempts = 10
        attempts = 0
        while len(unique_values) < n_samples - null_count and attempts < max_attempts:
            more = self._get_generator_method(spec.dtype)(spec, n_samples, None)
            unique_values.extend(v for v in more if v not in unique_values and v is not None)
            unique_values = list(set(unique_values))
            attempts += 1

        # Trim to required size
        unique_values = unique_values[: n_samples - null_count]

        # Reinsert nulls at random positions
        result = list(unique_values)
        for _ in range(null_count):
            pos = self._rng.integers(0, len(result) + 1)
            result.insert(pos, None)

        return result

    # Basic type generators
    def _generate_string(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        min_len = spec.min_length or 5
        max_len = spec.max_length or 20
        return [
            "".join(self._random.choices(string.ascii_letters, k=self._rng.integers(min_len, max_len + 1)))
            for _ in range(n)
        ]

    def _generate_int(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[int]:
        min_val = int(spec.min_value or 0)
        max_val = int(spec.max_value or 1000000)
        if spec.distribution == "normal":
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 6
            values = self._rng.normal(mean, std, n).astype(int)
            return [max(min_val, min(max_val, v)) for v in values]
        return self._rng.integers(min_val, max_val + 1, n).tolist()

    def _generate_float(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[float]:
        min_val = spec.min_value or 0.0
        max_val = spec.max_value or 1000.0
        if spec.distribution == "normal":
            mean = (min_val + max_val) / 2
            std = (max_val - min_val) / 6
            values = self._rng.normal(mean, std, n)
            return [round(max(min_val, min(max_val, v)), 2) for v in values]
        return [round(v, 2) for v in self._rng.uniform(min_val, max_val, n).tolist()]

    def _generate_bool(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[bool]:
        return self._rng.choice([True, False], n).tolist()

    def _generate_date(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[date]:
        start = date(1970, 1, 1)
        end = date(2030, 12, 31)
        delta = (end - start).days
        return [start + timedelta(days=int(self._rng.integers(0, delta))) for _ in range(n)]

    def _generate_datetime(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[datetime]:
        start = datetime(1970, 1, 1)
        end = datetime(2030, 12, 31)
        delta = (end - start).total_seconds()
        return [start + timedelta(seconds=int(self._rng.integers(0, int(delta)))) for _ in range(n)]

    def _generate_time(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[time]:
        return [
            time(
                int(self._rng.integers(0, 24)),
                int(self._rng.integers(0, 60)),
                int(self._rng.integers(0, 60)),
            )
            for _ in range(n)
        ]

    # Semantic type generators
    def _generate_email(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [
            f"{self._random.choice(self._first_names).lower()}.{self._random.choice(self._last_names).lower()}{self._rng.integers(1, 100)}@{self._random.choice(self._domains)}"
            for _ in range(n)
        ]

    def _generate_phone(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [
            f"+1-{self._rng.integers(200, 999)}-{self._rng.integers(200, 999)}-{self._rng.integers(1000, 9999)}"
            for _ in range(n)
        ]

    def _generate_name(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [
            f"{self._random.choice(self._first_names)} {self._random.choice(self._last_names)}"
            for _ in range(n)
        ]

    def _generate_first_name(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [self._random.choice(self._first_names) for _ in range(n)]

    def _generate_last_name(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [self._random.choice(self._last_names) for _ in range(n)]

    def _generate_address(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [
            f"{self._rng.integers(1, 9999)} {self._random.choice(self._last_names)} {self._random.choice(self._street_types)}"
            for _ in range(n)
        ]

    def _generate_city(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [self._random.choice(self._cities) for _ in range(n)]

    def _generate_country(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [self._random.choice(self._countries) for _ in range(n)]

    def _generate_zip_code(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [f"{self._rng.integers(10000, 99999)}" for _ in range(n)]

    def _generate_uuid(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [str(uuid.uuid4()) for _ in range(n)]

    def _generate_url(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [
            f"https://www.{self._random.choice(self._domains)}/{self._random.choice(self._words)}"
            for _ in range(n)
        ]

    def _generate_ip_address(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [
            f"{self._rng.integers(1, 255)}.{self._rng.integers(0, 255)}.{self._rng.integers(0, 255)}.{self._rng.integers(1, 255)}"
            for _ in range(n)
        ]

    # Financial generators
    def _generate_credit_card(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        # Generate fake credit card numbers (not valid for actual use)
        return [
            f"4{self._rng.integers(100, 999)}-{self._rng.integers(1000, 9999)}-{self._rng.integers(1000, 9999)}-{self._rng.integers(1000, 9999)}"
            for _ in range(n)
        ]

    def _generate_currency(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[float]:
        min_val = spec.min_value or 0.0
        max_val = spec.max_value or 10000.0
        return [round(v, 2) for v in self._rng.uniform(min_val, max_val, n).tolist()]

    def _generate_price(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[float]:
        min_val = spec.min_value or 0.01
        max_val = spec.max_value or 999.99
        return [round(v, 2) for v in self._rng.uniform(min_val, max_val, n).tolist()]

    def _generate_account_number(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [f"{self._rng.integers(1000000000, 9999999999)}" for _ in range(n)]

    # Identifier generators
    def _generate_ssn(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        # Generate fake SSN-format numbers (not valid for actual use)
        return [
            f"{self._rng.integers(100, 999)}-{self._rng.integers(10, 99)}-{self._rng.integers(1000, 9999)}"
            for _ in range(n)
        ]

    def _generate_username(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        min_len = spec.min_length or 5
        max_len = spec.max_length or 15
        return [
            f"{self._random.choice(self._first_names).lower()}{self._rng.integers(1, 9999)}"[: self._rng.integers(min_len, max_len + 1)]
            for _ in range(n)
        ]

    def _generate_password(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        # Generate password hashes (not actual passwords)
        return [hashlib.sha256(str(self._rng.integers(0, 1000000)).encode()).hexdigest()[:32] for _ in range(n)]

    # Text generators
    def _generate_word(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [self._random.choice(self._words) for _ in range(n)]

    def _generate_sentence(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [
            " ".join(self._random.choices(self._words, k=self._rng.integers(5, 15))).capitalize() + "."
            for _ in range(n)
        ]

    def _generate_paragraph(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        return [
            " ".join(
                " ".join(self._random.choices(self._words, k=self._rng.integers(5, 12))).capitalize() + "."
                for _ in range(self._rng.integers(3, 7))
            )
            for _ in range(n)
        ]

    def _generate_text(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        min_len = spec.min_length or 50
        max_len = spec.max_length or 500
        return [
            " ".join(self._random.choices(self._words, k=self._rng.integers(min_len // 5, max_len // 5)))
            for _ in range(n)
        ]

    # Categorical generators
    def _generate_category(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        choices = spec.choices or ["A", "B", "C", "D", "E"]
        return self._random.choices(choices, k=n)

    def _generate_enum(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[str]:
        choices = spec.choices or ["option_1", "option_2", "option_3"]
        return self._random.choices(choices, k=n)

    # Special generators
    def _generate_primary_key(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[int]:
        return list(range(1, n + 1))

    def _generate_sequence(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[int]:
        start = int(spec.min_value or 1)
        return list(range(start, start + n))

    def _generate_foreign_key(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[Any]:
        if existing:
            return self._random.choices(existing, k=n)
        # Default: generate integers
        max_val = int(spec.max_value or 100)
        return self._rng.integers(1, max_val + 1, n).tolist()

    def _generate_constant(
        self,
        spec: ColumnSpec,
        n: int,
        existing: Optional[List[Any]] = None,
    ) -> List[Any]:
        value = spec.choices[0] if spec.choices else "constant"
        return [value] * n


class SchemaBasedGenerator:
    """Generate data based on a schema definition.

    Example:
        >>> schema = SchemaDefinition.from_dict({
        ...     "id": "primary_key",
        ...     "name": "name",
        ...     "email": "email!",
        ...     "age": "int:18-65",
        ... })
        >>> generator = SchemaBasedGenerator(schema)
        >>> df = generator.generate(100)
    """

    def __init__(self, schema: SchemaDefinition, seed: Optional[int] = None):
        """Initialize the generator.

        Args:
            schema: Schema definition
            seed: Random seed
        """
        self.schema = schema
        self.seed = seed
        self._column_generator = ColumnGenerator(seed)
        self._generated_data: Dict[str, List[Any]] = {}

    def generate(
        self,
        n_samples: int,
        parent_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """Generate a DataFrame based on the schema.

        Args:
            n_samples: Number of rows to generate
            parent_data: Parent table data for foreign keys

        Returns:
            Generated DataFrame
        """
        data: Dict[str, List[Any]] = {}

        for col_spec in self.schema.columns:
            # Handle foreign keys
            existing_values = None
            if col_spec.dtype == DataType.FOREIGN_KEY and parent_data and col_spec.foreign_key_ref:
                table, column = col_spec.foreign_key_ref.split(".")
                if table in parent_data:
                    existing_values = parent_data[table][column].tolist()

            values = self._column_generator.generate(col_spec, n_samples, existing_values)
            data[col_spec.name] = values

        return pd.DataFrame(data)


class TestDataGenerator:
    """High-level interface for generating test data.

    Example:
        >>> generator = TestDataGenerator(seed=42)
        >>> df = generator.from_dict({
        ...     "name": "name",
        ...     "age": "int:18-65",
        ...     "email": "email",
        ... }, n_samples=100)
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._column_generator = ColumnGenerator(seed)

    def from_schema(
        self,
        schema: SchemaDefinition,
        n_samples: int,
    ) -> pd.DataFrame:
        """Generate data from a schema definition.

        Args:
            schema: Schema definition
            n_samples: Number of samples

        Returns:
            Generated DataFrame
        """
        generator = SchemaBasedGenerator(schema, self.seed)
        return generator.generate(n_samples)

    def from_dict(
        self,
        spec: Dict[str, Union[str, Dict[str, Any]]],
        n_samples: int,
    ) -> pd.DataFrame:
        """Generate data from a dictionary specification.

        Args:
            spec: Column specifications
            n_samples: Number of samples

        Returns:
            Generated DataFrame

        Example:
            >>> generator = TestDataGenerator()
            >>> df = generator.from_dict({
            ...     "id": "primary_key",
            ...     "name": "name",
            ...     "age": "int:18-100",
            ...     "status": "enum:active,inactive,pending",
            ... }, n_samples=50)
        """
        schema = SchemaDefinition.from_dict(spec)
        return self.from_schema(schema, n_samples)

    def from_pandas(
        self,
        df: pd.DataFrame,
        n_samples: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data matching an existing DataFrame's structure.

        Args:
            df: Reference DataFrame
            n_samples: Number of samples (default: same as input)

        Returns:
            Generated DataFrame with same columns
        """
        n = n_samples or len(df)
        schema = self._infer_schema_from_dataframe(df)
        return self.from_schema(schema, n)

    def _infer_schema_from_dataframe(self, df: pd.DataFrame) -> SchemaDefinition:
        """Infer a schema from an existing DataFrame."""
        columns = []

        for col_name in df.columns:
            col = df[col_name]
            dtype = col.dtype

            # Determine data type
            if pd.api.types.is_integer_dtype(dtype):
                spec = ColumnSpec(
                    name=col_name,
                    dtype=DataType.INT,
                    min_value=col.min(),
                    max_value=col.max(),
                )
            elif pd.api.types.is_float_dtype(dtype):
                spec = ColumnSpec(
                    name=col_name,
                    dtype=DataType.FLOAT,
                    min_value=col.min(),
                    max_value=col.max(),
                )
            elif pd.api.types.is_bool_dtype(dtype):
                spec = ColumnSpec(name=col_name, dtype=DataType.BOOL)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                spec = ColumnSpec(name=col_name, dtype=DataType.DATETIME)
            else:
                # String/object - check for patterns
                spec = self._infer_string_column(col_name, col)

            columns.append(spec)

        return SchemaDefinition(name="inferred", columns=columns)

    def _infer_string_column(self, name: str, col: pd.Series) -> ColumnSpec:
        """Infer the type of a string column based on content."""
        sample = col.dropna().head(100)

        # Check for email pattern
        if sample.str.contains(r"@.*\.", regex=True, na=False).mean() > 0.8:
            return ColumnSpec(name=name, dtype=DataType.EMAIL)

        # Check for phone pattern
        if sample.str.contains(r"\d{3}.*\d{3}.*\d{4}", regex=True, na=False).mean() > 0.5:
            return ColumnSpec(name=name, dtype=DataType.PHONE)

        # Check for UUID pattern
        if sample.str.contains(r"^[0-9a-f]{8}-[0-9a-f]{4}", regex=True, na=False).mean() > 0.5:
            return ColumnSpec(name=name, dtype=DataType.UUID)

        # Check cardinality for categorical
        if col.nunique() < 20 and len(col) > 100:
            return ColumnSpec(
                name=name,
                dtype=DataType.ENUM,
                choices=col.dropna().unique().tolist()[:20],
            )

        # Default to string
        return ColumnSpec(
            name=name,
            dtype=DataType.STRING,
            min_length=int(col.str.len().min()) if not col.empty else 1,
            max_length=int(col.str.len().max()) if not col.empty else 50,
        )
