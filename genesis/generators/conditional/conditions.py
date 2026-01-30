"""Condition classes for conditional generation.

This module provides the Operator enum, Condition and ConditionSet dataclasses,
and the fluent ConditionBuilder API for creating filter conditions.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from genesis.core.exceptions import ValidationError


class Operator(Enum):
    """Condition operators for filtering."""

    EQ = "eq"  # Equal
    NE = "ne"  # Not equal
    GT = "gt"  # Greater than
    GE = "ge"  # Greater than or equal
    LT = "lt"  # Less than
    LE = "le"  # Less than or equal
    IN = "in"  # In list
    NOT_IN = "not_in"  # Not in list
    BETWEEN = "between"  # Between range (inclusive)
    LIKE = "like"  # String pattern matching
    IS_NULL = "is_null"  # Is null/NaN
    NOT_NULL = "not_null"  # Is not null/NaN


@dataclass
class Condition:
    """A single condition for filtering synthetic data.

    Examples:
        >>> Condition("age", Operator.GE, 18)
        >>> Condition("country", Operator.IN, ["US", "CA", "UK"])
        >>> Condition("salary", Operator.BETWEEN, (50000, 100000))
    """

    column: str
    operator: Operator
    value: Any

    def __post_init__(self) -> None:
        """Validate condition parameters."""
        if self.operator == Operator.BETWEEN:
            if not isinstance(self.value, (tuple, list)) or len(self.value) != 2:
                raise ValidationError(
                    f"BETWEEN operator requires a tuple/list of 2 values, got {self.value}"
                )
        if self.operator in (Operator.IN, Operator.NOT_IN):
            if not isinstance(self.value, (list, tuple, set)):
                raise ValidationError(
                    f"{self.operator.value} operator requires an iterable, got {type(self.value)}"
                )

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate condition against a DataFrame.

        Args:
            df: DataFrame to evaluate

        Returns:
            Boolean Series indicating which rows satisfy the condition
        """
        if self.column not in df.columns:
            raise ValidationError(f"Column '{self.column}' not found in data")

        col = df[self.column]

        if self.operator == Operator.EQ:
            return col == self.value
        elif self.operator == Operator.NE:
            return col != self.value
        elif self.operator == Operator.GT:
            return col > self.value
        elif self.operator == Operator.GE:
            return col >= self.value
        elif self.operator == Operator.LT:
            return col < self.value
        elif self.operator == Operator.LE:
            return col <= self.value
        elif self.operator == Operator.IN:
            return col.isin(self.value)
        elif self.operator == Operator.NOT_IN:
            return ~col.isin(self.value)
        elif self.operator == Operator.BETWEEN:
            return (col >= self.value[0]) & (col <= self.value[1])
        elif self.operator == Operator.LIKE:
            return col.astype(str).str.contains(self.value, regex=True, na=False)
        elif self.operator == Operator.IS_NULL:
            return col.isna()
        elif self.operator == Operator.NOT_NULL:
            return col.notna()
        else:
            raise ValidationError(f"Unknown operator: {self.operator}")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Condition":
        """Create condition from dictionary.

        Args:
            d: Dictionary with 'column', 'operator', 'value' keys

        Returns:
            Condition instance
        """
        return cls(
            column=d["column"],
            operator=Operator(d["operator"]) if isinstance(d["operator"], str) else d["operator"],
            value=d["value"],
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column": self.column,
            "operator": self.operator.value,
            "value": self.value,
        }


@dataclass
class ConditionSet:
    """A set of conditions combined with AND logic.

    Examples:
        >>> conditions = ConditionSet([
        ...     Condition("age", Operator.GE, 18),
        ...     Condition("country", Operator.EQ, "US"),
        ... ])
    """

    conditions: List[Condition] = field(default_factory=list)

    def add(self, condition: Condition) -> "ConditionSet":
        """Add a condition to the set."""
        self.conditions.append(condition)
        return self

    def evaluate(self, df: pd.DataFrame) -> pd.Series:
        """Evaluate all conditions against a DataFrame.

        Args:
            df: DataFrame to evaluate

        Returns:
            Boolean Series (True where ALL conditions are satisfied)
        """
        if not self.conditions:
            return pd.Series([True] * len(df), index=df.index)

        result = self.conditions[0].evaluate(df)
        for cond in self.conditions[1:]:
            result = result & cond.evaluate(df)
        return result

    def __len__(self) -> int:
        return len(self.conditions)

    @classmethod
    def from_dict(cls, conditions_dict: Dict[str, Any]) -> "ConditionSet":
        """Create ConditionSet from a simple dictionary format.

        Args:
            conditions_dict: Dictionary where keys are column names
                             and values are condition values or tuples (operator, value)

        Examples:
            >>> ConditionSet.from_dict({"age": 25})  # age == 25
            >>> ConditionSet.from_dict({"age": (">=", 18)})  # age >= 18
            >>> ConditionSet.from_dict({"country": ("in", ["US", "UK"])})
            >>> ConditionSet.from_dict({"salary": ("between", (50000, 100000))})
        """
        operator_map = {
            "=": Operator.EQ,
            "==": Operator.EQ,
            "!=": Operator.NE,
            "<>": Operator.NE,
            ">": Operator.GT,
            ">=": Operator.GE,
            "<": Operator.LT,
            "<=": Operator.LE,
            "in": Operator.IN,
            "not_in": Operator.NOT_IN,
            "between": Operator.BETWEEN,
            "like": Operator.LIKE,
            "is_null": Operator.IS_NULL,
            "not_null": Operator.NOT_NULL,
        }

        conditions = []
        for column, value in conditions_dict.items():
            if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], str):
                op_str, val = value
                op = operator_map.get(op_str.lower())
                if op is None:
                    raise ValidationError(f"Unknown operator: {op_str}")
                conditions.append(Condition(column, op, val))
            else:
                # Simple equality
                conditions.append(Condition(column, Operator.EQ, value))

        return cls(conditions)


class ConditionBuilder:
    """Fluent builder for creating conditions.

    Example:
        >>> conditions = (ConditionBuilder()
        ...     .where("age").gte(18)
        ...     .where("country").in_(["US", "UK"])
        ...     .where("salary").between(50000, 100000)
        ...     .build())
    """

    def __init__(self) -> None:
        self._conditions: List[Condition] = []
        self._current_column: Optional[str] = None

    def where(self, column: str) -> "ConditionBuilder":
        """Start a condition for a column."""
        self._current_column = column
        return self

    def eq(self, value: Any) -> "ConditionBuilder":
        """Equal to value."""
        self._add_condition(Operator.EQ, value)
        return self

    def ne(self, value: Any) -> "ConditionBuilder":
        """Not equal to value."""
        self._add_condition(Operator.NE, value)
        return self

    def gt(self, value: Any) -> "ConditionBuilder":
        """Greater than value."""
        self._add_condition(Operator.GT, value)
        return self

    def gte(self, value: Any) -> "ConditionBuilder":
        """Greater than or equal to value."""
        self._add_condition(Operator.GE, value)
        return self

    def lt(self, value: Any) -> "ConditionBuilder":
        """Less than value."""
        self._add_condition(Operator.LT, value)
        return self

    def lte(self, value: Any) -> "ConditionBuilder":
        """Less than or equal to value."""
        self._add_condition(Operator.LE, value)
        return self

    def in_(self, values: List[Any]) -> "ConditionBuilder":
        """In list of values."""
        self._add_condition(Operator.IN, values)
        return self

    def not_in(self, values: List[Any]) -> "ConditionBuilder":
        """Not in list of values."""
        self._add_condition(Operator.NOT_IN, values)
        return self

    def between(self, low: Any, high: Any) -> "ConditionBuilder":
        """Between low and high (inclusive)."""
        self._add_condition(Operator.BETWEEN, (low, high))
        return self

    def like(self, pattern: str) -> "ConditionBuilder":
        """Match regex pattern."""
        self._add_condition(Operator.LIKE, pattern)
        return self

    def is_null(self) -> "ConditionBuilder":
        """Is null/NaN."""
        self._add_condition(Operator.IS_NULL, None)
        return self

    def not_null(self) -> "ConditionBuilder":
        """Is not null/NaN."""
        self._add_condition(Operator.NOT_NULL, None)
        return self

    def _add_condition(self, operator: Operator, value: Any) -> None:
        """Add a condition to the list."""
        if self._current_column is None:
            raise ValidationError("Call where() before adding conditions")
        self._conditions.append(Condition(self._current_column, operator, value))

    def build(self) -> ConditionSet:
        """Build the ConditionSet."""
        return ConditionSet(self._conditions.copy())

    def __repr__(self) -> str:
        return f"ConditionBuilder({len(self._conditions)} conditions)"
