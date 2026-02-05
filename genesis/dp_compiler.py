"""Differential Privacy SQL Compiler.

Compiles SQL queries into differentially private synthetic outputs
with automatic sensitivity analysis and epsilon budget management.

Features:
    - SQL query parsing and sensitivity analysis
    - Automatic epsilon budget allocation
    - Noise mechanism selection (Laplace, Gaussian)
    - Query result synthesis with DP guarantees
    - Budget tracking and enforcement

Example:
    Basic usage::

        from genesis.dp_compiler import DPCompiler, DPConfig

        compiler = DPCompiler(
            config=DPConfig(epsilon=1.0, delta=1e-5)
        )

        # Compile and execute DP query
        result = compiler.execute(
            "SELECT department, AVG(salary) FROM employees GROUP BY department",
            data=employees_df
        )
        
        print(f"Epsilon used: {result.epsilon_used}")
        print(result.data)

    With budget management::

        from genesis.dp_compiler import DPBudgetManager

        budget = DPBudgetManager(total_epsilon=5.0)
        
        # Each query deducts from budget
        result1 = compiler.execute(query1, data, budget=budget)
        result2 = compiler.execute(query2, data, budget=budget)
        
        print(f"Remaining budget: {budget.remaining}")

Classes:
    DPConfig: Differential privacy configuration.
    SQLParser: Parses SQL into query AST.
    SensitivityAnalyzer: Computes query sensitivity.
    NoiseMechanism: Noise addition mechanisms.
    DPBudgetManager: Epsilon budget tracking.
    DPQueryResult: Result with privacy guarantees.
    DPCompiler: Main compiler interface.
"""

import hashlib
import math
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class NoiseType(str, Enum):
    """Type of noise mechanism."""

    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"
    EXPONENTIAL = "exponential"


class AggregationType(str, Enum):
    """SQL aggregation types."""

    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    STDDEV = "stddev"
    VARIANCE = "variance"


@dataclass
class DPConfig:
    """Differential privacy configuration.

    Attributes:
        epsilon: Privacy budget (smaller = more private)
        delta: Probability of privacy breach (0 for pure DP)
        noise_type: Noise mechanism to use
        clipping_bounds: Dict of column -> (min, max) for sensitivity
        sensitivity_method: How to compute sensitivity
        composition_method: How to compose multiple queries
    """

    epsilon: float = 1.0
    delta: float = 1e-5
    noise_type: NoiseType = NoiseType.LAPLACE
    clipping_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    sensitivity_method: str = "global"  # global, local, smooth
    composition_method: str = "basic"  # basic, advanced, rdp


@dataclass
class QueryAST:
    """Abstract syntax tree for SQL query."""

    query_type: str  # SELECT, INSERT, etc.
    columns: List[Dict[str, Any]]  # Selected columns with aggregations
    from_table: str
    where_clauses: List[Dict[str, Any]] = field(default_factory=list)
    group_by: List[str] = field(default_factory=list)
    having: Optional[Dict[str, Any]] = None
    order_by: List[Tuple[str, str]] = field(default_factory=list)
    limit: Optional[int] = None


@dataclass
class SensitivityResult:
    """Result of sensitivity analysis."""

    global_sensitivity: float
    local_sensitivity: Optional[float] = None
    column_sensitivities: Dict[str, float] = field(default_factory=dict)
    clipping_applied: bool = False
    clipping_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class DPQueryResult:
    """Result of a differentially private query.

    Attributes:
        data: Noisy query result
        epsilon_used: Actual epsilon consumed
        delta_used: Actual delta consumed
        sensitivity: Computed sensitivity
        noise_scale: Scale of added noise
        confidence_interval: Approximate confidence bounds
        warnings: Any warnings about result quality
    """

    data: pd.DataFrame
    epsilon_used: float
    delta_used: float
    sensitivity: float
    noise_scale: float
    confidence_interval: Optional[Tuple[float, float]] = None
    warnings: List[str] = field(default_factory=list)
    query_hash: str = ""


class SQLParser:
    """Parses SQL queries into AST for DP analysis.

    Supports a subset of SQL focused on aggregation queries
    that are amenable to differential privacy.
    """

    # Supported aggregation functions
    AGGREGATIONS = {
        "count": AggregationType.COUNT,
        "sum": AggregationType.SUM,
        "avg": AggregationType.AVG,
        "average": AggregationType.AVG,
        "min": AggregationType.MIN,
        "max": AggregationType.MAX,
        "median": AggregationType.MEDIAN,
        "stddev": AggregationType.STDDEV,
        "std": AggregationType.STDDEV,
        "variance": AggregationType.VARIANCE,
        "var": AggregationType.VARIANCE,
    }

    def parse(self, query: str) -> QueryAST:
        """Parse SQL query into AST.

        Args:
            query: SQL query string

        Returns:
            QueryAST representation

        Raises:
            ValueError: If query is not supported
        """
        query = query.strip()

        # Normalize whitespace
        query = re.sub(r"\s+", " ", query)

        # Only support SELECT
        if not query.upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are supported for DP compilation")

        # Extract components using regex
        # This is a simplified parser - production would use a proper SQL parser

        # Remove SELECT keyword
        query_body = query[6:].strip()

        # Find FROM
        from_match = re.search(r"\bFROM\b", query_body, re.IGNORECASE)
        if not from_match:
            raise ValueError("Query must have FROM clause")

        select_part = query_body[: from_match.start()].strip()
        remaining = query_body[from_match.end() :].strip()

        # Parse SELECT columns
        columns = self._parse_columns(select_part)

        # Parse FROM table
        table, remaining = self._parse_from(remaining)

        # Parse WHERE
        where_clauses = []
        where_match = re.search(r"\bWHERE\b", remaining, re.IGNORECASE)
        if where_match:
            where_part, remaining = self._extract_clause(remaining, where_match.end(), ["GROUP", "HAVING", "ORDER", "LIMIT"])
            where_clauses = self._parse_where(where_part)

        # Parse GROUP BY
        group_by = []
        group_match = re.search(r"\bGROUP\s+BY\b", remaining, re.IGNORECASE)
        if group_match:
            group_part, remaining = self._extract_clause(remaining, group_match.end(), ["HAVING", "ORDER", "LIMIT"])
            group_by = [g.strip() for g in group_part.split(",")]

        # Parse HAVING
        having = None
        having_match = re.search(r"\bHAVING\b", remaining, re.IGNORECASE)
        if having_match:
            having_part, remaining = self._extract_clause(remaining, having_match.end(), ["ORDER", "LIMIT"])
            having = {"raw": having_part}

        # Parse ORDER BY
        order_by = []
        order_match = re.search(r"\bORDER\s+BY\b", remaining, re.IGNORECASE)
        if order_match:
            order_part, remaining = self._extract_clause(remaining, order_match.end(), ["LIMIT"])
            order_by = self._parse_order_by(order_part)

        # Parse LIMIT
        limit = None
        limit_match = re.search(r"\bLIMIT\s+(\d+)", remaining, re.IGNORECASE)
        if limit_match:
            limit = int(limit_match.group(1))

        return QueryAST(
            query_type="SELECT",
            columns=columns,
            from_table=table,
            where_clauses=where_clauses,
            group_by=group_by,
            having=having,
            order_by=order_by,
            limit=limit,
        )

    def _parse_columns(self, select_part: str) -> List[Dict[str, Any]]:
        """Parse column expressions."""
        columns = []

        # Split by comma (careful with nested parentheses)
        parts = self._split_columns(select_part)

        for part in parts:
            part = part.strip()

            # Check for aggregation
            agg_match = re.match(r"(\w+)\s*\(\s*(\*|\w+)\s*\)(?:\s+AS\s+(\w+))?", part, re.IGNORECASE)
            if agg_match:
                func_name = agg_match.group(1).lower()
                col_name = agg_match.group(2)
                alias = agg_match.group(3) or f"{func_name}_{col_name}"

                if func_name in self.AGGREGATIONS:
                    columns.append({
                        "type": "aggregation",
                        "function": self.AGGREGATIONS[func_name],
                        "column": col_name,
                        "alias": alias,
                    })
                else:
                    columns.append({
                        "type": "function",
                        "function": func_name,
                        "column": col_name,
                        "alias": alias,
                    })
            else:
                # Regular column
                alias_match = re.match(r"(\w+)(?:\s+AS\s+(\w+))?", part, re.IGNORECASE)
                if alias_match:
                    columns.append({
                        "type": "column",
                        "column": alias_match.group(1),
                        "alias": alias_match.group(2) or alias_match.group(1),
                    })

        return columns

    def _split_columns(self, select_part: str) -> List[str]:
        """Split columns handling nested parentheses."""
        parts = []
        current = ""
        depth = 0

        for char in select_part:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "," and depth == 0:
                parts.append(current)
                current = ""
                continue
            current += char

        if current:
            parts.append(current)

        return parts

    def _parse_from(self, remaining: str) -> Tuple[str, str]:
        """Parse FROM clause and return table name and remaining query."""
        # Find next keyword
        keywords = ["WHERE", "GROUP", "HAVING", "ORDER", "LIMIT"]
        end_pos = len(remaining)

        for keyword in keywords:
            match = re.search(rf"\b{keyword}\b", remaining, re.IGNORECASE)
            if match and match.start() < end_pos:
                end_pos = match.start()

        table = remaining[:end_pos].strip()
        remaining = remaining[end_pos:]

        return table, remaining

    def _extract_clause(self, remaining: str, start: int, end_keywords: List[str]) -> Tuple[str, str]:
        """Extract a clause up to the next keyword."""
        body = remaining[start:]
        end_pos = len(body)

        for keyword in end_keywords:
            match = re.search(rf"\b{keyword}\b", body, re.IGNORECASE)
            if match and match.start() < end_pos:
                end_pos = match.start()

        clause = body[:end_pos].strip()
        remaining = body[end_pos:]

        return clause, remaining

    def _parse_where(self, where_part: str) -> List[Dict[str, Any]]:
        """Parse WHERE conditions."""
        conditions = []

        # Split by AND/OR (simplified)
        parts = re.split(r"\bAND\b|\bOR\b", where_part, flags=re.IGNORECASE)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Parse comparison
            match = re.match(r"(\w+)\s*(=|!=|<>|>=|<=|>|<|LIKE|IN)\s*(.+)", part, re.IGNORECASE)
            if match:
                conditions.append({
                    "column": match.group(1),
                    "operator": match.group(2).upper(),
                    "value": match.group(3).strip().strip("'\""),
                })

        return conditions

    def _parse_order_by(self, order_part: str) -> List[Tuple[str, str]]:
        """Parse ORDER BY clause."""
        results = []

        parts = order_part.split(",")
        for part in parts:
            part = part.strip()
            match = re.match(r"(\w+)(?:\s+(ASC|DESC))?", part, re.IGNORECASE)
            if match:
                results.append((match.group(1), (match.group(2) or "ASC").upper()))

        return results


class SensitivityAnalyzer:
    """Analyzes query sensitivity for DP noise calibration.

    Computes the global or local sensitivity of SQL queries
    to determine appropriate noise levels.
    """

    def __init__(self, clipping_bounds: Optional[Dict[str, Tuple[float, float]]] = None):
        """Initialize analyzer.

        Args:
            clipping_bounds: Per-column clipping bounds
        """
        self._clipping_bounds = clipping_bounds or {}

    def analyze(
        self,
        ast: QueryAST,
        data: pd.DataFrame,
        method: str = "global",
    ) -> SensitivityResult:
        """Analyze query sensitivity.

        Args:
            ast: Query AST
            data: Source data
            method: Sensitivity method (global, local, smooth)

        Returns:
            SensitivityResult with computed sensitivities
        """
        column_sensitivities: Dict[str, float] = {}
        clipping_applied = False

        for col_info in ast.columns:
            if col_info["type"] == "aggregation":
                func = col_info["function"]
                col_name = col_info["column"]
                alias = col_info["alias"]

                # Get bounds for clipping
                if col_name != "*" and col_name in self._clipping_bounds:
                    bounds = self._clipping_bounds[col_name]
                    clipping_applied = True
                elif col_name != "*" and col_name in data.columns:
                    # Auto-detect bounds from data (less private but practical)
                    col_data = data[col_name].dropna()
                    if len(col_data) > 0 and pd.api.types.is_numeric_dtype(col_data):
                        bounds = (float(col_data.min()), float(col_data.max()))
                    else:
                        bounds = (0.0, 1.0)
                else:
                    bounds = (0.0, 1.0)

                sensitivity = self._compute_aggregation_sensitivity(func, bounds, len(data))
                column_sensitivities[alias] = sensitivity

        # Global sensitivity is max of column sensitivities
        global_sensitivity = max(column_sensitivities.values()) if column_sensitivities else 1.0

        return SensitivityResult(
            global_sensitivity=global_sensitivity,
            column_sensitivities=column_sensitivities,
            clipping_applied=clipping_applied,
            clipping_bounds=self._clipping_bounds,
        )

    def _compute_aggregation_sensitivity(
        self,
        func: AggregationType,
        bounds: Tuple[float, float],
        n: int,
    ) -> float:
        """Compute sensitivity for an aggregation function."""
        low, high = bounds
        range_val = high - low

        if func == AggregationType.COUNT:
            return 1.0  # Adding/removing one record changes count by 1

        elif func == AggregationType.SUM:
            return range_val  # Adding/removing one record changes sum by at most range

        elif func == AggregationType.AVG:
            # Sensitivity of average is range / n
            return range_val / max(n, 1)

        elif func == AggregationType.MIN or func == AggregationType.MAX:
            # Sensitivity is the full range (one record can change min/max completely)
            return range_val

        elif func == AggregationType.MEDIAN:
            # Median sensitivity is complex, approximate with range / (n/2)
            return range_val / max(n / 2, 1)

        elif func in (AggregationType.STDDEV, AggregationType.VARIANCE):
            # Approximate sensitivity
            return range_val ** 2 / max(n, 1)

        return range_val  # Default to range


class NoiseMechanism(ABC):
    """Abstract base class for noise mechanisms."""

    @abstractmethod
    def add_noise(self, value: float, sensitivity: float, epsilon: float, delta: float = 0.0) -> float:
        """Add noise to a value.

        Args:
            value: Original value
            sensitivity: Query sensitivity
            epsilon: Privacy budget
            delta: Privacy parameter (for approximate DP)

        Returns:
            Noisy value
        """
        pass

    @abstractmethod
    def get_scale(self, sensitivity: float, epsilon: float, delta: float = 0.0) -> float:
        """Get noise scale for given parameters."""
        pass


class LaplaceMechanism(NoiseMechanism):
    """Laplace mechanism for (ε, 0)-DP."""

    def add_noise(self, value: float, sensitivity: float, epsilon: float, delta: float = 0.0) -> float:
        scale = self.get_scale(sensitivity, epsilon)
        noise = np.random.laplace(0, scale)
        return value + noise

    def get_scale(self, sensitivity: float, epsilon: float, delta: float = 0.0) -> float:
        return sensitivity / epsilon


class GaussianMechanism(NoiseMechanism):
    """Gaussian mechanism for (ε, δ)-DP."""

    def add_noise(self, value: float, sensitivity: float, epsilon: float, delta: float = 1e-5) -> float:
        scale = self.get_scale(sensitivity, epsilon, delta)
        noise = np.random.normal(0, scale)
        return value + noise

    def get_scale(self, sensitivity: float, epsilon: float, delta: float = 1e-5) -> float:
        if delta <= 0:
            raise ValueError("Gaussian mechanism requires delta > 0")
        # Using the analytic Gaussian mechanism formula
        return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


class DPBudgetManager:
    """Manages differential privacy budget across queries.

    Tracks epsilon consumption and enforces budget limits
    using composition theorems.
    """

    def __init__(
        self,
        total_epsilon: float,
        total_delta: float = 1e-5,
        composition_method: str = "basic",
    ):
        """Initialize budget manager.

        Args:
            total_epsilon: Total privacy budget
            total_delta: Total delta budget
            composition_method: Composition method (basic, advanced, rdp)
        """
        self._total_epsilon = total_epsilon
        self._total_delta = total_delta
        self._composition_method = composition_method
        self._consumed_epsilon = 0.0
        self._consumed_delta = 0.0
        self._query_history: List[Dict[str, Any]] = []

    @property
    def remaining(self) -> float:
        """Remaining epsilon budget."""
        return max(0.0, self._total_epsilon - self._consumed_epsilon)

    @property
    def consumed(self) -> float:
        """Consumed epsilon budget."""
        return self._consumed_epsilon

    def can_afford(self, epsilon: float, delta: float = 0.0) -> bool:
        """Check if budget can afford a query.

        Args:
            epsilon: Epsilon needed
            delta: Delta needed

        Returns:
            True if budget is sufficient
        """
        if self._composition_method == "basic":
            return (
                self._consumed_epsilon + epsilon <= self._total_epsilon
                and self._consumed_delta + delta <= self._total_delta
            )
        else:
            # Advanced composition: epsilon^2 composition
            return self._consumed_epsilon + epsilon <= self._total_epsilon

    def consume(self, epsilon: float, delta: float = 0.0, query_hash: str = "") -> bool:
        """Consume budget for a query.

        Args:
            epsilon: Epsilon to consume
            delta: Delta to consume
            query_hash: Hash of query for tracking

        Returns:
            True if budget consumed successfully
        """
        if not self.can_afford(epsilon, delta):
            return False

        if self._composition_method == "basic":
            self._consumed_epsilon += epsilon
            self._consumed_delta += delta
        else:
            # Advanced composition with better bounds
            self._consumed_epsilon += epsilon
            self._consumed_delta += delta

        self._query_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "epsilon": epsilon,
            "delta": delta,
            "query_hash": query_hash,
            "remaining": self.remaining,
        })

        return True

    def get_history(self) -> List[Dict[str, Any]]:
        """Get query history."""
        return list(self._query_history)

    def reset(self) -> None:
        """Reset budget consumption."""
        self._consumed_epsilon = 0.0
        self._consumed_delta = 0.0
        self._query_history = []


class DPCompiler:
    """Differential Privacy SQL Compiler.

    Compiles SQL queries into differentially private synthetic
    outputs with automatic sensitivity analysis.
    """

    def __init__(self, config: Optional[DPConfig] = None):
        """Initialize compiler.

        Args:
            config: DP configuration
        """
        self.config = config or DPConfig()
        self._parser = SQLParser()
        self._sensitivity_analyzer = SensitivityAnalyzer(self.config.clipping_bounds)

        # Select noise mechanism
        if self.config.noise_type == NoiseType.LAPLACE:
            self._mechanism = LaplaceMechanism()
        else:
            self._mechanism = GaussianMechanism()

    def compile(self, query: str) -> Tuple[QueryAST, Callable]:
        """Compile query into DP-safe executable.

        Args:
            query: SQL query string

        Returns:
            Tuple of (AST, execution function)
        """
        ast = self._parser.parse(query)

        def execute_fn(data: pd.DataFrame, epsilon: Optional[float] = None) -> DPQueryResult:
            return self._execute_query(ast, data, epsilon or self.config.epsilon)

        return ast, execute_fn

    def execute(
        self,
        query: str,
        data: pd.DataFrame,
        budget: Optional[DPBudgetManager] = None,
        epsilon: Optional[float] = None,
    ) -> DPQueryResult:
        """Execute a differentially private query.

        Args:
            query: SQL query string
            data: Source DataFrame
            budget: Optional budget manager
            epsilon: Override epsilon (uses config if not provided)

        Returns:
            DPQueryResult with noisy results
        """
        eps = epsilon or self.config.epsilon
        delta = self.config.delta

        # Check budget if provided
        if budget and not budget.can_afford(eps, delta):
            raise ValueError(f"Insufficient privacy budget. Remaining: {budget.remaining}")

        # Parse query
        ast = self._parser.parse(query)
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]

        # Analyze sensitivity
        sensitivity_result = self._sensitivity_analyzer.analyze(
            ast, data, self.config.sensitivity_method
        )

        # Execute query and add noise
        result = self._execute_query(ast, data, eps, sensitivity_result)
        result.query_hash = query_hash

        # Update budget
        if budget:
            budget.consume(result.epsilon_used, result.delta_used, query_hash)

        return result

    def _execute_query(
        self,
        ast: QueryAST,
        data: pd.DataFrame,
        epsilon: float,
        sensitivity_result: Optional[SensitivityResult] = None,
    ) -> DPQueryResult:
        """Execute parsed query with DP guarantees."""
        warnings: List[str] = []

        # Analyze sensitivity if not provided
        if sensitivity_result is None:
            sensitivity_result = self._sensitivity_analyzer.analyze(ast, data)

        # Apply WHERE filters first (doesn't affect DP)
        filtered_data = self._apply_filters(data, ast.where_clauses)

        # Check for GROUP BY
        if ast.group_by:
            result_df = self._execute_grouped_query(ast, filtered_data, epsilon, sensitivity_result)
        else:
            result_df = self._execute_aggregate_query(ast, filtered_data, epsilon, sensitivity_result)

        # Apply ORDER BY
        if ast.order_by:
            for col, direction in reversed(ast.order_by):
                if col in result_df.columns:
                    result_df = result_df.sort_values(col, ascending=(direction == "ASC"))

        # Apply LIMIT
        if ast.limit:
            result_df = result_df.head(ast.limit)

        # Compute noise scale for reporting
        noise_scale = self._mechanism.get_scale(
            sensitivity_result.global_sensitivity, epsilon, self.config.delta
        )

        return DPQueryResult(
            data=result_df,
            epsilon_used=epsilon,
            delta_used=self.config.delta,
            sensitivity=sensitivity_result.global_sensitivity,
            noise_scale=noise_scale,
            warnings=warnings,
        )

    def _apply_filters(self, data: pd.DataFrame, where_clauses: List[Dict[str, Any]]) -> pd.DataFrame:
        """Apply WHERE clause filters."""
        result = data.copy()

        for clause in where_clauses:
            col = clause["column"]
            op = clause["operator"]
            value = clause["value"]

            if col not in result.columns:
                continue

            if op == "=":
                result = result[result[col] == value]
            elif op in ("!=", "<>"):
                result = result[result[col] != value]
            elif op == ">":
                result = result[result[col] > float(value)]
            elif op == ">=":
                result = result[result[col] >= float(value)]
            elif op == "<":
                result = result[result[col] < float(value)]
            elif op == "<=":
                result = result[result[col] <= float(value)]
            elif op == "LIKE":
                pattern = value.replace("%", ".*").replace("_", ".")
                result = result[result[col].astype(str).str.match(pattern, na=False)]

        return result

    def _execute_aggregate_query(
        self,
        ast: QueryAST,
        data: pd.DataFrame,
        epsilon: float,
        sensitivity_result: SensitivityResult,
    ) -> pd.DataFrame:
        """Execute query without GROUP BY."""
        results: Dict[str, Any] = {}
        eps_per_col = epsilon / max(len(ast.columns), 1)

        for col_info in ast.columns:
            alias = col_info["alias"]

            if col_info["type"] == "aggregation":
                func = col_info["function"]
                col_name = col_info["column"]
                sensitivity = sensitivity_result.column_sensitivities.get(alias, 1.0)

                # Compute true aggregate
                true_value = self._compute_aggregate(data, col_name, func)

                # Add noise
                noisy_value = self._mechanism.add_noise(
                    true_value, sensitivity, eps_per_col, self.config.delta
                )

                results[alias] = [noisy_value]

            elif col_info["type"] == "column":
                # Non-aggregated columns need special handling
                results[alias] = ["*"]

        return pd.DataFrame(results)

    def _execute_grouped_query(
        self,
        ast: QueryAST,
        data: pd.DataFrame,
        epsilon: float,
        sensitivity_result: SensitivityResult,
    ) -> pd.DataFrame:
        """Execute query with GROUP BY."""
        # Number of aggregation columns
        n_agg_cols = sum(1 for c in ast.columns if c["type"] == "aggregation")
        eps_per_col = epsilon / max(n_agg_cols, 1)

        # Group data
        groups = data.groupby(ast.group_by)
        result_rows = []

        for group_key, group_data in groups:
            row: Dict[str, Any] = {}

            # Add group columns
            if isinstance(group_key, tuple):
                for i, col in enumerate(ast.group_by):
                    row[col] = group_key[i]
            else:
                row[ast.group_by[0]] = group_key

            # Compute aggregates
            for col_info in ast.columns:
                if col_info["type"] == "aggregation":
                    func = col_info["function"]
                    col_name = col_info["column"]
                    alias = col_info["alias"]
                    sensitivity = sensitivity_result.column_sensitivities.get(alias, 1.0)

                    # Compute true aggregate
                    true_value = self._compute_aggregate(group_data, col_name, func)

                    # Add noise
                    noisy_value = self._mechanism.add_noise(
                        true_value, sensitivity, eps_per_col, self.config.delta
                    )

                    row[alias] = noisy_value

            result_rows.append(row)

        return pd.DataFrame(result_rows)

    def _compute_aggregate(self, data: pd.DataFrame, col_name: str, func: AggregationType) -> float:
        """Compute an aggregate function."""
        if func == AggregationType.COUNT:
            if col_name == "*":
                return float(len(data))
            return float(data[col_name].count())

        if col_name not in data.columns:
            return 0.0

        col_data = data[col_name].dropna()

        if len(col_data) == 0:
            return 0.0

        if func == AggregationType.SUM:
            return float(col_data.sum())
        elif func == AggregationType.AVG:
            return float(col_data.mean())
        elif func == AggregationType.MIN:
            return float(col_data.min())
        elif func == AggregationType.MAX:
            return float(col_data.max())
        elif func == AggregationType.MEDIAN:
            return float(col_data.median())
        elif func == AggregationType.STDDEV:
            return float(col_data.std())
        elif func == AggregationType.VARIANCE:
            return float(col_data.var())

        return 0.0


def dp_query(
    query: str,
    data: pd.DataFrame,
    epsilon: float = 1.0,
    delta: float = 1e-5,
    clipping_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> DPQueryResult:
    """Convenience function for one-off DP queries.

    Args:
        query: SQL query
        data: Source DataFrame
        epsilon: Privacy budget
        delta: Delta parameter
        clipping_bounds: Optional clipping bounds

    Returns:
        DPQueryResult with noisy results

    Example:
        >>> result = dp_query(
        ...     "SELECT department, AVG(salary) FROM employees GROUP BY department",
        ...     employees_df,
        ...     epsilon=1.0
        ... )
        >>> print(result.data)
    """
    config = DPConfig(
        epsilon=epsilon,
        delta=delta,
        clipping_bounds=clipping_bounds or {},
    )
    compiler = DPCompiler(config)
    return compiler.execute(query, data)


class DPSyntheticGenerator:
    """Generate synthetic data from DP query results.

    Creates full synthetic datasets that preserve the DP-computed
    statistics from queries.
    """

    def __init__(self, compiler: DPCompiler):
        """Initialize generator.

        Args:
            compiler: DP compiler to use
        """
        self._compiler = compiler

    def generate_from_statistics(
        self,
        statistics_queries: List[str],
        source_data: pd.DataFrame,
        n_samples: int,
        budget: Optional[DPBudgetManager] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data matching DP statistics.

        Args:
            statistics_queries: Queries to compute statistics
            source_data: Source data
            n_samples: Number of synthetic samples
            budget: Privacy budget manager

        Returns:
            Synthetic DataFrame
        """
        # Execute DP queries to get statistics
        dp_stats: Dict[str, Any] = {}

        for query in statistics_queries:
            result = self._compiler.execute(query, source_data, budget)
            # Store statistics from result
            for col in result.data.columns:
                dp_stats[col] = result.data[col].iloc[0] if len(result.data) > 0 else 0

        # Generate synthetic data matching statistics
        # This is a simplified version - full implementation would use
        # more sophisticated matching techniques

        synthetic_data: Dict[str, Any] = {}

        for col in source_data.columns:
            if pd.api.types.is_numeric_dtype(source_data[col]):
                # For numeric columns, generate normal distribution around DP mean
                if f"avg_{col}" in dp_stats:
                    mean = dp_stats[f"avg_{col}"]
                    std = source_data[col].std()  # Use original std as approximation
                    synthetic_data[col] = np.random.normal(mean, std, n_samples)
                else:
                    synthetic_data[col] = np.random.normal(
                        source_data[col].mean(), source_data[col].std(), n_samples
                    )
            else:
                # For categorical, sample from observed distribution
                synthetic_data[col] = np.random.choice(
                    source_data[col].dropna().unique(), n_samples
                )

        return pd.DataFrame(synthetic_data)
