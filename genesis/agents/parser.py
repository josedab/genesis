"""Configuration parser for agent responses."""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from genesis.core.config import GeneratorConfig
from genesis.core.constraints import Constraint
from genesis.core.exceptions import ValidationError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnSpec:
    """Specification for a column in the schema."""

    name: str
    type: str  # numeric, categorical, datetime, text
    description: str = ""
    nullable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "nullable": self.nullable,
        }


@dataclass
class GenerationConfig:
    """Parsed configuration for synthetic data generation."""

    schema: List[ColumnSpec] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    n_samples: int = 1000
    generator_method: str = "auto"
    generator_epochs: int = 300
    clarification_needed: Optional[str] = None
    explanation: str = ""
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """Check if configuration is complete (no clarification needed)."""
        return self.clarification_needed is None

    def to_generator_config(self) -> GeneratorConfig:
        """Convert to GeneratorConfig."""
        return GeneratorConfig(
            method=self.generator_method,
            epochs=self.generator_epochs,
        )

    def to_constraints(self) -> List[Constraint]:
        """Convert constraint specs to Constraint objects."""
        result = []
        for c in self.constraints:
            constraint_type = c.get("type", "")
            column = c.get("column", "")
            params = c.get("params", {})

            if constraint_type == "positive":
                result.append(Constraint.positive(column))
            elif constraint_type == "non_negative":
                result.append(Constraint.non_negative(column))
            elif constraint_type == "range":
                result.append(
                    Constraint.range(
                        column,
                        min_value=params.get("min"),
                        max_value=params.get("max"),
                    )
                )
            elif constraint_type == "unique":
                result.append(Constraint.unique(column))
            elif constraint_type == "regex":
                result.append(Constraint.regex(column, params.get("pattern", ".*")))
            elif constraint_type == "one_of":
                result.append(Constraint.one_of(column, params.get("values", [])))
            else:
                logger.warning(f"Unknown constraint type: {constraint_type}")

        return result

    def to_conditions_dict(self) -> Dict[str, Any]:
        """Convert conditions to format expected by conditional sampler."""
        result = {}
        for col, value in self.conditions.items():
            if isinstance(value, list) and len(value) == 2:
                # Operator format: [">=", 18]
                op, val = value
                result[col] = (op, val)
            else:
                # Simple equality
                result[col] = value
        return result


class ConfigParser:
    """Parser for LLM-generated configurations."""

    def __init__(self) -> None:
        self._last_error: Optional[str] = None

    def parse(self, response: str) -> GenerationConfig:
        """Parse LLM response into GenerationConfig.

        Args:
            response: Raw LLM response (may contain JSON)

        Returns:
            GenerationConfig instance

        Raises:
            ValidationError: If response cannot be parsed
        """
        # Extract JSON from response
        json_str = self._extract_json(response)

        if json_str is None:
            raise ValidationError(f"Could not extract JSON from response: {response[:200]}...")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON: {e}") from e

        return self._parse_config(data)

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON object from text.

        Handles:
        - Pure JSON
        - JSON in markdown code blocks
        - JSON embedded in text
        """
        text = text.strip()

        # Try parsing directly
        if text.startswith("{"):
            # Find matching closing brace
            depth = 0
            for i, c in enumerate(text):
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[: i + 1]

        # Try extracting from markdown code block
        code_block_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try finding JSON object in text
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        return None

    def _parse_config(self, data: Dict[str, Any]) -> GenerationConfig:
        """Parse validated JSON into GenerationConfig."""
        config = GenerationConfig(raw_config=data)

        # Parse schema
        schema_data = data.get("schema", {})
        columns = schema_data.get("columns", [])
        for col in columns:
            config.schema.append(
                ColumnSpec(
                    name=col.get("name", ""),
                    type=col.get("type", "text"),
                    description=col.get("description", ""),
                    nullable=col.get("nullable", True),
                )
            )

        # Parse conditions
        config.conditions = data.get("conditions", {})

        # Parse constraints
        config.constraints = data.get("constraints", [])

        # Parse generation parameters
        config.n_samples = data.get("n_samples", 1000)

        gen_config = data.get("generator_config", {})
        config.generator_method = gen_config.get("method", "auto")
        config.generator_epochs = gen_config.get("epochs", 300)

        # Parse clarification
        config.clarification_needed = data.get("clarification_needed")
        config.explanation = data.get("explanation", "")

        return config

    def validate_config(self, config: GenerationConfig) -> Tuple[bool, List[str]]:
        """Validate a generation config.

        Args:
            config: Configuration to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check n_samples
        if config.n_samples <= 0:
            errors.append("n_samples must be positive")
        if config.n_samples > 10_000_000:
            errors.append("n_samples exceeds maximum (10 million)")

        # Check generator method
        valid_methods = {"auto", "ctgan", "tvae", "gaussian_copula"}
        if config.generator_method not in valid_methods:
            errors.append(f"Invalid generator method: {config.generator_method}")

        # Check schema columns have names
        for i, col in enumerate(config.schema):
            if not col.name:
                errors.append(f"Column {i} has no name")

        # Check constraint columns exist in schema if schema is defined
        if config.schema:
            schema_cols = {c.name for c in config.schema}
            for constraint in config.constraints:
                col = constraint.get("column", "")
                if col and col not in schema_cols:
                    errors.append(f"Constraint references unknown column: {col}")

        return len(errors) == 0, errors
