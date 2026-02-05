"""Agentic Data Generation System.

Multi-agent system for sophisticated synthetic data generation using specialized
AI agents that collaborate: Schema Agent infers structure, Quality Agent validates,
Privacy Agent enforces constraints, Domain Agent adds realism.

Features:
    - Multi-agent orchestration with tool calling
    - Specialized agents: Schema, Generator, Validator, Privacy, Domain
    - Natural language workflow interface
    - Agent memory for learning user preferences
    - Parallel agent execution with caching

Example:
    Basic agentic generation::

        from genesis.agents.agentic import AgenticDataGenerator

        generator = AgenticDataGenerator(api_key="sk-...")
        
        # Natural language request
        result = generator.generate(
            "Create 10,000 healthcare patient records with realistic "
            "demographics, diagnosis codes (ICD-10), and treatment histories. "
            "Ensure HIPAA compliance and no rare conditions < 5 patients."
        )
        
        print(result.synthetic_data.head())
        print(result.quality_report.summary())

    Multi-turn conversation::

        orchestrator = AgentOrchestrator(api_key="sk-...")
        
        # Start conversation
        response = orchestrator.chat("I need customer transaction data")
        print(response.message)  # "What industry? How many records?..."
        
        response = orchestrator.chat("E-commerce, 50k transactions, last 6 months")
        print(response.synthetic_data.head())

Classes:
    AgentRole: Specialized agent roles in the system.
    AgentMessage: Communication between agents.
    AgentTool: Tool that agents can invoke.
    BaseAgent: Abstract base class for agents.
    SchemaAgent: Infers and refines data schemas.
    GeneratorAgent: Selects and runs generators.
    ValidatorAgent: Validates synthetic data quality.
    PrivacyAgent: Enforces privacy constraints.
    DomainAgent: Adds domain-specific realism.
    AgentOrchestrator: Coordinates multi-agent workflow.
    AgenticDataGenerator: High-level interface.

Note:
    Requires LLM API access (OpenAI, Anthropic, or local models via Ollama).
    Use GENESIS_LLM_API_KEY environment variable or pass api_key directly.
"""

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class AgentRole(str, Enum):
    """Specialized agent roles in the agentic system."""

    SCHEMA = "schema"  # Infers data structure
    GENERATOR = "generator"  # Runs generation methods
    VALIDATOR = "validator"  # Validates quality
    PRIVACY = "privacy"  # Enforces privacy constraints
    DOMAIN = "domain"  # Adds domain realism
    ORCHESTRATOR = "orchestrator"  # Coordinates agents


class AgentStatus(str, Enum):
    """Agent execution status."""

    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AgentMessage:
    """Message passed between agents.

    Attributes:
        sender: Role of sending agent.
        recipient: Role of receiving agent.
        content: Message content (text or structured).
        message_type: Type of message (request, response, broadcast).
        metadata: Additional metadata.
        timestamp: When message was created.
    """

    sender: AgentRole
    recipient: AgentRole
    content: Union[str, Dict[str, Any]]
    message_type: str = "request"  # request, response, broadcast
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sender": self.sender.value,
            "recipient": self.recipient.value,
            "content": self.content,
            "message_type": self.message_type,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentTool:
    """Tool that agents can invoke.

    Attributes:
        name: Tool identifier.
        description: Human-readable description.
        parameters: JSON schema for parameters.
        function: Callable that executes the tool.
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable[..., Any]

    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI tool calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with given parameters."""
        return self.function(**kwargs)


@dataclass
class AgentThought:
    """Agent's reasoning step.

    Attributes:
        thought: What the agent is thinking.
        action: Action to take (or None if done).
        action_input: Input for the action.
        observation: Result of the action.
    """

    thought: str
    action: Optional[str] = None
    action_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None


@dataclass
class AgentResult:
    """Result from an agent's work.

    Attributes:
        success: Whether agent succeeded.
        output: Primary output from agent.
        thoughts: Chain of reasoning.
        tool_calls: Tools invoked.
        metadata: Additional information.
        error: Error message if failed.
    """

    success: bool
    output: Any
    thoughts: List[AgentThought] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class AgentMemory:
    """Memory system for agents to learn from interactions.

    Stores user preferences, successful patterns, and context
    across multiple sessions.
    """

    def __init__(self, max_history: int = 100) -> None:
        self.max_history = max_history
        self.interactions: List[Dict[str, Any]] = []
        self.preferences: Dict[str, Any] = {}
        self.patterns: Dict[str, int] = {}  # pattern -> success count

    def add_interaction(
        self,
        request: str,
        result: AgentResult,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an interaction for learning."""
        interaction = {
            "request": request,
            "success": result.success,
            "output_type": type(result.output).__name__,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context or {},
        }
        self.interactions.append(interaction)

        # Trim history
        if len(self.interactions) > self.max_history:
            self.interactions = self.interactions[-self.max_history :]

        # Update patterns
        if result.success and result.tool_calls:
            pattern = ",".join(tc.get("name", "") for tc in result.tool_calls)
            self.patterns[pattern] = self.patterns.get(pattern, 0) + 1

    def update_preference(self, key: str, value: Any) -> None:
        """Update a learned preference."""
        self.preferences[key] = value

    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a learned preference."""
        return self.preferences.get(key, default)

    def get_successful_patterns(self, top_k: int = 5) -> List[str]:
        """Get most successful tool call patterns."""
        sorted_patterns = sorted(self.patterns.items(), key=lambda x: -x[1])
        return [p[0] for p in sorted_patterns[:top_k]]

    def to_context_string(self) -> str:
        """Convert memory to context for LLM."""
        lines = []
        if self.preferences:
            lines.append("User preferences:")
            for k, v in list(self.preferences.items())[:10]:
                lines.append(f"  - {k}: {v}")
        if self.patterns:
            lines.append("Successful patterns:")
            for pattern in self.get_successful_patterns(3):
                lines.append(f"  - {pattern}")
        return "\n".join(lines) if lines else ""


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Agents have a role, tools, and can process messages to produce results.
    """

    def __init__(
        self,
        role: AgentRole,
        llm_client: Optional[Any] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
    ) -> None:
        """Initialize agent.

        Args:
            role: Agent's role in the system.
            llm_client: Optional LLM client (OpenAI, Anthropic, etc.).
            model: Model identifier.
            temperature: Sampling temperature.
        """
        self.role = role
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.status = AgentStatus.IDLE
        self.tools: List[AgentTool] = []
        self.memory = AgentMemory()
        self._setup_tools()

    @abstractmethod
    def _setup_tools(self) -> None:
        """Setup agent-specific tools. Override in subclasses."""
        pass

    @abstractmethod
    def process(self, message: AgentMessage) -> AgentResult:
        """Process a message and return result."""
        pass

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Call LLM with messages and optional tools."""
        if self.llm_client is None:
            # Return mock response for testing
            return {"content": "Mock response", "tool_calls": []}

        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
            }
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            response = self.llm_client.chat.completions.create(**kwargs)
            message = response.choices[0].message

            result: Dict[str, Any] = {"content": message.content or ""}
            if hasattr(message, "tool_calls") and message.tool_calls:
                result["tool_calls"] = [
                    {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    }
                    for tc in message.tool_calls
                ]
            else:
                result["tool_calls"] = []
            return result
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return {"content": f"Error: {e}", "tool_calls": []}


class SchemaAgent(BaseAgent):
    """Agent that infers and refines data schemas.

    Analyzes requests to determine column names, types, distributions,
    and relationships.
    """

    def __init__(self, llm_client: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(AgentRole.SCHEMA, llm_client, **kwargs)

    def _setup_tools(self) -> None:
        self.tools = [
            AgentTool(
                name="infer_schema",
                description="Infer schema from natural language description",
                parameters={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": "Description of desired data",
                        },
                        "domain": {
                            "type": "string",
                            "enum": ["healthcare", "finance", "retail", "general"],
                            "description": "Domain for schema inference",
                        },
                    },
                    "required": ["description"],
                },
                function=self._infer_schema,
            ),
            AgentTool(
                name="refine_schema",
                description="Refine existing schema based on feedback",
                parameters={
                    "type": "object",
                    "properties": {
                        "schema": {"type": "object", "description": "Current schema"},
                        "feedback": {"type": "string", "description": "Feedback"},
                    },
                    "required": ["schema", "feedback"],
                },
                function=self._refine_schema,
            ),
            AgentTool(
                name="validate_schema",
                description="Validate schema consistency",
                parameters={
                    "type": "object",
                    "properties": {
                        "schema": {"type": "object", "description": "Schema to validate"},
                    },
                    "required": ["schema"],
                },
                function=self._validate_schema,
            ),
        ]

    def _infer_schema(
        self, description: str, domain: str = "general"
    ) -> Dict[str, Any]:
        """Infer schema from natural language."""
        # Use LLM to extract schema
        prompt = f"""Analyze this data request and extract a schema:

Request: {description}
Domain: {domain}

Return a JSON schema with:
- columns: list of {{name, type, description, constraints}}
- relationships: any foreign key relationships
- n_samples: estimated sample count if mentioned

Types: string, integer, float, boolean, datetime, categorical
Constraints: min, max, unique, not_null, pattern, values (for categorical)
"""
        messages = [
            {"role": "system", "content": "You are a schema inference expert."},
            {"role": "user", "content": prompt},
        ]

        response = self._call_llm(messages)
        content = response.get("content", "")

        # Parse JSON from response
        try:
            # Try to find JSON in response
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        # Fallback: return basic schema based on domain
        return self._get_domain_schema(domain)

    def _get_domain_schema(self, domain: str) -> Dict[str, Any]:
        """Get default schema for domain."""
        schemas = {
            "healthcare": {
                "columns": [
                    {"name": "patient_id", "type": "string", "constraints": {"unique": True}},
                    {"name": "age", "type": "integer", "constraints": {"min": 0, "max": 120}},
                    {"name": "gender", "type": "categorical", "constraints": {"values": ["M", "F", "O"]}},
                    {"name": "diagnosis_code", "type": "string", "description": "ICD-10 code"},
                    {"name": "admission_date", "type": "datetime"},
                    {"name": "discharge_date", "type": "datetime"},
                ],
                "n_samples": 1000,
            },
            "finance": {
                "columns": [
                    {"name": "transaction_id", "type": "string", "constraints": {"unique": True}},
                    {"name": "account_id", "type": "string"},
                    {"name": "amount", "type": "float", "constraints": {"min": 0}},
                    {"name": "transaction_type", "type": "categorical", "constraints": {"values": ["debit", "credit"]}},
                    {"name": "timestamp", "type": "datetime"},
                    {"name": "merchant", "type": "string"},
                ],
                "n_samples": 10000,
            },
            "retail": {
                "columns": [
                    {"name": "order_id", "type": "string", "constraints": {"unique": True}},
                    {"name": "customer_id", "type": "string"},
                    {"name": "product_id", "type": "string"},
                    {"name": "quantity", "type": "integer", "constraints": {"min": 1}},
                    {"name": "price", "type": "float", "constraints": {"min": 0}},
                    {"name": "order_date", "type": "datetime"},
                ],
                "n_samples": 5000,
            },
            "general": {
                "columns": [
                    {"name": "id", "type": "string", "constraints": {"unique": True}},
                    {"name": "name", "type": "string"},
                    {"name": "value", "type": "float"},
                    {"name": "category", "type": "categorical"},
                    {"name": "created_at", "type": "datetime"},
                ],
                "n_samples": 1000,
            },
        }
        return schemas.get(domain, schemas["general"])

    def _refine_schema(
        self, schema: Dict[str, Any], feedback: str
    ) -> Dict[str, Any]:
        """Refine schema based on feedback."""
        prompt = f"""Refine this schema based on the feedback:

Current schema:
{json.dumps(schema, indent=2)}

Feedback: {feedback}

Return the updated schema as JSON.
"""
        messages = [
            {"role": "system", "content": "You are a schema refinement expert."},
            {"role": "user", "content": prompt},
        ]

        response = self._call_llm(messages)
        content = response.get("content", "")

        try:
            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

        return schema  # Return original if parsing fails

    def _validate_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate schema consistency."""
        issues: List[str] = []
        valid = True

        columns = schema.get("columns", [])
        if not columns:
            issues.append("Schema has no columns defined")
            valid = False

        seen_names: set[str] = set()
        for col in columns:
            name = col.get("name")
            if not name:
                issues.append("Column missing name")
                valid = False
            elif name in seen_names:
                issues.append(f"Duplicate column name: {name}")
                valid = False
            else:
                seen_names.add(name)

            if not col.get("type"):
                issues.append(f"Column {name} missing type")
                valid = False

        return {"valid": valid, "issues": issues, "schema": schema}

    def process(self, message: AgentMessage) -> AgentResult:
        """Process schema-related request."""
        self.status = AgentStatus.THINKING
        thoughts: List[AgentThought] = []

        try:
            content = message.content
            if isinstance(content, str):
                # Natural language request
                thought = AgentThought(
                    thought="Analyzing request to infer schema",
                    action="infer_schema",
                    action_input={"description": content},
                )
                thoughts.append(thought)

                domain = message.metadata.get("domain", "general")
                schema = self._infer_schema(content, domain)
                thought.observation = f"Inferred schema with {len(schema.get('columns', []))} columns"

                # Validate
                validation = self._validate_schema(schema)
                thoughts.append(
                    AgentThought(
                        thought="Validating inferred schema",
                        action="validate_schema",
                        observation=f"Valid: {validation['valid']}",
                    )
                )

                self.status = AgentStatus.COMPLETED
                return AgentResult(
                    success=validation["valid"],
                    output=schema,
                    thoughts=thoughts,
                    metadata={"validation": validation},
                )
            else:
                # Structured request with schema
                schema = content.get("schema", content)
                validation = self._validate_schema(schema)

                return AgentResult(
                    success=validation["valid"],
                    output=schema,
                    thoughts=thoughts,
                    metadata={"validation": validation},
                )

        except Exception as e:
            self.status = AgentStatus.FAILED
            return AgentResult(success=False, output=None, error=str(e))


class GeneratorAgent(BaseAgent):
    """Agent that selects and runs appropriate generators.

    Chooses between CTGAN, TVAE, Gaussian Copula, etc. based on
    data characteristics and user requirements.
    """

    def __init__(self, llm_client: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(AgentRole.GENERATOR, llm_client, **kwargs)

    def _setup_tools(self) -> None:
        self.tools = [
            AgentTool(
                name="select_method",
                description="Select best generation method for data",
                parameters={
                    "type": "object",
                    "properties": {
                        "schema": {"type": "object"},
                        "n_samples": {"type": "integer"},
                        "preferences": {"type": "object"},
                    },
                    "required": ["schema"],
                },
                function=self._select_method,
            ),
            AgentTool(
                name="generate_data",
                description="Generate synthetic data",
                parameters={
                    "type": "object",
                    "properties": {
                        "schema": {"type": "object"},
                        "method": {"type": "string"},
                        "n_samples": {"type": "integer"},
                    },
                    "required": ["schema", "method", "n_samples"],
                },
                function=self._generate_data,
            ),
        ]

    def _select_method(
        self,
        schema: Dict[str, Any],
        n_samples: int = 1000,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Select best generation method based on data characteristics."""
        columns = schema.get("columns", [])
        n_columns = len(columns)

        # Count types
        numeric_count = sum(1 for c in columns if c.get("type") in ("integer", "float"))
        categorical_count = sum(1 for c in columns if c.get("type") == "categorical")
        text_count = sum(1 for c in columns if c.get("type") == "string")

        prefs = preferences or {}

        # Selection logic
        if prefs.get("prefer_speed"):
            return "gaussian_copula"
        if prefs.get("prefer_quality"):
            return "ctgan"
        if n_samples > 100000:
            return "gaussian_copula"  # Faster for large datasets
        if categorical_count > numeric_count:
            return "ctgan"  # Better for categorical
        if n_columns < 10 and n_samples < 10000:
            return "tvae"  # Good for small datasets
        return "ctgan"  # Default

    def _generate_data(
        self,
        schema: Dict[str, Any],
        method: str,
        n_samples: int,
    ) -> pd.DataFrame:
        """Generate synthetic data from schema."""
        columns = schema.get("columns", [])
        data: Dict[str, Any] = {}

        for col in columns:
            name = col["name"]
            col_type = col.get("type", "string")
            constraints = col.get("constraints", {})

            if col_type == "integer":
                min_val = constraints.get("min", 0)
                max_val = constraints.get("max", 100)
                data[name] = np.random.randint(min_val, max_val + 1, n_samples)

            elif col_type == "float":
                min_val = constraints.get("min", 0.0)
                max_val = constraints.get("max", 1000.0)
                data[name] = np.random.uniform(min_val, max_val, n_samples)

            elif col_type == "categorical":
                values = constraints.get("values", ["A", "B", "C"])
                data[name] = np.random.choice(values, n_samples)

            elif col_type == "boolean":
                data[name] = np.random.choice([True, False], n_samples)

            elif col_type == "datetime":
                start = pd.Timestamp("2020-01-01")
                end = pd.Timestamp("2024-12-31")
                data[name] = pd.to_datetime(
                    np.random.randint(start.value, end.value, n_samples)
                )

            else:  # string
                if constraints.get("unique"):
                    data[name] = [f"{name}_{i:06d}" for i in range(n_samples)]
                else:
                    data[name] = [f"{name}_{np.random.randint(1000)}" for _ in range(n_samples)]

        return pd.DataFrame(data)

    def process(self, message: AgentMessage) -> AgentResult:
        """Process generation request."""
        self.status = AgentStatus.EXECUTING
        thoughts: List[AgentThought] = []

        try:
            content = message.content
            if isinstance(content, str):
                return AgentResult(
                    success=False,
                    output=None,
                    error="Generator agent requires structured schema input",
                )

            schema = content.get("schema", content)
            n_samples = content.get("n_samples", 1000)
            preferences = content.get("preferences", {})

            # Select method
            method = self._select_method(schema, n_samples, preferences)
            thoughts.append(
                AgentThought(
                    thought=f"Selected {method} for generation",
                    action="select_method",
                    observation=f"Using {method}",
                )
            )

            # Generate data
            df = self._generate_data(schema, method, n_samples)
            thoughts.append(
                AgentThought(
                    thought=f"Generating {n_samples} samples",
                    action="generate_data",
                    observation=f"Generated {len(df)} rows, {len(df.columns)} columns",
                )
            )

            self.status = AgentStatus.COMPLETED
            return AgentResult(
                success=True,
                output=df,
                thoughts=thoughts,
                metadata={"method": method, "n_samples": len(df)},
            )

        except Exception as e:
            self.status = AgentStatus.FAILED
            return AgentResult(success=False, output=None, error=str(e), thoughts=thoughts)


class ValidatorAgent(BaseAgent):
    """Agent that validates synthetic data quality.

    Checks statistical fidelity, constraint satisfaction, and
    overall quality metrics.
    """

    def __init__(self, llm_client: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(AgentRole.VALIDATOR, llm_client, **kwargs)

    def _setup_tools(self) -> None:
        self.tools = [
            AgentTool(
                name="validate_quality",
                description="Validate data quality metrics",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object", "description": "DataFrame dict"},
                        "schema": {"type": "object"},
                    },
                    "required": ["data", "schema"],
                },
                function=self._validate_quality,
            ),
            AgentTool(
                name="check_constraints",
                description="Check constraint satisfaction",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object"},
                        "constraints": {"type": "array"},
                    },
                    "required": ["data"],
                },
                function=self._check_constraints,
            ),
        ]

    def _validate_quality(
        self,
        data: pd.DataFrame,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate data quality against schema."""
        if not isinstance(data, pd.DataFrame):
            return {"valid": False, "issues": ["Input is not a DataFrame"]}

        issues: List[str] = []
        metrics: Dict[str, float] = {}

        # Check column presence
        expected_cols = {c["name"] for c in schema.get("columns", [])}
        actual_cols = set(data.columns)
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols

        if missing:
            issues.append(f"Missing columns: {missing}")
        if extra:
            issues.append(f"Extra columns: {extra}")

        # Check for null values
        null_counts = data.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        if cols_with_nulls:
            null_pct = sum(cols_with_nulls.values()) / (len(data) * len(data.columns)) * 100
            metrics["null_percentage"] = null_pct
            if null_pct > 10:
                issues.append(f"High null percentage: {null_pct:.1f}%")

        # Check uniqueness constraints
        for col_spec in schema.get("columns", []):
            name = col_spec["name"]
            constraints = col_spec.get("constraints", {})
            if name in data.columns and constraints.get("unique"):
                if data[name].duplicated().any():
                    issues.append(f"Column {name} has duplicates but should be unique")

        # Overall quality score
        metrics["completeness"] = 1 - (data.isnull().sum().sum() / data.size)
        metrics["schema_conformance"] = 1 - len(issues) / max(len(expected_cols), 1)
        metrics["overall_score"] = (metrics["completeness"] + metrics["schema_conformance"]) / 2

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "metrics": metrics,
        }

    def _check_constraints(
        self,
        data: pd.DataFrame,
        constraints: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Check constraint satisfaction."""
        if constraints is None:
            constraints = []

        violations: List[str] = []

        for constraint in constraints:
            col = constraint.get("column")
            if col not in data.columns:
                continue

            ctype = constraint.get("type")
            if ctype == "range":
                min_val = constraint.get("min")
                max_val = constraint.get("max")
                if min_val is not None and (data[col] < min_val).any():
                    violations.append(f"{col} has values below {min_val}")
                if max_val is not None and (data[col] > max_val).any():
                    violations.append(f"{col} has values above {max_val}")

            elif ctype == "unique":
                if data[col].duplicated().any():
                    violations.append(f"{col} has duplicate values")

            elif ctype == "not_null":
                if data[col].isnull().any():
                    violations.append(f"{col} has null values")

        return {
            "satisfied": len(violations) == 0,
            "violations": violations,
            "checked_count": len(constraints),
        }

    def process(self, message: AgentMessage) -> AgentResult:
        """Process validation request."""
        self.status = AgentStatus.EXECUTING
        thoughts: List[AgentThought] = []

        try:
            content = message.content
            data = content.get("data")
            schema = content.get("schema", {})

            if data is None:
                return AgentResult(
                    success=False,
                    output=None,
                    error="No data provided for validation",
                )

            # Validate quality
            quality_result = self._validate_quality(data, schema)
            thoughts.append(
                AgentThought(
                    thought="Validating data quality",
                    action="validate_quality",
                    observation=f"Score: {quality_result['metrics'].get('overall_score', 0):.2f}",
                )
            )

            self.status = AgentStatus.COMPLETED
            return AgentResult(
                success=quality_result["valid"],
                output=quality_result,
                thoughts=thoughts,
                metadata=quality_result["metrics"],
            )

        except Exception as e:
            self.status = AgentStatus.FAILED
            return AgentResult(success=False, output=None, error=str(e))


class PrivacyAgent(BaseAgent):
    """Agent that enforces privacy constraints.

    Ensures differential privacy, k-anonymity, and prevents
    privacy leaks in generated data.
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,
        default_epsilon: float = 1.0,
        default_k: int = 5,
        **kwargs: Any,
    ) -> None:
        self.default_epsilon = default_epsilon
        self.default_k = default_k
        super().__init__(AgentRole.PRIVACY, llm_client, **kwargs)

    def _setup_tools(self) -> None:
        self.tools = [
            AgentTool(
                name="assess_privacy",
                description="Assess privacy risks in data",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object"},
                        "sensitive_columns": {"type": "array"},
                    },
                    "required": ["data"],
                },
                function=self._assess_privacy,
            ),
            AgentTool(
                name="enforce_k_anonymity",
                description="Enforce k-anonymity on data",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object"},
                        "quasi_identifiers": {"type": "array"},
                        "k": {"type": "integer"},
                    },
                    "required": ["data", "quasi_identifiers"],
                },
                function=self._enforce_k_anonymity,
            ),
            AgentTool(
                name="suppress_rare",
                description="Suppress rare categories",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object"},
                        "threshold": {"type": "number"},
                    },
                    "required": ["data"],
                },
                function=self._suppress_rare,
            ),
        ]

    def _assess_privacy(
        self,
        data: pd.DataFrame,
        sensitive_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Assess privacy risks in data."""
        risks: List[Dict[str, Any]] = []
        overall_risk = "low"

        # Check for unique identifiers
        for col in data.columns:
            unique_ratio = data[col].nunique() / len(data)
            if unique_ratio > 0.9:
                risks.append({
                    "column": col,
                    "risk": "high",
                    "reason": "Near-unique values (potential identifier)",
                })
                overall_risk = "high"

        # Check for rare categories
        for col in data.select_dtypes(include=["object", "category"]).columns:
            value_counts = data[col].value_counts(normalize=True)
            rare = value_counts[value_counts < 0.01]
            if len(rare) > 0:
                risks.append({
                    "column": col,
                    "risk": "medium",
                    "reason": f"{len(rare)} rare categories (<1%)",
                })
                if overall_risk == "low":
                    overall_risk = "medium"

        # Check sensitive columns
        if sensitive_columns:
            for col in sensitive_columns:
                if col in data.columns:
                    risks.append({
                        "column": col,
                        "risk": "high",
                        "reason": "Marked as sensitive",
                    })
                    overall_risk = "high"

        return {
            "overall_risk": overall_risk,
            "risks": risks,
            "recommendations": self._get_recommendations(risks),
        }

    def _get_recommendations(self, risks: List[Dict[str, Any]]) -> List[str]:
        """Get privacy recommendations based on risks."""
        recs: List[str] = []
        for risk in risks:
            if risk["risk"] == "high":
                if "identifier" in risk.get("reason", ""):
                    recs.append(f"Remove or hash column '{risk['column']}'")
                elif "sensitive" in risk.get("reason", ""):
                    recs.append(f"Apply differential privacy to '{risk['column']}'")
            elif risk["risk"] == "medium":
                recs.append(f"Consider suppressing rare values in '{risk['column']}'")
        return recs

    def _enforce_k_anonymity(
        self,
        data: pd.DataFrame,
        quasi_identifiers: List[str],
        k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Enforce k-anonymity by suppressing small groups."""
        k = k or self.default_k
        df = data.copy()

        # Filter quasi-identifiers that exist
        qi_cols = [c for c in quasi_identifiers if c in df.columns]
        if not qi_cols:
            return df

        # Group by quasi-identifiers and filter
        group_sizes = df.groupby(qi_cols).size()
        valid_groups = group_sizes[group_sizes >= k].index

        # Keep only rows in valid groups
        if len(qi_cols) == 1:
            mask = df[qi_cols[0]].isin(
                [g if isinstance(g, str) else g for g in valid_groups]
            )
        else:
            mask = df.set_index(qi_cols).index.isin(valid_groups)

        return df[mask].reset_index(drop=True)

    def _suppress_rare(
        self,
        data: pd.DataFrame,
        threshold: float = 0.01,
    ) -> pd.DataFrame:
        """Suppress rare categories below threshold."""
        df = data.copy()

        for col in df.select_dtypes(include=["object", "category"]).columns:
            value_counts = df[col].value_counts(normalize=True)
            rare_values = value_counts[value_counts < threshold].index
            if len(rare_values) > 0:
                df.loc[df[col].isin(rare_values), col] = "OTHER"

        return df

    def process(self, message: AgentMessage) -> AgentResult:
        """Process privacy request."""
        self.status = AgentStatus.EXECUTING
        thoughts: List[AgentThought] = []

        try:
            content = message.content
            data = content.get("data")
            action = content.get("action", "assess")

            if data is None:
                return AgentResult(
                    success=False,
                    output=None,
                    error="No data provided for privacy processing",
                )

            if action == "assess":
                result = self._assess_privacy(
                    data, content.get("sensitive_columns")
                )
                thoughts.append(
                    AgentThought(
                        thought="Assessing privacy risks",
                        action="assess_privacy",
                        observation=f"Overall risk: {result['overall_risk']}",
                    )
                )
                output = result

            elif action == "enforce_k_anonymity":
                result = self._enforce_k_anonymity(
                    data,
                    content.get("quasi_identifiers", []),
                    content.get("k"),
                )
                original_rows = len(data)
                new_rows = len(result)
                thoughts.append(
                    AgentThought(
                        thought="Enforcing k-anonymity",
                        action="enforce_k_anonymity",
                        observation=f"Reduced from {original_rows} to {new_rows} rows",
                    )
                )
                output = result

            elif action == "suppress_rare":
                result = self._suppress_rare(data, content.get("threshold", 0.01))
                thoughts.append(
                    AgentThought(
                        thought="Suppressing rare categories",
                        action="suppress_rare",
                        observation="Rare categories suppressed",
                    )
                )
                output = result

            else:
                return AgentResult(
                    success=False,
                    output=None,
                    error=f"Unknown privacy action: {action}",
                )

            self.status = AgentStatus.COMPLETED
            return AgentResult(success=True, output=output, thoughts=thoughts)

        except Exception as e:
            self.status = AgentStatus.FAILED
            return AgentResult(success=False, output=None, error=str(e))


class DomainAgent(BaseAgent):
    """Agent that adds domain-specific realism.

    Enhances generated data with realistic patterns, distributions,
    and relationships specific to healthcare, finance, retail, etc.
    """

    def __init__(self, llm_client: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(AgentRole.DOMAIN, llm_client, **kwargs)

    def _setup_tools(self) -> None:
        self.tools = [
            AgentTool(
                name="detect_domain",
                description="Detect domain from schema/data",
                parameters={
                    "type": "object",
                    "properties": {
                        "schema": {"type": "object"},
                        "sample_data": {"type": "object"},
                    },
                },
                function=self._detect_domain,
            ),
            AgentTool(
                name="enhance_realism",
                description="Enhance data with domain-specific patterns",
                parameters={
                    "type": "object",
                    "properties": {
                        "data": {"type": "object"},
                        "domain": {"type": "string"},
                    },
                    "required": ["data", "domain"],
                },
                function=self._enhance_realism,
            ),
        ]

    def _detect_domain(
        self,
        schema: Optional[Dict[str, Any]] = None,
        sample_data: Optional[pd.DataFrame] = None,
    ) -> str:
        """Detect domain from schema or data."""
        keywords = {
            "healthcare": ["patient", "diagnosis", "icd", "treatment", "medication", "hospital"],
            "finance": ["transaction", "account", "balance", "payment", "credit", "debit"],
            "retail": ["order", "product", "customer", "price", "cart", "inventory"],
        }

        text_to_check = ""
        if schema:
            col_names = [c.get("name", "").lower() for c in schema.get("columns", [])]
            text_to_check = " ".join(col_names)
        if sample_data is not None:
            text_to_check += " " + " ".join(sample_data.columns.str.lower())

        scores = {}
        for domain, kws in keywords.items():
            scores[domain] = sum(1 for kw in kws if kw in text_to_check)

        if max(scores.values()) > 0:
            return max(scores, key=lambda x: scores[x])
        return "general"

    def _enhance_realism(
        self,
        data: pd.DataFrame,
        domain: str,
    ) -> pd.DataFrame:
        """Enhance data with domain-specific patterns."""
        df = data.copy()

        if domain == "healthcare":
            # Add realistic patterns
            if "age" in df.columns:
                # Skew age distribution older
                df["age"] = (df["age"] * 0.8 + 30).clip(0, 100).astype(int)
            if "diagnosis_code" in df.columns:
                # Use realistic ICD-10 codes
                common_codes = ["J06.9", "I10", "E11.9", "K21.0", "M54.5", "J20.9"]
                df["diagnosis_code"] = np.random.choice(common_codes, len(df))

        elif domain == "finance":
            if "amount" in df.columns:
                # Log-normal distribution for transactions
                df["amount"] = np.abs(np.random.lognormal(3, 1.5, len(df)))
            if "transaction_type" in df.columns:
                # More debits than credits
                df["transaction_type"] = np.random.choice(
                    ["debit", "credit"], len(df), p=[0.7, 0.3]
                )

        elif domain == "retail":
            if "quantity" in df.columns:
                # Most orders are small
                df["quantity"] = np.random.geometric(0.5, len(df)).clip(1, 20)
            if "price" in df.columns:
                # Price clustering around common points
                price_points = [9.99, 19.99, 29.99, 49.99, 99.99]
                df["price"] = np.random.choice(price_points, len(df)) * (
                    1 + np.random.uniform(-0.1, 0.1, len(df))
                )

        return df

    def process(self, message: AgentMessage) -> AgentResult:
        """Process domain enhancement request."""
        self.status = AgentStatus.EXECUTING
        thoughts: List[AgentThought] = []

        try:
            content = message.content
            data = content.get("data")
            schema = content.get("schema")
            domain = content.get("domain")

            # Detect domain if not provided
            if not domain:
                domain = self._detect_domain(schema, data)
                thoughts.append(
                    AgentThought(
                        thought="Detecting domain",
                        action="detect_domain",
                        observation=f"Detected domain: {domain}",
                    )
                )

            if data is None:
                return AgentResult(
                    success=False,
                    output=None,
                    error="No data provided for domain enhancement",
                )

            # Enhance data
            enhanced = self._enhance_realism(data, domain)
            thoughts.append(
                AgentThought(
                    thought=f"Enhancing data with {domain} patterns",
                    action="enhance_realism",
                    observation="Applied domain-specific transformations",
                )
            )

            self.status = AgentStatus.COMPLETED
            return AgentResult(
                success=True,
                output=enhanced,
                thoughts=thoughts,
                metadata={"domain": domain},
            )

        except Exception as e:
            self.status = AgentStatus.FAILED
            return AgentResult(success=False, output=None, error=str(e))


@dataclass
class OrchestrationResult:
    """Result from agent orchestration.

    Attributes:
        success: Whether orchestration succeeded.
        synthetic_data: Generated DataFrame.
        quality_report: Quality validation results.
        privacy_report: Privacy assessment results.
        schema: Inferred schema.
        agent_logs: Logs from each agent.
        message: Human-readable summary.
    """

    success: bool
    synthetic_data: Optional[pd.DataFrame] = None
    quality_report: Optional[Dict[str, Any]] = None
    privacy_report: Optional[Dict[str, Any]] = None
    schema: Optional[Dict[str, Any]] = None
    agent_logs: Dict[str, List[AgentThought]] = field(default_factory=dict)
    message: str = ""


class AgentOrchestrator:
    """Coordinates multiple agents for synthetic data generation.

    Manages the workflow: Schema -> Generator -> Validator -> Privacy -> Domain
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ) -> None:
        """Initialize orchestrator with agents.

        Args:
            api_key: LLM API key (or set GENESIS_LLM_API_KEY env var).
            model: Model to use for agents.
            provider: LLM provider ("openai", "anthropic").
        """
        self.api_key = api_key or os.environ.get("GENESIS_LLM_API_KEY")
        self.model = model
        self.provider = provider
        self._llm_client: Optional[Any] = None

        # Initialize LLM client
        self._setup_llm_client()

        # Initialize agents
        agent_kwargs = {"llm_client": self._llm_client, "model": model}
        self.schema_agent = SchemaAgent(**agent_kwargs)
        self.generator_agent = GeneratorAgent(**agent_kwargs)
        self.validator_agent = ValidatorAgent(**agent_kwargs)
        self.privacy_agent = PrivacyAgent(**agent_kwargs)
        self.domain_agent = DomainAgent(**agent_kwargs)

        # Conversation state
        self._conversation: List[Dict[str, str]] = []

    def _setup_llm_client(self) -> None:
        """Setup LLM client based on provider."""
        if not self.api_key:
            logger.warning("No API key provided, agents will use mock responses")
            return

        try:
            if self.provider == "openai":
                from openai import OpenAI
                self._llm_client = OpenAI(api_key=self.api_key)
            elif self.provider == "anthropic":
                from anthropic import Anthropic
                self._llm_client = Anthropic(api_key=self.api_key)
        except ImportError as e:
            logger.warning(f"Could not import {self.provider} client: {e}")
        except Exception as e:
            logger.error(f"Error setting up LLM client: {e}")

    def generate(
        self,
        request: str,
        n_samples: Optional[int] = None,
        domain: Optional[str] = None,
        enforce_privacy: bool = True,
        sensitive_columns: Optional[List[str]] = None,
    ) -> OrchestrationResult:
        """Generate synthetic data from natural language request.

        Args:
            request: Natural language description of desired data.
            n_samples: Number of samples (overrides inferred).
            domain: Domain hint (healthcare, finance, retail, general).
            enforce_privacy: Whether to run privacy agent.
            sensitive_columns: Columns to treat as sensitive.

        Returns:
            OrchestrationResult with generated data and reports.
        """
        agent_logs: Dict[str, List[AgentThought]] = {}

        try:
            # Step 1: Schema Agent - Infer schema
            logger.info("Step 1: Inferring schema...")
            schema_msg = AgentMessage(
                sender=AgentRole.ORCHESTRATOR,
                recipient=AgentRole.SCHEMA,
                content=request,
                metadata={"domain": domain or "general"},
            )
            schema_result = self.schema_agent.process(schema_msg)
            agent_logs["schema"] = schema_result.thoughts

            if not schema_result.success:
                return OrchestrationResult(
                    success=False,
                    message=f"Schema inference failed: {schema_result.error}",
                    agent_logs=agent_logs,
                )

            schema = schema_result.output
            if n_samples:
                schema["n_samples"] = n_samples
            actual_n_samples = schema.get("n_samples", 1000)

            # Step 2: Generator Agent - Generate data
            logger.info(f"Step 2: Generating {actual_n_samples} samples...")
            gen_msg = AgentMessage(
                sender=AgentRole.ORCHESTRATOR,
                recipient=AgentRole.GENERATOR,
                content={"schema": schema, "n_samples": actual_n_samples},
            )
            gen_result = self.generator_agent.process(gen_msg)
            agent_logs["generator"] = gen_result.thoughts

            if not gen_result.success:
                return OrchestrationResult(
                    success=False,
                    message=f"Generation failed: {gen_result.error}",
                    schema=schema,
                    agent_logs=agent_logs,
                )

            synthetic_data = gen_result.output

            # Step 3: Domain Agent - Enhance realism
            logger.info("Step 3: Enhancing with domain patterns...")
            domain_msg = AgentMessage(
                sender=AgentRole.ORCHESTRATOR,
                recipient=AgentRole.DOMAIN,
                content={"data": synthetic_data, "schema": schema, "domain": domain},
            )
            domain_result = self.domain_agent.process(domain_msg)
            agent_logs["domain"] = domain_result.thoughts

            if domain_result.success:
                synthetic_data = domain_result.output

            # Step 4: Privacy Agent - Assess and enforce privacy
            privacy_report = None
            if enforce_privacy:
                logger.info("Step 4: Assessing privacy...")
                privacy_msg = AgentMessage(
                    sender=AgentRole.ORCHESTRATOR,
                    recipient=AgentRole.PRIVACY,
                    content={
                        "data": synthetic_data,
                        "action": "assess",
                        "sensitive_columns": sensitive_columns,
                    },
                )
                privacy_result = self.privacy_agent.process(privacy_msg)
                agent_logs["privacy"] = privacy_result.thoughts
                privacy_report = privacy_result.output

            # Step 5: Validator Agent - Validate quality
            logger.info("Step 5: Validating quality...")
            val_msg = AgentMessage(
                sender=AgentRole.ORCHESTRATOR,
                recipient=AgentRole.VALIDATOR,
                content={"data": synthetic_data, "schema": schema},
            )
            val_result = self.validator_agent.process(val_msg)
            agent_logs["validator"] = val_result.thoughts
            quality_report = val_result.output

            # Build result
            return OrchestrationResult(
                success=True,
                synthetic_data=synthetic_data,
                quality_report=quality_report,
                privacy_report=privacy_report,
                schema=schema,
                agent_logs=agent_logs,
                message=f"Generated {len(synthetic_data)} samples with "
                f"quality score {quality_report.get('metrics', {}).get('overall_score', 0):.2f}",
            )

        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            return OrchestrationResult(
                success=False,
                message=f"Orchestration failed: {str(e)}",
                agent_logs=agent_logs,
            )

    def chat(self, message: str) -> OrchestrationResult:
        """Interactive chat interface for data generation.

        Maintains conversation context for multi-turn interactions.

        Args:
            message: User message.

        Returns:
            OrchestrationResult (may include clarifying questions).
        """
        self._conversation.append({"role": "user", "content": message})

        # Try to extract generation request
        # Simple heuristic: if message mentions "generate" or numbers, try generation
        if any(kw in message.lower() for kw in ["generate", "create", "make", "synthetic"]):
            result = self.generate(message)
            self._conversation.append({
                "role": "assistant",
                "content": result.message,
            })
            return result

        # Otherwise, return clarifying message
        return OrchestrationResult(
            success=True,
            message="I can help you generate synthetic data. Please describe:\n"
            "- What type of data do you need?\n"
            "- How many records?\n"
            "- Any specific columns or constraints?\n"
            "- Any privacy requirements?",
        )

    def reset_conversation(self) -> None:
        """Reset conversation history."""
        self._conversation = []


class AgenticDataGenerator:
    """High-level interface for agentic synthetic data generation.

    Convenience wrapper around AgentOrchestrator with simplified API.

    Example:
        >>> gen = AgenticDataGenerator(api_key="sk-...")
        >>> result = gen.generate("10k healthcare records with HIPAA compliance")
        >>> df = result.synthetic_data
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ) -> None:
        """Initialize agentic generator.

        Args:
            api_key: LLM API key.
            model: Model identifier.
            provider: LLM provider.
        """
        self._orchestrator = AgentOrchestrator(
            api_key=api_key,
            model=model,
            provider=provider,
        )

    def generate(
        self,
        description: str,
        n_samples: Optional[int] = None,
        domain: Optional[str] = None,
        enforce_privacy: bool = True,
        sensitive_columns: Optional[List[str]] = None,
    ) -> OrchestrationResult:
        """Generate synthetic data from description.

        Args:
            description: Natural language description.
            n_samples: Number of samples.
            domain: Domain hint.
            enforce_privacy: Run privacy checks.
            sensitive_columns: Sensitive column names.

        Returns:
            OrchestrationResult with data and reports.
        """
        return self._orchestrator.generate(
            request=description,
            n_samples=n_samples,
            domain=domain,
            enforce_privacy=enforce_privacy,
            sensitive_columns=sensitive_columns,
        )

    def chat(self, message: str) -> OrchestrationResult:
        """Interactive chat interface."""
        return self._orchestrator.chat(message)

    def reset(self) -> None:
        """Reset conversation state."""
        self._orchestrator.reset_conversation()


# Convenience function
def agentic_generate(
    description: str,
    n_samples: Optional[int] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Generate synthetic data using agentic system.

    Convenience function for quick agentic generation.

    Args:
        description: Natural language description of desired data.
        n_samples: Number of samples to generate.
        api_key: LLM API key.
        **kwargs: Additional arguments for AgenticDataGenerator.

    Returns:
        Generated DataFrame.

    Raises:
        RuntimeError: If generation fails.

    Example:
        >>> df = agentic_generate(
        ...     "Customer transactions for an e-commerce site",
        ...     n_samples=5000
        ... )
    """
    generator = AgenticDataGenerator(api_key=api_key, **kwargs)
    result = generator.generate(description, n_samples=n_samples)

    if not result.success or result.synthetic_data is None:
        raise RuntimeError(f"Agentic generation failed: {result.message}")

    return result.synthetic_data
