"""Tests for the Synthetic Data Agent."""

import json

import numpy as np
import pandas as pd
import pytest

from genesis.agents.parser import ColumnSpec, ConfigParser, GenerationConfig
from genesis.agents.prompts import PromptTemplates
from genesis.agents.synthetic_agent import AgentResponse, SyntheticDataAgent


class TestConfigParser:
    """Tests for the configuration parser."""

    def test_parse_simple_json(self) -> None:
        """Test parsing simple JSON response."""
        response = json.dumps(
            {
                "schema": {
                    "columns": [
                        {"name": "age", "type": "numeric", "description": "Age in years"},
                        {"name": "name", "type": "text", "description": "Full name"},
                    ]
                },
                "conditions": {},
                "constraints": [],
                "n_samples": 1000,
                "generator_config": {"method": "auto", "epochs": 300},
                "clarification_needed": None,
                "explanation": "Simple dataset generation",
            }
        )

        parser = ConfigParser()
        config = parser.parse(response)

        assert config.n_samples == 1000
        assert len(config.schema) == 2
        assert config.schema[0].name == "age"
        assert config.generator_method == "auto"
        assert config.is_complete

    def test_parse_json_in_markdown(self) -> None:
        """Test parsing JSON wrapped in markdown code block."""
        response = """Here's the configuration:

```json
{
    "schema": {"columns": []},
    "conditions": {},
    "constraints": [],
    "n_samples": 500,
    "generator_config": {"method": "ctgan"},
    "clarification_needed": null,
    "explanation": "Config ready"
}
```

Let me know if you need changes."""

        parser = ConfigParser()
        config = parser.parse(response)

        assert config.n_samples == 500
        assert config.generator_method == "ctgan"

    def test_parse_with_conditions(self) -> None:
        """Test parsing configuration with conditions."""
        response = json.dumps(
            {
                "schema": {"columns": []},
                "conditions": {
                    "age": [">=", 18],
                    "country": "US",
                    "income": ["between", [30000, 100000]],
                },
                "constraints": [],
                "n_samples": 1000,
                "generator_config": {"method": "auto"},
                "clarification_needed": None,
                "explanation": "",
            }
        )

        parser = ConfigParser()
        config = parser.parse(response)

        assert config.conditions["age"] == [">=", 18]
        assert config.conditions["country"] == "US"

        # Convert to conditions dict
        conditions_dict = config.to_conditions_dict()
        assert conditions_dict["age"] == (">=", 18)
        assert conditions_dict["country"] == "US"

    def test_parse_with_constraints(self) -> None:
        """Test parsing configuration with constraints."""
        response = json.dumps(
            {
                "schema": {
                    "columns": [
                        {"name": "price", "type": "numeric"},
                        {"name": "quantity", "type": "numeric"},
                    ]
                },
                "conditions": {},
                "constraints": [
                    {"type": "positive", "column": "price"},
                    {"type": "range", "column": "quantity", "params": {"min": 1, "max": 100}},
                ],
                "n_samples": 1000,
                "generator_config": {"method": "auto"},
                "clarification_needed": None,
                "explanation": "",
            }
        )

        parser = ConfigParser()
        config = parser.parse(response)

        assert len(config.constraints) == 2
        constraints = config.to_constraints()
        assert len(constraints) == 2

    def test_parse_with_clarification(self) -> None:
        """Test parsing response that needs clarification."""
        response = json.dumps(
            {
                "schema": {"columns": []},
                "conditions": {},
                "constraints": [],
                "n_samples": 1000,
                "generator_config": {"method": "auto"},
                "clarification_needed": "How many categories should the 'status' column have?",
                "explanation": "Need more information",
            }
        )

        parser = ConfigParser()
        config = parser.parse(response)

        assert not config.is_complete
        assert config.clarification_needed == "How many categories should the 'status' column have?"

    def test_validate_config_valid(self) -> None:
        """Test validation of valid config."""
        config = GenerationConfig(
            n_samples=1000,
            generator_method="auto",
            schema=[ColumnSpec("col1", "numeric")],
        )

        parser = ConfigParser()
        is_valid, errors = parser.validate_config(config)

        assert is_valid
        assert len(errors) == 0

    def test_validate_config_invalid_samples(self) -> None:
        """Test validation catches invalid n_samples."""
        config = GenerationConfig(n_samples=-1)

        parser = ConfigParser()
        is_valid, errors = parser.validate_config(config)

        assert not is_valid
        assert any("n_samples" in e for e in errors)

    def test_validate_config_invalid_method(self) -> None:
        """Test validation catches invalid generator method."""
        config = GenerationConfig(generator_method="invalid_method")

        parser = ConfigParser()
        is_valid, errors = parser.validate_config(config)

        assert not is_valid
        assert any("method" in e for e in errors)


class TestPromptTemplates:
    """Tests for prompt templates."""

    def test_get_parse_prompt_basic(self) -> None:
        """Test basic prompt generation."""
        prompt = PromptTemplates.get_parse_prompt(
            request="Generate 1000 customer records",
        )

        assert "Generate 1000 customer records" in prompt
        assert "No additional context" in prompt

    def test_get_parse_prompt_with_schema(self) -> None:
        """Test prompt generation with schema context."""
        schema = {
            "columns": [
                {"name": "id", "type": "numeric"},
                {"name": "name", "type": "text"},
            ]
        }

        prompt = PromptTemplates.get_parse_prompt(
            request="Generate more data",
            schema=schema,
        )

        assert "Current schema information" in prompt

    def test_get_parse_prompt_with_sample(self) -> None:
        """Test prompt generation with sample data."""
        prompt = PromptTemplates.get_parse_prompt(
            request="Generate similar data",
            sample_data="   id  name\n0   1  John\n1   2  Jane",
        )

        assert "Sample of the data" in prompt

    def test_get_refine_prompt(self) -> None:
        """Test refine prompt generation."""
        prompt = PromptTemplates.get_refine_prompt(
            original_request="Generate customer data",
            question="How many categories?",
            answer="3 categories: Bronze, Silver, Gold",
            previous_config={"n_samples": 1000},
        )

        assert "Generate customer data" in prompt
        assert "How many categories?" in prompt
        assert "3 categories" in prompt
        assert "1000" in prompt


class TestSyntheticDataAgent:
    """Tests for the SyntheticDataAgent class."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample base data."""
        np.random.seed(42)
        return pd.DataFrame(
            {
                "age": np.random.randint(18, 80, 100),
                "income": np.random.normal(50000, 20000, 100),
                "category": np.random.choice(["A", "B", "C"], 100),
            }
        )

    def test_init_with_base_data(self, sample_data: pd.DataFrame) -> None:
        """Test agent initialization with base data."""
        agent = SyntheticDataAgent(base_data=sample_data)

        assert agent._schema_info is not None
        assert len(agent._schema_info["columns"]) == 3
        assert agent._sample_str is not None

    def test_infer_type(self, sample_data: pd.DataFrame) -> None:
        """Test type inference."""
        agent = SyntheticDataAgent(base_data=sample_data)

        schema = agent._schema_info
        col_types = {c["name"]: c["type"] for c in schema["columns"]}

        assert col_types["age"] == "numeric"
        assert col_types["income"] == "numeric"
        assert col_types["category"] == "categorical"

    def test_reset_clears_state(self, sample_data: pd.DataFrame) -> None:
        """Test that reset clears agent state."""
        agent = SyntheticDataAgent(base_data=sample_data)
        agent._state.original_request = "test"
        agent._state.is_complete = True

        agent.reset()

        assert agent._state.original_request is None
        assert not agent._state.is_complete
        assert len(agent.history) == 0


class TestAgentResponse:
    """Tests for AgentResponse class."""

    def test_repr_clarification(self) -> None:
        """Test repr when clarification needed."""
        config = GenerationConfig(
            clarification_needed="What format?",
        )
        agent = SyntheticDataAgent()
        response = AgentResponse(
            config=config,
            needs_clarification=True,
            clarification_question="What format?",
            explanation="",
            data=None,
            agent=agent,
        )

        repr_str = repr(response)
        assert "needs_clarification=True" in repr_str
        assert "What format?" in repr_str

    def test_repr_config_ready(self) -> None:
        """Test repr when config is ready."""
        config = GenerationConfig(n_samples=5000)
        agent = SyntheticDataAgent()
        response = AgentResponse(
            config=config,
            needs_clarification=False,
            clarification_question=None,
            explanation="Ready",
            data=None,
            agent=agent,
        )

        repr_str = repr(response)
        assert "config_ready=True" in repr_str
        assert "5000" in repr_str

    def test_repr_with_data(self) -> None:
        """Test repr when data is generated."""
        config = GenerationConfig(n_samples=100)
        agent = SyntheticDataAgent()
        data = pd.DataFrame({"a": range(100), "b": range(100)})
        response = AgentResponse(
            config=config,
            needs_clarification=False,
            clarification_question=None,
            explanation="Done",
            data=data,
            agent=agent,
        )

        repr_str = repr(response)
        assert "data=" in repr_str
        assert "(100, 2)" in repr_str


class TestColumnSpec:
    """Tests for ColumnSpec dataclass."""

    def test_to_dict(self) -> None:
        """Test column spec serialization."""
        col = ColumnSpec(
            name="age",
            type="numeric",
            description="Age in years",
            nullable=False,
        )

        d = col.to_dict()

        assert d["name"] == "age"
        assert d["type"] == "numeric"
        assert d["description"] == "Age in years"
        assert d["nullable"] is False


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_is_complete_true(self) -> None:
        """Test is_complete when no clarification needed."""
        config = GenerationConfig(clarification_needed=None)
        assert config.is_complete

    def test_is_complete_false(self) -> None:
        """Test is_complete when clarification needed."""
        config = GenerationConfig(clarification_needed="What format?")
        assert not config.is_complete

    def test_to_generator_config(self) -> None:
        """Test conversion to GeneratorConfig."""
        config = GenerationConfig(
            generator_method="ctgan",
            generator_epochs=500,
        )

        gen_config = config.to_generator_config()

        # Method is an enum, compare to string value
        assert gen_config.method.value == "ctgan" or gen_config.method == "ctgan"
        assert gen_config.epochs == 500
