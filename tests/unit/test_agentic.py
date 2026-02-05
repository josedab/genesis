"""Tests for Agentic Data Generation system."""

import pytest
import pandas as pd
import numpy as np

from genesis.agents.agentic import (
    AgentMessage,
    AgentOrchestrator,
    AgentResult,
    AgentRole,
    AgentStatus,
    AgentThought,
    AgentTool,
    AgenticDataGenerator,
    AgentMemory,
    BaseAgent,
    DomainAgent,
    GeneratorAgent,
    OrchestrationResult,
    PrivacyAgent,
    SchemaAgent,
    ValidatorAgent,
    agentic_generate,
)


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_create_message(self):
        msg = AgentMessage(
            sender=AgentRole.SCHEMA,
            recipient=AgentRole.GENERATOR,
            content="Test message",
        )
        assert msg.sender == AgentRole.SCHEMA
        assert msg.recipient == AgentRole.GENERATOR
        assert msg.content == "Test message"
        assert msg.message_type == "request"

    def test_message_to_dict(self):
        msg = AgentMessage(
            sender=AgentRole.SCHEMA,
            recipient=AgentRole.GENERATOR,
            content={"key": "value"},
            metadata={"extra": True},
        )
        d = msg.to_dict()
        assert d["sender"] == "schema"
        assert d["recipient"] == "generator"
        assert d["content"]["key"] == "value"


class TestAgentTool:
    """Tests for AgentTool."""

    def test_create_tool(self):
        def my_func(x: int) -> int:
            return x * 2

        tool = AgentTool(
            name="multiply",
            description="Multiply by 2",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
            function=my_func,
        )
        assert tool.name == "multiply"
        assert tool.execute(x=5) == 10

    def test_tool_to_openai_format(self):
        tool = AgentTool(
            name="test_tool",
            description="A test tool",
            parameters={"type": "object"},
            function=lambda: None,
        )
        fmt = tool.to_openai_tool()
        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "test_tool"


class TestAgentMemory:
    """Tests for AgentMemory."""

    def test_add_interaction(self):
        memory = AgentMemory(max_history=10)
        result = AgentResult(success=True, output="test")
        memory.add_interaction("test request", result)
        assert len(memory.interactions) == 1

    def test_memory_max_history(self):
        memory = AgentMemory(max_history=5)
        for i in range(10):
            result = AgentResult(success=True, output=i)
            memory.add_interaction(f"request {i}", result)
        assert len(memory.interactions) == 5

    def test_preferences(self):
        memory = AgentMemory()
        memory.update_preference("method", "ctgan")
        assert memory.get_preference("method") == "ctgan"
        assert memory.get_preference("missing", "default") == "default"


class TestSchemaAgent:
    """Tests for SchemaAgent."""

    def test_infer_schema_healthcare(self):
        agent = SchemaAgent()
        schema = agent._infer_schema(
            "Patient records with age and diagnosis",
            domain="healthcare"
        )
        assert "columns" in schema
        assert len(schema["columns"]) > 0

    def test_infer_schema_finance(self):
        agent = SchemaAgent()
        schema = agent._get_domain_schema("finance")
        col_names = [c["name"] for c in schema["columns"]]
        assert "transaction_id" in col_names

    def test_validate_schema_valid(self):
        agent = SchemaAgent()
        schema = {
            "columns": [
                {"name": "id", "type": "string"},
                {"name": "value", "type": "float"},
            ]
        }
        result = agent._validate_schema(schema)
        assert result["valid"] is True

    def test_validate_schema_duplicate_columns(self):
        agent = SchemaAgent()
        schema = {
            "columns": [
                {"name": "id", "type": "string"},
                {"name": "id", "type": "integer"},  # Duplicate
            ]
        }
        result = agent._validate_schema(schema)
        assert result["valid"] is False
        assert any("Duplicate" in issue for issue in result["issues"])

    def test_process_message(self):
        agent = SchemaAgent()
        msg = AgentMessage(
            sender=AgentRole.ORCHESTRATOR,
            recipient=AgentRole.SCHEMA,
            content="Customer data with names and ages",
            metadata={"domain": "general"},
        )
        result = agent.process(msg)
        assert result.success
        assert "columns" in result.output


class TestGeneratorAgent:
    """Tests for GeneratorAgent."""

    def test_select_method_speed_preference(self):
        agent = GeneratorAgent()
        schema = {"columns": [{"name": "a", "type": "float"}]}
        method = agent._select_method(schema, preferences={"prefer_speed": True})
        assert method == "gaussian_copula"

    def test_select_method_quality_preference(self):
        agent = GeneratorAgent()
        schema = {"columns": [{"name": "a", "type": "float"}]}
        method = agent._select_method(schema, preferences={"prefer_quality": True})
        assert method == "ctgan"

    def test_generate_data(self):
        agent = GeneratorAgent()
        schema = {
            "columns": [
                {"name": "id", "type": "string", "constraints": {"unique": True}},
                {"name": "age", "type": "integer", "constraints": {"min": 0, "max": 100}},
                {"name": "score", "type": "float", "constraints": {"min": 0, "max": 1}},
                {"name": "category", "type": "categorical", "constraints": {"values": ["A", "B"]}},
            ]
        }
        df = agent._generate_data(schema, "ctgan", n_samples=100)
        assert len(df) == 100
        assert list(df.columns) == ["id", "age", "score", "category"]
        assert df["age"].min() >= 0
        assert df["age"].max() <= 100

    def test_process_message(self):
        agent = GeneratorAgent()
        schema = {
            "columns": [
                {"name": "value", "type": "float"},
            ]
        }
        msg = AgentMessage(
            sender=AgentRole.ORCHESTRATOR,
            recipient=AgentRole.GENERATOR,
            content={"schema": schema, "n_samples": 50},
        )
        result = agent.process(msg)
        assert result.success
        assert isinstance(result.output, pd.DataFrame)
        assert len(result.output) == 50


class TestValidatorAgent:
    """Tests for ValidatorAgent."""

    def test_validate_quality(self):
        agent = ValidatorAgent()
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
        schema = {
            "columns": [
                {"name": "id", "type": "integer"},
                {"name": "name", "type": "string"},
            ]
        }
        result = agent._validate_quality(df, schema)
        assert result["valid"] is True
        assert result["metrics"]["completeness"] == 1.0

    def test_validate_missing_columns(self):
        agent = ValidatorAgent()
        df = pd.DataFrame({"id": [1, 2, 3]})
        schema = {
            "columns": [
                {"name": "id", "type": "integer"},
                {"name": "missing_col", "type": "string"},
            ]
        }
        result = agent._validate_quality(df, schema)
        assert result["valid"] is False

    def test_check_constraints(self):
        agent = ValidatorAgent()
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
        constraints = [{"column": "value", "type": "range", "min": 0, "max": 10}]
        result = agent._check_constraints(df, constraints)
        assert result["satisfied"] is True


class TestPrivacyAgent:
    """Tests for PrivacyAgent."""

    def test_assess_privacy_unique_identifier(self):
        agent = PrivacyAgent()
        df = pd.DataFrame({"id": range(100), "value": [1] * 100})
        result = agent._assess_privacy(df)
        assert result["overall_risk"] == "high"
        assert any(r["column"] == "id" for r in result["risks"])

    def test_assess_privacy_sensitive_columns(self):
        agent = PrivacyAgent()
        df = pd.DataFrame({"ssn": ["123-45-6789"] * 10, "name": ["John"] * 10})
        result = agent._assess_privacy(df, sensitive_columns=["ssn"])
        assert result["overall_risk"] == "high"

    def test_enforce_k_anonymity(self):
        agent = PrivacyAgent()
        df = pd.DataFrame({
            "age": [25, 25, 25, 30, 30, 30, 30, 30, 35],  # 35 is alone
            "city": ["A"] * 9,
        })
        result = agent._enforce_k_anonymity(df, ["age"], k=3)
        assert 35 not in result["age"].values  # Single record removed
        assert len(result) == 8

    def test_suppress_rare(self):
        agent = PrivacyAgent()
        df = pd.DataFrame({
            "category": ["common"] * 95 + ["rare"] * 5,
        })
        result = agent._suppress_rare(df, threshold=0.1)
        assert "rare" not in result["category"].values


class TestDomainAgent:
    """Tests for DomainAgent."""

    def test_detect_domain_healthcare(self):
        agent = DomainAgent()
        schema = {
            "columns": [
                {"name": "patient_id"},
                {"name": "diagnosis_code"},
            ]
        }
        domain = agent._detect_domain(schema=schema)
        assert domain == "healthcare"

    def test_detect_domain_finance(self):
        agent = DomainAgent()
        df = pd.DataFrame(columns=["transaction_id", "account_balance"])
        domain = agent._detect_domain(sample_data=df)
        assert domain == "finance"

    def test_enhance_realism_finance(self):
        agent = DomainAgent()
        df = pd.DataFrame({
            "amount": [100.0] * 100,
            "transaction_type": ["debit"] * 100,
        })
        result = agent._enhance_realism(df, "finance")
        # Amount should now have log-normal distribution
        assert result["amount"].std() > 0


class TestAgentOrchestrator:
    """Tests for AgentOrchestrator."""

    def test_generate_basic(self):
        orchestrator = AgentOrchestrator()  # No API key, uses mocks
        result = orchestrator.generate(
            "Generate 100 customer records",
            n_samples=100,
        )
        assert result.success
        assert result.synthetic_data is not None
        assert len(result.synthetic_data) == 100

    def test_generate_with_domain(self):
        orchestrator = AgentOrchestrator()
        result = orchestrator.generate(
            "Healthcare patient data",
            n_samples=50,
            domain="healthcare",
        )
        assert result.success
        assert result.schema is not None

    def test_generate_with_privacy(self):
        orchestrator = AgentOrchestrator()
        result = orchestrator.generate(
            "Customer data",
            n_samples=100,
            enforce_privacy=True,
            sensitive_columns=["email"],
        )
        assert result.success
        assert result.privacy_report is not None

    def test_chat_interface(self):
        orchestrator = AgentOrchestrator()
        result = orchestrator.chat("Hello, I need some data")
        assert result.success
        assert "describe" in result.message.lower() or "help" in result.message.lower()


class TestAgenticDataGenerator:
    """Tests for AgenticDataGenerator convenience class."""

    def test_generate(self):
        gen = AgenticDataGenerator()
        result = gen.generate("100 retail orders", n_samples=100)
        assert result.success
        assert result.synthetic_data is not None

    def test_chat_and_reset(self):
        gen = AgenticDataGenerator()
        gen.chat("Hello")
        gen.reset()
        # Should not raise


class TestAgenticGenerateFunction:
    """Tests for agentic_generate convenience function."""

    def test_basic_generation(self):
        df = agentic_generate("Simple test data", n_samples=50)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 50

    def test_with_domain(self):
        df = agentic_generate(
            "Finance transaction data",
            n_samples=100,
            provider="openai",  # Uses mock without API key
        )
        assert len(df) == 100


class TestOrchestrationResult:
    """Tests for OrchestrationResult."""

    def test_create_result(self):
        result = OrchestrationResult(
            success=True,
            synthetic_data=pd.DataFrame({"a": [1, 2, 3]}),
            message="Success",
        )
        assert result.success
        assert len(result.synthetic_data) == 3

    def test_failed_result(self):
        result = OrchestrationResult(
            success=False,
            message="Generation failed",
        )
        assert not result.success
        assert result.synthetic_data is None
