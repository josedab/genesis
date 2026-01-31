"""Tests for Visual Pipeline Builder module."""


import numpy as np
import pandas as pd
import pytest

from genesis.pipeline import (
    NodePort,
    NodeType,
    Pipeline,
    PipelineBuilder,
    PipelineConnection,
    PipelineExecutor,
    PipelineNode,
    create_simple_pipeline,
    get_node_templates,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "a": np.random.randint(0, 100, 100),
            "b": np.random.normal(0, 1, 100),
            "c": np.random.choice(["X", "Y", "Z"], 100),
        }
    )


class TestNodePort:
    """Tests for NodePort."""

    def test_creation(self) -> None:
        """Test port creation."""
        port = NodePort(
            id="port1",
            name="Input",
            port_type="input",
            data_type="dataframe",
        )

        assert port.id == "port1"
        assert port.name == "Input"
        assert port.required is True


class TestPipelineNode:
    """Tests for PipelineNode."""

    def test_to_dict(self) -> None:
        """Test serialization."""
        node = PipelineNode(
            id="node1",
            node_type=NodeType.GENERATOR,
            name="Generator",
            config={"n_samples": 100},
            position={"x": 100, "y": 100},
        )

        d = node.to_dict()

        assert d["id"] == "node1"
        assert d["node_type"] == "generator"
        assert d["config"]["n_samples"] == 100

    def test_from_dict(self) -> None:
        """Test deserialization."""
        data = {
            "id": "node1",
            "node_type": "filter",
            "name": "Filter",
            "config": {"condition": "a > 50"},
            "position": {"x": 0, "y": 0},
            "inputs": [],
            "outputs": [],
        }

        node = PipelineNode.from_dict(data)

        assert node.id == "node1"
        assert node.node_type == NodeType.FILTER


class TestPipeline:
    """Tests for Pipeline."""

    def test_validation_empty(self) -> None:
        """Test validating empty pipeline."""
        pipeline = Pipeline(
            id="p1",
            name="Test",
            description="",
            nodes=[],
            connections=[],
        )

        result = pipeline.validate()

        # Empty pipeline is valid
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validation_invalid_connection(self) -> None:
        """Test validating invalid connection."""
        node = PipelineNode(
            id="node1",
            node_type=NodeType.DATA_SOURCE,
            name="Source",
            config={},
            position={"x": 0, "y": 0},
        )

        conn = PipelineConnection(
            id="conn1",
            source_node="nonexistent",
            source_port="output",
            target_node="node1",
            target_port="input",
        )

        pipeline = Pipeline(
            id="p1",
            name="Test",
            description="",
            nodes=[node],
            connections=[conn],
        )

        result = pipeline.validate()

        assert not result.is_valid
        assert len(result.errors) > 0
        assert any("unknown source" in e.lower() for e in result.errors)

    def test_execution_order(self) -> None:
        """Test topological execution order."""
        node1 = PipelineNode(
            id="node1",
            node_type=NodeType.DATA_SOURCE,
            name="Source",
            config={},
            position={"x": 0, "y": 0},
        )
        node2 = PipelineNode(
            id="node2",
            node_type=NodeType.FILTER,
            name="Filter",
            config={},
            position={"x": 100, "y": 0},
        )

        conn = PipelineConnection(
            id="conn1",
            source_node="node1",
            source_port="data",
            target_node="node2",
            target_port="input",
        )

        pipeline = Pipeline(
            id="p1",
            name="Test",
            description="",
            nodes=[node2, node1],  # Wrong order
            connections=[conn],
        )

        order = pipeline.get_execution_order()

        assert order.index("node1") < order.index("node2")

    def test_to_dict(self) -> None:
        """Test serialization."""
        pipeline = Pipeline(
            id="p1",
            name="Test",
            description="Test pipeline",
            nodes=[],
            connections=[],
        )

        d = pipeline.to_dict()

        assert d["id"] == "p1"
        assert d["name"] == "Test"


class TestPipelineBuilder:
    """Tests for PipelineBuilder."""

    def test_add_node(self) -> None:
        """Test adding nodes."""
        builder = PipelineBuilder("Test")

        node_id = builder.add_node(
            NodeType.DATA_SOURCE,
            {"variable_name": "data"},
        )

        assert node_id == "node_0"
        assert len(builder.pipeline.nodes) == 1

    def test_connect(self) -> None:
        """Test connecting nodes."""
        builder = PipelineBuilder("Test")
        source = builder.add_node(NodeType.DATA_SOURCE, {"variable_name": "data"})
        gen = builder.add_node(NodeType.GENERATOR, {"n_samples": 100})

        conn_id = builder.connect(source, "data", gen, "training_data")

        assert conn_id == "conn_0"
        assert len(builder.pipeline.connections) == 1

    def test_build_valid(self) -> None:
        """Test building valid pipeline."""
        builder = PipelineBuilder("Test")
        builder.add_node(NodeType.DATA_SOURCE, {"variable_name": "data"})

        pipeline = builder.build()

        assert isinstance(pipeline, Pipeline)

    def test_build_with_cycle_raises(self) -> None:
        """Test that cycles are detected."""
        builder = PipelineBuilder("Test")
        n1 = builder.add_node(NodeType.FILTER, {"condition": "True"})
        n2 = builder.add_node(NodeType.FILTER, {"condition": "True"})

        builder.connect(n1, "output", n2, "input")
        builder.connect(n2, "output", n1, "input")  # Cycle

        with pytest.raises(ValueError, match="cycle"):
            builder.build()


class TestPipelineExecutor:
    """Tests for PipelineExecutor."""

    def test_execute_data_source(self, sample_data: pd.DataFrame) -> None:
        """Test executing data source node."""
        builder = PipelineBuilder("Test")
        builder.add_node(NodeType.DATA_SOURCE, {"variable_name": "input_data"})
        pipeline = builder.build()

        executor = PipelineExecutor()
        outputs = executor.execute(pipeline, inputs={"input_data": sample_data})

        assert "node_0" in outputs
        assert outputs["node_0"]["data"] is not None

    def test_execute_filter(self, sample_data: pd.DataFrame) -> None:
        """Test executing filter node."""
        builder = PipelineBuilder("Test")
        source = builder.add_node(NodeType.DATA_SOURCE, {"variable_name": "data"})
        filt = builder.add_node(NodeType.FILTER, {"condition": "a > 50"})
        builder.connect(source, "data", filt, "input")
        pipeline = builder.build()

        executor = PipelineExecutor()
        outputs = executor.execute(pipeline, inputs={"data": sample_data})

        filtered = outputs["node_1"]["output"]
        assert filtered is not None
        assert all(filtered["a"] > 50)


class TestCreateSimplePipeline:
    """Tests for create_simple_pipeline function."""

    def test_creates_pipeline(self) -> None:
        """Test simple pipeline creation."""
        pipeline = create_simple_pipeline(
            name="Test",
            input_variable="data",
            method="gaussian_copula",
            n_samples=100,
        )

        assert pipeline.name == "Test"
        assert len(pipeline.nodes) >= 2


class TestGetNodeTemplates:
    """Tests for get_node_templates function."""

    def test_returns_templates(self) -> None:
        """Test getting node templates."""
        templates = get_node_templates()

        assert "data_source" in templates
        assert "generator" in templates
        assert "filter" in templates

        # Check template structure
        gen_template = templates["generator"]
        assert "name" in gen_template
        assert "config_schema" in gen_template
        assert "inputs" in gen_template
        assert "outputs" in gen_template
