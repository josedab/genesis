"""Tests for Causality-Aware Synthesis module."""

import numpy as np
import pandas as pd
import pytest

from genesis.causality import (
    CausalDiscovery,
    CausalEffect,
    CausalGenerator,
    CausalMetrics,
    CausalModel,
    CausalNode,
    Edge,
    EdgeType,
    FairnessAnalyzer,
    NodeType,
    create_causal_model,
    discover_causal_structure,
    generate_causal_data,
)


class TestCausalNode:
    """Tests for CausalNode."""

    def test_create_exogenous_node(self):
        """Test creating exogenous node."""
        node = CausalNode(
            name="age",
            node_type=NodeType.EXOGENOUS,
        )

        assert node.name == "age"
        assert node.node_type == NodeType.EXOGENOUS
        assert node.parents == []

    def test_create_endogenous_node(self):
        """Test creating endogenous node with parents."""
        node = CausalNode(
            name="income",
            node_type=NodeType.ENDOGENOUS,
            parents=["age", "education"],
        )

        assert node.name == "income"
        assert "age" in node.parents
        assert "education" in node.parents

    def test_to_dict(self):
        """Test node serialization."""
        node = CausalNode(name="test", noise_scale=0.5)
        data = node.to_dict()

        assert data["name"] == "test"
        assert data["noise_scale"] == 0.5


class TestEdge:
    """Tests for Edge."""

    def test_create_direct_edge(self):
        """Test creating direct causal edge."""
        edge = Edge(source="age", target="income")

        assert edge.source == "age"
        assert edge.target == "income"
        assert edge.edge_type == EdgeType.DIRECT

    def test_edge_with_strength(self):
        """Test edge with specified strength."""
        edge = Edge(
            source="treatment",
            target="outcome",
            strength=0.75,
        )

        assert edge.strength == 0.75


class TestCausalModel:
    """Tests for CausalModel."""

    def test_add_node(self):
        """Test adding nodes to model."""
        model = CausalModel(name="test")
        model.add_node(CausalNode(name="X"))
        model.add_node(CausalNode(name="Y", parents=["X"]))

        assert "X" in model.nodes
        assert "Y" in model.nodes
        assert "X" in model.get_parents("Y")

    def test_get_children(self):
        """Test getting child nodes."""
        model = CausalModel()
        model.add_node(CausalNode(name="X"))
        model.add_node(CausalNode(name="Y", parents=["X"]))
        model.add_node(CausalNode(name="Z", parents=["X"]))

        children = model.get_children("X")
        assert "Y" in children
        assert "Z" in children

    def test_get_ancestors(self):
        """Test getting ancestor nodes."""
        model = CausalModel()
        model.add_node(CausalNode(name="A"))
        model.add_node(CausalNode(name="B", parents=["A"]))
        model.add_node(CausalNode(name="C", parents=["B"]))

        ancestors = model.get_ancestors("C")
        assert "A" in ancestors
        assert "B" in ancestors

    def test_get_descendants(self):
        """Test getting descendant nodes."""
        model = CausalModel()
        model.add_node(CausalNode(name="A"))
        model.add_node(CausalNode(name="B", parents=["A"]))
        model.add_node(CausalNode(name="C", parents=["B"]))

        descendants = model.get_descendants("A")
        assert "B" in descendants
        assert "C" in descendants

    def test_topological_sort(self):
        """Test topological sorting."""
        model = CausalModel()
        model.add_node(CausalNode(name="X"))
        model.add_node(CausalNode(name="Y", parents=["X"]))
        model.add_node(CausalNode(name="Z", parents=["X", "Y"]))

        order = model.topological_sort()

        assert order.index("X") < order.index("Y")
        assert order.index("Y") < order.index("Z")

    def test_is_valid_dag(self):
        """Test DAG validity check."""
        model = CausalModel()
        model.add_node(CausalNode(name="X"))
        model.add_node(CausalNode(name="Y", parents=["X"]))

        assert model.is_valid_dag()

    def test_markov_blanket(self):
        """Test Markov blanket computation."""
        model = CausalModel()
        model.add_node(CausalNode(name="A"))
        model.add_node(CausalNode(name="B", parents=["A"]))
        model.add_node(CausalNode(name="C", parents=["A", "B"]))
        model.add_node(CausalNode(name="D", parents=["B"]))

        blanket = model.get_markov_blanket("B")

        # Parents: A; Children: C, D; Co-parents of children: A
        assert "A" in blanket
        assert "C" in blanket
        assert "D" in blanket

    def test_to_dot(self):
        """Test DOT format export."""
        model = CausalModel(name="test_graph")
        model.add_node(CausalNode(name="X", node_type=NodeType.EXOGENOUS))
        model.add_node(CausalNode(name="Y", parents=["X"]))

        dot = model.to_dot()

        assert "digraph" in dot
        assert '"X"' in dot
        assert '"Y"' in dot
        assert "->" in dot

    def test_from_dict(self):
        """Test loading model from dictionary."""
        data = {
            "name": "loaded_model",
            "nodes": {
                "X": {"node_type": "exogenous"},
                "Y": {"node_type": "endogenous", "parents": ["X"]},
            },
        }

        model = CausalModel.from_dict(data)

        assert model.name == "loaded_model"
        assert "X" in model.nodes
        assert "Y" in model.nodes


class TestCausalGenerator:
    """Tests for CausalGenerator."""

    @pytest.fixture
    def simple_model(self):
        """Create simple causal model."""
        model = CausalModel()
        model.add_node(CausalNode(name="X", node_type=NodeType.EXOGENOUS))
        model.add_node(CausalNode(name="Y", parents=["X"]))
        model.add_node(CausalNode(name="Z", parents=["X", "Y"]))
        return model

    @pytest.fixture
    def sample_data(self):
        """Create sample data matching model structure."""
        np.random.seed(42)
        n = 1000
        X = np.random.normal(0, 1, n)
        Y = 0.5 * X + np.random.normal(0, 0.5, n)
        Z = 0.3 * X + 0.4 * Y + np.random.normal(0, 0.5, n)
        return pd.DataFrame({"X": X, "Y": Y, "Z": Z})

    def test_fit(self, simple_model, sample_data):
        """Test fitting causal generator."""
        generator = CausalGenerator(simple_model)
        generator.fit(sample_data)

        assert generator.fitted
        assert "Y" in generator._coefficients
        assert "X" in generator._coefficients["Y"]

    def test_generate(self, simple_model, sample_data):
        """Test generating observational data."""
        generator = CausalGenerator(simple_model)
        generator.fit(sample_data)

        synthetic = generator.generate(n_samples=500, seed=42)

        assert len(synthetic) == 500
        assert list(synthetic.columns) == ["X", "Y", "Z"]

    def test_intervene(self, simple_model, sample_data):
        """Test interventional generation."""
        generator = CausalGenerator(simple_model)
        generator.fit(sample_data)

        # do(X=2)
        intervened = generator.intervene({"X": 2}, n_samples=500, seed=42)

        assert len(intervened) == 500
        # All X values should be 2
        assert np.allclose(intervened["X"], 2)
        # Y and Z should vary (but be influenced by X=2)
        assert intervened["Y"].std() > 0

    def test_counterfactual(self, simple_model, sample_data):
        """Test counterfactual generation."""
        generator = CausalGenerator(simple_model)
        generator.fit(sample_data)

        factual = {"X": 1.0, "Y": 0.6, "Z": 0.5}
        counterfactual = generator.counterfactual(
            factual=factual,
            intervention={"X": 2.0},
        )

        assert "X" in counterfactual
        assert counterfactual["X"] == 2.0
        # Y and Z should change based on new X
        assert counterfactual["Y"] != factual["Y"]

    def test_estimate_ate(self, simple_model, sample_data):
        """Test ATE estimation."""
        generator = CausalGenerator(simple_model)
        generator.fit(sample_data)

        effect = generator.estimate_ate("X", "Y", n_samples=5000)

        assert effect.treatment == "X"
        assert effect.outcome == "Y"
        assert effect.effect_type == "ATE"
        # Effect should be close to 0.5 (our true coefficient)
        assert abs(effect.estimate - 0.5) < 0.2


class TestCausalDiscovery:
    """Tests for CausalDiscovery."""

    def test_pc_algorithm(self):
        """Test PC algorithm for structure discovery."""
        # Generate data with known structure
        np.random.seed(42)
        n = 1000
        X = np.random.normal(0, 1, n)
        Y = 0.7 * X + np.random.normal(0, 0.5, n)
        Z = 0.5 * Y + np.random.normal(0, 0.5, n)
        data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})

        discovery = CausalDiscovery(method="pc", alpha=0.05)
        model = discovery.discover(data)

        # Should discover some edges
        assert len(model.edges) > 0
        assert len(model.nodes) == 3

    def test_lingam_algorithm(self):
        """Test LiNGAM algorithm."""
        np.random.seed(42)
        n = 500
        X = np.random.uniform(-1, 1, n)
        Y = 0.5 * X + np.random.uniform(-0.5, 0.5, n)
        data = pd.DataFrame({"X": X, "Y": Y})

        discovery = CausalDiscovery(method="lingam")
        model = discovery.discover(data)

        assert len(model.nodes) == 2


class TestCausalMetrics:
    """Tests for CausalMetrics."""

    def test_structural_hamming_distance_identical(self):
        """Test SHD for identical graphs."""
        model1 = CausalModel()
        model1.add_node(CausalNode(name="X"))
        model1.add_node(CausalNode(name="Y", parents=["X"]))

        model2 = CausalModel()
        model2.add_node(CausalNode(name="X"))
        model2.add_node(CausalNode(name="Y", parents=["X"]))

        shd = CausalMetrics.structural_hamming_distance(model1, model2)
        assert shd == 0

    def test_structural_hamming_distance_different(self):
        """Test SHD for different graphs."""
        model1 = CausalModel()
        model1.add_node(CausalNode(name="X"))
        model1.add_node(CausalNode(name="Y", parents=["X"]))

        model2 = CausalModel()
        model2.add_node(CausalNode(name="X"))
        model2.add_node(CausalNode(name="Y"))  # No edge

        shd = CausalMetrics.structural_hamming_distance(model1, model2)
        assert shd > 0

    def test_causal_effect_error(self):
        """Test causal effect error computation."""
        error = CausalMetrics.causal_effect_error(1.0, 0.9)
        assert abs(error - 0.1) < 1e-10


class TestFairnessAnalyzer:
    """Tests for FairnessAnalyzer."""

    @pytest.fixture
    def fairness_model(self):
        """Create model for fairness analysis."""
        model = CausalModel()
        model.add_node(CausalNode(name="gender", node_type=NodeType.EXOGENOUS))
        model.add_node(CausalNode(name="education", parents=["gender"]))
        model.add_node(CausalNode(name="income", parents=["gender", "education"]))
        return model

    @pytest.fixture
    def fairness_data(self):
        """Create data for fairness analysis."""
        np.random.seed(42)
        n = 1000
        gender = np.random.binomial(1, 0.5, n)
        education = gender * 2 + np.random.normal(0, 1, n)
        income = 0.3 * gender + 0.5 * education + np.random.normal(0, 1, n)
        return pd.DataFrame({
            "gender": gender,
            "education": education,
            "income": income,
        })

    def test_analyze_direct_discrimination(self, fairness_model, fairness_data):
        """Test direct discrimination analysis."""
        generator = CausalGenerator(fairness_model)
        generator.fit(fairness_data)

        analyzer = FairnessAnalyzer(fairness_model)
        result = analyzer.analyze_direct_discrimination(
            fairness_data,
            sensitive="gender",
            outcome="income",
            generator=generator,
        )

        assert "sensitive_attribute" in result
        assert result["sensitive_attribute"] == "gender"
        assert "disparity" in result
        assert "group_means" in result

    def test_analyze_indirect_discrimination(self, fairness_model, fairness_data):
        """Test indirect discrimination analysis."""
        generator = CausalGenerator(fairness_model)
        generator.fit(fairness_data)

        analyzer = FairnessAnalyzer(fairness_model)
        result = analyzer.analyze_indirect_discrimination(
            fairness_data,
            sensitive="gender",
            outcome="income",
            mediator="education",
            generator=generator,
        )

        assert "mediator" in result
        assert result["mediator"] == "education"
        assert "indirect_effect" in result


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_causal_model(self):
        """Test creating model from edge list."""
        edges = [("X", "Y"), ("Y", "Z"), ("X", "Z")]
        model = create_causal_model(edges, exogenous=["X"])

        assert model.nodes["X"].node_type == NodeType.EXOGENOUS
        assert "X" in model.get_parents("Y")
        assert "X" in model.get_parents("Z")

    def test_discover_causal_structure(self):
        """Test causal structure discovery."""
        np.random.seed(42)
        data = pd.DataFrame({
            "A": np.random.normal(0, 1, 200),
            "B": np.random.normal(0, 1, 200),
        })
        data["C"] = 0.5 * data["A"] + 0.3 * data["B"] + np.random.normal(0, 0.5, 200)

        model = discover_causal_structure(data, method="pc")

        assert len(model.nodes) == 3

    def test_generate_causal_data(self):
        """Test generating data from causal model."""
        model = CausalModel()
        model.add_node(CausalNode(name="X", node_type=NodeType.EXOGENOUS))
        model.add_node(CausalNode(name="Y", parents=["X"]))

        data = generate_causal_data(model, n_samples=100, seed=42)

        assert len(data) == 100
        assert "X" in data.columns
        assert "Y" in data.columns


class TestCausalEffect:
    """Tests for CausalEffect dataclass."""

    def test_to_dict(self):
        """Test effect serialization."""
        effect = CausalEffect(
            treatment="T",
            outcome="Y",
            effect_type="ATE",
            estimate=0.5,
            ci_lower=0.3,
            ci_upper=0.7,
        )

        data = effect.to_dict()

        assert data["treatment"] == "T"
        assert data["outcome"] == "Y"
        assert data["estimate"] == 0.5
