"""Causality-Aware Synthesis.

Preserve causal relationships (not just correlations) using Structural Causal
Models (SCM) and Directed Acyclic Graphs (DAG) modeling. Enables counterfactual
generation and interventional sampling.

Features:
    - Structural Causal Model (SCM) definition
    - DAG specification and validation
    - Causal discovery integration
    - Interventional sampling (do-calculus)
    - Counterfactual generation
    - Confounding preservation
    - Causal fidelity metrics
    - Fairness analysis with causal lens
    - Visualization of causal structure

Example:
    Define causal model and generate data::

        from genesis.causality import (
            CausalModel, CausalGenerator, CausalNode, Edge
        )

        # Define causal structure
        model = CausalModel()
        model.add_node(CausalNode("age", node_type="exogenous"))
        model.add_node(CausalNode("income", parents=["age"]))
        model.add_node(CausalNode("purchase", parents=["age", "income"]))

        # Create generator
        generator = CausalGenerator(model)
        generator.fit(data)

        # Generate observational data
        synthetic = generator.generate(n_samples=1000)

        # Generate interventional data (what if age = 30?)
        intervened = generator.intervene({"age": 30}, n_samples=1000)

        # Generate counterfactual (what would have happened?)
        counterfactual = generator.counterfactual(
            factual={"age": 25, "income": 50000, "purchase": 1},
            intervention={"age": 35},
        )

Classes:
    CausalNode: A node in the causal graph.
    Edge: An edge in the causal graph.
    CausalModel: Structural Causal Model definition.
    CausalGenerator: Generator preserving causal relationships.
    CausalDiscovery: Automatic causal structure learning.
    CausalMetrics: Metrics for causal fidelity.
    FairnessAnalyzer: Causal fairness analysis.

Note:
    Requires understanding of causal inference concepts.
    See Pearl's "Causality" for theoretical background.
"""

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit as sigmoid

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class NodeType(str, Enum):
    """Types of causal nodes."""

    EXOGENOUS = "exogenous"  # External causes (no parents in model)
    ENDOGENOUS = "endogenous"  # Caused by other variables
    LATENT = "latent"  # Unobserved confounders
    OUTCOME = "outcome"  # Primary outcome variable
    TREATMENT = "treatment"  # Intervention variable


class DistributionType(str, Enum):
    """Types of probability distributions."""

    NORMAL = "normal"
    BERNOULLI = "bernoulli"
    CATEGORICAL = "categorical"
    POISSON = "poisson"
    EXPONENTIAL = "exponential"
    UNIFORM = "uniform"
    CUSTOM = "custom"


class EdgeType(str, Enum):
    """Types of causal edges."""

    DIRECT = "direct"  # Direct causal effect
    CONFOUNDED = "confounded"  # Bidirectional (latent confounder)
    SELECTION = "selection"  # Selection bias


@dataclass
class CausalNode:
    """A node in the causal graph.

    Attributes:
        name: Node name (variable name).
        node_type: Type of causal node.
        distribution: Probability distribution type.
        parents: Parent node names.
        structural_equation: Function defining causal mechanism.
        noise_scale: Scale of exogenous noise.
        domain: Value domain (continuous range or categories).
    """

    name: str
    node_type: NodeType = NodeType.ENDOGENOUS
    distribution: DistributionType = DistributionType.NORMAL
    parents: List[str] = field(default_factory=list)
    structural_equation: Optional[Callable] = None
    noise_scale: float = 1.0
    domain: Optional[Union[Tuple[float, float], List[Any]]] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "node_type": self.node_type.value,
            "distribution": self.distribution.value,
            "parents": self.parents,
            "noise_scale": self.noise_scale,
            "description": self.description,
        }


@dataclass
class Edge:
    """An edge in the causal graph.

    Attributes:
        source: Source node name.
        target: Target node name.
        edge_type: Type of edge.
        strength: Edge strength/coefficient.
        functional_form: Functional form description.
    """

    source: str
    target: str
    edge_type: EdgeType = EdgeType.DIRECT
    strength: float = 1.0
    functional_form: str = "linear"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "edge_type": self.edge_type.value,
            "strength": self.strength,
            "functional_form": self.functional_form,
        }


@dataclass
class CausalEffect:
    """Estimated causal effect.

    Attributes:
        treatment: Treatment variable.
        outcome: Outcome variable.
        effect_type: Type of effect (ATE, ATT, CATE).
        estimate: Point estimate.
        ci_lower: Confidence interval lower bound.
        ci_upper: Confidence interval upper bound.
        p_value: Statistical significance.
    """

    treatment: str
    outcome: str
    effect_type: str  # ATE, ATT, CATE
    estimate: float
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    p_value: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "treatment": self.treatment,
            "outcome": self.outcome,
            "effect_type": self.effect_type,
            "estimate": self.estimate,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "p_value": self.p_value,
        }


class CausalModel:
    """Structural Causal Model (SCM).

    Represents causal relationships as a DAG with structural equations.
    """

    def __init__(self, name: str = "unnamed") -> None:
        """Initialize causal model.

        Args:
            name: Model name.
        """
        self.name = name
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[Edge] = []
        self._adjacency: Dict[str, List[str]] = {}  # parent -> children
        self._parents: Dict[str, List[str]] = {}  # child -> parents

    def add_node(self, node: CausalNode) -> None:
        """Add node to model.

        Args:
            node: CausalNode to add.
        """
        self.nodes[node.name] = node
        self._parents[node.name] = node.parents.copy()

        if node.name not in self._adjacency:
            self._adjacency[node.name] = []

        # Update adjacency for parents
        for parent in node.parents:
            if parent not in self._adjacency:
                self._adjacency[parent] = []
            self._adjacency[parent].append(node.name)

            # Add edge
            self.edges.append(Edge(source=parent, target=node.name))

    def add_edge(self, edge: Edge) -> None:
        """Add edge to model.

        Args:
            edge: Edge to add.
        """
        self.edges.append(edge)

        if edge.source not in self._adjacency:
            self._adjacency[edge.source] = []
        self._adjacency[edge.source].append(edge.target)

        if edge.target not in self._parents:
            self._parents[edge.target] = []
        self._parents[edge.target].append(edge.source)

    def get_parents(self, node: str) -> List[str]:
        """Get parent nodes."""
        return self._parents.get(node, [])

    def get_children(self, node: str) -> List[str]:
        """Get child nodes."""
        return self._adjacency.get(node, [])

    def get_ancestors(self, node: str) -> Set[str]:
        """Get all ancestor nodes."""
        ancestors = set()
        to_visit = list(self.get_parents(node))

        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.get_parents(current))

        return ancestors

    def get_descendants(self, node: str) -> Set[str]:
        """Get all descendant nodes."""
        descendants = set()
        to_visit = list(self.get_children(node))

        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                to_visit.extend(self.get_children(current))

        return descendants

    def topological_sort(self) -> List[str]:
        """Get topologically sorted node order."""
        # Kahn's algorithm
        in_degree = {n: len(self.get_parents(n)) for n in self.nodes}
        queue = [n for n, d in in_degree.items() if d == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            for child in self.get_children(node):
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(result) != len(self.nodes):
            raise ValueError("Causal graph contains cycles")

        return result

    def is_valid_dag(self) -> bool:
        """Check if graph is a valid DAG."""
        try:
            self.topological_sort()
            return True
        except ValueError:
            return False

    def get_markov_blanket(self, node: str) -> Set[str]:
        """Get Markov blanket (parents, children, and co-parents)."""
        blanket = set()

        # Parents
        blanket.update(self.get_parents(node))

        # Children
        children = self.get_children(node)
        blanket.update(children)

        # Co-parents (parents of children)
        for child in children:
            blanket.update(self.get_parents(child))

        blanket.discard(node)
        return blanket

    def d_separated(self, x: str, y: str, z: Set[str]) -> bool:
        """Check if X and Y are d-separated given Z.

        Uses the Bayes-Ball algorithm.

        Args:
            x: Source node.
            y: Target node.
            z: Conditioning set.

        Returns:
            True if d-separated.
        """
        # Simplified check - full implementation would use Bayes-Ball
        # For now, check if all paths blocked
        paths = self._find_paths(x, y)

        for path in paths:
            if not self._path_blocked(path, z):
                return False

        return True

    def _find_paths(self, x: str, y: str) -> List[List[str]]:
        """Find all undirected paths between x and y."""
        paths = []

        def dfs(current: str, target: str, visited: Set[str], path: List[str]):
            if current == target:
                paths.append(path.copy())
                return

            visited.add(current)

            # Try parents
            for parent in self.get_parents(current):
                if parent not in visited:
                    dfs(parent, target, visited, path + [parent])

            # Try children
            for child in self.get_children(current):
                if child not in visited:
                    dfs(child, target, visited, path + [child])

            visited.remove(current)

        dfs(x, y, set(), [x])
        return paths

    def _path_blocked(self, path: List[str], z: Set[str]) -> bool:
        """Check if path is blocked by conditioning set."""
        if len(path) < 3:
            return len(path) == 2 and (path[0] in z or path[1] in z)

        for i in range(1, len(path) - 1):
            prev_node = path[i - 1]
            node = path[i]
            next_node = path[i + 1]

            is_parent_of_node = node in self.get_children(prev_node)
            is_child_of_node = node in self.get_parents(next_node)

            # Chain or fork: blocked if node in Z
            if is_parent_of_node != is_child_of_node:
                if node in z:
                    return True
            # Collider: blocked if node not in Z and no descendants in Z
            else:
                if node not in z and not any(d in z for d in self.get_descendants(node)):
                    return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "nodes": {n: node.to_dict() for n, node in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }

    def to_dot(self) -> str:
        """Convert to DOT format for visualization."""
        lines = [f'digraph "{self.name}" {{']
        lines.append("  rankdir=TB;")

        # Add nodes
        for name, node in self.nodes.items():
            shape = "ellipse"
            if node.node_type == NodeType.EXOGENOUS:
                shape = "box"
            elif node.node_type == NodeType.OUTCOME:
                shape = "doubleoctagon"
            elif node.node_type == NodeType.TREATMENT:
                shape = "diamond"

            lines.append(f'  "{name}" [shape={shape}];')

        # Add edges
        for edge in self.edges:
            style = "solid"
            if edge.edge_type == EdgeType.CONFOUNDED:
                style = "dashed"

            lines.append(f'  "{edge.source}" -> "{edge.target}" [style={style}];')

        lines.append("}")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CausalModel":
        """Create model from dictionary."""
        model = cls(name=data.get("name", "unnamed"))

        for name, node_data in data.get("nodes", {}).items():
            node = CausalNode(
                name=name,
                node_type=NodeType(node_data.get("node_type", "endogenous")),
                distribution=DistributionType(node_data.get("distribution", "normal")),
                parents=node_data.get("parents", []),
                noise_scale=node_data.get("noise_scale", 1.0),
            )
            model.add_node(node)

        return model


class CausalGenerator:
    """Generator preserving causal relationships.

    Generates synthetic data respecting the causal structure.
    """

    def __init__(self, model: CausalModel) -> None:
        """Initialize causal generator.

        Args:
            model: Causal model defining structure.
        """
        self.model = model
        self.fitted = False

        # Learned parameters
        self._coefficients: Dict[str, Dict[str, float]] = {}  # node -> {parent: coef}
        self._intercepts: Dict[str, float] = {}
        self._noise_scales: Dict[str, float] = {}
        self._distributions: Dict[str, Any] = {}

    def fit(self, data: pd.DataFrame) -> "CausalGenerator":
        """Fit structural equations from data.

        Args:
            data: Training data.

        Returns:
            self
        """
        # Topological order ensures parents fitted before children
        order = self.model.topological_sort()

        for node_name in order:
            node = self.model.nodes.get(node_name)
            if not node or node_name not in data.columns:
                continue

            parents = self.model.get_parents(node_name)
            valid_parents = [p for p in parents if p in data.columns]

            if not valid_parents:
                # Exogenous: fit marginal distribution
                self._fit_marginal(node_name, data[node_name], node)
            else:
                # Endogenous: fit structural equation
                self._fit_structural_equation(node_name, data, valid_parents, node)

        self.fitted = True
        logger.info(f"Fitted causal generator with {len(order)} nodes")

        return self

    def _fit_marginal(
        self,
        name: str,
        values: pd.Series,
        node: CausalNode,
    ) -> None:
        """Fit marginal distribution for exogenous node."""
        if node.distribution == DistributionType.BERNOULLI:
            p = values.mean()
            self._distributions[name] = {"type": "bernoulli", "p": p}
        elif node.distribution == DistributionType.CATEGORICAL:
            counts = values.value_counts(normalize=True)
            self._distributions[name] = {
                "type": "categorical",
                "categories": counts.index.tolist(),
                "probs": counts.values.tolist(),
            }
        else:
            # Default to normal
            mean = values.mean()
            std = values.std()
            self._distributions[name] = {"type": "normal", "mean": mean, "std": std}

        self._intercepts[name] = values.mean() if node.distribution == DistributionType.NORMAL else 0
        self._noise_scales[name] = values.std() if node.distribution == DistributionType.NORMAL else 1

    def _fit_structural_equation(
        self,
        name: str,
        data: pd.DataFrame,
        parents: List[str],
        node: CausalNode,
    ) -> None:
        """Fit structural equation from data."""
        y = data[name].values
        X = data[parents].values

        # Linear regression for structural equation
        X_with_intercept = np.column_stack([np.ones(len(X)), X])

        try:
            # OLS fit
            coeffs, residuals, rank, s = np.linalg.lstsq(X_with_intercept, y, rcond=None)

            self._intercepts[name] = coeffs[0]
            self._coefficients[name] = dict(zip(parents, coeffs[1:]))

            # Noise from residuals
            predictions = X_with_intercept @ coeffs
            residuals = y - predictions
            self._noise_scales[name] = np.std(residuals)

        except np.linalg.LinAlgError:
            # Fallback
            self._intercepts[name] = np.mean(y)
            self._coefficients[name] = {p: 0.0 for p in parents}
            self._noise_scales[name] = np.std(y)

    def generate(
        self,
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate observational data.

        Args:
            n_samples: Number of samples.
            seed: Random seed.

        Returns:
            Generated DataFrame.
        """
        if not self.fitted:
            raise ValueError("Generator not fitted. Call fit() first.")

        if seed is not None:
            np.random.seed(seed)

        data = {}
        order = self.model.topological_sort()

        for node_name in order:
            node = self.model.nodes.get(node_name)
            if not node:
                continue

            parents = self.model.get_parents(node_name)

            if not parents or node_name not in self._coefficients:
                # Sample from marginal
                data[node_name] = self._sample_marginal(node_name, n_samples)
            else:
                # Apply structural equation
                value = self._intercepts.get(node_name, 0)

                for parent in parents:
                    if parent in data and parent in self._coefficients.get(node_name, {}):
                        coef = self._coefficients[node_name][parent]
                        value = value + coef * data[parent]

                # Add noise
                noise_scale = self._noise_scales.get(node_name, 1.0)
                noise = np.random.normal(0, noise_scale, n_samples)
                value = value + noise

                # Apply distribution-specific transformations
                if node.distribution == DistributionType.BERNOULLI:
                    value = (sigmoid(value) > np.random.random(n_samples)).astype(int)
                elif node.distribution == DistributionType.POISSON:
                    value = np.random.poisson(np.exp(value))

                data[node_name] = value

        return pd.DataFrame(data)

    def _sample_marginal(self, name: str, n_samples: int) -> np.ndarray:
        """Sample from marginal distribution."""
        dist = self._distributions.get(name, {"type": "normal", "mean": 0, "std": 1})

        if dist["type"] == "bernoulli":
            return np.random.binomial(1, dist["p"], n_samples)
        elif dist["type"] == "categorical":
            return np.random.choice(dist["categories"], n_samples, p=dist["probs"])
        else:
            return np.random.normal(dist["mean"], dist["std"], n_samples)

    def intervene(
        self,
        interventions: Dict[str, Any],
        n_samples: int = 1000,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Generate data under intervention (do-calculus).

        Sets treatment variables to specific values, breaking causal links
        from their parents.

        Args:
            interventions: Dict of {variable: value} for do(X=x).
            n_samples: Number of samples.
            seed: Random seed.

        Returns:
            Interventional data.
        """
        if not self.fitted:
            raise ValueError("Generator not fitted. Call fit() first.")

        if seed is not None:
            np.random.seed(seed)

        data = {}
        order = self.model.topological_sort()

        for node_name in order:
            node = self.model.nodes.get(node_name)
            if not node:
                continue

            # Check if this node is intervened
            if node_name in interventions:
                # Set to intervention value (breaking parent links)
                value = interventions[node_name]
                if isinstance(value, (int, float)):
                    data[node_name] = np.full(n_samples, value)
                else:
                    data[node_name] = np.array([value] * n_samples)
                continue

            # Normal generation
            parents = self.model.get_parents(node_name)

            if not parents or node_name not in self._coefficients:
                data[node_name] = self._sample_marginal(node_name, n_samples)
            else:
                value = self._intercepts.get(node_name, 0)

                for parent in parents:
                    if parent in data and parent in self._coefficients.get(node_name, {}):
                        coef = self._coefficients[node_name][parent]
                        value = value + coef * data[parent]

                noise_scale = self._noise_scales.get(node_name, 1.0)
                noise = np.random.normal(0, noise_scale, n_samples)
                data[node_name] = value + noise

        return pd.DataFrame(data)

    def counterfactual(
        self,
        factual: Dict[str, Any],
        intervention: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate counterfactual values.

        Given observed factual values and an intervention, compute what
        other values would have been.

        Args:
            factual: Observed factual values for all variables.
            intervention: Counterfactual intervention {var: new_value}.

        Returns:
            Counterfactual values for all variables.
        """
        if not self.fitted:
            raise ValueError("Generator not fitted. Call fit() first.")

        # Step 1: Abduction - infer noise terms from factual
        noise = {}
        order = self.model.topological_sort()

        for node_name in order:
            if node_name not in factual:
                continue

            parents = self.model.get_parents(node_name)
            actual = factual[node_name]

            if not parents or node_name not in self._coefficients:
                # Exogenous noise
                dist = self._distributions.get(node_name, {"type": "normal", "mean": 0, "std": 1})
                if dist["type"] == "normal":
                    noise[node_name] = actual - dist["mean"]
                else:
                    noise[node_name] = 0
            else:
                # Endogenous noise = actual - deterministic part
                deterministic = self._intercepts.get(node_name, 0)
                for parent in parents:
                    if parent in factual and parent in self._coefficients.get(node_name, {}):
                        deterministic += self._coefficients[node_name][parent] * factual[parent]

                noise[node_name] = actual - deterministic

        # Step 2: Action - apply intervention
        counterfactual = {}

        for node_name in order:
            if node_name in intervention:
                counterfactual[node_name] = intervention[node_name]
                continue

            parents = self.model.get_parents(node_name)

            if not parents or node_name not in self._coefficients:
                # Keep same as factual
                counterfactual[node_name] = factual.get(node_name, 0)
            else:
                # Step 3: Prediction - use noise with new parent values
                value = self._intercepts.get(node_name, 0)

                for parent in parents:
                    if parent in counterfactual and parent in self._coefficients.get(node_name, {}):
                        value += self._coefficients[node_name][parent] * counterfactual[parent]

                # Add original noise
                value += noise.get(node_name, 0)
                counterfactual[node_name] = value

        return counterfactual

    def estimate_ate(
        self,
        treatment: str,
        outcome: str,
        data: Optional[pd.DataFrame] = None,
        n_samples: int = 10000,
    ) -> CausalEffect:
        """Estimate Average Treatment Effect.

        Args:
            treatment: Treatment variable name.
            outcome: Outcome variable name.
            data: Optional data for estimation.
            n_samples: Samples for Monte Carlo estimation.

        Returns:
            CausalEffect with ATE estimate.
        """
        # Generate potential outcomes using interventions
        # Y(1) - do(T=1)
        y1 = self.intervene({treatment: 1}, n_samples=n_samples)
        # Y(0) - do(T=0)
        y0 = self.intervene({treatment: 0}, n_samples=n_samples)

        if outcome not in y1.columns or outcome not in y0.columns:
            raise ValueError(f"Outcome {outcome} not found in generated data")

        ate = y1[outcome].mean() - y0[outcome].mean()

        # Bootstrap confidence interval
        bootstrap_ates = []
        for _ in range(100):
            idx1 = np.random.choice(n_samples, n_samples, replace=True)
            idx0 = np.random.choice(n_samples, n_samples, replace=True)
            bootstrap_ate = y1[outcome].iloc[idx1].mean() - y0[outcome].iloc[idx0].mean()
            bootstrap_ates.append(bootstrap_ate)

        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)

        # Simple z-test p-value
        se = np.std(bootstrap_ates)
        z = ate / se if se > 0 else 0
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        return CausalEffect(
            treatment=treatment,
            outcome=outcome,
            effect_type="ATE",
            estimate=ate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            p_value=p_value,
        )


class CausalDiscovery:
    """Automatic causal structure learning.

    Discovers causal DAG from observational data.
    """

    def __init__(
        self,
        method: str = "pc",
        alpha: float = 0.05,
    ) -> None:
        """Initialize causal discovery.

        Args:
            method: Discovery method (pc, ges, fci, lingam).
            alpha: Significance level for independence tests.
        """
        self.method = method
        self.alpha = alpha

    def discover(
        self,
        data: pd.DataFrame,
        prior_knowledge: Optional[Dict[str, Any]] = None,
    ) -> CausalModel:
        """Discover causal structure from data.

        Args:
            data: Observational data.
            prior_knowledge: Optional prior constraints.

        Returns:
            Discovered CausalModel.
        """
        if self.method == "pc":
            return self._pc_algorithm(data, prior_knowledge)
        elif self.method == "lingam":
            return self._lingam(data)
        else:
            logger.warning(f"Unknown method {self.method}, using PC")
            return self._pc_algorithm(data, prior_knowledge)

    def _pc_algorithm(
        self,
        data: pd.DataFrame,
        prior_knowledge: Optional[Dict[str, Any]] = None,
    ) -> CausalModel:
        """PC algorithm for causal discovery.

        Simplified implementation.
        """
        columns = data.columns.tolist()
        n = len(columns)

        # Initialize complete undirected graph
        adj = {c: set(columns) - {c} for c in columns}

        # Phase 1: Remove edges based on conditional independence
        for sep_size in range(n):
            edges_to_test = [
                (x, y) for x in columns for y in adj[x] if x < y
            ]

            for x, y in edges_to_test:
                if y not in adj[x]:
                    continue

                # Find possible separating sets
                neighbors = (adj[x] | adj[y]) - {x, y}

                for sep_set in self._subsets(neighbors, sep_size):
                    if self._conditionally_independent(data, x, y, set(sep_set)):
                        adj[x].discard(y)
                        adj[y].discard(x)
                        break

        # Phase 2: Orient edges (simplified)
        model = CausalModel(name="discovered")

        # Add nodes
        for col in columns:
            model.add_node(CausalNode(name=col))

        # Orient based on prior knowledge or heuristics
        oriented = set()
        for x in columns:
            for y in adj[x]:
                if (x, y) not in oriented and (y, x) not in oriented:
                    # Heuristic: order by position in columns
                    if columns.index(x) < columns.index(y):
                        model.add_edge(Edge(source=x, target=y))
                    else:
                        model.add_edge(Edge(source=y, target=x))
                    oriented.add((x, y))
                    oriented.add((y, x))

        return model

    def _lingam(self, data: pd.DataFrame) -> CausalModel:
        """LiNGAM algorithm for linear non-Gaussian models."""
        # Simplified ICA-based approach
        from scipy import linalg

        X = data.values
        columns = data.columns.tolist()
        n_vars = X.shape[1]

        # Standardize
        X = (X - X.mean(axis=0)) / X.std(axis=0)

        # ICA approximation using FastICA-like approach
        W = np.eye(n_vars)

        # Estimate causal order using kurtosis
        kurtosis = stats.kurtosis(X, axis=0)
        order = np.argsort(np.abs(kurtosis))[::-1]

        model = CausalModel(name="lingam_discovered")

        for col in columns:
            model.add_node(CausalNode(name=col))

        # Add edges based on estimated order
        for i, idx in enumerate(order):
            child = columns[idx]
            for j in range(i + 1, len(order)):
                parent = columns[order[j]]
                # Simple regression test
                corr = np.corrcoef(X[:, idx], X[:, order[j]])[0, 1]
                if abs(corr) > 0.1:  # Threshold
                    model.add_edge(Edge(source=parent, target=child, strength=corr))

        return model

    def _subsets(self, items: set, size: int):
        """Generate all subsets of given size."""
        items = list(items)
        if size > len(items):
            return

        from itertools import combinations
        for subset in combinations(items, size):
            yield subset

    def _conditionally_independent(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        z: Set[str],
    ) -> bool:
        """Test conditional independence X ⊥ Y | Z."""
        if not z:
            # Marginal independence
            corr = np.corrcoef(data[x], data[y])[0, 1]
            n = len(data)
            t_stat = corr * np.sqrt((n - 2) / (1 - corr**2 + 1e-10))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            return p_value > self.alpha

        # Partial correlation test
        z_cols = list(z)

        # Regress X on Z
        X_z = data[z_cols].values
        x_vals = data[x].values
        y_vals = data[y].values

        # Add intercept
        X_z_int = np.column_stack([np.ones(len(X_z)), X_z])

        # Residuals
        coef_x = np.linalg.lstsq(X_z_int, x_vals, rcond=None)[0]
        coef_y = np.linalg.lstsq(X_z_int, y_vals, rcond=None)[0]

        res_x = x_vals - X_z_int @ coef_x
        res_y = y_vals - X_z_int @ coef_y

        # Correlation of residuals
        partial_corr = np.corrcoef(res_x, res_y)[0, 1]

        n = len(data)
        df = n - len(z) - 2
        t_stat = partial_corr * np.sqrt(df / (1 - partial_corr**2 + 1e-10))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        return p_value > self.alpha


class CausalMetrics:
    """Metrics for evaluating causal fidelity."""

    @staticmethod
    def structural_hamming_distance(
        true_model: CausalModel,
        learned_model: CausalModel,
    ) -> int:
        """Compute Structural Hamming Distance between DAGs.

        Args:
            true_model: Ground truth causal model.
            learned_model: Learned causal model.

        Returns:
            SHD (lower is better).
        """
        true_edges = set((e.source, e.target) for e in true_model.edges)
        learned_edges = set((e.source, e.target) for e in learned_model.edges)

        # Missing edges
        missing = true_edges - learned_edges

        # Extra edges
        extra = learned_edges - true_edges

        # Reversed edges (count as 2)
        reversed_edges = sum(
            1 for (a, b) in missing if (b, a) in learned_edges
        )

        return len(missing) + len(extra) - reversed_edges

    @staticmethod
    def causal_effect_error(
        true_effect: float,
        estimated_effect: float,
    ) -> float:
        """Compute relative error in causal effect estimation.

        Args:
            true_effect: True causal effect.
            estimated_effect: Estimated causal effect.

        Returns:
            Relative error.
        """
        if abs(true_effect) < 1e-10:
            return abs(estimated_effect)
        return abs(estimated_effect - true_effect) / abs(true_effect)

    @staticmethod
    def interventional_mse(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        treatment: str,
        outcome: str,
    ) -> float:
        """Compute MSE of interventional distributions.

        Args:
            real_data: Real interventional data.
            synthetic_data: Synthetic interventional data.
            treatment: Treatment variable.
            outcome: Outcome variable.

        Returns:
            MSE.
        """
        # Compare outcome distributions at different treatment levels
        mse = 0
        for t_val in real_data[treatment].unique():
            real_outcomes = real_data[real_data[treatment] == t_val][outcome]
            synth_outcomes = synthetic_data[synthetic_data[treatment] == t_val][outcome]

            if len(synth_outcomes) > 0:
                mse += (real_outcomes.mean() - synth_outcomes.mean()) ** 2

        return mse / len(real_data[treatment].unique())


class FairnessAnalyzer:
    """Causal fairness analysis.

    Analyzes fairness using causal lens to distinguish between
    different types of discrimination.
    """

    def __init__(self, model: CausalModel) -> None:
        """Initialize fairness analyzer.

        Args:
            model: Causal model.
        """
        self.model = model

    def analyze_direct_discrimination(
        self,
        data: pd.DataFrame,
        sensitive: str,
        outcome: str,
        generator: CausalGenerator,
    ) -> Dict[str, Any]:
        """Analyze direct discrimination.

        Direct effect of sensitive attribute on outcome.

        Args:
            data: Data to analyze.
            sensitive: Sensitive attribute (e.g., gender, race).
            outcome: Outcome variable.
            generator: Fitted causal generator.

        Returns:
            Analysis results.
        """
        # Get unique values of sensitive attribute
        s_values = data[sensitive].unique()

        if len(s_values) != 2:
            warnings.warn("Direct discrimination analysis assumes binary sensitive attribute")

        # Estimate controlled direct effect
        # Intervention on sensitive while controlling mediators
        mediators = self._find_mediators(sensitive, outcome)

        results = {
            "sensitive_attribute": sensitive,
            "outcome": outcome,
            "mediators": list(mediators),
        }

        # Compare outcomes across sensitive groups
        group_means = {}
        for s_val in s_values:
            group_data = data[data[sensitive] == s_val]
            group_means[s_val] = group_data[outcome].mean()

        results["group_means"] = group_means
        results["disparity"] = max(group_means.values()) - min(group_means.values())

        # Estimate causal direct effect
        if len(s_values) == 2:
            s0, s1 = sorted(s_values)
            effect = generator.estimate_ate(sensitive, outcome)
            results["direct_effect"] = effect.to_dict()

        return results

    def analyze_indirect_discrimination(
        self,
        data: pd.DataFrame,
        sensitive: str,
        outcome: str,
        mediator: str,
        generator: CausalGenerator,
    ) -> Dict[str, Any]:
        """Analyze indirect discrimination through mediator.

        Args:
            data: Data to analyze.
            sensitive: Sensitive attribute.
            outcome: Outcome variable.
            mediator: Mediating variable.
            generator: Fitted causal generator.

        Returns:
            Analysis results.
        """
        # Natural indirect effect
        s_values = sorted(data[sensitive].unique())

        results = {
            "sensitive_attribute": sensitive,
            "outcome": outcome,
            "mediator": mediator,
        }

        if len(s_values) == 2:
            s0, s1 = s_values

            # Effect through mediator
            # Compare: E[Y | do(S=s1)] vs E[Y | do(S=s0), do(M=M(S=s1))]
            # Simplified: regress mediator on sensitive, then outcome on mediator

            # Effect of sensitive on mediator
            sm_effect = generator.estimate_ate(sensitive, mediator)
            results["sensitive_mediator_effect"] = sm_effect.to_dict()

            # Effect of mediator on outcome
            mo_effect = generator.estimate_ate(mediator, outcome)
            results["mediator_outcome_effect"] = mo_effect.to_dict()

            # Indirect effect = product (for linear models)
            indirect = sm_effect.estimate * mo_effect.estimate
            results["indirect_effect"] = indirect

        return results

    def _find_mediators(self, source: str, target: str) -> Set[str]:
        """Find mediating variables between source and target."""
        mediators = set()

        descendants = self.model.get_descendants(source)
        ancestors = self.model.get_ancestors(target)

        mediators = descendants & ancestors

        return mediators


# Convenience functions

def create_causal_model(
    edges: List[Tuple[str, str]],
    exogenous: Optional[List[str]] = None,
) -> CausalModel:
    """Create causal model from edge list.

    Args:
        edges: List of (source, target) tuples.
        exogenous: List of exogenous node names.

    Returns:
        CausalModel.
    """
    model = CausalModel()
    exogenous = set(exogenous or [])

    # Collect all nodes
    all_nodes = set()
    parents_map = {}

    for source, target in edges:
        all_nodes.add(source)
        all_nodes.add(target)

        if target not in parents_map:
            parents_map[target] = []
        parents_map[target].append(source)

    # Add nodes
    for node in all_nodes:
        node_type = NodeType.EXOGENOUS if node in exogenous else NodeType.ENDOGENOUS
        parents = parents_map.get(node, [])

        if not parents and node not in exogenous:
            node_type = NodeType.EXOGENOUS

        model.add_node(CausalNode(
            name=node,
            node_type=node_type,
            parents=parents,
        ))

    return model


def discover_causal_structure(
    data: pd.DataFrame,
    method: str = "pc",
) -> CausalModel:
    """Discover causal structure from data.

    Args:
        data: Observational data.
        method: Discovery method.

    Returns:
        Discovered CausalModel.
    """
    discovery = CausalDiscovery(method=method)
    return discovery.discover(data)


def generate_causal_data(
    model: CausalModel,
    n_samples: int = 1000,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic data from causal model.

    Note: Model must have structural equations defined or be fitted.

    Args:
        model: Causal model.
        n_samples: Number of samples.
        seed: Random seed.

    Returns:
        Generated DataFrame.
    """
    generator = CausalGenerator(model)

    # Simple generation without fitting (uses random coefficients)
    if seed is not None:
        np.random.seed(seed)

    data = {}
    order = model.topological_sort()

    for node_name in order:
        node = model.nodes.get(node_name)
        if not node:
            continue

        parents = model.get_parents(node_name)

        if not parents:
            # Exogenous: sample from distribution
            if node.distribution == DistributionType.NORMAL:
                data[node_name] = np.random.normal(0, 1, n_samples)
            elif node.distribution == DistributionType.BERNOULLI:
                data[node_name] = np.random.binomial(1, 0.5, n_samples)
            else:
                data[node_name] = np.random.normal(0, 1, n_samples)
        else:
            # Endogenous: linear combination of parents + noise
            value = np.zeros(n_samples)
            for parent in parents:
                if parent in data:
                    coef = np.random.uniform(0.3, 0.8)  # Random coefficient
                    value = value + coef * data[parent]

            value = value + np.random.normal(0, node.noise_scale, n_samples)
            data[node_name] = value

    return pd.DataFrame(data)
