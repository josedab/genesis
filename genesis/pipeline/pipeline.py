"""Pipeline definition and graph operations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

from genesis.pipeline.nodes import (
    NODE_TEMPLATES,
    NodePort,
    NodeType,
    PipelineConnection,
    PipelineNode,
    ValidationResult,
)


@dataclass
class Pipeline:
    """A complete pipeline definition.

    Attributes:
        id: Unique pipeline identifier
        name: Human-readable name
        description: Pipeline description
        nodes: List of pipeline nodes
        connections: List of connections between nodes
        created_at: Creation timestamp
        updated_at: Last update timestamp
        version: Pipeline schema version
        metadata: Additional metadata

    Example:
        >>> pipeline = Pipeline.load("my_pipeline.yaml")
        >>> result = pipeline.validate()
        >>> if result.is_valid:
        ...     outputs = pipeline.execute()
    """

    id: str
    name: str
    description: str
    nodes: List[PipelineNode]
    connections: List[PipelineConnection]
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": [n.to_dict() for n in self.nodes],
            "connections": [c.to_dict() for c in self.connections],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pipeline":
        """Create from dictionary."""
        nodes = [PipelineNode.from_dict(n) for n in data.get("nodes", [])]
        connections = [PipelineConnection(**c) for c in data.get("connections", [])]

        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            nodes=nodes,
            connections=connections,
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            version=data.get("version", "1.0"),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def load(cls, path: str) -> "Pipeline":
        """Load pipeline from a YAML or JSON file.

        Args:
            path: Path to the pipeline configuration file

        Returns:
            Pipeline instance
        """
        import json

        import yaml

        with open(path) as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        return cls.from_dict(data)

    def save(self, path: str) -> None:
        """Save pipeline to a YAML or JSON file.

        Args:
            path: Path to save the pipeline configuration
        """
        import json

        import yaml

        with open(path, "w") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                yaml.safe_dump(self.to_dict(), f, default_flow_style=False)
            else:
                json.dump(self.to_dict(), f, indent=2)

    def validate(self) -> ValidationResult:
        """Validate pipeline structure.

        Checks:
            - All connections reference valid nodes
            - No cycles exist in the pipeline
            - All required inputs are connected

        Returns:
            ValidationResult with is_valid and errors
        """
        errors = []
        node_ids = {n.id for n in self.nodes}

        # Check all connections reference valid nodes
        for conn in self.connections:
            if conn.source_node not in node_ids:
                errors.append(f"Connection references unknown source node: {conn.source_node}")
            if conn.target_node not in node_ids:
                errors.append(f"Connection references unknown target node: {conn.target_node}")

        # Check for cycles
        if self._has_cycle():
            errors.append("Pipeline contains a cycle")

        # Check required inputs are connected
        connected_inputs = {(c.target_node, c.target_port) for c in self.connections}

        for node in self.nodes:
            for port in node.inputs:
                if port.required and (node.id, port.id) not in connected_inputs:
                    errors.append(f"Node '{node.name}' has unconnected required input: {port.name}")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def execute(self) -> Dict[str, Any]:
        """Execute the pipeline.

        Returns:
            Dictionary of node results
        """
        from genesis.pipeline.executor import PipelineExecutor

        executor = PipelineExecutor()
        return executor.execute(self)

    def _has_cycle(self) -> bool:
        """Check for cycles in the pipeline using DFS."""
        # Build adjacency list
        node_ids = {n.id for n in self.nodes}
        graph: Dict[str, List[str]] = {n.id: [] for n in self.nodes}

        for conn in self.connections:
            # Skip connections with invalid nodes
            if conn.source_node not in node_ids or conn.target_node not in node_ids:
                continue
            graph[conn.source_node].append(conn.target_node)

        # DFS for cycle detection
        visited: set[str] = set()
        rec_stack: set[str] = set()

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node_id in graph:
            if node_id not in visited:
                if dfs(node_id):
                    return True

        return False

    def get_execution_order(self) -> List[str]:
        """Get topological order for execution using Kahn's algorithm.

        Returns:
            List of node IDs in execution order
        """
        # Build in-degree map
        in_degree = {n.id: 0 for n in self.nodes}
        graph: Dict[str, List[str]] = {n.id: [] for n in self.nodes}

        for conn in self.connections:
            graph[conn.source_node].append(conn.target_node)
            in_degree[conn.target_node] += 1

        # Kahn's algorithm
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)

            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return order


def create_simple_pipeline(
    name: str,
    input_variable: str,
    method: str = "auto",
    n_samples: int = 1000,
    output_path: str | None = None,
) -> Pipeline:
    """Create a simple generation pipeline.

    This convenience function creates a basic pipeline that:
    1. Loads data from a variable
    2. Generates synthetic data
    3. Optionally saves to a file

    Args:
        name: Pipeline name
        input_variable: Input data variable name
        method: Generation method (auto, gaussian_copula, ctgan, tvae)
        n_samples: Number of samples to generate
        output_path: Optional output file path

    Returns:
        Configured Pipeline ready for execution

    Example:
        >>> pipeline = create_simple_pipeline(
        ...     "quick_gen",
        ...     "customer_data",
        ...     n_samples=5000,
        ...     output_path="synthetic.csv"
        ... )
        >>> pipeline.execute()
    """
    from genesis.pipeline.builder import PipelineBuilder

    builder = PipelineBuilder(name)

    # Add nodes
    source = builder.add_node(
        NodeType.DATA_SOURCE,
        {"variable_name": input_variable},
    )

    generator = builder.add_node(
        NodeType.GENERATOR,
        {"method": method, "n_samples": n_samples},
    )

    # Connect
    builder.connect(source, "data", generator, "training_data")

    # Add output if specified
    if output_path:
        output = builder.add_node(
            NodeType.FILE_OUTPUT,
            {"file_path": output_path, "file_format": "csv"},
        )
        builder.connect(generator, "synthetic_data", output, "data")

    return builder.build()
