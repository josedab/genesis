"""Fluent pipeline builder API."""

import uuid
from typing import Any, Dict, Optional

from genesis.pipeline.nodes import (
    NODE_TEMPLATES,
    NodePort,
    NodeType,
    PipelineConnection,
    PipelineNode,
)
from genesis.pipeline.pipeline import Pipeline


class PipelineBuilder:
    """Builder for creating pipelines programmatically.

    Provides a fluent API for constructing complex data generation pipelines.
    Validates the pipeline structure on build.

    Example:
        >>> builder = PipelineBuilder("customer_gen", "Generate synthetic customers")
        >>> pipeline = (
        ...     builder
        ...     .add_node(NodeType.FILE_INPUT, {"file_path": "data.csv"})
        ...     .add_node(NodeType.GENERATOR, {"method": "ctgan", "n_samples": 1000})
        ...     .connect("node_0", "data", "node_1", "training_data")
        ...     .build()
        ... )
    """

    def __init__(self, name: str, description: str = ""):
        """Initialize builder.

        Args:
            name: Pipeline name
            description: Pipeline description
        """
        self.pipeline = Pipeline(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            nodes=[],
            connections=[],
        )
        self._node_counter = 0

    def add_node(
        self,
        node_type: NodeType,
        config: Dict[str, Any],
        name: Optional[str] = None,
        position: Optional[Dict[str, float]] = None,
    ) -> str:
        """Add a node to the pipeline.

        Args:
            node_type: Type of node from NodeType enum
            config: Node configuration dictionary
            name: Node name (optional, defaults to template name)
            position: Visual position as {x, y} dict (optional)

        Returns:
            Node ID for use in connections
        """
        node_id = f"node_{self._node_counter}"
        self._node_counter += 1

        template = NODE_TEMPLATES.get(node_type, {})

        inputs = [
            NodePort(**{**p, "id": f"{node_id}_{p['id']}"}) for p in template.get("inputs", [])
        ]
        outputs = [
            NodePort(**{**p, "id": f"{node_id}_{p['id']}"}) for p in template.get("outputs", [])
        ]

        node = PipelineNode(
            id=node_id,
            node_type=node_type,
            name=name or template.get("name", node_type.value),
            config=config,
            position=position or {"x": self._node_counter * 200, "y": 100},
            inputs=inputs,
            outputs=outputs,
        )

        self.pipeline.nodes.append(node)
        return node_id

    def connect(
        self,
        source_node: str,
        source_port: str,
        target_node: str,
        target_port: str,
    ) -> str:
        """Connect two nodes.

        Args:
            source_node: Source node ID
            source_port: Source port name (e.g., "data", "output")
            target_node: Target node ID
            target_port: Target port name (e.g., "input", "training_data")

        Returns:
            Connection ID
        """
        conn_id = f"conn_{len(self.pipeline.connections)}"

        # Find full port IDs
        source_port_id = f"{source_node}_{source_port}"
        target_port_id = f"{target_node}_{target_port}"

        connection = PipelineConnection(
            id=conn_id,
            source_node=source_node,
            source_port=source_port_id,
            target_node=target_node,
            target_port=target_port_id,
        )

        self.pipeline.connections.append(connection)
        return conn_id

    def build(self) -> Pipeline:
        """Build and validate pipeline.

        Returns:
            Pipeline instance

        Raises:
            ValueError: If pipeline is invalid
        """
        result = self.pipeline.validate()
        if not result.is_valid:
            raise ValueError(f"Invalid pipeline: {'; '.join(result.errors)}")

        return self.pipeline
