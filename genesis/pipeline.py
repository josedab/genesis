"""Visual Pipeline Builder Backend.

A fluent API and execution engine for building complex synthetic data
generation workflows. Supports visual pipeline construction with nodes
for data sources, transformations, generation, evaluation, and output.

Features:
    - Fluent builder API for programmatic pipeline construction
    - Multiple node types (source, transform, synthesize, evaluate, sink)
    - YAML serialization for pipeline definitions
    - Topological execution with dependency resolution
    - Parallel execution for independent nodes
    - Validation and error handling

Example:
    Building a simple pipeline::

        from genesis.pipeline import PipelineBuilder

        pipeline = (
            PipelineBuilder()
            .source("customers.csv")
            .transform("clean", {"drop_na": True})
            .synthesize(method="ctgan", n_samples=10000)
            .evaluate()
            .sink("synthetic_customers.csv")
            .build()
        )

        result = pipeline.execute()
        print(f"Quality score: {result['evaluate']['overall_score']:.1%}")

    Branching pipeline::

        pipeline = (
            PipelineBuilder()
            .source("data.csv", name="source")
            .synthesize(method="ctgan", input="source", name="ctgan_out")
            .synthesize(method="tvae", input="source", name="tvae_out")
            .sink("ctgan.csv", input="ctgan_out")
            .sink("tvae.csv", input="tvae_out")
            .build()
        )

    Loading from YAML::

        pipeline = Pipeline.load("my_pipeline.yaml")
        pipeline.execute()

Classes:
    NodeType: Enum of available node types.
    PipelineNode: Base class for pipeline nodes.
    PipelineBuilder: Fluent API for building pipelines.
    Pipeline: Executable pipeline with validation.
    PipelineExecutor: Executes pipeline graphs.

Functions:
    create_simple_pipeline: Quick pipeline for basic generation.
    create_evaluation_pipeline: Pipeline for quality evaluation.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


@dataclass
class ValidationResult:
    """Result of pipeline validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)


class NodeType(str, Enum):
    """Types of pipeline nodes.

    Categories:
        Input: DATA_SOURCE, FILE_INPUT, DATABASE_INPUT
        Transform: FILTER, SELECT_COLUMNS, RENAME_COLUMNS, etc.
        Generate: SYNTHESIZE, AUGMENT, CONDITIONAL_GENERATE
        Evaluate: QUALITY_CHECK, PRIVACY_AUDIT, DRIFT_DETECT
        Output: FILE_OUTPUT, DATABASE_OUTPUT, API_OUTPUT
    """

    # Input nodes
    DATA_SOURCE = "data_source"
    FILE_INPUT = "file_input"
    DATABASE_INPUT = "database_input"

    # Transform nodes
    FILTER = "filter"
    SELECT_COLUMNS = "select_columns"
    RENAME_COLUMNS = "rename_columns"
    TYPE_CAST = "type_cast"
    AGGREGATE = "aggregate"
    JOIN = "join"

    # Generator nodes
    GENERATOR = "generator"
    CONDITIONAL_GENERATOR = "conditional_generator"
    AUGMENTATION = "augmentation"

    # Privacy nodes
    PRIVACY_FILTER = "privacy_filter"
    ANONYMIZE = "anonymize"
    DIFFERENTIAL_PRIVACY = "differential_privacy"

    # Quality nodes
    QUALITY_CHECK = "quality_check"
    VALIDATION = "validation"

    # Output nodes
    DATA_OUTPUT = "data_output"
    FILE_OUTPUT = "file_output"
    DATABASE_OUTPUT = "database_output"


@dataclass
class NodePort:
    """Input or output port of a node."""

    id: str
    name: str
    port_type: str  # "input" or "output"
    data_type: str  # "dataframe", "config", "model"
    required: bool = True
    multi: bool = False  # Can accept multiple connections


@dataclass
class PipelineNode:
    """A node in the pipeline."""

    id: str
    node_type: NodeType
    name: str
    config: Dict[str, Any]
    position: Dict[str, float]  # x, y coordinates
    inputs: List[NodePort] = field(default_factory=list)
    outputs: List[NodePort] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "name": self.name,
            "config": self.config,
            "position": self.position,
            "inputs": [
                {
                    "id": p.id,
                    "name": p.name,
                    "port_type": p.port_type,
                    "data_type": p.data_type,
                    "required": p.required,
                }
                for p in self.inputs
            ],
            "outputs": [
                {
                    "id": p.id,
                    "name": p.name,
                    "port_type": p.port_type,
                    "data_type": p.data_type,
                    "required": p.required,
                }
                for p in self.outputs
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineNode":
        """Create from dictionary."""
        inputs = [NodePort(**p) for p in data.get("inputs", [])]
        outputs = [NodePort(**p) for p in data.get("outputs", [])]

        return cls(
            id=data["id"],
            node_type=NodeType(data["node_type"]),
            name=data["name"],
            config=data.get("config", {}),
            position=data.get("position", {"x": 0, "y": 0}),
            inputs=inputs,
            outputs=outputs,
        )


@dataclass
class PipelineConnection:
    """Connection between nodes."""

    id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source_node": self.source_node,
            "source_port": self.source_port,
            "target_node": self.target_node,
            "target_port": self.target_port,
        }


@dataclass
class Pipeline:
    """A complete pipeline definition."""

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
        """Convert to dictionary."""
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

    def validate(self) -> "ValidationResult":
        """Validate pipeline structure.

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
        from genesis.pipeline import PipelineExecutor

        executor = PipelineExecutor()
        return executor.execute(self)

    def _has_cycle(self) -> bool:
        """Check for cycles in the pipeline."""
        # Build adjacency list
        node_ids = {n.id for n in self.nodes}
        graph = {n.id: [] for n in self.nodes}

        for conn in self.connections:
            # Skip connections with invalid nodes
            if conn.source_node not in node_ids or conn.target_node not in node_ids:
                continue
            graph[conn.source_node].append(conn.target_node)

        # DFS for cycle detection
        visited = set()
        rec_stack = set()

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
        """Get topological order for execution.

        Returns:
            List of node IDs in execution order
        """
        # Build in-degree map
        in_degree = {n.id: 0 for n in self.nodes}
        graph = {n.id: [] for n in self.nodes}

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


# Node templates
NODE_TEMPLATES = {
    NodeType.DATA_SOURCE: {
        "name": "Data Source",
        "description": "Load data from memory",
        "config_schema": {
            "variable_name": {"type": "string", "required": True},
        },
        "inputs": [],
        "outputs": [
            {"id": "data", "name": "Data", "port_type": "output", "data_type": "dataframe"},
        ],
    },
    NodeType.FILE_INPUT: {
        "name": "File Input",
        "description": "Load data from file",
        "config_schema": {
            "file_path": {"type": "string", "required": True},
            "file_format": {"type": "enum", "values": ["csv", "parquet", "json"], "default": "csv"},
        },
        "inputs": [],
        "outputs": [
            {"id": "data", "name": "Data", "port_type": "output", "data_type": "dataframe"},
        ],
    },
    NodeType.FILTER: {
        "name": "Filter",
        "description": "Filter rows based on condition",
        "config_schema": {
            "condition": {"type": "string", "required": True},
        },
        "inputs": [
            {"id": "input", "name": "Input", "port_type": "input", "data_type": "dataframe"},
        ],
        "outputs": [
            {"id": "output", "name": "Output", "port_type": "output", "data_type": "dataframe"},
        ],
    },
    NodeType.SELECT_COLUMNS: {
        "name": "Select Columns",
        "description": "Select specific columns",
        "config_schema": {
            "columns": {"type": "array", "items": "string", "required": True},
        },
        "inputs": [
            {"id": "input", "name": "Input", "port_type": "input", "data_type": "dataframe"},
        ],
        "outputs": [
            {"id": "output", "name": "Output", "port_type": "output", "data_type": "dataframe"},
        ],
    },
    NodeType.GENERATOR: {
        "name": "Synthetic Generator",
        "description": "Generate synthetic data",
        "config_schema": {
            "method": {
                "type": "enum",
                "values": ["auto", "gaussian_copula", "ctgan", "tvae"],
                "default": "auto",
            },
            "n_samples": {"type": "integer", "required": True, "min": 1},
            "discrete_columns": {"type": "array", "items": "string"},
        },
        "inputs": [
            {
                "id": "training_data",
                "name": "Training Data",
                "port_type": "input",
                "data_type": "dataframe",
            },
        ],
        "outputs": [
            {
                "id": "synthetic_data",
                "name": "Synthetic Data",
                "port_type": "output",
                "data_type": "dataframe",
            },
        ],
    },
    NodeType.CONDITIONAL_GENERATOR: {
        "name": "Conditional Generator",
        "description": "Generate data with conditions",
        "config_schema": {
            "method": {
                "type": "enum",
                "values": ["auto", "gaussian_copula", "ctgan"],
                "default": "auto",
            },
            "n_samples": {"type": "integer", "required": True, "min": 1},
            "conditions": {"type": "object", "required": True},
        },
        "inputs": [
            {
                "id": "training_data",
                "name": "Training Data",
                "port_type": "input",
                "data_type": "dataframe",
            },
        ],
        "outputs": [
            {
                "id": "synthetic_data",
                "name": "Synthetic Data",
                "port_type": "output",
                "data_type": "dataframe",
            },
        ],
    },
    NodeType.QUALITY_CHECK: {
        "name": "Quality Check",
        "description": "Evaluate synthetic data quality",
        "config_schema": {
            "metrics": {
                "type": "array",
                "items": "string",
                "default": ["statistical", "ml_utility", "privacy"],
            },
            "threshold": {"type": "number", "min": 0, "max": 1, "default": 0.8},
        },
        "inputs": [
            {
                "id": "real_data",
                "name": "Real Data",
                "port_type": "input",
                "data_type": "dataframe",
            },
            {
                "id": "synthetic_data",
                "name": "Synthetic Data",
                "port_type": "input",
                "data_type": "dataframe",
            },
        ],
        "outputs": [
            {
                "id": "report",
                "name": "Quality Report",
                "port_type": "output",
                "data_type": "report",
            },
            {
                "id": "passed_data",
                "name": "Passed Data",
                "port_type": "output",
                "data_type": "dataframe",
            },
        ],
    },
    NodeType.FILE_OUTPUT: {
        "name": "File Output",
        "description": "Save data to file",
        "config_schema": {
            "file_path": {"type": "string", "required": True},
            "file_format": {"type": "enum", "values": ["csv", "parquet", "json"], "default": "csv"},
        },
        "inputs": [
            {"id": "data", "name": "Data", "port_type": "input", "data_type": "dataframe"},
        ],
        "outputs": [],
    },
    NodeType.PRIVACY_FILTER: {
        "name": "Privacy Filter",
        "description": "Apply privacy protections",
        "config_schema": {
            "method": {
                "type": "enum",
                "values": ["k_anonymity", "l_diversity", "differential_privacy"],
            },
            "k": {"type": "integer", "min": 2, "default": 5},
            "epsilon": {"type": "number", "min": 0.01, "default": 1.0},
            "quasi_identifiers": {"type": "array", "items": "string"},
        },
        "inputs": [
            {"id": "input", "name": "Input", "port_type": "input", "data_type": "dataframe"},
        ],
        "outputs": [
            {
                "id": "output",
                "name": "Protected Data",
                "port_type": "output",
                "data_type": "dataframe",
            },
        ],
    },
    NodeType.AUGMENTATION: {
        "name": "Data Augmentation",
        "description": "Augment imbalanced data",
        "config_schema": {
            "target_column": {"type": "string", "required": True},
            "target_ratio": {"type": "number", "min": 0, "max": 1, "default": 1.0},
            "strategy": {
                "type": "enum",
                "values": ["oversample", "undersample", "hybrid"],
                "default": "oversample",
            },
        },
        "inputs": [
            {"id": "input", "name": "Input", "port_type": "input", "data_type": "dataframe"},
        ],
        "outputs": [
            {
                "id": "output",
                "name": "Augmented Data",
                "port_type": "output",
                "data_type": "dataframe",
            },
        ],
    },
}


class PipelineBuilder:
    """Builder for creating pipelines programmatically."""

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
            node_type: Type of node
            config: Node configuration
            name: Node name (optional)
            position: Visual position (optional)

        Returns:
            Node ID
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
            source_port: Source port name
            target_node: Target node ID
            target_port: Target port name

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


class PipelineExecutor:
    """Execute pipelines."""

    def __init__(self):
        """Initialize executor."""
        self._node_handlers: Dict[NodeType, Callable] = {
            NodeType.DATA_SOURCE: self._handle_data_source,
            NodeType.FILE_INPUT: self._handle_file_input,
            NodeType.FILTER: self._handle_filter,
            NodeType.SELECT_COLUMNS: self._handle_select_columns,
            NodeType.GENERATOR: self._handle_generator,
            NodeType.QUALITY_CHECK: self._handle_quality_check,
            NodeType.FILE_OUTPUT: self._handle_file_output,
            NodeType.AUGMENTATION: self._handle_augmentation,
        }

        self._context: Dict[str, Any] = {}

    def execute(
        self,
        pipeline: Pipeline,
        inputs: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """Execute a pipeline.

        Args:
            pipeline: Pipeline to execute
            inputs: Input data by variable name

        Returns:
            Dict of outputs by node ID
        """
        self._context = {"inputs": inputs or {}}
        outputs = {}

        # Get execution order
        order = pipeline.get_execution_order()
        node_map = {n.id: n for n in pipeline.nodes}

        # Build connection map
        conn_map = {}  # target_port -> source output
        for conn in pipeline.connections:
            conn_map[conn.target_port] = (conn.source_node, conn.source_port)

        # Execute nodes
        for node_id in order:
            node = node_map[node_id]

            # Gather inputs
            node_inputs = {}
            for port in node.inputs:
                if port.id in conn_map:
                    source_node, source_port = conn_map[port.id]
                    if source_node in outputs:
                        port_name = port.id.replace(f"{node_id}_", "")
                        source_port_name = source_port.replace(f"{source_node}_", "")
                        node_inputs[port_name] = outputs[source_node].get(source_port_name)

            # Execute node
            handler = self._node_handlers.get(node.node_type)
            if handler:
                result = handler(node, node_inputs)
                outputs[node_id] = result
            else:
                outputs[node_id] = {}

        return outputs

    def _handle_data_source(self, node: PipelineNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data source node."""
        var_name = node.config.get("variable_name")
        data = self._context.get("inputs", {}).get(var_name)
        return {"data": data}

    def _handle_file_input(self, node: PipelineNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file input node."""
        path = node.config.get("file_path")
        fmt = node.config.get("file_format", "csv")

        if fmt == "csv":
            data = pd.read_csv(path)
        elif fmt == "parquet":
            data = pd.read_parquet(path)
        elif fmt == "json":
            data = pd.read_json(path)
        else:
            data = pd.read_csv(path)

        return {"data": data}

    def _handle_filter(self, node: PipelineNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle filter node."""
        data = inputs.get("input")
        condition = node.config.get("condition", "True")

        if data is not None:
            filtered = data.query(condition)
            return {"output": filtered}

        return {"output": None}

    def _handle_select_columns(self, node: PipelineNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle select columns node."""
        data = inputs.get("input")
        columns = node.config.get("columns", [])

        if data is not None and columns:
            selected = data[columns]
            return {"output": selected}

        return {"output": data}

    def _handle_generator(self, node: PipelineNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generator node."""
        training_data = inputs.get("training_data")

        if training_data is None:
            return {"synthetic_data": None}

        method = node.config.get("method", "auto")
        n_samples = node.config.get("n_samples", len(training_data))
        discrete_columns = node.config.get("discrete_columns", [])

        # Use AutoML or specific method
        if method == "auto":
            from genesis.automl import AutoMLSynthesizer

            gen = AutoMLSynthesizer()
        else:
            from genesis.generators.tabular import CTGANGenerator, GaussianCopulaGenerator

            if method == "ctgan":
                gen = CTGANGenerator()
            else:
                gen = GaussianCopulaGenerator()

        gen.fit(training_data, discrete_columns=discrete_columns)
        synthetic = gen.generate(n_samples)

        return {"synthetic_data": synthetic}

    def _handle_quality_check(self, node: PipelineNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle quality check node."""
        real = inputs.get("real_data")
        synthetic = inputs.get("synthetic_data")

        if real is None or synthetic is None:
            return {"report": None, "passed_data": synthetic}

        from genesis.evaluation import QualityEvaluator

        evaluator = QualityEvaluator(real, synthetic)
        report = evaluator.evaluate()

        threshold = node.config.get("threshold", 0.8)
        passed = report.overall_score >= threshold

        return {
            "report": report.to_dict(),
            "passed_data": synthetic if passed else None,
        }

    def _handle_file_output(self, node: PipelineNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file output node."""
        data = inputs.get("data")

        if data is not None:
            path = node.config.get("file_path")
            fmt = node.config.get("file_format", "csv")

            if fmt == "csv":
                data.to_csv(path, index=False)
            elif fmt == "parquet":
                data.to_parquet(path, index=False)
            elif fmt == "json":
                data.to_json(path, orient="records")

        return {}

    def _handle_augmentation(self, node: PipelineNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle augmentation node."""
        data = inputs.get("input")

        if data is None:
            return {"output": None}

        from genesis.augmentation import AugmentationStrategy, SyntheticAugmenter

        target_column = node.config.get("target_column")
        target_ratio = node.config.get("target_ratio", 1.0)
        strategy = AugmentationStrategy(node.config.get("strategy", "oversample"))

        augmenter = SyntheticAugmenter(target_ratio=target_ratio)
        augmenter.fit(data, target_column)
        result = augmenter.augment(strategy=strategy)

        return {"output": result.augmented_data}


def get_node_templates() -> Dict[str, Any]:
    """Get all node templates for UI.

    Returns:
        Dict of node templates
    """
    return {k.value: v for k, v in NODE_TEMPLATES.items()}


def create_simple_pipeline(
    name: str,
    input_variable: str,
    method: str = "auto",
    n_samples: int = 1000,
    output_path: Optional[str] = None,
) -> Pipeline:
    """Create a simple generation pipeline.

    Args:
        name: Pipeline name
        input_variable: Input data variable name
        method: Generation method
        n_samples: Number of samples
        output_path: Optional output file path

    Returns:
        Pipeline
    """
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
