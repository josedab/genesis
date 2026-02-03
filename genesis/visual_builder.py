"""Low-Code Visual Builder Backend for Genesis.

This module provides a REST API backend for a visual drag-and-drop
pipeline builder with live preview and collaboration features.

Example:
    >>> from genesis.visual_builder import VisualBuilderAPI
    >>>
    >>> # Start the API server
    >>> api = VisualBuilderAPI()
    >>> api.run(port=8080)
    >>>
    >>> # Or use programmatically
    >>> from genesis.visual_builder import VisualPipeline, NodeDefinition
    >>>
    >>> pipeline = VisualPipeline("my_pipeline")
    >>> pipeline.add_node(NodeDefinition(
    ...     id="source",
    ...     type="data_source",
    ...     config={"file_path": "data.csv"}
    ... ))
    >>> result = pipeline.execute()
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from genesis.core.base import SyntheticGenerator
from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.core.exceptions import ConfigurationError, ValidationError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class NodeCategory(Enum):
    """Categories of nodes in the visual builder."""

    DATA_SOURCE = "data_source"
    TRANSFORM = "transform"
    GENERATOR = "generator"
    EVALUATOR = "evaluator"
    OUTPUT = "output"
    CONTROL = "control"


class NodeType(Enum):
    """Types of nodes available in the visual builder."""

    # Data Sources
    CSV_SOURCE = "csv_source"
    PARQUET_SOURCE = "parquet_source"
    JSON_SOURCE = "json_source"
    DATABASE_SOURCE = "database_source"
    API_SOURCE = "api_source"

    # Transforms
    FILTER = "filter"
    SELECT_COLUMNS = "select_columns"
    RENAME_COLUMNS = "rename_columns"
    TYPE_CAST = "type_cast"
    FILL_MISSING = "fill_missing"
    DROP_DUPLICATES = "drop_duplicates"
    SAMPLE = "sample"
    MERGE = "merge"
    AGGREGATE = "aggregate"
    CUSTOM_TRANSFORM = "custom_transform"

    # Generators
    CTGAN = "ctgan"
    TVAE = "tvae"
    GAUSSIAN_COPULA = "gaussian_copula"
    AUTO_SELECT = "auto_select"
    CONDITIONAL = "conditional"

    # Evaluators
    QUALITY_EVALUATOR = "quality_evaluator"
    PRIVACY_EVALUATOR = "privacy_evaluator"
    STATISTICAL_EVALUATOR = "statistical_evaluator"
    COMPARISON = "comparison"

    # Outputs
    CSV_OUTPUT = "csv_output"
    PARQUET_OUTPUT = "parquet_output"
    DATABASE_OUTPUT = "database_output"
    PREVIEW = "preview"

    # Control
    BRANCH = "branch"
    MERGE_BRANCH = "merge_branch"
    LOOP = "loop"
    CONDITIONAL_BRANCH = "conditional_branch"


@dataclass
class PortDefinition:
    """Definition of an input or output port on a node."""

    id: str
    name: str
    data_type: str  # 'dataframe', 'config', 'report', 'any'
    required: bool = True
    multiple: bool = False  # Can accept multiple connections

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "data_type": self.data_type,
            "required": self.required,
            "multiple": self.multiple,
        }


@dataclass
class ParameterDefinition:
    """Definition of a configurable parameter for a node."""

    id: str
    name: str
    param_type: str  # 'string', 'number', 'boolean', 'select', 'multiselect', 'code'
    default: Any = None
    required: bool = False
    options: Optional[List[Any]] = None  # For select/multiselect
    min_value: Optional[float] = None  # For number
    max_value: Optional[float] = None
    placeholder: Optional[str] = None
    description: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "param_type": self.param_type,
            "default": self.default,
            "required": self.required,
            "options": self.options,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "placeholder": self.placeholder,
            "description": self.description,
        }


@dataclass
class NodeTemplate:
    """Template defining a type of node for the visual builder."""

    node_type: NodeType
    category: NodeCategory
    name: str
    description: str
    icon: str = "📦"
    color: str = "#4A90D9"
    inputs: List[PortDefinition] = field(default_factory=list)
    outputs: List[PortDefinition] = field(default_factory=list)
    parameters: List[ParameterDefinition] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_type": self.node_type.value,
            "category": self.category.value,
            "name": self.name,
            "description": self.description,
            "icon": self.icon,
            "color": self.color,
            "inputs": [i.to_dict() for i in self.inputs],
            "outputs": [o.to_dict() for o in self.outputs],
            "parameters": [p.to_dict() for p in self.parameters],
        }


# Node templates library
NODE_TEMPLATES: Dict[NodeType, NodeTemplate] = {
    # Data Sources
    NodeType.CSV_SOURCE: NodeTemplate(
        node_type=NodeType.CSV_SOURCE,
        category=NodeCategory.DATA_SOURCE,
        name="CSV Source",
        description="Load data from a CSV file",
        icon="📄",
        color="#4CAF50",
        inputs=[],
        outputs=[
            PortDefinition("data", "Data", "dataframe"),
        ],
        parameters=[
            ParameterDefinition("file_path", "File Path", "string", required=True, placeholder="/path/to/data.csv"),
            ParameterDefinition("delimiter", "Delimiter", "string", default=","),
            ParameterDefinition("encoding", "Encoding", "select", default="utf-8", options=["utf-8", "latin-1", "ascii"]),
            ParameterDefinition("has_header", "Has Header", "boolean", default=True),
        ],
    ),
    NodeType.DATABASE_SOURCE: NodeTemplate(
        node_type=NodeType.DATABASE_SOURCE,
        category=NodeCategory.DATA_SOURCE,
        name="Database Source",
        description="Load data from a database",
        icon="🗄️",
        color="#4CAF50",
        inputs=[],
        outputs=[
            PortDefinition("data", "Data", "dataframe"),
        ],
        parameters=[
            ParameterDefinition("connection_string", "Connection String", "string", required=True),
            ParameterDefinition("query", "SQL Query", "code", required=True, placeholder="SELECT * FROM table"),
        ],
    ),

    # Transforms
    NodeType.FILTER: NodeTemplate(
        node_type=NodeType.FILTER,
        category=NodeCategory.TRANSFORM,
        name="Filter",
        description="Filter rows based on a condition",
        icon="🔍",
        color="#FF9800",
        inputs=[
            PortDefinition("data", "Data", "dataframe"),
        ],
        outputs=[
            PortDefinition("data", "Filtered Data", "dataframe"),
        ],
        parameters=[
            ParameterDefinition("condition", "Condition", "code", required=True, placeholder="column > 0"),
        ],
    ),
    NodeType.SELECT_COLUMNS: NodeTemplate(
        node_type=NodeType.SELECT_COLUMNS,
        category=NodeCategory.TRANSFORM,
        name="Select Columns",
        description="Select specific columns",
        icon="📋",
        color="#FF9800",
        inputs=[
            PortDefinition("data", "Data", "dataframe"),
        ],
        outputs=[
            PortDefinition("data", "Selected Data", "dataframe"),
        ],
        parameters=[
            ParameterDefinition("columns", "Columns", "multiselect", required=True),
        ],
    ),
    NodeType.FILL_MISSING: NodeTemplate(
        node_type=NodeType.FILL_MISSING,
        category=NodeCategory.TRANSFORM,
        name="Fill Missing",
        description="Handle missing values",
        icon="🔧",
        color="#FF9800",
        inputs=[
            PortDefinition("data", "Data", "dataframe"),
        ],
        outputs=[
            PortDefinition("data", "Cleaned Data", "dataframe"),
        ],
        parameters=[
            ParameterDefinition("strategy", "Strategy", "select", default="mean", options=["mean", "median", "mode", "constant", "drop"]),
            ParameterDefinition("fill_value", "Fill Value", "string", placeholder="Value for constant strategy"),
        ],
    ),
    NodeType.SAMPLE: NodeTemplate(
        node_type=NodeType.SAMPLE,
        category=NodeCategory.TRANSFORM,
        name="Sample",
        description="Sample rows from data",
        icon="🎲",
        color="#FF9800",
        inputs=[
            PortDefinition("data", "Data", "dataframe"),
        ],
        outputs=[
            PortDefinition("data", "Sampled Data", "dataframe"),
        ],
        parameters=[
            ParameterDefinition("n_samples", "Number of Samples", "number", default=1000, min_value=1),
            ParameterDefinition("random_state", "Random Seed", "number"),
            ParameterDefinition("replace", "With Replacement", "boolean", default=False),
        ],
    ),

    # Generators
    NodeType.AUTO_SELECT: NodeTemplate(
        node_type=NodeType.AUTO_SELECT,
        category=NodeCategory.GENERATOR,
        name="Auto Generator",
        description="Automatically select best generation method",
        icon="🤖",
        color="#9C27B0",
        inputs=[
            PortDefinition("data", "Training Data", "dataframe"),
        ],
        outputs=[
            PortDefinition("synthetic", "Synthetic Data", "dataframe"),
            PortDefinition("generator", "Fitted Generator", "generator"),
        ],
        parameters=[
            ParameterDefinition("n_samples", "Number of Samples", "number", default=1000, min_value=1, required=True),
            ParameterDefinition("privacy_level", "Privacy Level", "select", default="medium", options=["none", "low", "medium", "high", "maximum"]),
            ParameterDefinition("quality_preference", "Prefer Quality", "boolean", default=True),
        ],
    ),
    NodeType.CTGAN: NodeTemplate(
        node_type=NodeType.CTGAN,
        category=NodeCategory.GENERATOR,
        name="CTGAN Generator",
        description="Conditional Tabular GAN for mixed data",
        icon="🧠",
        color="#9C27B0",
        inputs=[
            PortDefinition("data", "Training Data", "dataframe"),
        ],
        outputs=[
            PortDefinition("synthetic", "Synthetic Data", "dataframe"),
            PortDefinition("generator", "Fitted Generator", "generator"),
        ],
        parameters=[
            ParameterDefinition("n_samples", "Number of Samples", "number", default=1000, min_value=1, required=True),
            ParameterDefinition("epochs", "Training Epochs", "number", default=300, min_value=1, max_value=10000),
            ParameterDefinition("batch_size", "Batch Size", "number", default=500, min_value=1),
            ParameterDefinition("discrete_columns", "Discrete Columns", "multiselect"),
            ParameterDefinition("privacy_level", "Privacy Level", "select", default="none", options=["none", "low", "medium", "high", "maximum"]),
        ],
    ),
    NodeType.GAUSSIAN_COPULA: NodeTemplate(
        node_type=NodeType.GAUSSIAN_COPULA,
        category=NodeCategory.GENERATOR,
        name="Gaussian Copula",
        description="Statistical method preserving correlations",
        icon="📊",
        color="#9C27B0",
        inputs=[
            PortDefinition("data", "Training Data", "dataframe"),
        ],
        outputs=[
            PortDefinition("synthetic", "Synthetic Data", "dataframe"),
        ],
        parameters=[
            ParameterDefinition("n_samples", "Number of Samples", "number", default=1000, min_value=1, required=True),
        ],
    ),

    # Evaluators
    NodeType.QUALITY_EVALUATOR: NodeTemplate(
        node_type=NodeType.QUALITY_EVALUATOR,
        category=NodeCategory.EVALUATOR,
        name="Quality Evaluator",
        description="Evaluate synthetic data quality",
        icon="✅",
        color="#2196F3",
        inputs=[
            PortDefinition("real", "Real Data", "dataframe"),
            PortDefinition("synthetic", "Synthetic Data", "dataframe"),
        ],
        outputs=[
            PortDefinition("report", "Quality Report", "report"),
        ],
        parameters=[
            ParameterDefinition("metrics", "Metrics", "multiselect", default=["statistical", "ml_utility", "privacy"], options=["statistical", "ml_utility", "privacy", "all"]),
        ],
    ),

    # Outputs
    NodeType.CSV_OUTPUT: NodeTemplate(
        node_type=NodeType.CSV_OUTPUT,
        category=NodeCategory.OUTPUT,
        name="CSV Output",
        description="Save data to CSV file",
        icon="💾",
        color="#607D8B",
        inputs=[
            PortDefinition("data", "Data", "dataframe"),
        ],
        outputs=[],
        parameters=[
            ParameterDefinition("file_path", "File Path", "string", required=True, placeholder="/path/to/output.csv"),
            ParameterDefinition("include_index", "Include Index", "boolean", default=False),
        ],
    ),
    NodeType.PREVIEW: NodeTemplate(
        node_type=NodeType.PREVIEW,
        category=NodeCategory.OUTPUT,
        name="Preview",
        description="Preview data in the builder",
        icon="👁️",
        color="#607D8B",
        inputs=[
            PortDefinition("data", "Data", "dataframe"),
        ],
        outputs=[],
        parameters=[
            ParameterDefinition("n_rows", "Rows to Show", "number", default=10, min_value=1, max_value=1000),
        ],
    ),
}


@dataclass
class NodeInstance:
    """An instance of a node in a pipeline."""

    id: str
    node_type: NodeType
    position: Dict[str, float]  # {"x": 100, "y": 200}
    config: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "node_type": self.node_type.value,
            "position": self.position,
            "config": self.config,
            "label": self.label or NODE_TEMPLATES[self.node_type].name,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInstance":
        return cls(
            id=data["id"],
            node_type=NodeType(data["node_type"]),
            position=data.get("position", {"x": 0, "y": 0}),
            config=data.get("config", {}),
            label=data.get("label"),
        )


@dataclass
class Connection:
    """A connection between two nodes."""

    id: str
    source_node: str
    source_port: str
    target_node: str
    target_port: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_node": self.source_node,
            "source_port": self.source_port,
            "target_node": self.target_node,
            "target_port": self.target_port,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Connection":
        return cls(
            id=data["id"],
            source_node=data["source_node"],
            source_port=data["source_port"],
            target_node=data["target_node"],
            target_port=data["target_port"],
        )


@dataclass
class VisualPipelineDefinition:
    """Definition of a visual pipeline."""

    id: str
    name: str
    description: str = ""
    nodes: List[NodeInstance] = field(default_factory=list)
    connections: List[Connection] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, data: Dict[str, Any]) -> "VisualPipelineDefinition":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            nodes=[NodeInstance.from_dict(n) for n in data.get("nodes", [])],
            connections=[Connection.from_dict(c) for c in data.get("connections", [])],
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
            version=data.get("version", 1),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "VisualPipelineDefinition":
        return cls.from_dict(json.loads(json_str))


class NodeExecutor:
    """Executes individual nodes in a pipeline."""

    def __init__(self):
        self._executors: Dict[NodeType, Callable] = {
            # Data Sources
            NodeType.CSV_SOURCE: self._execute_csv_source,
            NodeType.DATABASE_SOURCE: self._execute_database_source,
            # Transforms
            NodeType.FILTER: self._execute_filter,
            NodeType.SELECT_COLUMNS: self._execute_select_columns,
            NodeType.FILL_MISSING: self._execute_fill_missing,
            NodeType.SAMPLE: self._execute_sample,
            # Generators
            NodeType.AUTO_SELECT: self._execute_auto_generator,
            NodeType.CTGAN: self._execute_ctgan,
            NodeType.GAUSSIAN_COPULA: self._execute_gaussian_copula,
            # Evaluators
            NodeType.QUALITY_EVALUATOR: self._execute_quality_evaluator,
            # Outputs
            NodeType.CSV_OUTPUT: self._execute_csv_output,
            NodeType.PREVIEW: self._execute_preview,
        }

    def execute(
        self,
        node: NodeInstance,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a node.

        Args:
            node: Node instance to execute
            inputs: Input data for the node

        Returns:
            Dictionary of output port -> data
        """
        executor = self._executors.get(node.node_type)

        if executor is None:
            raise ConfigurationError(f"No executor for node type: {node.node_type}")

        return executor(node, inputs)

    # Source executors
    def _execute_csv_source(
        self,
        node: NodeInstance,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        file_path = node.config.get("file_path")
        delimiter = node.config.get("delimiter", ",")
        encoding = node.config.get("encoding", "utf-8")
        has_header = node.config.get("has_header", True)

        df = pd.read_csv(
            file_path,
            delimiter=delimiter,
            encoding=encoding,
            header=0 if has_header else None,
        )

        return {"data": df}

    def _execute_database_source(
        self,
        node: NodeInstance,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        from sqlalchemy import create_engine

        connection_string = node.config.get("connection_string")
        query = node.config.get("query")

        engine = create_engine(connection_string)
        df = pd.read_sql(query, engine)

        return {"data": df}

    # Transform executors
    def _execute_filter(
        self,
        node: NodeInstance,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        df = inputs.get("data")
        condition = node.config.get("condition")

        filtered_df = df.query(condition)

        return {"data": filtered_df}

    def _execute_select_columns(
        self,
        node: NodeInstance,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        df = inputs.get("data")
        columns = node.config.get("columns", [])

        selected_df = df[columns]

        return {"data": selected_df}

    def _execute_fill_missing(
        self,
        node: NodeInstance,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        df = inputs.get("data").copy()
        strategy = node.config.get("strategy", "mean")
        fill_value = node.config.get("fill_value")

        if strategy == "drop":
            df = df.dropna()
        elif strategy == "constant":
            df = df.fillna(fill_value)
        elif strategy == "mean":
            df = df.fillna(df.mean(numeric_only=True))
        elif strategy == "median":
            df = df.fillna(df.median(numeric_only=True))
        elif strategy == "mode":
            df = df.fillna(df.mode().iloc[0])

        return {"data": df}

    def _execute_sample(
        self,
        node: NodeInstance,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        df = inputs.get("data")
        n_samples = node.config.get("n_samples", 1000)
        random_state = node.config.get("random_state")
        replace = node.config.get("replace", False)

        sampled_df = df.sample(
            n=min(n_samples, len(df)),
            random_state=random_state,
            replace=replace,
        )

        return {"data": sampled_df}

    # Generator executors
    def _execute_auto_generator(
        self,
        node: NodeInstance,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        df = inputs.get("data")
        n_samples = node.config.get("n_samples", 1000)
        privacy_level = node.config.get("privacy_level", "none")

        from genesis.core.types import PrivacyLevel

        privacy = PrivacyConfig(privacy_level=PrivacyLevel(privacy_level))
        generator = SyntheticGenerator(method="auto", privacy=privacy)
        generator.fit(df)
        synthetic = generator.generate(n_samples=n_samples)

        return {"synthetic": synthetic, "generator": generator}

    def _execute_ctgan(
        self,
        node: NodeInstance,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        df = inputs.get("data")
        n_samples = node.config.get("n_samples", 1000)
        epochs = node.config.get("epochs", 300)
        discrete_columns = node.config.get("discrete_columns", [])

        config = GeneratorConfig(method="ctgan", epochs=epochs)
        generator = SyntheticGenerator(method="ctgan", config=config)
        generator.fit(df, discrete_columns=discrete_columns)
        synthetic = generator.generate(n_samples=n_samples)

        return {"synthetic": synthetic, "generator": generator}

    def _execute_gaussian_copula(
        self,
        node: NodeInstance,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        df = inputs.get("data")
        n_samples = node.config.get("n_samples", 1000)

        generator = SyntheticGenerator(method="gaussian_copula")
        generator.fit(df)
        synthetic = generator.generate(n_samples=n_samples)

        return {"synthetic": synthetic}

    # Evaluator executors
    def _execute_quality_evaluator(
        self,
        node: NodeInstance,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        real = inputs.get("real")
        synthetic = inputs.get("synthetic")

        from genesis.evaluation.evaluator import QualityEvaluator

        evaluator = QualityEvaluator(real, synthetic)
        report = evaluator.evaluate()

        return {"report": report}

    # Output executors
    def _execute_csv_output(
        self,
        node: NodeInstance,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        df = inputs.get("data")
        file_path = node.config.get("file_path")
        include_index = node.config.get("include_index", False)

        df.to_csv(file_path, index=include_index)

        return {}

    def _execute_preview(
        self,
        node: NodeInstance,
        inputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        df = inputs.get("data")
        n_rows = node.config.get("n_rows", 10)

        preview = df.head(n_rows).to_dict(orient="records")

        return {"preview": preview}


class VisualPipelineExecutor:
    """Executes visual pipelines."""

    def __init__(self):
        self._node_executor = NodeExecutor()

    def execute(
        self,
        pipeline: VisualPipelineDefinition,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """Execute a visual pipeline.

        Args:
            pipeline: Pipeline definition to execute
            progress_callback: Optional callback for progress updates

        Returns:
            Dictionary of results
        """
        # Build execution graph
        node_map = {n.id: n for n in pipeline.nodes}
        outputs: Dict[str, Dict[str, Any]] = {}
        results: Dict[str, Any] = {}

        # Get execution order (topological sort)
        execution_order = self._topological_sort(pipeline)

        # Execute nodes in order
        total_nodes = len(execution_order)
        for i, node_id in enumerate(execution_order):
            node = node_map[node_id]

            if progress_callback:
                progress_callback(f"Executing {node.label or node.node_type.value}", (i + 1) / total_nodes)

            # Gather inputs
            inputs = self._gather_inputs(node_id, pipeline.connections, outputs)

            # Execute node
            try:
                node_outputs = self._node_executor.execute(node, inputs)
                outputs[node_id] = node_outputs

                # Store results for output nodes
                if NODE_TEMPLATES[node.node_type].category == NodeCategory.OUTPUT:
                    results[node_id] = node_outputs

            except Exception as e:
                logger.error(f"Error executing node {node_id}: {e}")
                results[node_id] = {"error": str(e)}

        return results

    def _topological_sort(self, pipeline: VisualPipelineDefinition) -> List[str]:
        """Get nodes in topological order."""
        # Build adjacency list
        dependencies: Dict[str, Set[str]] = {n.id: set() for n in pipeline.nodes}

        for conn in pipeline.connections:
            dependencies[conn.target_node].add(conn.source_node)

        # Kahn's algorithm
        order = []
        no_deps = [n.id for n in pipeline.nodes if not dependencies[n.id]]

        while no_deps:
            node_id = no_deps.pop(0)
            order.append(node_id)

            for n_id, deps in dependencies.items():
                if node_id in deps:
                    deps.remove(node_id)
                    if not deps and n_id not in order and n_id not in no_deps:
                        no_deps.append(n_id)

        return order

    def _gather_inputs(
        self,
        node_id: str,
        connections: List[Connection],
        outputs: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Gather inputs for a node from connected outputs."""
        inputs = {}

        for conn in connections:
            if conn.target_node == node_id:
                source_outputs = outputs.get(conn.source_node, {})
                if conn.source_port in source_outputs:
                    inputs[conn.target_port] = source_outputs[conn.source_port]

        return inputs


class VisualPipeline:
    """High-level API for creating and executing visual pipelines."""

    def __init__(self, name: str, description: str = ""):
        """Initialize a visual pipeline.

        Args:
            name: Pipeline name
            description: Pipeline description
        """
        self._definition = VisualPipelineDefinition(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
        )
        self._executor = VisualPipelineExecutor()
        self._next_position = {"x": 100, "y": 100}

    def add_node(
        self,
        node_type: Union[NodeType, str],
        config: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
        position: Optional[Dict[str, float]] = None,
    ) -> str:
        """Add a node to the pipeline.

        Args:
            node_type: Type of node to add
            config: Node configuration
            label: Optional label for the node
            position: Optional position {"x": float, "y": float}

        Returns:
            Node ID
        """
        if isinstance(node_type, str):
            node_type = NodeType(node_type)

        node_id = str(uuid.uuid4())[:8]

        # Auto-position if not specified
        if position is None:
            position = self._next_position.copy()
            self._next_position["y"] += 100

        node = NodeInstance(
            id=node_id,
            node_type=node_type,
            position=position,
            config=config or {},
            label=label,
        )

        self._definition.nodes.append(node)
        return node_id

    def connect(
        self,
        source_node: str,
        target_node: str,
        source_port: str = "data",
        target_port: str = "data",
    ) -> str:
        """Connect two nodes.

        Args:
            source_node: Source node ID
            target_node: Target node ID
            source_port: Source port name
            target_port: Target port name

        Returns:
            Connection ID
        """
        conn_id = str(uuid.uuid4())[:8]

        connection = Connection(
            id=conn_id,
            source_node=source_node,
            source_port=source_port,
            target_node=target_node,
            target_port=target_port,
        )

        self._definition.connections.append(connection)
        return conn_id

    def execute(
        self,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the pipeline.

        Args:
            progress_callback: Optional progress callback

        Returns:
            Execution results
        """
        return self._executor.execute(self._definition, progress_callback)

    def validate(self) -> List[str]:
        """Validate the pipeline.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check for disconnected nodes (except sources)
        node_ids = {n.id for n in self._definition.nodes}
        target_nodes = {c.target_node for c in self._definition.connections}

        for node in self._definition.nodes:
            template = NODE_TEMPLATES.get(node.node_type)
            if template and template.inputs and node.id not in target_nodes:
                errors.append(f"Node '{node.label or node.id}' has unconnected required inputs")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary."""
        return self._definition.to_dict()

    def to_json(self) -> str:
        """Convert pipeline to JSON."""
        return self._definition.to_json()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualPipeline":
        """Create pipeline from dictionary."""
        pipeline = cls(data["name"], data.get("description", ""))
        pipeline._definition = VisualPipelineDefinition.from_dict(data)
        return pipeline

    @classmethod
    def from_json(cls, json_str: str) -> "VisualPipeline":
        """Create pipeline from JSON."""
        return cls.from_dict(json.loads(json_str))


class VisualBuilderAPI:
    """REST API backend for the visual builder.

    Provides endpoints for:
    - Pipeline CRUD operations
    - Node template discovery
    - Pipeline execution
    - Live preview
    """

    def __init__(self, storage_path: str = "./pipelines"):
        """Initialize the API.

        Args:
            storage_path: Path to store pipeline definitions
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._executor = VisualPipelineExecutor()

    def get_node_templates(self) -> List[Dict[str, Any]]:
        """Get all available node templates.

        Returns:
            List of node template dictionaries
        """
        return [template.to_dict() for template in NODE_TEMPLATES.values()]

    def get_node_template(self, node_type: str) -> Optional[Dict[str, Any]]:
        """Get a specific node template.

        Args:
            node_type: Node type string

        Returns:
            Node template dictionary or None
        """
        try:
            nt = NodeType(node_type)
            return NODE_TEMPLATES[nt].to_dict()
        except (ValueError, KeyError):
            return None

    def create_pipeline(
        self,
        name: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """Create a new pipeline.

        Args:
            name: Pipeline name
            description: Pipeline description

        Returns:
            Created pipeline definition
        """
        pipeline = VisualPipelineDefinition(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
        )

        # Save to storage
        self._save_pipeline(pipeline)

        return pipeline.to_dict()

    def get_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get a pipeline by ID.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Pipeline definition or None
        """
        pipeline = self._load_pipeline(pipeline_id)
        return pipeline.to_dict() if pipeline else None

    def update_pipeline(
        self,
        pipeline_id: str,
        updates: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Update a pipeline.

        Args:
            pipeline_id: Pipeline ID
            updates: Updates to apply

        Returns:
            Updated pipeline or None
        """
        pipeline = self._load_pipeline(pipeline_id)
        if not pipeline:
            return None

        # Apply updates
        if "name" in updates:
            pipeline.name = updates["name"]
        if "description" in updates:
            pipeline.description = updates["description"]
        if "nodes" in updates:
            pipeline.nodes = [NodeInstance.from_dict(n) for n in updates["nodes"]]
        if "connections" in updates:
            pipeline.connections = [Connection.from_dict(c) for c in updates["connections"]]

        pipeline.updated_at = datetime.utcnow().isoformat()
        pipeline.version += 1

        self._save_pipeline(pipeline)

        return pipeline.to_dict()

    def delete_pipeline(self, pipeline_id: str) -> bool:
        """Delete a pipeline.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            True if deleted, False if not found
        """
        file_path = self.storage_path / f"{pipeline_id}.json"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipelines.

        Returns:
            List of pipeline summaries
        """
        pipelines = []
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    pipelines.append({
                        "id": data["id"],
                        "name": data["name"],
                        "description": data.get("description", ""),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "node_count": len(data.get("nodes", [])),
                    })
            except Exception as e:
                logger.warning(f"Failed to load pipeline {file_path}: {e}")

        return pipelines

    def execute_pipeline(
        self,
        pipeline_id: str,
        async_execution: bool = False,
    ) -> Dict[str, Any]:
        """Execute a pipeline.

        Args:
            pipeline_id: Pipeline ID
            async_execution: If True, return execution ID immediately

        Returns:
            Execution results or execution ID
        """
        pipeline = self._load_pipeline(pipeline_id)
        if not pipeline:
            return {"error": "Pipeline not found"}

        if async_execution:
            # In a real implementation, this would queue the execution
            execution_id = str(uuid.uuid4())
            return {"execution_id": execution_id, "status": "queued"}

        # Synchronous execution
        try:
            results = self._executor.execute(pipeline)
            return {"status": "completed", "results": self._serialize_results(results)}
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    def preview_node(
        self,
        pipeline_id: str,
        node_id: str,
        n_rows: int = 10,
    ) -> Dict[str, Any]:
        """Get a preview of a node's output.

        Args:
            pipeline_id: Pipeline ID
            node_id: Node ID to preview
            n_rows: Number of rows to preview

        Returns:
            Preview data
        """
        pipeline = self._load_pipeline(pipeline_id)
        if not pipeline:
            return {"error": "Pipeline not found"}

        # Execute up to the target node
        # (Simplified - in production, cache intermediate results)
        try:
            results = self._executor.execute(pipeline)

            # Find the node's output
            for result_id, result in results.items():
                if result_id == node_id and "preview" in result:
                    return {"preview": result["preview"][:n_rows]}

            return {"error": "Node output not available"}
        except Exception as e:
            return {"error": str(e)}

    def _save_pipeline(self, pipeline: VisualPipelineDefinition) -> None:
        """Save a pipeline to storage."""
        file_path = self.storage_path / f"{pipeline.id}.json"
        with open(file_path, "w") as f:
            json.dump(pipeline.to_dict(), f, indent=2)

    def _load_pipeline(self, pipeline_id: str) -> Optional[VisualPipelineDefinition]:
        """Load a pipeline from storage."""
        file_path = self.storage_path / f"{pipeline_id}.json"
        if not file_path.exists():
            return None

        with open(file_path, "r") as f:
            data = json.load(f)
            return VisualPipelineDefinition.from_dict(data)

    def _serialize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize execution results for JSON response."""
        serialized = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                serialized[key] = {
                    "type": "dataframe",
                    "shape": list(value.shape),
                    "columns": list(value.columns),
                    "preview": value.head(10).to_dict(orient="records"),
                }
            elif isinstance(value, dict):
                serialized[key] = self._serialize_results(value)
            else:
                serialized[key] = str(value)
        return serialized

    def create_fastapi_app(self) -> Any:
        """Create a FastAPI application for the visual builder.

        Returns:
            FastAPI application
        """
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
        except ImportError:
            raise ImportError("FastAPI required: pip install fastapi uvicorn")

        app = FastAPI(
            title="Genesis Visual Builder API",
            description="REST API for the Genesis visual pipeline builder",
            version="1.0.0",
        )

        # CORS
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/api/templates")
        def get_templates():
            return self.get_node_templates()

        @app.get("/api/templates/{node_type}")
        def get_template(node_type: str):
            template = self.get_node_template(node_type)
            if not template:
                raise HTTPException(status_code=404, detail="Template not found")
            return template

        @app.get("/api/pipelines")
        def list_pipelines_endpoint():
            return self.list_pipelines()

        @app.post("/api/pipelines")
        def create_pipeline_endpoint(data: dict):
            return self.create_pipeline(
                data.get("name", "Untitled"),
                data.get("description", ""),
            )

        @app.get("/api/pipelines/{pipeline_id}")
        def get_pipeline_endpoint(pipeline_id: str):
            pipeline = self.get_pipeline(pipeline_id)
            if not pipeline:
                raise HTTPException(status_code=404, detail="Pipeline not found")
            return pipeline

        @app.put("/api/pipelines/{pipeline_id}")
        def update_pipeline_endpoint(pipeline_id: str, data: dict):
            pipeline = self.update_pipeline(pipeline_id, data)
            if not pipeline:
                raise HTTPException(status_code=404, detail="Pipeline not found")
            return pipeline

        @app.delete("/api/pipelines/{pipeline_id}")
        def delete_pipeline_endpoint(pipeline_id: str):
            if not self.delete_pipeline(pipeline_id):
                raise HTTPException(status_code=404, detail="Pipeline not found")
            return {"status": "deleted"}

        @app.post("/api/pipelines/{pipeline_id}/execute")
        def execute_pipeline_endpoint(pipeline_id: str, async_exec: bool = False):
            return self.execute_pipeline(pipeline_id, async_exec)

        @app.get("/api/pipelines/{pipeline_id}/preview/{node_id}")
        def preview_node_endpoint(pipeline_id: str, node_id: str, n_rows: int = 10):
            return self.preview_node(pipeline_id, node_id, n_rows)

        return app

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Run the API server.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        try:
            import uvicorn
        except ImportError:
            raise ImportError("uvicorn required: pip install uvicorn")

        app = self.create_fastapi_app()
        uvicorn.run(app, host=host, port=port)


__all__ = [
    # Core classes
    "VisualPipeline",
    "VisualPipelineDefinition",
    "VisualPipelineExecutor",
    "VisualBuilderAPI",
    # Node components
    "NodeInstance",
    "NodeTemplate",
    "Connection",
    "PortDefinition",
    "ParameterDefinition",
    # Enums
    "NodeCategory",
    "NodeType",
    # Executors
    "NodeExecutor",
    # Templates
    "NODE_TEMPLATES",
]
