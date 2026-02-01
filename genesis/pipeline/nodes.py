"""Pipeline node definitions and templates."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List


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
    """Input or output port of a node.

    Attributes:
        id: Unique port identifier
        name: Human-readable name
        port_type: Either "input" or "output"
        data_type: Type of data (e.g., "dataframe", "config", "model")
        required: Whether the port must be connected
        multi: Whether the port accepts multiple connections
    """

    id: str
    name: str
    port_type: str  # "input" or "output"
    data_type: str  # "dataframe", "config", "model"
    required: bool = True
    multi: bool = False  # Can accept multiple connections


@dataclass
class PipelineNode:
    """A node in the pipeline.

    Attributes:
        id: Unique node identifier
        node_type: Type of node
        name: Human-readable name
        config: Node configuration
        position: Visual position (x, y coordinates)
        inputs: List of input ports
        outputs: List of output ports
    """

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
    """Connection between nodes.

    Attributes:
        id: Unique connection identifier
        source_node: ID of the source node
        source_port: ID of the source port
        target_node: ID of the target node
        target_port: ID of the target port
    """

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


# Node templates define the structure of each node type
NODE_TEMPLATES: Dict[NodeType, Dict[str, Any]] = {
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


def get_node_templates() -> Dict[str, Any]:
    """Get all node templates for UI.

    Returns:
        Dict of node templates keyed by node type value
    """
    return {k.value: v for k, v in NODE_TEMPLATES.items()}
