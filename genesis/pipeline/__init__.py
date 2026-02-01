"""Visual Pipeline Builder Backend.

A fluent API and execution engine for building complex synthetic data
generation workflows. Supports visual pipeline construction with nodes
for data sources, transformations, generation, evaluation, and output.

Example:
    Building a simple pipeline::

        from genesis.pipeline import PipelineBuilder

        pipeline = (
            PipelineBuilder("my_pipeline")
            .add_source("customers.csv")
            .add_generator(method="ctgan", n_samples=10000)
            .add_output("synthetic_customers.csv")
            .build()
        )

        result = pipeline.execute()

    Loading from YAML::

        pipeline = Pipeline.load("my_pipeline.yaml")
        pipeline.execute()
"""

from genesis.pipeline.builder import PipelineBuilder
from genesis.pipeline.executor import PipelineExecutor
from genesis.pipeline.nodes import (
    NODE_TEMPLATES,
    NodePort,
    NodeType,
    PipelineConnection,
    PipelineNode,
    ValidationResult,
    get_node_templates,
)
from genesis.pipeline.pipeline import Pipeline, create_simple_pipeline

__all__ = [
    # Core types
    "NodeType",
    "NodePort",
    "PipelineNode",
    "PipelineConnection",
    "ValidationResult",
    # Pipeline
    "Pipeline",
    "PipelineBuilder",
    "PipelineExecutor",
    # Templates
    "NODE_TEMPLATES",
    "get_node_templates",
    # Convenience
    "create_simple_pipeline",
]
