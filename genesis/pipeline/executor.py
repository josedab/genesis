"""Pipeline execution engine."""

from typing import Any, Callable, Dict, Optional

import pandas as pd

from genesis.pipeline.nodes import NodeType, PipelineNode
from genesis.pipeline.pipeline import Pipeline


class PipelineExecutor:
    """Execute pipelines by processing nodes in topological order.

    The executor traverses the pipeline graph, executing each node and
    passing data between connected ports.

    Attributes:
        _node_handlers: Registry of handler functions for each node type
        _context: Execution context with inputs and intermediate results

    Example:
        >>> executor = PipelineExecutor()
        >>> results = executor.execute(pipeline, inputs={"my_data": df})
        >>> print(results["node_2"]["synthetic_data"])
    """

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
        outputs: Dict[str, Any] = {}

        # Get execution order
        order = pipeline.get_execution_order()
        node_map = {n.id: n for n in pipeline.nodes}

        # Build connection map
        conn_map: Dict[str, tuple[str, str]] = {}  # target_port -> (source_node, source_port)
        for conn in pipeline.connections:
            conn_map[conn.target_port] = (conn.source_node, conn.source_port)

        # Execute nodes
        for node_id in order:
            node = node_map[node_id]

            # Gather inputs from connected nodes
            node_inputs: Dict[str, Any] = {}
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
        """Handle data source node - loads data from context."""
        var_name = node.config.get("variable_name")
        data = self._context.get("inputs", {}).get(var_name)
        return {"data": data}

    def _handle_file_input(self, node: PipelineNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle file input node - reads data from file."""
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
        """Handle filter node - applies pandas query."""
        data = inputs.get("input")
        condition = node.config.get("condition", "True")

        if data is not None:
            filtered = data.query(condition)
            return {"output": filtered}

        return {"output": None}

    def _handle_select_columns(self, node: PipelineNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle select columns node - subsets columns."""
        data = inputs.get("input")
        columns = node.config.get("columns", [])

        if data is not None and columns:
            selected = data[columns]
            return {"output": selected}

        return {"output": data}

    def _handle_generator(self, node: PipelineNode, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Handle generator node - creates synthetic data."""
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
        """Handle quality check node - evaluates synthetic data."""
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
        """Handle file output node - saves data to file."""
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
        """Handle augmentation node - balances dataset."""
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
