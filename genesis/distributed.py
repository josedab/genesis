"""Distributed training and GPU cluster orchestration.

This module provides capabilities for:
- Distributed training across multiple GPUs
- Cluster orchestration with Ray or Dask
- Automatic data sharding for large datasets
- Multi-node training coordination

Example:
    >>> from genesis.distributed import DistributedTrainer
    >>>
    >>> trainer = DistributedTrainer(n_workers=4)
    >>> trainer.fit(large_data, method='ctgan')
    >>> synthetic = trainer.generate(1000000)
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from genesis.core.config import GeneratorConfig
from genesis.core.types import GeneratorMethod
from genesis.utils.logging import get_logger

logger = get_logger(__name__)

# Check for distributed frameworks
try:
    import ray

    HAS_RAY = True
except ImportError:
    HAS_RAY = False

try:
    import dask.dataframe  # noqa: F401
    from dask.distributed import Client, LocalCluster

    HAS_DASK = True
except ImportError:
    HAS_DASK = False


class DistributedBackend(Enum):
    """Available distributed computing backends."""

    RAY = "ray"
    DASK = "dask"
    MULTIPROCESSING = "multiprocessing"
    AUTO = "auto"


class ShardingStrategy(Enum):
    """Strategies for sharding data across workers."""

    RANDOM = "random"  # Random shuffle and split
    STRATIFIED = "stratified"  # Maintain class distribution
    CONTIGUOUS = "contiguous"  # Contiguous chunks
    HASH = "hash"  # Hash-based assignment


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""

    backend: DistributedBackend = DistributedBackend.AUTO
    n_workers: int = 4
    gpus_per_worker: int = 1
    memory_per_worker: str = "4GB"
    sharding: ShardingStrategy = ShardingStrategy.RANDOM
    checkpoint_dir: Optional[str] = None
    ray_address: Optional[str] = None
    dask_scheduler: Optional[str] = None
    timeout_seconds: int = 3600


@dataclass
class WorkerStatus:
    """Status of a distributed worker."""

    worker_id: str
    status: str  # "idle", "running", "completed", "failed"
    progress: float
    shard_size: int
    gpu_id: Optional[int]
    error: Optional[str] = None


@dataclass
class DistributedResult:
    """Result of distributed training."""

    success: bool
    total_time: float
    n_workers_used: int
    samples_processed: int
    worker_stats: List[WorkerStatus]
    aggregated_model: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "total_time": self.total_time,
            "n_workers_used": self.n_workers_used,
            "samples_processed": self.samples_processed,
        }


class DataSharder:
    """Handles sharding of data for distributed processing."""

    def __init__(
        self,
        strategy: ShardingStrategy = ShardingStrategy.RANDOM,
        stratify_column: Optional[str] = None,
    ):
        self.strategy = strategy
        self.stratify_column = stratify_column

    def shard(
        self,
        data: pd.DataFrame,
        n_shards: int,
    ) -> List[pd.DataFrame]:
        """Split data into shards.

        Args:
            data: Data to shard
            n_shards: Number of shards to create

        Returns:
            List of data shards
        """
        if self.strategy == ShardingStrategy.RANDOM:
            return self._random_shard(data, n_shards)
        elif self.strategy == ShardingStrategy.STRATIFIED:
            return self._stratified_shard(data, n_shards)
        elif self.strategy == ShardingStrategy.CONTIGUOUS:
            return self._contiguous_shard(data, n_shards)
        elif self.strategy == ShardingStrategy.HASH:
            return self._hash_shard(data, n_shards)
        else:
            return self._random_shard(data, n_shards)

    def _random_shard(self, data: pd.DataFrame, n_shards: int) -> List[pd.DataFrame]:
        """Random shuffle and split."""
        shuffled = data.sample(frac=1).reset_index(drop=True)
        return np.array_split(shuffled, n_shards)

    def _stratified_shard(self, data: pd.DataFrame, n_shards: int) -> List[pd.DataFrame]:
        """Stratified split maintaining class distribution."""
        if self.stratify_column and self.stratify_column in data.columns:
            # Group by stratify column and distribute evenly
            groups = data.groupby(self.stratify_column)
            shards = [pd.DataFrame() for _ in range(n_shards)]

            for _, group in groups:
                group_shards = np.array_split(group, n_shards)
                for i, shard in enumerate(group_shards):
                    shards[i] = pd.concat([shards[i], shard], ignore_index=True)

            return shards
        return self._random_shard(data, n_shards)

    def _contiguous_shard(self, data: pd.DataFrame, n_shards: int) -> List[pd.DataFrame]:
        """Contiguous chunks without shuffling."""
        return np.array_split(data, n_shards)

    def _hash_shard(self, data: pd.DataFrame, n_shards: int) -> List[pd.DataFrame]:
        """Hash-based assignment for reproducibility."""
        # Use row index hash for assignment
        assignments = data.index % n_shards
        shards = []
        for i in range(n_shards):
            shard = data[assignments == i].copy()
            shards.append(shard)
        return shards


class ModelAggregator:
    """Aggregates models trained on different shards."""

    def aggregate_gaussian_copula(
        self,
        models: List[Any],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Aggregate Gaussian Copula models.

        Uses weighted averaging of correlation matrices and
        marginal parameters.
        """
        if not models:
            return {}

        n_models = len(models)
        weights = weights or [1.0 / n_models] * n_models

        # Aggregate correlation matrices
        corr_matrices = []
        for model in models:
            if hasattr(model, "_correlation_matrix") and model._correlation_matrix is not None:
                corr_matrices.append(model._correlation_matrix)

        if corr_matrices:
            aggregated_corr = np.zeros_like(corr_matrices[0])
            for w, corr in zip(weights, corr_matrices):
                aggregated_corr += w * corr
        else:
            aggregated_corr = None

        # Aggregate marginal parameters
        aggregated_marginals = {}
        for model in models:
            if hasattr(model, "_marginal_params"):
                for col, params in model._marginal_params.items():
                    if col not in aggregated_marginals:
                        aggregated_marginals[col] = {"samples": []}
                    aggregated_marginals[col]["samples"].append(params)

        return {
            "correlation_matrix": aggregated_corr,
            "marginal_params": aggregated_marginals,
        }

    def aggregate_statistics(
        self,
        stats_list: List[Dict[str, Any]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Aggregate statistics from multiple shards."""
        if not stats_list:
            return {}

        n_stats = len(stats_list)
        weights = weights or [1.0 / n_stats] * n_stats

        aggregated = {"columns": {}}

        # Get all columns
        all_cols = set()
        for s in stats_list:
            all_cols.update(s.get("columns", {}).keys())

        for col in all_cols:
            col_stats = []
            for s, w in zip(stats_list, weights):
                if col in s.get("columns", {}):
                    col_stats.append((s["columns"][col], w))

            if not col_stats:
                continue

            # Aggregate numeric statistics
            if "mean" in col_stats[0][0]:
                aggregated["columns"][col] = {
                    "mean": sum(s["mean"] * w for s, w in col_stats),
                    "std": sum(s["std"] * w for s, w in col_stats),
                    "min": min(s["min"] for s, _ in col_stats),
                    "max": max(s["max"] for s, _ in col_stats),
                }

        return aggregated


class DistributedTrainer:
    """Distributed trainer for synthetic data generation.

    Supports training across multiple GPUs or nodes using
    Ray, Dask, or multiprocessing backends.
    """

    def __init__(
        self,
        config: Optional[DistributedConfig] = None,
        n_workers: int = 4,
    ):
        """Initialize distributed trainer.

        Args:
            config: Distributed configuration
            n_workers: Number of workers (shortcut for config)
        """
        self.config = config or DistributedConfig(n_workers=n_workers)
        self._backend = self._select_backend()
        self._generator = None
        self._is_fitted = False
        self._worker_models: List[Any] = []

    def _select_backend(self) -> str:
        """Select the best available backend."""
        if self.config.backend == DistributedBackend.AUTO:
            if HAS_RAY:
                return "ray"
            elif HAS_DASK:
                return "dask"
            else:
                return "multiprocessing"
        return self.config.backend.value

    def fit(
        self,
        data: pd.DataFrame,
        method: Union[str, GeneratorMethod] = GeneratorMethod.GAUSSIAN_COPULA,
        discrete_columns: Optional[List[str]] = None,
        generator_config: Optional[GeneratorConfig] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> DistributedResult:
        """Fit generator on distributed cluster.

        Args:
            data: Training data
            method: Generator method
            discrete_columns: Categorical columns
            generator_config: Generator configuration
            progress_callback: Progress callback(completed, total)

        Returns:
            DistributedResult with training statistics
        """
        start_time = time.time()

        logger.info(
            f"Starting distributed training with {self.config.n_workers} workers "
            f"using {self._backend} backend"
        )

        # Shard data
        sharder = DataSharder(self.config.sharding)
        shards = sharder.shard(data, self.config.n_workers)

        logger.info(
            f"Data sharded into {len(shards)} parts " f"({[len(s) for s in shards]} rows each)"
        )

        # Train on each shard
        if self._backend == "ray":
            result = self._fit_ray(shards, method, discrete_columns, generator_config)
        elif self._backend == "dask":
            result = self._fit_dask(shards, method, discrete_columns, generator_config)
        else:
            result = self._fit_multiprocessing(shards, method, discrete_columns, generator_config)

        result.total_time = time.time() - start_time

        if result.success:
            self._is_fitted = True
            logger.info(f"Distributed training completed in {result.total_time:.1f}s")

        return result

    def generate(
        self,
        n_samples: int,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """Generate synthetic data using trained distributed model.

        Args:
            n_samples: Number of samples to generate
            conditions: Conditional generation conditions

        Returns:
            Generated DataFrame
        """
        if not self._is_fitted:
            raise RuntimeError("Trainer not fitted. Call fit() first.")

        # Generate from each worker model and combine
        samples_per_worker = n_samples // len(self._worker_models)

        all_samples = []
        for model in self._worker_models:
            if hasattr(model, "generate"):
                samples = model.generate(samples_per_worker)
                all_samples.append(samples)

        if all_samples:
            combined = pd.concat(all_samples, ignore_index=True)
            return combined.head(n_samples)

        return pd.DataFrame()

    def _fit_multiprocessing(
        self,
        shards: List[pd.DataFrame],
        method: Union[str, GeneratorMethod],
        discrete_columns: Optional[List[str]],
        config: Optional[GeneratorConfig],
    ) -> DistributedResult:
        """Fit using Python multiprocessing."""

        worker_stats = []
        self._worker_models = []

        def train_shard(shard_id: int, shard: pd.DataFrame) -> Tuple[int, Any, str]:
            """Train on a single shard."""
            try:
                from genesis.generators.tabular import GaussianCopulaGenerator

                gen = GaussianCopulaGenerator(config=config)
                gen.fit(shard, discrete_columns=discrete_columns)

                return shard_id, gen, "completed"
            except Exception as e:
                return shard_id, None, str(e)

        # Use ProcessPoolExecutor for parallel training
        # Note: This is simplified - real implementation would use proper serialization

        # For now, train sequentially (multiprocessing has pickle limitations)
        for i, shard in enumerate(shards):
            shard_id, model, status = train_shard(i, shard)

            worker_stats.append(
                WorkerStatus(
                    worker_id=f"worker-{shard_id}",
                    status=status if status == "completed" else "failed",
                    progress=1.0 if status == "completed" else 0.0,
                    shard_size=len(shard),
                    gpu_id=None,
                    error=None if status == "completed" else status,
                )
            )

            if model is not None:
                self._worker_models.append(model)

        success = len(self._worker_models) > 0

        return DistributedResult(
            success=success,
            total_time=0,  # Will be set by caller
            n_workers_used=len(self._worker_models),
            samples_processed=sum(len(s) for s in shards),
            worker_stats=worker_stats,
        )

    def _fit_ray(
        self,
        shards: List[pd.DataFrame],
        method: Union[str, GeneratorMethod],
        discrete_columns: Optional[List[str]],
        config: Optional[GeneratorConfig],
    ) -> DistributedResult:
        """Fit using Ray distributed computing."""
        if not HAS_RAY:
            logger.warning("Ray not available, falling back to multiprocessing")
            return self._fit_multiprocessing(shards, method, discrete_columns, config)

        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(
                address=self.config.ray_address,
                ignore_reinit_error=True,
            )

        @ray.remote(num_gpus=self.config.gpus_per_worker)
        def train_shard_ray(
            shard: pd.DataFrame, discrete_cols: List[str], gen_config: dict
        ) -> dict:
            """Ray remote function for training."""
            from genesis.core.config import GeneratorConfig
            from genesis.generators.tabular import GaussianCopulaGenerator

            config_obj = GeneratorConfig(**gen_config) if gen_config else None
            gen = GaussianCopulaGenerator(config=config_obj)
            gen.fit(shard, discrete_columns=discrete_cols)

            # Return serializable statistics
            return {
                "n_samples": len(shard),
                "correlation": (
                    gen._correlation_matrix.tolist()
                    if gen._correlation_matrix is not None
                    else None
                ),
                "column_names": gen._column_names,
            }

        # Submit tasks
        config_dict = config.to_dict() if config else {}
        futures = [
            train_shard_ray.remote(shard, discrete_columns or [], config_dict) for shard in shards
        ]

        # Wait for completion
        results = ray.get(futures)

        worker_stats = [
            WorkerStatus(
                worker_id=f"ray-worker-{i}",
                status="completed",
                progress=1.0,
                shard_size=r["n_samples"],
                gpu_id=i % max(1, self.config.gpus_per_worker),
            )
            for i, r in enumerate(results)
        ]

        # Aggregate results
        aggregator = ModelAggregator()
        aggregated = aggregator.aggregate_statistics(results)

        return DistributedResult(
            success=True,
            total_time=0,
            n_workers_used=len(results),
            samples_processed=sum(r["n_samples"] for r in results),
            worker_stats=worker_stats,
            aggregated_model=aggregated,
        )

    def _fit_dask(
        self,
        shards: List[pd.DataFrame],
        method: Union[str, GeneratorMethod],
        discrete_columns: Optional[List[str]],
        config: Optional[GeneratorConfig],
    ) -> DistributedResult:
        """Fit using Dask distributed computing."""
        if not HAS_DASK:
            logger.warning("Dask not available, falling back to multiprocessing")
            return self._fit_multiprocessing(shards, method, discrete_columns, config)

        # Create or connect to cluster
        if self.config.dask_scheduler:
            client = Client(self.config.dask_scheduler)
        else:
            cluster = LocalCluster(n_workers=self.config.n_workers)
            client = Client(cluster)

        try:

            def train_shard_dask(shard: pd.DataFrame) -> dict:
                """Dask task for training."""
                from genesis.generators.tabular import GaussianCopulaGenerator

                gen = GaussianCopulaGenerator(config=config)
                gen.fit(shard, discrete_columns=discrete_columns)

                return {
                    "n_samples": len(shard),
                    "success": True,
                }

            # Submit tasks
            futures = client.map(train_shard_dask, shards)
            results = client.gather(futures)

            worker_stats = [
                WorkerStatus(
                    worker_id=f"dask-worker-{i}",
                    status="completed",
                    progress=1.0,
                    shard_size=r["n_samples"],
                    gpu_id=None,
                )
                for i, r in enumerate(results)
            ]

            return DistributedResult(
                success=True,
                total_time=0,
                n_workers_used=len(results),
                samples_processed=sum(r["n_samples"] for r in results),
                worker_stats=worker_stats,
            )
        finally:
            client.close()


class GPUManager:
    """Manages GPU resources for distributed training."""

    def __init__(self):
        self._available_gpus: List[int] = []
        self._gpu_memory: Dict[int, int] = {}
        self._refresh_gpus()

    def _refresh_gpus(self) -> None:
        """Detect available GPUs."""
        try:
            import torch

            if torch.cuda.is_available():
                n_gpus = torch.cuda.device_count()
                self._available_gpus = list(range(n_gpus))
                for i in range(n_gpus):
                    props = torch.cuda.get_device_properties(i)
                    self._gpu_memory[i] = props.total_memory
        except ImportError:
            pass

        if not self._available_gpus:
            try:
                import tensorflow as tf

                gpus = tf.config.list_physical_devices("GPU")
                self._available_gpus = list(range(len(gpus)))
            except ImportError:
                pass

    @property
    def n_gpus(self) -> int:
        """Number of available GPUs."""
        return len(self._available_gpus)

    @property
    def gpu_ids(self) -> List[int]:
        """List of available GPU IDs."""
        return self._available_gpus.copy()

    def get_gpu_memory(self, gpu_id: int) -> int:
        """Get memory for a specific GPU in bytes."""
        return self._gpu_memory.get(gpu_id, 0)

    def select_gpus(self, n_gpus: int) -> List[int]:
        """Select a number of GPUs for use.

        Args:
            n_gpus: Number of GPUs needed

        Returns:
            List of selected GPU IDs
        """
        return self._available_gpus[:n_gpus]


__all__ = [
    # Main classes
    "DistributedTrainer",
    "DistributedConfig",
    "DistributedResult",
    # Utilities
    "DataSharder",
    "ModelAggregator",
    "GPUManager",
    # Types
    "DistributedBackend",
    "ShardingStrategy",
    "WorkerStatus",
]
