"""Auto-tuning hyperparameters for synthetic data generators.

This module provides automatic hyperparameter optimization using Optuna
or a built-in simple search when Optuna is not available.

Example:
    >>> from genesis.tuning import AutoTuner
    >>>
    >>> tuner = AutoTuner(method='ctgan')
    >>> best_config = tuner.tune(data, n_trials=20)
    >>>
    >>> # Use the optimized config
    >>> gen = SyntheticGenerator(config=best_config)
    >>> gen.fit(data)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from genesis.core.config import GeneratorConfig
from genesis.core.types import GeneratorMethod
from genesis.utils.logging import get_logger

logger = get_logger(__name__)

# Check for Optuna
try:
    import optuna
    from optuna.samplers import TPESampler

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class TuningPreset(Enum):
    """Presets for tuning speed/quality tradeoff."""

    FAST = "fast"  # Quick search, fewer trials
    BALANCED = "balanced"  # Default balance
    QUALITY = "quality"  # Thorough search, more trials


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""

    n_trials: int = 20
    timeout: Optional[float] = None  # Seconds
    n_jobs: int = 1  # Parallel trials
    preset: TuningPreset = TuningPreset.BALANCED
    validation_split: float = 0.2
    metric: str = "statistical_fidelity"  # or "ml_utility", "combined"
    early_stopping_rounds: int = 5
    random_seed: Optional[int] = None

    @classmethod
    def from_preset(cls, preset: TuningPreset) -> "TuningConfig":
        """Create config from preset."""
        presets = {
            TuningPreset.FAST: cls(n_trials=10, timeout=300),
            TuningPreset.BALANCED: cls(n_trials=20, timeout=600),
            TuningPreset.QUALITY: cls(n_trials=50, timeout=1800),
        }
        return presets.get(preset, cls())


@dataclass
class TuningResult:
    """Result of hyperparameter tuning."""

    best_config: GeneratorConfig
    best_score: float
    n_trials: int
    total_time: float
    trials_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_config": self.best_config.to_dict(),
            "best_score": self.best_score,
            "n_trials": self.n_trials,
            "total_time": self.total_time,
            "trials_history": self.trials_history,
        }


class SearchSpace:
    """Defines hyperparameter search spaces for different generators."""

    @staticmethod
    def get_ctgan_space() -> Dict[str, Any]:
        """Get search space for CTGAN."""
        return {
            "epochs": {"type": "int", "low": 100, "high": 500, "step": 50},
            "batch_size": {"type": "categorical", "choices": [100, 200, 500, 1000]},
            "generator_dim": {
                "type": "categorical",
                "choices": [(128, 128), (256, 256), (256, 256, 256), (512, 256, 128)],
            },
            "discriminator_dim": {
                "type": "categorical",
                "choices": [(128, 128), (256, 256), (256, 256, 256), (512, 256, 128)],
            },
            "embedding_dim": {"type": "categorical", "choices": [64, 128, 256]},
            "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
            "pac": {"type": "categorical", "choices": [1, 5, 10]},
        }

    @staticmethod
    def get_tvae_space() -> Dict[str, Any]:
        """Get search space for TVAE."""
        return {
            "epochs": {"type": "int", "low": 100, "high": 500, "step": 50},
            "batch_size": {"type": "categorical", "choices": [100, 200, 500, 1000]},
            "embedding_dim": {"type": "categorical", "choices": [64, 128, 256]},
            "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
        }

    @staticmethod
    def get_gaussian_copula_space() -> Dict[str, Any]:
        """Get search space for Gaussian Copula (limited params)."""
        return {
            # Gaussian Copula has few tunable parameters
        }

    @staticmethod
    def get_space(method: GeneratorMethod) -> Dict[str, Any]:
        """Get search space for a generator method."""
        spaces = {
            GeneratorMethod.CTGAN: SearchSpace.get_ctgan_space,
            GeneratorMethod.TVAE: SearchSpace.get_tvae_space,
            GeneratorMethod.GAUSSIAN_COPULA: SearchSpace.get_gaussian_copula_space,
        }
        getter = spaces.get(method, SearchSpace.get_ctgan_space)
        return getter()


class SimpleSearcher:
    """Simple grid/random search when Optuna is not available."""

    def __init__(
        self,
        search_space: Dict[str, Any],
        n_trials: int = 10,
        random_seed: Optional[int] = None,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.rng = np.random.RandomState(random_seed)

    def sample(self) -> Dict[str, Any]:
        """Sample a random configuration."""
        config = {}
        for name, spec in self.search_space.items():
            if spec["type"] == "int":
                values = list(range(spec["low"], spec["high"] + 1, spec.get("step", 1)))
                config[name] = self.rng.choice(values)
            elif spec["type"] == "categorical":
                config[name] = self.rng.choice(spec["choices"])
            elif spec["type"] == "loguniform":
                log_low = np.log(spec["low"])
                log_high = np.log(spec["high"])
                config[name] = np.exp(self.rng.uniform(log_low, log_high))
            elif spec["type"] == "float":
                config[name] = self.rng.uniform(spec["low"], spec["high"])
        return config


class AutoTuner:
    """Automatic hyperparameter tuner for synthetic data generators.

    Uses Optuna when available, falls back to simple random search otherwise.
    """

    def __init__(
        self,
        method: Union[str, GeneratorMethod] = GeneratorMethod.CTGAN,
        config: Optional[TuningConfig] = None,
    ):
        """Initialize the auto-tuner.

        Args:
            method: Generator method to tune
            config: Tuning configuration
        """
        if isinstance(method, str):
            method = GeneratorMethod(method.lower())

        self.method = method
        self.config = config or TuningConfig()
        self.search_space = SearchSpace.get_space(method)

        self._best_score = float("-inf")
        self._trials_history: List[Dict[str, Any]] = []

    def tune(
        self,
        data: pd.DataFrame,
        discrete_columns: Optional[List[str]] = None,
        validation_data: Optional[pd.DataFrame] = None,
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> TuningResult:
        """Run hyperparameter tuning.

        Args:
            data: Training data
            discrete_columns: Categorical columns
            validation_data: Validation data (if None, splits from data)
            progress_callback: Callback(trial, total, best_score)

        Returns:
            TuningResult with best configuration
        """
        start_time = time.time()

        # Split data if no validation set provided
        if validation_data is None:
            split_idx = int(len(data) * (1 - self.config.validation_split))
            train_data = data.iloc[:split_idx]
            val_data = data.iloc[split_idx:]
        else:
            train_data = data
            val_data = validation_data

        logger.info(
            f"Starting hyperparameter tuning for {self.method.value} "
            f"with {self.config.n_trials} trials"
        )

        if HAS_OPTUNA:
            result = self._tune_with_optuna(
                train_data, val_data, discrete_columns, progress_callback
            )
        else:
            logger.info("Optuna not available, using simple random search")
            result = self._tune_simple(train_data, val_data, discrete_columns, progress_callback)

        result.total_time = time.time() - start_time
        logger.info(
            f"Tuning complete: best score = {result.best_score:.4f} " f"in {result.total_time:.1f}s"
        )

        return result

    def _tune_with_optuna(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        discrete_columns: Optional[List[str]],
        progress_callback: Optional[Callable],
    ) -> TuningResult:
        """Tune using Optuna."""

        def objective(trial: "optuna.Trial") -> float:
            # Sample hyperparameters
            params = {}
            for name, spec in self.search_space.items():
                if spec["type"] == "int":
                    params[name] = trial.suggest_int(
                        name, spec["low"], spec["high"], step=spec.get("step", 1)
                    )
                elif spec["type"] == "categorical":
                    params[name] = trial.suggest_categorical(name, spec["choices"])
                elif spec["type"] == "loguniform":
                    params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=True)
                elif spec["type"] == "float":
                    params[name] = trial.suggest_float(name, spec["low"], spec["high"])

            # Evaluate configuration
            score = self._evaluate_config(params, train_data, val_data, discrete_columns)

            # Track history
            self._trials_history.append(
                {
                    "trial": trial.number,
                    "params": params.copy(),
                    "score": score,
                }
            )

            if progress_callback:
                progress_callback(
                    trial.number + 1, self.config.n_trials, max(self._best_score, score)
                )

            if score > self._best_score:
                self._best_score = score

            return score

        # Create study
        sampler = TPESampler(seed=self.config.random_seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Suppress Optuna logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Run optimization
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs,
            show_progress_bar=False,
        )

        # Build best config
        best_config = self._params_to_config(study.best_params)

        return TuningResult(
            best_config=best_config,
            best_score=study.best_value,
            n_trials=len(study.trials),
            total_time=0,  # Will be set by caller
            trials_history=self._trials_history,
        )

    def _tune_simple(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        discrete_columns: Optional[List[str]],
        progress_callback: Optional[Callable],
    ) -> TuningResult:
        """Tune using simple random search."""
        searcher = SimpleSearcher(
            self.search_space,
            n_trials=self.config.n_trials,
            random_seed=self.config.random_seed,
        )

        best_params = {}
        best_score = float("-inf")

        for trial in range(self.config.n_trials):
            params = searcher.sample()

            score = self._evaluate_config(params, train_data, val_data, discrete_columns)

            self._trials_history.append(
                {
                    "trial": trial,
                    "params": params.copy(),
                    "score": score,
                }
            )

            if score > best_score:
                best_score = score
                best_params = params.copy()

            if progress_callback:
                progress_callback(trial + 1, self.config.n_trials, best_score)

        best_config = self._params_to_config(best_params)

        return TuningResult(
            best_config=best_config,
            best_score=best_score,
            n_trials=self.config.n_trials,
            total_time=0,
            trials_history=self._trials_history,
        )

    def _evaluate_config(
        self,
        params: Dict[str, Any],
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        discrete_columns: Optional[List[str]],
    ) -> float:
        """Evaluate a configuration by training and measuring quality."""
        from genesis.evaluation.statistical import compute_statistical_fidelity
        from genesis.generators.auto import select_generator

        try:
            # Create config from params
            config = self._params_to_config(params)

            # Create and train generator
            generator = select_generator(
                data=train_data,
                method=self.method,
                config=config,
            )

            # Reduced epochs for faster evaluation during tuning
            generator.fit(train_data, discrete_columns=discrete_columns)

            # Generate synthetic data
            synthetic = generator.generate(len(val_data))

            # Compute score
            if self.config.metric == "statistical_fidelity":
                result = compute_statistical_fidelity(val_data, synthetic)
                score = result.overall_score
            else:
                # Default to statistical fidelity
                result = compute_statistical_fidelity(val_data, synthetic)
                score = result.overall_score

            return score

        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0

    def _params_to_config(self, params: Dict[str, Any]) -> GeneratorConfig:
        """Convert parameter dict to GeneratorConfig."""
        config_params = {"method": self.method}

        # Map parameters
        if "epochs" in params:
            config_params["epochs"] = params["epochs"]
        if "batch_size" in params:
            config_params["batch_size"] = params["batch_size"]
        if "generator_dim" in params:
            config_params["generator_dim"] = params["generator_dim"]
        if "discriminator_dim" in params:
            config_params["discriminator_dim"] = params["discriminator_dim"]
        if "embedding_dim" in params:
            config_params["embedding_dim"] = params["embedding_dim"]
        if "learning_rate" in params:
            config_params["learning_rate"] = params["learning_rate"]
        if "pac" in params:
            config_params["pac"] = params["pac"]

        return GeneratorConfig(**config_params)


def auto_tune(
    data: pd.DataFrame,
    method: Union[str, GeneratorMethod] = GeneratorMethod.CTGAN,
    preset: TuningPreset = TuningPreset.BALANCED,
    discrete_columns: Optional[List[str]] = None,
    **kwargs,
) -> GeneratorConfig:
    """Convenience function for quick auto-tuning.

    Args:
        data: Training data
        method: Generator method to tune
        preset: Tuning preset (fast/balanced/quality)
        discrete_columns: Categorical columns
        **kwargs: Additional TuningConfig parameters

    Returns:
        Optimized GeneratorConfig
    """
    config = TuningConfig.from_preset(preset)
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    tuner = AutoTuner(method=method, config=config)
    result = tuner.tune(data, discrete_columns=discrete_columns)

    return result.best_config


__all__ = [
    "AutoTuner",
    "TuningConfig",
    "TuningResult",
    "TuningPreset",
    "SearchSpace",
    "auto_tune",
]
