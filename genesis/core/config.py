"""Configuration classes for Genesis."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from genesis.core.types import BackendType, GeneratorMethod, PrivacyLevel


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generators.

    Attributes:
        method: Generation method to use ('auto', 'ctgan', 'tvae', 'gaussian_copula')
        backend: Deep learning backend ('auto', 'pytorch', 'tensorflow')
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        generator_dim: Dimensions of generator network layers
        discriminator_dim: Dimensions of discriminator network layers
        embedding_dim: Dimension of embedding layer
        random_seed: Random seed for reproducibility
        verbose: Whether to print training progress
        device: Device to use ('cpu', 'cuda', 'auto')
        n_critics: Number of critic updates per generator update (for WGAN)
        pac: Number of samples to pack together (PacGAN)
    """

    method: Union[str, GeneratorMethod] = GeneratorMethod.AUTO
    backend: Union[str, BackendType] = BackendType.AUTO
    epochs: int = 300
    batch_size: int = 500
    learning_rate: float = 2e-4
    generator_dim: Tuple[int, ...] = (256, 256)
    discriminator_dim: Tuple[int, ...] = (256, 256)
    embedding_dim: int = 128
    random_seed: Optional[int] = None
    verbose: bool = True
    device: str = "auto"
    n_critics: int = 1
    pac: int = 10

    def __post_init__(self) -> None:
        if isinstance(self.method, str):
            self.method = GeneratorMethod(self.method.lower())
        if isinstance(self.backend, str):
            self.backend = BackendType(self.backend.lower())

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "method": (
                self.method.value if isinstance(self.method, GeneratorMethod) else self.method
            ),
            "backend": (
                self.backend.value if isinstance(self.backend, BackendType) else self.backend
            ),
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "generator_dim": self.generator_dim,
            "discriminator_dim": self.discriminator_dim,
            "embedding_dim": self.embedding_dim,
            "random_seed": self.random_seed,
            "verbose": self.verbose,
            "device": self.device,
            "n_critics": self.n_critics,
            "pac": self.pac,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GeneratorConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class PrivacyConfig:
    """Configuration for privacy protection.

    Attributes:
        enable_differential_privacy: Whether to use differential privacy
        epsilon: Privacy budget (lower = more privacy)
        delta: Probability of privacy breach
        max_grad_norm: Maximum gradient norm for DP-SGD
        k_anonymity: Minimum anonymity group size
        l_diversity: Minimum diversity for sensitive attributes
        suppress_rare_categories: Whether to suppress rare categories
        rare_threshold: Threshold for considering a category as rare
        noise_multiplier: Noise multiplier for DP
        privacy_level: Preset privacy level
        sensitive_columns: List of columns containing sensitive data
        quasi_identifiers: List of columns that could identify individuals
    """

    enable_differential_privacy: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    k_anonymity: Optional[int] = None
    l_diversity: Optional[int] = None
    suppress_rare_categories: bool = False
    rare_threshold: float = 0.01
    noise_multiplier: Optional[float] = None
    privacy_level: Union[str, PrivacyLevel] = PrivacyLevel.NONE
    sensitive_columns: List[str] = field(default_factory=list)
    quasi_identifiers: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.privacy_level, str):
            self.privacy_level = PrivacyLevel(self.privacy_level.lower())

        # Apply presets based on privacy level
        if self.privacy_level == PrivacyLevel.LOW:
            if self.epsilon == 1.0:  # Default, not explicitly set
                self.epsilon = 10.0
        elif self.privacy_level == PrivacyLevel.MEDIUM:
            if self.epsilon == 1.0:
                self.epsilon = 1.0
            self.enable_differential_privacy = True
        elif self.privacy_level == PrivacyLevel.HIGH:
            if self.epsilon == 1.0:
                self.epsilon = 0.1
            self.enable_differential_privacy = True
            self.suppress_rare_categories = True
        elif self.privacy_level == PrivacyLevel.MAXIMUM:
            if self.epsilon == 1.0:
                self.epsilon = 0.01
            self.enable_differential_privacy = True
            self.suppress_rare_categories = True
            if self.k_anonymity is None:
                self.k_anonymity = 10

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "enable_differential_privacy": self.enable_differential_privacy,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "max_grad_norm": self.max_grad_norm,
            "k_anonymity": self.k_anonymity,
            "l_diversity": self.l_diversity,
            "suppress_rare_categories": self.suppress_rare_categories,
            "rare_threshold": self.rare_threshold,
            "noise_multiplier": self.noise_multiplier,
            "privacy_level": (
                self.privacy_level.value
                if isinstance(self.privacy_level, PrivacyLevel)
                else self.privacy_level
            ),
            "sensitive_columns": self.sensitive_columns,
            "quasi_identifiers": self.quasi_identifiers,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PrivacyConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)


@dataclass
class TimeSeriesConfig:
    """Configuration for time series generation.

    Attributes:
        sequence_length: Length of each time series sequence
        n_features: Number of features in the time series
        hidden_dim: Hidden dimension for RNN/GRU layers
        n_layers: Number of RNN/GRU layers
        temporal_order: Order for ARIMA model
        seasonal_period: Seasonal period for decomposition
        preserve_trend: Whether to preserve trend component
        preserve_seasonality: Whether to preserve seasonal component
    """

    sequence_length: int = 24
    n_features: Optional[int] = None
    hidden_dim: int = 24
    n_layers: int = 3
    temporal_order: Tuple[int, int, int] = (1, 0, 1)  # ARIMA (p, d, q)
    seasonal_period: Optional[int] = None
    preserve_trend: bool = True
    preserve_seasonality: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "temporal_order": self.temporal_order,
            "seasonal_period": self.seasonal_period,
            "preserve_trend": self.preserve_trend,
            "preserve_seasonality": self.preserve_seasonality,
        }


@dataclass
class TextGenerationConfig:
    """Configuration for text generation.

    Attributes:
        backend: Backend to use ('openai', 'huggingface')
        model_name: Name of the model to use
        temperature: Sampling temperature (higher = more random)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        api_key: API key for OpenAI (if using OpenAI backend)
        batch_size: Batch size for generation
        privacy_filter: Whether to filter PII from generated text
    """

    backend: str = "huggingface"
    model_name: str = "gpt2"
    temperature: float = 0.7
    max_tokens: int = 256
    top_p: float = 0.9
    top_k: int = 50
    api_key: Optional[str] = None
    batch_size: int = 10
    privacy_filter: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "backend": self.backend,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "batch_size": self.batch_size,
            "privacy_filter": self.privacy_filter,
        }


@dataclass
class EvaluationConfig:
    """Configuration for quality evaluation.

    Attributes:
        statistical_tests: List of statistical tests to run
        ml_models: List of ML models for utility evaluation
        privacy_metrics: List of privacy metrics to compute
        target_column: Target column for ML utility evaluation
        test_size: Test set size for ML evaluation
        n_folds: Number of folds for cross-validation
        random_seed: Random seed for reproducibility
    """

    statistical_tests: List[str] = field(
        default_factory=lambda: ["ks_test", "chi_squared", "correlation"]
    )
    ml_models: List[str] = field(default_factory=lambda: ["random_forest", "logistic_regression"])
    privacy_metrics: List[str] = field(
        default_factory=lambda: ["dcr", "reidentification", "attribute_disclosure"]
    )
    target_column: Optional[str] = None
    test_size: float = 0.2
    n_folds: int = 5
    random_seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "statistical_tests": self.statistical_tests,
            "ml_models": self.ml_models,
            "privacy_metrics": self.privacy_metrics,
            "target_column": self.target_column,
            "test_size": self.test_size,
            "n_folds": self.n_folds,
            "random_seed": self.random_seed,
        }


@dataclass
class GenesisConfig:
    """Unified configuration for Genesis synthetic data generation.

    This class provides a single configuration object that encompasses all
    aspects of synthetic data generation, including training, privacy,
    evaluation, and specialized data types.

    Using GenesisConfig simplifies the API by avoiding the need to manage
    multiple configuration objects.

    Example:
        >>> config = GenesisConfig(
        ...     training={"epochs": 500, "batch_size": 256},
        ...     privacy={"level": "high"},
        ...     evaluation={"target_column": "income"}
        ... )
        >>> generator = CTGANGenerator.from_config(config)

        >>> # Or build programmatically
        >>> config = GenesisConfig()
        >>> config.training.epochs = 500
        >>> config.privacy.privacy_level = PrivacyLevel.HIGH

    Attributes:
        training: Generator training configuration
        privacy: Privacy protection configuration
        evaluation: Quality evaluation configuration
        timeseries: Time series generation configuration (optional)
        text: Text generation configuration (optional)
        metadata: Additional user-defined metadata
    """

    training: GeneratorConfig = field(default_factory=GeneratorConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    timeseries: Optional[TimeSeriesConfig] = None
    text: Optional[TextGenerationConfig] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        training: Optional[Union[GeneratorConfig, Dict[str, Any]]] = None,
        privacy: Optional[Union[PrivacyConfig, Dict[str, Any]]] = None,
        evaluation: Optional[Union[EvaluationConfig, Dict[str, Any]]] = None,
        timeseries: Optional[Union[TimeSeriesConfig, Dict[str, Any]]] = None,
        text: Optional[Union[TextGenerationConfig, Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize unified configuration.

        Args:
            training: Training configuration (dict or GeneratorConfig)
            privacy: Privacy configuration (dict or PrivacyConfig)
            evaluation: Evaluation configuration (dict or EvaluationConfig)
            timeseries: Time series config (dict or TimeSeriesConfig), optional
            text: Text generation config (dict or TextGenerationConfig), optional
            metadata: Additional user-defined metadata
        """
        # Handle training config
        if training is None:
            self.training = GeneratorConfig()
        elif isinstance(training, dict):
            self.training = GeneratorConfig.from_dict(training)
        else:
            self.training = training

        # Handle privacy config
        if privacy is None:
            self.privacy = PrivacyConfig()
        elif isinstance(privacy, dict):
            self.privacy = PrivacyConfig.from_dict(privacy)
        else:
            self.privacy = privacy

        # Handle evaluation config
        if evaluation is None:
            self.evaluation = EvaluationConfig()
        elif isinstance(evaluation, dict):
            self.evaluation = EvaluationConfig(**evaluation)
        else:
            self.evaluation = evaluation

        # Handle optional timeseries config
        if timeseries is None:
            self.timeseries = None
        elif isinstance(timeseries, dict):
            self.timeseries = TimeSeriesConfig(**timeseries)
        else:
            self.timeseries = timeseries

        # Handle optional text config
        if text is None:
            self.text = None
        elif isinstance(text, dict):
            self.text = TextGenerationConfig(**text)
        else:
            self.text = text

        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        result = {
            "training": self.training.to_dict(),
            "privacy": self.privacy.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "metadata": self.metadata,
        }
        if self.timeseries is not None:
            result["timeseries"] = self.timeseries.to_dict()
        if self.text is not None:
            result["text"] = self.text.to_dict()
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "GenesisConfig":
        """Create configuration from dictionary.

        Args:
            config_dict: Dictionary with configuration values

        Returns:
            GenesisConfig instance
        """
        return cls(
            training=config_dict.get("training"),
            privacy=config_dict.get("privacy"),
            evaluation=config_dict.get("evaluation"),
            timeseries=config_dict.get("timeseries"),
            text=config_dict.get("text"),
            metadata=config_dict.get("metadata", {}),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "GenesisConfig":
        """Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            GenesisConfig instance
        """
        import yaml

        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_json(cls, path: str) -> "GenesisConfig":
        """Load configuration from JSON file.

        Args:
            path: Path to JSON configuration file

        Returns:
            GenesisConfig instance
        """
        import json

        with open(path) as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file.

        Args:
            path: Path to save YAML configuration
        """
        import yaml

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def save_json(self, path: str) -> None:
        """Save configuration to JSON file.

        Args:
            path: Path to save JSON configuration
        """
        import json

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def with_training(self, **kwargs: Any) -> "GenesisConfig":
        """Return a copy with updated training configuration.

        Args:
            **kwargs: Training configuration parameters to update

        Returns:
            New GenesisConfig with updated training settings
        """
        new_training_dict = self.training.to_dict()
        new_training_dict.update(kwargs)
        return GenesisConfig(
            training=new_training_dict,
            privacy=self.privacy,
            evaluation=self.evaluation,
            timeseries=self.timeseries,
            text=self.text,
            metadata=self.metadata,
        )

    def with_privacy(self, **kwargs: Any) -> "GenesisConfig":
        """Return a copy with updated privacy configuration.

        Args:
            **kwargs: Privacy configuration parameters to update

        Returns:
            New GenesisConfig with updated privacy settings
        """
        new_privacy_dict = self.privacy.to_dict()
        new_privacy_dict.update(kwargs)
        return GenesisConfig(
            training=self.training,
            privacy=new_privacy_dict,
            evaluation=self.evaluation,
            timeseries=self.timeseries,
            text=self.text,
            metadata=self.metadata,
        )

    @classmethod
    def quick(
        cls,
        epochs: int = 100,
        privacy_level: str = "none",
        verbose: bool = True,
    ) -> "GenesisConfig":
        """Create a quick configuration with common defaults.

        Convenience method for rapid prototyping.

        Args:
            epochs: Number of training epochs
            privacy_level: Privacy level ('none', 'low', 'medium', 'high', 'maximum')
            verbose: Whether to print progress

        Returns:
            GenesisConfig configured for quick experimentation
        """
        return cls(
            training={"epochs": epochs, "verbose": verbose},
            privacy={"privacy_level": privacy_level},
        )

    @classmethod
    def production(
        cls,
        privacy_level: str = "high",
        random_seed: int = 42,
    ) -> "GenesisConfig":
        """Create a production-ready configuration.

        Convenience method for production deployments with
        sensible defaults for reproducibility and privacy.

        Args:
            privacy_level: Privacy level (default: 'high')
            random_seed: Random seed for reproducibility

        Returns:
            GenesisConfig configured for production use
        """
        return cls(
            training={
                "epochs": 300,
                "random_seed": random_seed,
                "verbose": False,
            },
            privacy={"privacy_level": privacy_level},
        )
