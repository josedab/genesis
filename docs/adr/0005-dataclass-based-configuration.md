# ADR-0005: Dataclass-Based Configuration

## Status

Accepted

## Context

Genesis requires configuration at multiple levels:
- Generator settings (epochs, batch size, learning rate, architecture)
- Privacy controls (epsilon, delta, k-anonymity level)
- Evaluation parameters (metrics to compute, thresholds)
- Time series options (sequence length, temporal columns)
- Text generation settings (model, temperature, max tokens)

Common configuration approaches include:

1. **Dictionary/kwargs**: `fit(data, epochs=100, batch_size=500, ...)`
2. **YAML/JSON files**: `fit(data, config='config.yaml')`
3. **Pydantic models**: `fit(data, config=Config(epochs=100))`
4. **Dataclasses**: `fit(data, config=GeneratorConfig(epochs=100))`

Requirements:
- Type safety and IDE autocompletion
- Clear defaults visible in code
- Serializable to/from JSON/YAML
- Validation of invalid values
- No external dependencies for core config

## Decision

We use **Python dataclasses** for all configuration:

```python
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union
from genesis.core.types import GeneratorMethod, BackendType, PrivacyLevel

@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generators."""
    
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
    
    def __post_init__(self) -> None:
        # Validation and normalization
        if isinstance(self.method, str):
            self.method = GeneratorMethod(self.method.lower())
        if isinstance(self.backend, str):
            self.backend = BackendType(self.backend.lower())
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")


@dataclass
class PrivacyConfig:
    """Configuration for privacy protection."""
    
    level: PrivacyLevel = PrivacyLevel.NONE
    epsilon: float = 1.0
    delta: float = 1e-5
    k_anonymity: Optional[int] = None
    l_diversity: Optional[int] = None
    sensitive_columns: list = field(default_factory=list)
    
    @classmethod
    def from_level(cls, level: PrivacyLevel) -> "PrivacyConfig":
        """Create config from preset privacy level."""
        presets = {
            PrivacyLevel.LOW: cls(level=level, epsilon=10.0),
            PrivacyLevel.MEDIUM: cls(level=level, epsilon=1.0),
            PrivacyLevel.HIGH: cls(level=level, epsilon=0.1, k_anonymity=5),
            PrivacyLevel.MAXIMUM: cls(level=level, epsilon=0.01, k_anonymity=10),
        }
        return presets.get(level, cls())
```

Usage:

```python
from genesis import SyntheticGenerator, GeneratorConfig, PrivacyConfig

# Explicit configuration
config = GeneratorConfig(
    method='ctgan',
    epochs=500,
    batch_size=1000,
    generator_dim=(512, 512, 256),
)
privacy = PrivacyConfig(epsilon=0.5, k_anonymity=5)

gen = SyntheticGenerator(config=config, privacy=privacy)

# Or use convenience parameters
gen = SyntheticGenerator(method='ctgan', epochs=500)

# Preset privacy levels
privacy = PrivacyConfig.from_level(PrivacyLevel.HIGH)
```

## Consequences

### Positive

- **IDE autocompletion**: all parameters discoverable via autocomplete
- **Type checking**: mypy catches type errors at lint time
- **Clear defaults**: visible in class definition, not scattered in code
- **Self-documenting**: docstrings appear in IDE hover
- **No dependencies**: dataclasses are stdlib (Python 3.7+)
- **Serializable**: `asdict()` and `from_dict()` for JSON/YAML
- **Immutable option**: can use `frozen=True` for thread safety
- **Validation**: `__post_init__` runs on creation

### Negative

- **Verbosity**: more code than kwargs for simple cases
- **Learning curve**: users must import config classes
- **Nested configs**: deeply nested configs can be awkward

### Mitigations

- Top-level convenience parameters for common options:
  ```python
  # These are equivalent:
  SyntheticGenerator(method='ctgan', epochs=100)
  SyntheticGenerator(config=GeneratorConfig(method='ctgan', epochs=100))
  ```

- Factory methods for common configurations:
  ```python
  config = GeneratorConfig.for_small_data()  # Optimized for <1000 rows
  config = GeneratorConfig.for_large_data()  # Optimized for >1M rows
  privacy = PrivacyConfig.from_level(PrivacyLevel.HIGH)
  ```

## Configuration Hierarchy

```
GeneratorConfig          # Core generation settings
├── method               # ctgan, tvae, gaussian_copula, etc.
├── backend              # pytorch, tensorflow, auto
├── epochs, batch_size   # Training parameters
└── architecture dims    # Neural network structure

PrivacyConfig            # Privacy controls
├── level                # none, low, medium, high, maximum
├── epsilon, delta       # Differential privacy budget
├── k_anonymity          # K-anonymity requirement
└── sensitive_columns    # Columns requiring extra protection

TimeSeriesConfig         # Time series specific
├── sequence_length      # Length of sequences
├── time_column          # Column containing timestamps
└── entity_columns       # Grouping columns

EvaluationConfig         # Quality evaluation settings
├── metrics              # Which metrics to compute
├── n_samples            # Samples for evaluation
└── reference_data       # Holdout data for comparison
```
