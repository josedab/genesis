"""Genesis: Synthetic Data Generation Platform.

Genesis is a comprehensive synthetic data generation platform that creates
realistic, privacy-safe data for ML training, testing, and development.

Example:
    >>> from genesis import SyntheticGenerator, PrivacyConfig
    >>>
    >>> generator = SyntheticGenerator(method='auto')
    >>> generator.fit(real_data)
    >>> synthetic_data = generator.generate(n_samples=10000)
    >>> report = generator.quality_report()
    >>> print(report.summary())
"""

# Agents
from genesis.agents import SyntheticDataAgent
from genesis.augmentation import augment_imbalanced

# v1.4.0 Convenience Functions (top-level exports)
from genesis.automl import auto_synthesize

# Core
from genesis.core.base import BaseGenerator, SyntheticGenerator
from genesis.core.config import (
    EvaluationConfig,
    GeneratorConfig,
    PrivacyConfig,
    TextGenerationConfig,
    TimeSeriesConfig,
)
from genesis.core.constraints import (
    BaseConstraint,
    Constraint,
    ConstraintSet,
)
from genesis.core.exceptions import (
    ConfigurationError,
    FittingError,
    GenerationError,
    GenesisError,
    NotFittedError,
    ValidationError,
)
from genesis.core.types import (
    BackendType,
    ColumnMetadata,
    ColumnType,
    DataSchema,
    GeneratorMethod,
    PrivacyLevel,
)
from genesis.drift import detect_drift

# Evaluation
from genesis.evaluation.evaluator import QualityEvaluator
from genesis.evaluation.report import QualityReport

# Conditional Generation
from genesis.generators.conditional import (
    Condition,
    ConditionalSampler,
    ConditionBuilder,
    ConditionSet,
    GuidedConditionalSampler,
    Operator,
    ScenarioGenerator,
    Upsampler,
    conditional_generate,
)

# Generators
from genesis.generators.tabular import (
    CTGANGenerator,
    GaussianCopulaGenerator,
    TVAEGenerator,
)
from genesis.generators.text import LLMTextGenerator
from genesis.generators.timeseries import (
    StatisticalTimeSeriesGenerator,
    TimeGANGenerator,
)

# Multi-table
from genesis.multitable.generator import MultiTableGenerator
from genesis.multitable.schema import RelationalSchema
from genesis.privacy_attacks import run_privacy_audit
from genesis.version import __version__

# Next-Gen Features (lazy imports for optional dependencies)
# These are available as: from genesis.lineage import DataLineage, etc.

__all__ = [
    # Version
    "__version__",
    # Main classes
    "SyntheticGenerator",
    "BaseGenerator",
    # Configuration
    "GeneratorConfig",
    "PrivacyConfig",
    "TimeSeriesConfig",
    "TextGenerationConfig",
    "EvaluationConfig",
    # Constraints
    "Constraint",
    "ConstraintSet",
    "BaseConstraint",
    # Types
    "ColumnType",
    "ColumnMetadata",
    "DataSchema",
    "GeneratorMethod",
    "BackendType",
    "PrivacyLevel",
    # Evaluation
    "QualityEvaluator",
    "QualityReport",
    # Generators
    "CTGANGenerator",
    "TVAEGenerator",
    "GaussianCopulaGenerator",
    "TimeGANGenerator",
    "StatisticalTimeSeriesGenerator",
    "LLMTextGenerator",
    # Conditional Generation
    "Condition",
    "ConditionBuilder",
    "ConditionSet",
    "ConditionalSampler",
    "GuidedConditionalSampler",
    "Operator",
    "Upsampler",
    "ScenarioGenerator",
    "conditional_generate",
    # Agents
    "SyntheticDataAgent",
    # Multi-table
    "MultiTableGenerator",
    "RelationalSchema",
    # v1.4.0 Convenience Functions
    "auto_synthesize",
    "augment_imbalanced",
    "run_privacy_audit",
    "detect_drift",
    # Exceptions
    "GenesisError",
    "ConfigurationError",
    "ValidationError",
    "FittingError",
    "GenerationError",
    "NotFittedError",
]

# Submodule documentation for IDE autocomplete
# Import with: from genesis.lineage import DataLineage
# Import with: from genesis.streaming import StreamingGenerator
# Import with: from genesis.federated import FederatedGenerator
# Import with: from genesis.dashboard import QualityDashboard
# Import with: from genesis.discovery import SchemaDiscovery
# Import with: from genesis.integrations import MLflowCallback, WandbCallback
# Import with: from genesis.generators.image import DiffusionImageGenerator
#
# v1.2.0 modules:
# Import with: from genesis.plugins import register_generator, get_generator
# Import with: from genesis.tuning import AutoTuner, auto_tune
# Import with: from genesis.compliance import PrivacyCertificate
# Import with: from genesis.monitoring import DriftDetector, DriftMonitor
# Import with: from genesis.debugger import SyntheticDebugger
# Import with: from genesis.anomaly import AnomalyGenerator
# Import with: from genesis.distributed import DistributedTrainer
# Import with: from genesis.crossmodal import CrossModalGenerator
# Import with: from genesis.schema_editor import SchemaDefinition
# Import with: from genesis.marketplace import Marketplace
#
# v1.4.0 modules:
# Import with: from genesis.automl import AutoMLSynthesizer, auto_synthesize
# Import with: from genesis.augmentation import SyntheticAugmenter, augment_imbalanced
# Import with: from genesis.privacy_attacks import PrivacyAttackTester, run_privacy_audit
# Import with: from genesis.llm_inference import LLMSchemaInferrer, infer_schema
# Import with: from genesis.drift import DriftAwareGenerator, detect_drift
# Import with: from genesis.versioning import DatasetRepository, VersionedGenerator
# Import with: from genesis.gpu import BatchedGenerator, create_gpu_generator
# Import with: from genesis.domains import DomainGenerator, HealthcareGenerator, FinanceGenerator
# Import with: from genesis.pipeline import PipelineBuilder, PipelineExecutor
