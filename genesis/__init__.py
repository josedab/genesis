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

# Convenience functions from new modules (v2.1.0)
from genesis.llm_finetuning import generate_safe_finetuning_data, audit_training_data
from genesis.realtime_api import create_realtime_app, save_model_for_realtime
from genesis.cicd import (
    generate_github_workflow,
    generate_gitlab_ci,
)
from genesis.dp_compiler import dp_query
from genesis.cloud_deploy import quick_deploy as cloud_deploy
from genesis.leaderboard import run_and_submit as submit_benchmark

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
    # v2.1.0 Next-Gen Convenience Functions
    "generate_safe_finetuning_data",
    "audit_training_data",
    "create_realtime_app",
    "save_model_for_realtime",
    "generate_github_workflow",
    "generate_gitlab_ci",
    "dp_query",
    "cloud_deploy",
    "submit_benchmark",
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
#
# v1.5.0 Next-Gen modules:
# Import with: from genesis.streaming.production import ProductionKafkaProducer, CheckpointManager
# Import with: from genesis.sla import SLAContract, SLAValidator, SLAEnforcedGenerator
# Import with: from genesis.zero_shot import ZeroShotSchemaGenerator, ZeroShotDataGenerator
# Import with: from genesis.connectors import SnowflakeConnector, BigQueryConnector, DatabricksConnector
# Import with: from genesis.fairness import FairnessAnalyzer, FairGenerator, CounterfactualGenerator
# Import with: from genesis.saas import TenantManager, APIKeyManager, UsageMeter
# Import with: from genesis.observability import GenesisTracer, MetricsCollector
# Import with: from genesis.delta import DeltaGenerator, ChangeTracker, SCDGenerator
# Import with: from genesis.explainability import ExplainableGenerator, AttributionTracker
#
# v2.0.0 Next-Gen modules (NEW):
# Import with: from genesis.agents.agentic import AgenticDataGenerator, AgentOrchestrator
# Import with: from genesis.benchmarking import BenchmarkSuite, BenchmarkMetrics
# Import with: from genesis.production_mirror import ProductionMirror, DriftAwareMirror
# Import with: from genesis.privacy_budget import PrivacyBudgetOrchestrator, CompositionCalculator
# Import with: from genesis.lakehouse import SyntheticLakehouse, LakehouseWriter
# Import with: from genesis.model_hub import ModelHub, ModelCard, PretrainedGenerator
# Import with: from genesis.compliance_as_code import PolicyValidator, PrivacyPolicy
# Import with: from genesis.marketplace_v2 import MarketplaceV2, Organization, PaymentProcessor
# Import with: from genesis.edge_generation import EdgeExporter, EdgeRuntime, export_to_onnx
# Import with: from genesis.causality import CausalModel, CausalGenerator, CausalDiscovery
#
# v2.1.0 Next-Gen modules (NEW - Implemented):
# Import with: from genesis.realtime_api import RealtimeGenerator, RealtimeConfig, create_realtime_app
# Import with: from genesis.cicd import SyntheticDataPipeline, SchemaDriftDetector, QualityGate
# Import with: from genesis.dp_compiler import DPCompiler, SQLParser, DPBudgetManager
# Import with: from genesis.federated_marketplace import FederatedMarketplace, ModelPackage
# Import with: from genesis.cloud_deploy import CloudDeployer, deploy_to_aws, deploy_to_gcp
# Import with: from genesis.leaderboard import Leaderboard, LeaderboardEntry, Submission
# Import with: from genesis.llm_finetuning import FineTuningDataGenerator, SafetyConfig, MemorizationDetector
