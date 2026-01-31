# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.4.0] - 2026-01-28

### Added

#### AutoML Synthesis (`genesis.automl`)
Automatic method selection based on dataset characteristics.

- `MetaFeatureExtractor`: Extracts meta-features (n_rows, n_columns, numeric_ratio, categorical_ratio, missing_ratio, correlations, cardinality) from datasets
- `MethodSelector`: Rule-based method selection with confidence scores and explanations
- `AutoMLSynthesizer`: End-to-end synthesizer that automatically selects and configures the best generation method
- `auto_synthesize()`: One-line convenience function for automatic synthesis
- Supports speed vs quality preference trade-offs

```python
from genesis import auto_synthesize
synthetic = auto_synthesize(df, n_samples=1000, prefer_quality=True)
```

#### Synthetic Data Augmentation (`genesis.augmentation`)
Balance imbalanced datasets with intelligent synthetic sample generation.

- `SyntheticAugmenter`: Core augmentation engine with multiple strategies
- `AugmentationPlanner`: Analyzes imbalance and recommends augmentation approach
- Strategies: `oversample`, `smote`, `combined`
- Target ratio control for precise class distribution
- Quality validation for generated samples

```python
from genesis import augment_imbalanced
balanced_df = augment_imbalanced(df, target_column="label", strategy="oversample")
```

#### Privacy Attack Testing (`genesis.privacy_attacks`)
Comprehensive privacy vulnerability assessment for synthetic data.

- `MembershipInferenceAttack`: Tests if training records can be identified
- `AttributeInferenceAttack`: Tests if sensitive attributes can be inferred
- `ReidentificationAttack`: Tests linkage risk via quasi-identifiers
- `PrivacyAttackTester`: Orchestrates multiple attacks
- `PrivacyAuditReport`: Comprehensive results with risk levels and recommendations
- `run_privacy_audit()`: One-line convenience function

```python
from genesis import run_privacy_audit
report = run_privacy_audit(real_df, synthetic_df, sensitive_columns=["ssn", "income"])
print(f"Passed: {report.passed}, Risk: {report.overall_risk}")
```

#### LLM-Powered Schema Inference (`genesis.llm_inference`)
Automatic semantic type detection using AI and pattern matching.

- `LLMSchemaInferrer`: OpenAI/Anthropic-powered schema inference
- `RuleBasedInferrer`: Pattern-based fallback with 30+ recognized types
- Recognizes: email, phone, ssn, address, credit_card, ip_address, uuid, etc.
- Automatic constraint detection
- Domain inference (healthcare, finance, retail)

```python
from genesis.llm_inference import LLMSchemaInferrer
inferrer = LLMSchemaInferrer(api_key=key)
schema = inferrer.infer(df)
```

#### Drift Detection (`genesis.drift`)
Monitor and adapt to changing data distributions.

- `DataDriftDetector`: Statistical drift detection (KS test, JS divergence, PSI)
- `DriftAwareGenerator`: Generates data adapting to detected drift
- `ContinuousMonitor`: Real-time streaming drift detection
- `detect_drift()`: Convenience function for quick drift analysis
- `calculate_psi()`: Population Stability Index calculation

```python
from genesis import detect_drift
report = detect_drift(baseline_df, current_df)
if report.has_significant_drift:
    print(f"Drifted columns: {report.drifted_columns}")
```

#### Dataset Versioning (`genesis.versioning`)
Git-like version control for synthetic datasets.

- `DatasetRepository`: Full repository management (init, commit, branch, merge, tag)
- `Commit`: Immutable dataset snapshots with metadata
- `DatasetDiff`: Compare versions (rows added/removed, column changes)
- `VersionedGenerator`: Auto-commit on generation
- Content-addressable storage for deduplication

```python
from genesis.versioning import DatasetRepository
repo = DatasetRepository.init("./data_repo")
repo.commit(df, message="Initial dataset")
repo.tag("v1.0")
```

#### GPU Acceleration (`genesis.gpu`)
High-performance generation for large datasets.

- `BatchedGenerator`: Memory-efficient batched generation on single GPU
- `MultiGPUGenerator`: Distribute across multiple GPUs
- `detect_gpus()`: GPU detection and capability checking
- `optimize_batch_size()`: Automatic batch size tuning
- `estimate_memory()`: Memory requirement estimation
- Mixed precision (FP16) support

```python
from genesis.gpu import BatchedGenerator
generator = BatchedGenerator(method="ctgan", device="cuda")
synthetic = generator.generate(1_000_000)  # Batched automatically
```

#### Domain-Specific Generators (`genesis.domains`)
Pre-configured generators for common business domains.

- `HealthcareGenerator`: Patient cohorts, clinical events, lab results, medications
- `FinanceGenerator`: Transactions, accounts, credit profiles, fraud scenarios
- `RetailGenerator`: Customers, orders, products, e-commerce datasets
- HIPAA-aware healthcare generation with privacy controls
- Realistic patterns: seasonal trends, fraud indicators, customer behavior

```python
from genesis.domains import FinanceGenerator
generator = FinanceGenerator()
transactions = generator.generate_transactions(10000, include_fraud=True, fraud_rate=0.02)
```

#### Pipeline Builder (`genesis.pipeline`)
Visual workflow construction for complex generation pipelines.

- `PipelineBuilder`: Fluent API for pipeline construction
- `Pipeline`: Executable pipeline with validation
- `PipelineExecutor`: Topological execution engine
- Node types: source, transform, synthesize, evaluate, sink
- YAML serialization for pipeline definitions
- Parallel execution for independent nodes

```python
from genesis.pipeline import PipelineBuilder
pipeline = (
    PipelineBuilder()
    .source("data.csv")
    .transform("clean", {"drop_na": True})
    .synthesize(method="ctgan", n_samples=10000)
    .evaluate()
    .sink("synthetic.csv")
    .build()
)
pipeline.execute()
```

#### CLI Commands
New commands for v1.4.0 features:

- `genesis automl`: Auto-select method and generate
- `genesis augment`: Augment imbalanced datasets
- `genesis privacy-audit`: Run privacy attack tests
- `genesis drift`: Detect distribution drift
- `genesis version`: Git-like dataset versioning
- `genesis domain`: Domain-specific generation
- `genesis pipeline`: Execute pipeline configurations

### Fixed
- Fixed numpy boolean subtract error in statistical evaluation
- Fixed `is True` vs `== True` assertions for numpy bools in tests
- Fixed single quasi-identifier handling in k-anonymity enforcement

### Changed
- Multi-table schema now supports auto-discovery of PKs and FKs
- Version bumped to 1.4.0
- Updated pyproject.toml with `streaming` and `reporting` optional dependencies

### Dependencies
- Added optional: `kafka-python>=2.0.0`, `websockets>=10.0` (streaming)
- Added optional: `weasyprint>=57.0` (reporting)

## [1.3.0] - 2026-01-15

### Added
- **Guided Conditional Generation** (`genesis.generators.conditional`)
  - `GuidedConditionalSampler` for constraint-satisfying generation
  - `ConditionBuilder` fluent API for building complex conditions
  - Strategies: iterative refinement, importance sampling
- **Natural Language Data Generation** (CLI & API)
  - `genesis chat` CLI command for conversational data generation
  - `/v1/generate/natural-language` REST API endpoint
  - Intent parsing for conditions, counts, and schemas
- **Interactive Quality Dashboard** (`genesis.dashboard`)
  - `generate_plotly_figures()` for interactive visualizations
  - `InteractiveDashboard` with FastAPI REST server
  - PDF export via `save_pdf()` method
- **Streaming Generation** (`genesis.streaming`)
  - `KafkaStreamingGenerator` for Kafka-based streaming
  - `WebSocketStreamingGenerator` for real-time WebSocket output
- **Blockchain-Style Data Lineage** (`genesis.lineage`)
  - `LineageChain` with SHA-256 hash chain verification
  - `LineageBlock` for immutable audit trail entries
  - Export/import with tamper detection
- **Federated Generation Enhancements** (`genesis.federated`)
  - `SecureAggregator` with differential privacy noise
  - `FederatedTrainingSimulator` for testing distributed scenarios
  - Non-IID data partitioning support
- **Cloud Deployment Kit**
  - Multi-stage Dockerfile (production/development/GPU targets)
  - Enhanced docker-compose.yml with profiles and services
  - Helm chart improvements: autoscaling, PDB, network policies
- **Schema Discovery** CLI command
  - `genesis discover` for automatic schema inference
  - Distribution analysis and profiling

### Changed
- Improved error handling in conditional generation for edge cases
- Better boolean dtype handling in statistics extraction

## [1.2.0] - 2026-01-28

### Added
- **Plugin System** (`genesis.plugins`)
  - Decorator-based registration: `@register_generator`, `@register_transformer`
  - Plugin discovery from directories
  - Support for custom generators, transformers, evaluators, constraints, callbacks
- **Auto-Tuning Hyperparameters** (`genesis.tuning`)
  - `AutoTuner` for automatic hyperparameter optimization
  - Optuna integration with fallback to random search
  - Presets: fast, balanced, quality
  - `auto_tune()` convenience function
- **Privacy Certificate Generation** (`genesis.compliance`)
  - `PrivacyCertificate` for compliance documentation
  - GDPR, HIPAA, CCPA framework support
  - Risk assessment with formal privacy guarantees
  - Export to HTML, JSON, Markdown
- **Drift Detection & Alerting** (`genesis.monitoring`)
  - `DriftDetector` for quality monitoring
  - Data drift, concept drift, quality drift detection
  - Alert handlers: logging, callback, webhook
  - `DriftMonitor` for continuous monitoring
- **Synthetic Data Debugger** (`genesis.debugger`)
  - `SyntheticDebugger` for diagnosing quality issues
  - Column-level diagnostics with similarity scores
  - Correlation analysis and suggestions
  - Actionable fix recommendations
- **Anomaly Synthesis** (`genesis.anomaly`)
  - `AnomalyGenerator` for synthetic outliers
  - Types: statistical, point, contextual, collective, adversarial
  - Profiles: fraud, intrusion, equipment failure, medical
  - `BalancedDatasetGenerator` for imbalanced learning
- **Distributed Training** (`genesis.distributed`)
  - `DistributedTrainer` for multi-GPU/multi-node training
  - Ray and Dask backend support
  - Automatic data sharding (random, stratified, contiguous, hash)
  - `GPUManager` for resource management
- **Cross-Modal Generation** (`genesis.crossmodal`)
  - `CrossModalGenerator` for paired multi-modal data
  - Joint latent space learning
  - `TabularTextGenerator` for tabular + text pairs
  - Modality encoders: tabular, text
- **Visual Schema Editor Backend** (`genesis.schema_editor`)
  - `SchemaDefinition` and `ColumnDefinition` classes
  - Schema inference from DataFrames
  - Export to Python code, YAML, JSON
  - `SchemaEditorAPI` for REST integration
- **Synthetic Data Marketplace Backend** (`genesis.marketplace`)
  - `Marketplace` for dataset listing and discovery
  - Quality verification and provenance tracking
  - License management (MIT, CC-BY, Apache, etc.)
  - Search with filtering and pagination

### Changed
- Updated version to 1.2.0

### Dependencies
- Added optional: `optuna>=3.0.0` (tuning)
- Added optional: `ray>=2.0.0`, `dask>=2023.1.0` (distributed)

## [1.1.0] - 2026-01-28

### Added
- **Conditional Generation API** (`genesis.generators.conditional`)
  - Condition operators: `=`, `>`, `>=`, `<`, `<=`, `in`, `between`, `like`
  - `generate_conditional()` method on all generators
  - `Upsampler` for class imbalance correction
  - `ScenarioGenerator` for batch scenario generation
- **LLM-Powered Agents** (`genesis.agents`)
  - `SyntheticDataAgent` for natural language data generation
  - OpenAI and Anthropic provider support
  - Conversational interface with clarification handling
- **MLflow/W&B Integration** (`genesis.integrations`)
  - `MLflowCallback` and `WandbCallback` for training progress
  - `log_generator_to_mlflow()` and `log_generator_to_wandb()` functions
  - Experiment tracking context managers
- **Cloud-Native Deployment**
  - REST API with FastAPI (`genesis.api`)
  - Docker and docker-compose configurations
  - Kubernetes Helm chart (`deployment/helm/genesis/`)
- **Data Lineage & Provenance** (`genesis.lineage`)
  - `DataLineage` tracker for audit trails
  - `DataManifest` for complete provenance documentation
  - SBOM-like export format for supply chain transparency
- **Automated Schema Discovery** (`genesis.discovery`)
  - `SchemaDiscovery.from_database()` for SQL databases
  - `SchemaDiscovery.from_csv_directory()` for file-based data
  - Automatic foreign key relationship inference
- **Data Quality Dashboard** (`genesis.dashboard`)
  - `QualityDashboard` with interactive HTML reports
  - Distribution comparison plots
  - Correlation matrix visualization
  - Privacy risk heatmaps
- **Streaming/Incremental Generation** (`genesis.streaming`)
  - `StreamingGenerator` for continuous generation
  - `partial_fit()` for incremental model updates
  - `generate_stream()` generator for batch iteration
  - Async generation with callbacks
- **Image Synthesis Module** (`genesis.generators.image`)
  - `DiffusionImageGenerator` with multiple providers
  - HuggingFace Diffusers (local Stable Diffusion)
  - OpenAI DALL-E API
  - Replicate API (SDXL, etc.)
- **Federated Synthetic Data** (`genesis.federated`)
  - `FederatedGenerator` for distributed training
  - `DataSite` for local data encapsulation
  - `ModelAggregator` for parameter aggregation
  - Privacy-preserving collaboration

### Changed
- Enhanced `SyntheticGenerator` with `generate_conditional()`, `upsample()`, and `generate_scenarios()` methods
- Updated `pyproject.toml` with new optional dependencies: `api`, `image`, `integrations`

### Dependencies
- Added optional: `fastapi>=0.100.0`, `uvicorn>=0.22.0`, `pydantic>=2.0.0` (api)
- Added optional: `diffusers>=0.20.0`, `Pillow>=9.0.0` (image)
- Added optional: `mlflow>=2.0.0`, `wandb>=0.15.0` (integrations)

## [1.0.0] - 2024-12-01

### Added
- Initial release of Genesis synthetic data generation platform
- **Tabular Data Generation**
  - CTGAN (Conditional Tabular GAN) implementation
  - TVAE (Tabular Variational Autoencoder) implementation
  - Gaussian Copula statistical method
  - Auto-selection of best generator based on data characteristics
- **Time Series Generation**
  - TimeGAN implementation for temporal data
  - Statistical methods (ARIMA-based) for simpler time series
- **Text Generation**
  - OpenAI API backend integration
  - HuggingFace transformers backend
  - Privacy-aware text generation
- **Privacy Features**
  - Differential privacy (DP-SGD) support
  - K-anonymity verification
  - L-diversity enforcement
  - Re-identification risk metrics
- **Quality Evaluation**
  - Statistical fidelity metrics (KS test, chi-squared, correlation)
  - ML utility metrics (TSTR, TRTS)
  - Privacy metrics (DCR, attribute disclosure)
  - Comprehensive quality reports (HTML, JSON)
- **Multi-table Synthesis**
  - Foreign key detection and preservation
  - Referential integrity enforcement
  - Cross-table correlation preservation
- **Constraints System**
  - Positive value constraints
  - Range constraints
  - Uniqueness constraints
  - Custom constraint support
- **CLI Interface**
  - `genesis generate` command
  - `genesis evaluate` command
  - `genesis analyze` command
  - `genesis report` command
- **Framework Support**
  - PyTorch backend
  - TensorFlow backend
  - Automatic backend selection
- **I/O Support**
  - Pandas DataFrame integration
  - CSV file support
  - Parquet file support
  - Database connections (SQLAlchemy)

### Documentation
- Comprehensive README
- Installation guide
- Quick start guide
- User guides for all features
- API reference
- Example Jupyter notebooks

## [Unreleased]

### Planned
- Additional image synthesis backends (FLUX, Midjourney API)
- Graph data synthesis
- Audio/video synthetic data generation
- Native Polars support
