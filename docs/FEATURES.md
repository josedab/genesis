# Genesis v1.3.0 + v1.4.0 Features Overview

This document provides a comprehensive overview of all features introduced in Genesis v1.3.0 and v1.4.0.

## Table of Contents

- [v1.3.0 Features](#v130-features)
  - [Guided Conditional Generation](#guided-conditional-generation)
  - [Natural Language Interface](#natural-language-interface)
  - [Interactive Quality Dashboard](#interactive-quality-dashboard)
  - [Streaming Generation](#streaming-generation)
  - [Federated Learning](#federated-learning)
  - [Data Lineage Tracking](#data-lineage-tracking)
- [v1.4.0 Features](#v140-features)
  - [AutoML Synthesis](#automl-synthesis)
  - [Synthetic Augmentation](#synthetic-augmentation)
  - [Privacy Attack Testing](#privacy-attack-testing)
  - [LLM Schema Inference](#llm-schema-inference)
  - [Drift Detection](#drift-detection)
  - [Dataset Versioning](#dataset-versioning)
  - [GPU Acceleration](#gpu-acceleration)
  - [Domain Generators](#domain-generators)
  - [Pipeline Builder](#pipeline-builder)
- [CLI Commands](#cli-commands)
- [Architecture](#architecture)
- [Migration Guide](#migration-guide)

---

## v1.3.0 Features

### Guided Conditional Generation

Generate synthetic data that satisfies specific constraints with smart sampling strategies.

```python
from genesis.generators.conditional import ConditionBuilder, GuidedConditionalSampler

# Build conditions with fluent API
conditions = (
    ConditionBuilder()
    .where("age").gte(21).lte(65)
    .where("income").gt(50000)
    .where("country").in_(["US", "UK", "CA"])
    .build()
)

# Smart sampling
sampler = GuidedConditionalSampler(generator, strategy="importance_sampling")
synthetic = sampler.sample(1000, conditions)
```

### Natural Language Interface

Generate synthetic data using plain English descriptions.

```bash
genesis chat -d reference.csv -o output.csv
```

```python
# In the chat interface:
> Generate 1000 customers aged 25-45 from California with income over 50k
```

### Interactive Quality Dashboard

Plotly-powered interactive visualizations with PDF export.

```python
from genesis.dashboard import QualityDashboard

dashboard = QualityDashboard(real_df, synthetic_df)
dashboard.run(port=8050)  # Interactive web dashboard
dashboard.save_report("report.pdf")  # PDF export
```

### Streaming Generation

Real-time synthetic data via Kafka or WebSocket.

```python
from genesis.streaming import KafkaStreamingGenerator

generator = KafkaStreamingGenerator(
    kafka_topic="synthetic-data",
    bootstrap_servers="localhost:9092"
)
generator.start(rate=100)  # 100 records/second
```

### Federated Learning

Privacy-preserving distributed training across data silos.

```python
from genesis.federated import FederatedTrainingSimulator, SecureAggregator

simulator = FederatedTrainingSimulator(n_clients=5)
aggregator = SecureAggregator(noise_scale=0.1)

model = simulator.train(
    client_data_list,
    aggregator=aggregator,
    rounds=10
)
```

### Data Lineage Tracking

Blockchain-style immutable audit trails.

```python
from genesis.lineage import LineageChain

chain = LineageChain()
chain.add_block("source", {"file": "original.csv", "rows": 10000})
chain.add_block("transform", {"operation": "filter", "condition": "age > 18"})
chain.add_block("generate", {"method": "ctgan", "samples": 5000})

# Verify integrity
assert chain.verify()
chain.export("lineage.json")
```

---

## v1.4.0 Features

### AutoML Synthesis

Automatic method selection based on dataset characteristics.

```python
from genesis import auto_synthesize

# One-line automatic synthesis
synthetic = auto_synthesize(df, n_samples=1000)

# With preferences
from genesis.automl import AutoMLSynthesizer

automl = AutoMLSynthesizer(prefer_quality=True)
automl.fit(df)
synthetic = automl.generate(1000)

print(f"Selected: {automl.selected_method}")  # e.g., CTGAN
print(f"Confidence: {automl.selection_confidence:.0%}")  # e.g., 85%
```

**Method Selection Logic:**

| Data Characteristic | Recommended Method |
|--------------------|-------------------|
| Large dataset (>100K rows) | CTGAN |
| Small dataset (<1K rows) | Gaussian Copula |
| High cardinality categories | CTGAN |
| Strong correlations | TVAE |
| Time series data | TimeGAN |

### Synthetic Augmentation

Balance imbalanced datasets with intelligent augmentation.

```python
from genesis import augment_imbalanced

# Balance an imbalanced dataset
balanced = augment_imbalanced(
    df, 
    target_column="fraud_label",
    strategy="oversample",
    target_ratio=1.0  # Equal class sizes
)

# Analyze imbalance first
from genesis.augmentation import AugmentationPlanner

planner = AugmentationPlanner()
plan = planner.analyze(df, "label")
print(f"Imbalance ratio: {plan.imbalance_ratio:.2f}")
print(f"Recommended strategy: {plan.recommended_strategy}")
```

### Privacy Attack Testing

Comprehensive privacy vulnerability assessment.

```python
from genesis import run_privacy_audit

report = run_privacy_audit(
    real_data=original_df,
    synthetic_data=synthetic_df,
    sensitive_columns=["ssn", "income"],
    quasi_identifiers=["age", "zipcode", "gender"]
)

print(f"Overall Risk: {report.overall_risk}")  # LOW/MEDIUM/HIGH/CRITICAL
print(f"Passed: {report.passed}")

# Individual attacks
print(f"Membership Inference: {report.membership_result.accuracy:.1%}")
print(f"Re-identification Rate: {report.reidentification_result.rate:.1%}")
```

### LLM Schema Inference

Auto-detect column semantics from names and samples.

```python
from genesis.llm_inference import LLMSchemaInferrer

inferrer = LLMSchemaInferrer(
    provider="openai",
    api_key=os.environ["OPENAI_API_KEY"]
)
schema = inferrer.infer(df)

for col, info in schema.items():
    print(f"{col}: {info.semantic_type}")
# user_email -> email
# phone_number -> phone
# created_at -> datetime
```

**Recognized Types:** email, phone, ssn, address, credit_card, ip_address, uuid, latitude, longitude, date, amount, and 20+ more.

### Drift Detection

Monitor and adapt to changing data distributions.

```python
from genesis import detect_drift

report = detect_drift(baseline_df, current_df)

if report.has_significant_drift:
    print(f"Drifted columns: {report.drifted_columns}")
    
    # Generate drift-adapted data
    from genesis.drift import DriftAwareGenerator
    
    generator = DriftAwareGenerator()
    generator.fit(baseline_df)
    synthetic = generator.generate(
        n_samples=1000,
        target_distribution=current_df,
        drift_adaptation="weighted"
    )
```

### Dataset Versioning

Git-like version control for synthetic datasets.

```python
from genesis.versioning import DatasetRepository

# Initialize
repo = DatasetRepository.init("./data_repo")

# Commit versions
repo.commit(df_v1, message="Initial generation")
repo.commit(df_v2, message="Added more samples")

# Branch and tag
repo.branch("experiment")
repo.checkout("experiment")
repo.tag("v1.0", message="Production release")

# Compare versions
diff = repo.diff("v1.0", "HEAD")
print(f"Rows added: {diff.rows_added}")
```

### GPU Acceleration

High-performance generation for large datasets.

```python
from genesis.gpu import BatchedGenerator, MultiGPUGenerator

# Single GPU with batching
generator = BatchedGenerator(method="ctgan", device="cuda")
generator.fit(large_df)
synthetic = generator.generate(1_000_000)  # Auto-batched

# Multi-GPU
generator = MultiGPUGenerator(
    method="ctgan",
    devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
)
generator.fit(massive_df)
synthetic = generator.generate(10_000_000)
```

### Domain Generators

Pre-configured generators for healthcare, finance, and retail.

```python
# Healthcare
from genesis.domains import HealthcareGenerator

healthcare = HealthcareGenerator()
patients = healthcare.generate_patient_cohort(1000)
labs = healthcare.generate_lab_results(5000)

# Finance
from genesis.domains import FinanceGenerator

finance = FinanceGenerator()
transactions = finance.generate_transactions(
    n_transactions=10000,
    include_fraud=True,
    fraud_rate=0.02
)

# Retail
from genesis.domains import RetailGenerator

retail = RetailGenerator()
ecommerce = retail.generate_ecommerce_dataset(
    n_customers=5000,
    n_products=500,
    n_orders=20000
)
```

### Pipeline Builder

Visual workflow construction for complex generation pipelines.

```python
from genesis.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .source("raw_data.csv")
    .transform("clean", {"drop_na": True})
    .transform("filter", {"condition": "age >= 18"})
    .synthesize(method="ctgan", n_samples=10000)
    .evaluate()
    .sink("synthetic_output.csv")
    .build()
)

result = pipeline.execute()
print(f"Quality: {result['evaluate']['overall_score']:.1%}")

# Save/load YAML
pipeline.save("my_pipeline.yaml")
loaded = Pipeline.load("my_pipeline.yaml")
```

---

## CLI Commands

### v1.3.0 Commands

```bash
# Interactive chat for NL generation
genesis chat -d data.csv -o synthetic.csv

# Launch quality dashboard
genesis dashboard -r real.csv -s synthetic.csv

# Discover database schema
genesis discover -c "postgresql://..." -o ./synthetic_db
```

### v1.4.0 Commands

```bash
# AutoML synthesis
genesis automl -i data.csv -o synthetic.csv -n 1000
genesis automl -i data.csv --recommend-only  # Just show recommendation

# Data augmentation
genesis augment -i imbalanced.csv -o balanced.csv -t label
genesis augment -i data.csv -t fraud --analyze-only

# Privacy audit
genesis privacy-audit -r original.csv -s synthetic.csv --sensitive ssn,income
genesis privacy-audit -r data.csv -s synthetic.csv -o report.html

# Drift detection
genesis drift -b baseline.csv -c current.csv
genesis drift -b baseline.csv -c current.csv --generate -n 10000

# Dataset versioning
genesis version init ./data_repo
genesis version commit -r ./data_repo -d synthetic.csv -m "Initial commit"
genesis version log -r ./data_repo
genesis version tag -r ./data_repo v1.0 -m "Release"

# Domain generation
genesis domain healthcare -t patient_cohort -n 1000 -o patients.csv
genesis domain finance -t transactions -n 10000 --fraud-rate 0.02 -o txns.csv
genesis domain retail -t orders -n 5000 -o orders.csv

# Pipeline execution
genesis pipeline -c pipeline.yaml
genesis pipeline -c pipeline.yaml --validate-only
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Genesis v1.4.0                           │
├─────────────────────────────────────────────────────────────────┤
│  CLI Layer                                                      │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ automl  │ │ augment │ │ drift   │ │ domain  │ │pipeline │   │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │
├───────┼──────────┼──────────┼──────────┼──────────┼─────────────┤
│  Core Modules                                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │   AutoML    │ │Augmentation │ │  Privacy    │               │
│  │ Synthesizer │ │  Planner    │ │  Attacks    │               │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘               │
│         │               │               │                       │
│  ┌──────┴───────────────┴───────────────┴──────┐               │
│  │              SyntheticGenerator              │               │
│  │    (CTGAN, TVAE, GaussianCopula, etc.)      │               │
│  └──────────────────────┬──────────────────────┘               │
│                         │                                       │
│  ┌──────────────────────┴──────────────────────┐               │
│  │           Supporting Services                │               │
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌───────┐ │               │
│  │  │ Drift  │ │Version │ │Pipeline│ │ GPU   │ │               │
│  │  │Detector│ │  Repo  │ │Executor│ │Accel. │ │               │
│  │  └────────┘ └────────┘ └────────┘ └───────┘ │               │
│  └─────────────────────────────────────────────┘               │
├─────────────────────────────────────────────────────────────────┤
│  Domain Generators                                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                  │
│  │ Healthcare │ │  Finance   │ │   Retail   │                  │
│  └────────────┘ └────────────┘ └────────────┘                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Migration Guide

### From v1.2.0 to v1.3.0

No breaking changes. New features are additive.

```python
# New imports available
from genesis.generators.conditional import ConditionBuilder
from genesis.streaming import KafkaStreamingGenerator
from genesis.lineage import LineageChain
from genesis.federated import SecureAggregator
```

### From v1.3.0 to v1.4.0

No breaking changes. New features are additive.

```python
# New imports available
from genesis import auto_synthesize, augment_imbalanced, run_privacy_audit, detect_drift
from genesis.automl import AutoMLSynthesizer
from genesis.augmentation import SyntheticAugmenter
from genesis.privacy_attacks import PrivacyAttackTester
from genesis.drift import DriftAwareGenerator
from genesis.versioning import DatasetRepository
from genesis.gpu import BatchedGenerator
from genesis.domains import HealthcareGenerator, FinanceGenerator, RetailGenerator
from genesis.pipeline import PipelineBuilder
```

### New Dependencies

```bash
# For streaming features
pip install genesis-synth[streaming]  # kafka-python, websockets

# For PDF reports
pip install genesis-synth[reporting]  # weasyprint

# Full installation
pip install genesis-synth[all]
```

---

## Performance Benchmarks

| Feature | Dataset Size | Time | Speedup |
|---------|-------------|------|---------|
| AutoML (method selection) | 100K rows | 2s | N/A |
| Augmentation (10x minority) | 50K rows | 15s | N/A |
| Privacy Audit (full) | 10K rows | 8s | N/A |
| Drift Detection | 100K rows | 3s | N/A |
| GPU Generation | 1M rows | 3min | 25x vs CPU |
| Multi-GPU (4x) | 10M rows | 12min | 4x vs single GPU |

---

## Support

- **Documentation**: [genesis-synth.github.io/genesis](https://genesis-synth.github.io/genesis)
- **Issues**: [github.com/genesis-synth/genesis/issues](https://github.com/genesis-synth/genesis/issues)
- **Discussions**: [github.com/genesis-synth/genesis/discussions](https://github.com/genesis-synth/genesis/discussions)
