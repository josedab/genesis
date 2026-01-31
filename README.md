# Genesis: Synthetic Data Generation Platform

[![PyPI version](https://badge.fury.io/py/genesis-synth.svg)](https://badge.fury.io/py/genesis-synth)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![CI](https://github.com/genesis-synth/genesis/actions/workflows/ci.yml/badge.svg)](https://github.com/genesis-synth/genesis/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/genesis-synth/genesis/branch/main/graph/badge.svg)](https://codecov.io/gh/genesis-synth/genesis)
[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://genesis-synth.github.io/genesis)

**Genesis** is a comprehensive synthetic data generation platform that creates realistic, privacy-safe data for ML training, testing, and development.

## 🌟 Features

### Core Capabilities
- **Multiple Data Types**: Tabular, time series, and text data generation
- **Privacy Guarantees**: Differential privacy, k-anonymity metrics
- **Statistical Fidelity**: Preserves correlations and distributions
- **ML Utility Focus**: Generated data works for model training
- **Easy to Use**: Pandas in, pandas out

### v1.2.0 Features (New!)
- **🔌 Plugin System**: Extend Genesis with custom generators, transformers, and evaluators
- **⚡ Auto-Tuning**: Automatic hyperparameter optimization with Optuna
- **📜 Privacy Certificates**: GDPR, HIPAA, CCPA compliance reports
- **📊 Drift Detection**: Monitor data quality drift over time
- **🔍 Synthetic Debugger**: Diagnose quality issues with actionable suggestions
- **🎯 Anomaly Synthesis**: Generate realistic anomalies for ML training
- **🚀 Distributed Training**: Scale across GPUs with Ray or Dask
- **🔗 Cross-Modal Generation**: Generate paired tabular + text data
- **📝 Schema Editor**: Visual schema editing backend
- **🏪 Marketplace**: Synthetic data marketplace backend

### v1.3.0 Features
- **🎯 Guided Conditional Generation**: Smart sampling with ConditionBuilder for rare scenarios
- **💬 Natural Language Interface**: Chat-based data generation with LLM integration
- **📊 Interactive Dashboard**: Plotly-powered quality visualization with PDF export
- **🌊 Streaming Generation**: Real-time synthetic data via Kafka and WebSocket
- **🔐 Federated Learning**: Privacy-preserving distributed training across data silos
- **🔗 Data Lineage Tracking**: Blockchain-style immutable audit trails

### v1.4.0 Features (Latest!)
- **🤖 AutoML Synthesis**: Automatic method selection based on data characteristics
- **⚖️ Synthetic Augmentation**: Balance imbalanced datasets with intelligent augmentation
- **🛡️ Privacy Attack Testing**: Membership inference, attribute inference, re-identification attacks
- **🧠 LLM Schema Inference**: Auto-detect column semantics from names and samples
- **📈 Drift-Aware Generation**: Detect and adapt to distribution shifts
- **📦 Dataset Versioning**: Git-like versioning with branches, tags, and diffs
- **⚡ GPU Acceleration**: Batched and multi-GPU generation for large datasets
- **🏥 Domain Generators**: Healthcare, finance, and retail-specific generators
- **🔧 Pipeline Builder**: Visual workflow builder for complex generation pipelines

## 📦 Installation

```bash
# Basic installation
pip install genesis-synth

# With PyTorch backend (recommended)
pip install genesis-synth[pytorch]

# With TensorFlow backend
pip install genesis-synth[tensorflow]

# With text generation support
pip install genesis-synth[llm]

# Full installation with all features
pip install genesis-synth[all]
```

## 🚀 Quick Start

```python
from genesis import SyntheticGenerator, PrivacyConfig

# Simple generation with smart defaults
generator = SyntheticGenerator(
    method='auto',  # Automatically selects best method
    privacy=PrivacyConfig(
        enable_differential_privacy=True,
        epsilon=1.0
    )
)

# Fit to real data
generator.fit(real_data)

# Generate synthetic data
synthetic_data = generator.generate(n_samples=10000)

# Get comprehensive quality report
report = generator.quality_report()
print(report.summary())
# Statistical Fidelity: 94.2%
# ML Utility: 97.1%
# Privacy Score: 99.8%
```

## 📊 Supported Generators

### Tabular Data
- **CTGAN**: Conditional Tabular GAN for mixed-type data
- **TVAE**: Variational Autoencoder for tabular data
- **Gaussian Copula**: Statistical method for preserving correlations

### Time Series
- **TimeGAN**: Generative Adversarial Network for time series
- **Statistical**: ARIMA-based methods with trend/seasonality preservation

### Text Data
- **LLM-based**: OpenAI and HuggingFace transformer backends
- **Privacy-aware**: Prevents PII leakage in generated text

## 🔒 Privacy Features

```python
from genesis import PrivacyConfig

privacy = PrivacyConfig(
    enable_differential_privacy=True,
    epsilon=1.0,              # Privacy budget
    delta=1e-5,
    k_anonymity=5,            # Minimum group size
    suppress_rare_categories=True,
    rare_threshold=0.01
)

generator = SyntheticGenerator(privacy=privacy)
```

## 📏 Quality Evaluation

```python
from genesis import QualityEvaluator

evaluator = QualityEvaluator(
    real_data=real_df,
    synthetic_data=synthetic_df
)

report = evaluator.evaluate()
print(report.statistical_fidelity)  # Distribution matching scores
print(report.ml_utility)            # Model performance comparison
print(report.privacy_metrics)       # Re-identification risk
```

## 🔧 Constraints

```python
from genesis import Constraint

generator.fit(
    data=real_df,
    discrete_columns=['gender', 'city'],
    constraints=[
        Constraint.positive('age'),
        Constraint.range('age', 0, 120),
        Constraint.unique('customer_id')
    ]
)
```

## 💻 CLI

```bash
# Generate synthetic data
genesis generate --input data.csv --output synthetic.csv --method ctgan --samples 10000

# Evaluate quality
genesis evaluate --real data.csv --synthetic synthetic.csv --output report.html

# Analyze dataset
genesis analyze --input data.csv --output analysis.json

# AutoML - auto-select best method
genesis automl -i data.csv -o synthetic.csv -n 1000

# Augment imbalanced data
genesis augment -i imbalanced.csv -o balanced.csv -t label

# Privacy audit
genesis privacy-audit -r original.csv -s synthetic.csv --sensitive ssn,income

# Detect drift
genesis drift -b baseline.csv -c current.csv

# Domain-specific generation
genesis domain healthcare -t patient_cohort -n 1000 -o patients.csv
genesis domain finance -t transactions -n 10000 -o transactions.csv
```

## 🆕 v1.2.0 Features

### Plugin System
```python
from genesis.plugins import register_generator, get_generator

@register_generator("my_generator", description="Custom generator")
class MyGenerator(BaseGenerator):
    def _fit_impl(self, data, discrete_columns, progress_callback):
        pass
    def _generate_impl(self, n_samples, conditions, progress_callback):
        return data.sample(n_samples, replace=True)

# Use it
gen = get_generator("my_generator")()
```

### Auto-Tuning
```python
from genesis.tuning import AutoTuner, TuningPreset

tuner = AutoTuner.from_preset(TuningPreset.BALANCED)
result = tuner.tune(data, method='ctgan')
print(f"Best score: {result.best_score}")

# Use optimized config
gen = SyntheticGenerator(config=result.best_config)
```

### Privacy Certificates
```python
from genesis.compliance import PrivacyCertificate, ComplianceFramework

cert = PrivacyCertificate(real_data, synthetic_data)
report = cert.generate(framework=ComplianceFramework.GDPR)

print(f"Compliant: {report.is_compliant}")
report.to_html("privacy_certificate.html")
```

### Drift Detection
```python
from genesis.monitoring import DriftDetector

detector = DriftDetector(reference_data=baseline)
report = detector.check(current_data=new_data)

if report.has_drift:
    print(f"Drift detected in: {report.drifted_columns}")
```

### Anomaly Synthesis
```python
from genesis.anomaly import AnomalyGenerator, AnomalyType

gen = AnomalyGenerator(normal_data)
anomalies = gen.generate(100, anomaly_type=AnomalyType.STATISTICAL)
```

### Distributed Training
```python
from genesis.distributed import DistributedTrainer, DistributedConfig

trainer = DistributedTrainer(
    config=DistributedConfig(backend='ray', n_workers=4, use_gpu=True)
)
trainer.fit(data, method='ctgan')
```

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [Quick Start](docs/quickstart.md)
- [User Guide](docs/user_guide/)
- [API Reference](docs/api/reference.md)
- [v1.2.0 Features](docs/api/v120_features.md)
- [v1.3.0 Features](docs/api/v130_features.md)
- [v1.4.0 Features](docs/api/v140_features.md)
- [Architecture Overview](docs/architecture/overview.md)
- [Architecture Decision Records](docs/adr/README.md)
- [Examples](examples/)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Genesis builds upon foundational work in synthetic data generation:
- [CTGAN](https://arxiv.org/abs/1907.00503) - Modeling Tabular data using Conditional GAN
- [TVAE](https://arxiv.org/abs/1907.00503) - Variational Autoencoder for Tabular Data
- [TimeGAN](https://arxiv.org/abs/1906.02691) - Time-series Generative Adversarial Networks
