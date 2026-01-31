# Genesis Examples

This page provides an index of example notebooks.

## Interactive Notebooks

All notebooks can be run directly in Google Colab or locally.

### Getting Started

- **[01 - Quickstart](https://github.com/genesis-synth/genesis/blob/main/examples/01_quickstart.ipynb)** - Basic usage in 5 minutes

### Tabular Data

- **[02 - Tabular Synthesis](https://github.com/genesis-synth/genesis/blob/main/examples/02_tabular_synthesis.ipynb)** - Compare CTGAN, TVAE, Gaussian Copula

### Time Series

- **[03 - Time Series](https://github.com/genesis-synth/genesis/blob/main/examples/03_time_series.ipynb)** - Statistical and TimeGAN methods

### Text Generation

- **[04 - Text Generation](https://github.com/genesis-synth/genesis/blob/main/examples/04_text_generation.ipynb)** - LLM-based synthesis

### Privacy

- **[05 - Privacy Configuration](https://github.com/genesis-synth/genesis/blob/main/examples/05_privacy_config.ipynb)** - Differential privacy, k-anonymity

### Domain Examples

- **[06 - Healthcare](https://github.com/genesis-synth/genesis/blob/main/examples/06_healthcare_example.ipynb)** - Patient records with privacy
- **[07 - Finance](https://github.com/genesis-synth/genesis/blob/main/examples/07_finance_example.ipynb)** - Transaction data for fraud detection
- **[08 - Multi-table](https://github.com/genesis-synth/genesis/blob/main/examples/08_multitable_example.ipynb)** - Relational data synthesis

## v1.2.0 Feature Examples

### Plugin System
\`\`\`python
from genesis.plugins import register_generator, list_generators

# List built-in generators
print(list_generators())

# Register custom generator
@register_generator("my_gen", description="My generator")
class MyGenerator(BaseGenerator):
    pass
\`\`\`

### Auto-Tuning
\`\`\`python
from genesis.tuning import AutoTuner

tuner = AutoTuner(method='ctgan')
result = tuner.tune(data, n_trials=20)
print(f"Best config: {result.best_config}")
\`\`\`

### Privacy Certificates
\`\`\`python
from genesis.compliance import PrivacyCertificate, ComplianceFramework

cert = PrivacyCertificate(real_data, synthetic_data)
report = cert.generate(framework=ComplianceFramework.GDPR)
report.to_html("gdpr_certificate.html")
\`\`\`

### Drift Detection
\`\`\`python
from genesis.monitoring import DriftDetector

detector = DriftDetector(reference_data=baseline)
report = detector.check(new_data)
print(report.drifted_columns)
\`\`\`

### Anomaly Synthesis
\`\`\`python
from genesis.anomaly import AnomalyGenerator, BalancedDatasetGenerator

# Generate anomalies
gen = AnomalyGenerator(normal_data)
anomalies = gen.generate(100)

# Balance imbalanced dataset
balancer = BalancedDatasetGenerator(data, label_column='fraud')
balanced = balancer.generate(target_ratio=0.3)
\`\`\`

### Distributed Training
\`\`\`python
from genesis.distributed import DistributedTrainer, DistributedConfig

trainer = DistributedTrainer(
    config=DistributedConfig(backend='ray', n_workers=4)
)
trainer.fit(data, method='ctgan')
synthetic = trainer.generate(10000)
\`\`\`

## Running Locally

\`\`\`bash
pip install genesis-synth[all]
jupyter lab examples/
\`\`\`
