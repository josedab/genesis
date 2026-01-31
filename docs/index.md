# Genesis Documentation

Welcome to Genesis, the comprehensive synthetic data generation platform.

## Quick Start

```python
from genesis import SyntheticGenerator

generator = SyntheticGenerator(method='auto')
generator.fit(real_data, discrete_columns=['gender', 'city'])
synthetic_data = generator.generate(n_samples=10000)
```

## Contents

- [Installation](installation.md) - How to install Genesis
- [Quickstart](quickstart.md) - Get started in 5 minutes
- [User Guide](user_guide/) - Detailed usage guides
- [API Reference](api/reference.md) - Complete API documentation
- [Features Overview](FEATURES.md) - All v1.3.0 and v1.4.0 features

## Core Features

### Tabular Data Generation
- **CTGAN**: Deep learning (GAN) for complex distributions
- **TVAE**: Variational autoencoder for balanced performance
- **Gaussian Copula**: Fast statistical method

### Time Series Generation
- **Statistical**: ARIMA, seasonal decomposition
- **TimeGAN**: Deep learning for temporal patterns

### Text Generation
- **OpenAI**: GPT-4/3.5 via API
- **HuggingFace**: Local model inference

### Privacy Features
- Differential privacy (DP-SGD)
- K-anonymity
- L-diversity
- Rare category suppression

### Quality Evaluation
- Statistical fidelity metrics
- ML utility (TSTR/TRTS)
- Privacy risk assessment

## What's New

### v1.4.0 Highlights
- **🤖 AutoML Synthesis**: Automatic method selection based on data characteristics
- **⚖️ Synthetic Augmentation**: Balance imbalanced datasets intelligently
- **🛡️ Privacy Attack Testing**: Comprehensive privacy vulnerability assessment
- **📈 Drift Detection**: Monitor and adapt to distribution shifts
- **📦 Dataset Versioning**: Git-like version control for datasets
- **⚡ GPU Acceleration**: Batched and multi-GPU generation
- **🏥 Domain Generators**: Healthcare, finance, and retail-specific generators
- **🔧 Pipeline Builder**: Visual workflow construction

[Full v1.4.0 Feature Guide →](api/v140_features.md)

### v1.3.0 Highlights
- **🎯 Guided Conditional Generation**: Smart sampling with ConditionBuilder
- **💬 Natural Language Interface**: Chat-based data generation
- **📊 Interactive Dashboard**: Plotly-powered quality visualization
- **🌊 Streaming Generation**: Real-time via Kafka and WebSocket
- **🔐 Federated Learning**: Privacy-preserving distributed training
- **🔗 Data Lineage Tracking**: Blockchain-style audit trails

[Full v1.3.0 Feature Guide →](api/v130_features.md)

### v1.2.0 Highlights
- Plugin system for custom generators
- Auto-tuning hyperparameters with Optuna
- Privacy certificates (GDPR, HIPAA, CCPA)
- Distributed training with Ray/Dask
- Anomaly synthesis for ML training

[Full v1.2.0 Feature Guide →](api/v120_features.md)

## Support

- [GitHub Issues](https://github.com/genesis-synth/genesis/issues)
- [Discussion Forum](https://github.com/genesis-synth/genesis/discussions)
