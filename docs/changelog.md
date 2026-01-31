# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Image synthesis support
- Streaming generation for large datasets
- Distributed training support
- Web UI dashboard
