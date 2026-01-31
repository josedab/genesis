---
sidebar_position: 104
title: Changelog
---

# Changelog

All notable changes to Genesis are documented here.

## [1.4.0] - 2024-XX-XX

### Added

#### AutoML Synthesis
- `auto_synthesize()` function for one-shot generation with optimal settings
- `AutoMLSynthesizer` class with automatic method selection
- Mode options: `fast`, `balanced`, `quality`
- Hyperparameter tuning support

#### Data Augmentation
- `augment_imbalanced()` function for balancing datasets
- Support for oversample, undersample, and hybrid strategies
- Quality-aware augmentation with thresholds
- Multi-class balancing support

#### Privacy Attack Testing
- `run_privacy_audit()` for comprehensive privacy evaluation
- Membership inference attack testing
- Attribute inference attack testing
- Singling out attack testing
- Linkage attack testing
- Privacy score calculation

#### LLM-Powered Inference
- `LLMInference` class for context-aware generation
- Support for OpenAI, Anthropic, and local LLMs (Ollama)
- Schema-guided text generation
- Data enhancement capabilities

#### Drift Detection
- `detect_drift()` function for statistical drift detection
- `DriftDetector` class with configurable thresholds
- Support for numeric (KS test) and categorical (JS divergence) columns
- Continuous monitoring with `DriftMonitor`

#### Dataset Versioning
- `DatasetRepository` for version control of datasets
- Content-addressable storage for deduplication
- Tagging and branching support
- Version comparison and rollback

#### GPU Acceleration
- `BatchedGenerator` for optimized large-scale generation
- Multi-GPU support with `DistributedGenerator`
- Mixed precision (FP16) training
- Memory optimization utilities

#### Domain Generators
- `NameGenerator` with 50+ locales
- `EmailGenerator` with name linking
- `PhoneGenerator` with format options
- `AddressGenerator` with structured output
- `DateGenerator` with age distributions
- `CompositeGenerator` for complete records

#### Pipeline Builder
- `Pipeline` class for workflow definition
- YAML configuration support
- Built-in steps for common operations
- Conditional and branching pipelines
- Error handling and retries

### Changed
- Improved CTGAN training performance (40% faster)
- Better memory efficiency for large datasets
- Enhanced constraint handling
- Updated evaluation metrics

### Fixed
- Fixed k-anonymity with single quasi-identifier
- Fixed scipy boolean array handling in statistical tests
- Fixed numpy bool assertion compatibility
- Improved categorical handling in evaluation

### Deprecated
- `categorical_columns` parameter (use `discrete_columns`)

---

## [1.3.0] - 2024-XX-XX

### Added

#### Conditional Generation
- `GuidedConditionalSampler` for condition-based generation
- Multiple sampling strategies: rejection, guided, importance
- Support for numeric comparisons and categorical conditions
- Scenario generation utilities

#### Streaming Generation
- `StreamingGenerator` for real-time data streams
- Kafka integration with `KafkaPublisher`
- WebSocket support with `WebSocketPublisher`
- Configurable batch sizes and intervals

#### Federated Learning
- `FederatedSynthesizer` for distributed training
- Multiple aggregation strategies: FedAvg, weighted, FedProx
- Privacy-preserving training with differential privacy
- Client-server architecture for cross-organization learning

#### Data Lineage Tracking
- `LineageChain` for immutable audit trails
- Blockchain-style event tracking with SHA-256 hashing
- `LineageTracker` for automatic provenance capture
- Export to JSON and HTML reports

#### Dashboard & Reporting
- `DashboardGenerator` for quality reports
- HTML and PDF export support
- Per-column distribution visualization
- Correlation comparison charts

#### Natural Language API
- `NaturalLanguageInterface` for query-based generation
- Natural language parsing to structured conditions
- Integration with main generator classes

### Changed
- Improved error messages
- Better handling of edge cases
- Performance optimizations

### Fixed
- Memory leaks in long-running generation
- Thread safety issues in parallel generation

---

## [1.2.0] - 2024-XX-XX

### Added
- Time series generation with `TimeSeriesGenerator`
- Text generation with `TextGenerator`
- Multi-table synthesis with `MultiTableGenerator`
- Basic privacy controls (differential privacy, k-anonymity)
- Quality evaluation with `QualityEvaluator`
- Constraint system for data validation

### Changed
- Improved CTGAN implementation
- Better handling of mixed data types
- Enhanced documentation

### Fixed
- Various bug fixes and improvements

---

## [1.1.0] - 2024-XX-XX

### Added
- TVAE generator method
- Gaussian Copula generator method
- CopulaGAN generator method
- Basic CLI interface
- Column type auto-detection

### Changed
- Refactored generator interface
- Improved API consistency

---

## [1.0.0] - 2024-XX-XX

### Added
- Initial release
- CTGAN-based synthetic data generation
- Basic API for fit/generate workflow
- Support for tabular data with mixed types
- Documentation and examples

---

## Upgrading

### From 1.3.x to 1.4.x

```python
# Old (still works but deprecated)
generator.fit(data, categorical_columns=['status'])

# New
generator.fit(data, discrete_columns=['status'])
```

### From 1.2.x to 1.3.x

```python
# Conditional generation is now available
from genesis import ConditionalGenerator

generator = ConditionalGenerator()
generator.fit(data)
synthetic = generator.generate(1000, conditions={'status': 'active'})
```

### From 1.1.x to 1.2.x

```python
# Privacy settings now use a dict
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'differential_privacy': {'epsilon': 1.0}
    }
)
```

---

## Versioning

Genesis follows [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality, backwards compatible
- **PATCH**: Bug fixes, backwards compatible

## Release Schedule

- **Major releases**: As needed for breaking changes
- **Minor releases**: Monthly with new features
- **Patch releases**: As needed for bug fixes

## Support Policy

| Version | Status | Support Until |
|---------|--------|---------------|
| 1.4.x | Current | Active |
| 1.3.x | Maintenance | 6 months after 1.5 |
| 1.2.x | End of Life | - |
| 1.1.x | End of Life | - |
| 1.0.x | End of Life | - |

## Links

- [GitHub Releases](https://github.com/genesis/genesis/releases)
- [PyPI](https://pypi.org/project/genesis-synth/)
- [Documentation](https://genesis.dev/docs)
