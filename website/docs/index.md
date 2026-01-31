---
sidebar_position: 1
slug: /
title: Genesis Documentation
---

# Genesis Documentation

**Genesis** is a comprehensive synthetic data generation platform that creates realistic, privacy-safe data for ML training, testing, and development.

<div className="hero-badges">

[![PyPI version](https://badge.fury.io/py/genesis-synth.svg)](https://pypi.org/project/genesis-synth/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-291%20passing-brightgreen.svg)](https://github.com/genesis-synth/genesis)

</div>

## Quick Install

```bash
pip install genesis-synth
```

## 30-Second Example

```python
from genesis import SyntheticGenerator, auto_synthesize
import pandas as pd

# Load your data
df = pd.read_csv("customers.csv")

# Option 1: AutoML (recommended) - automatically selects best method
synthetic = auto_synthesize(df, n_samples=1000)

# Option 2: Manual control
generator = SyntheticGenerator(method="ctgan")
generator.fit(df)
synthetic = generator.generate(1000)

# Evaluate quality
from genesis import QualityEvaluator
evaluator = QualityEvaluator(df, synthetic)
report = evaluator.evaluate()
print(f"Quality Score: {report.overall_score:.1%}")
```

## What's New in v1.4.0

| Feature | Description |
|---------|-------------|
| 🤖 **AutoML Synthesis** | Automatic method selection based on your data |
| ⚖️ **Data Augmentation** | Balance imbalanced datasets intelligently |
| 🛡️ **Privacy Attack Testing** | Validate synthetic data doesn't leak information |
| 📈 **Drift Detection** | Monitor and adapt to distribution changes |
| 📦 **Dataset Versioning** | Git-like version control for datasets |
| 🏥 **Domain Generators** | Healthcare, Finance, Retail-specific generators |
| 🔧 **Pipeline Builder** | Visual workflow construction |

## Choose Your Path

<div className="row">
  <div className="col col--4">
    <div className="card">
      <div className="card__header">
        <h3>🚀 Quick Start</h3>
      </div>
      <div className="card__body">
        <p>Generate your first synthetic dataset in 5 minutes.</p>
      </div>
      <div className="card__footer">
        <a className="button button--primary button--block" href="/docs/getting-started/quickstart">Get Started</a>
      </div>
    </div>
  </div>
  <div className="col col--4">
    <div className="card">
      <div className="card__header">
        <h3>📖 Core Concepts</h3>
      </div>
      <div className="card__body">
        <p>Understand how synthetic data generation works.</p>
      </div>
      <div className="card__footer">
        <a className="button button--secondary button--block" href="/docs/concepts/overview">Learn Concepts</a>
      </div>
    </div>
  </div>
  <div className="col col--4">
    <div className="card">
      <div className="card__header">
        <h3>📚 API Reference</h3>
      </div>
      <div className="card__body">
        <p>Complete reference for all classes and functions.</p>
      </div>
      <div className="card__footer">
        <a className="button button--secondary button--block" href="/docs/api/reference">View API</a>
      </div>
    </div>
  </div>
</div>

## Supported Data Types

| Type | Generators | Use Cases |
|------|------------|-----------|
| **Tabular** | CTGAN, TVAE, GaussianCopula | Customer records, transactions, surveys |
| **Time Series** | TimeGAN, Statistical | Stock prices, sensor data, logs |
| **Text** | LLM-based | Reviews, descriptions, comments |
| **Multi-Table** | MultiTableGenerator | Relational databases with foreign keys |
| **Domain-Specific** | Healthcare, Finance, Retail | Industry-specific realistic data |

## Why Genesis?

- **🔒 Privacy First**: Built-in differential privacy, k-anonymity, and privacy attack testing
- **🤖 AutoML**: Automatically selects the best generation method for your data
- **📊 Quality Metrics**: Comprehensive evaluation with statistical tests and ML efficacy
- **🔌 Extensible**: Plugin architecture for custom generators and constraints
- **🚀 Production Ready**: REST API, CLI, distributed training, and monitoring
- **📝 Compliant**: Generate GDPR, HIPAA, and CCPA compliant synthetic data

## Community

- **GitHub**: [github.com/genesis-synth/genesis](https://github.com/genesis-synth/genesis)
- **Discussions**: [Ask questions and share ideas](https://github.com/genesis-synth/genesis/discussions)
- **Issues**: [Report bugs and request features](https://github.com/genesis-synth/genesis/issues)

## License

Genesis is open source software licensed under the [MIT License](https://github.com/genesis-synth/genesis/blob/main/LICENSE).
