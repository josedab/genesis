---
sidebar_position: 102
title: FAQ
---

# Frequently Asked Questions

Quick answers to common questions about Genesis.

## General

### What is synthetic data?

Synthetic data is artificially generated data that mimics the statistical properties of real data without containing any actual records from the original dataset. It looks and behaves like real data for analytics and ML, but no individual from the original dataset can be identified.

### Why use synthetic data instead of real data?

**Privacy & Compliance**: Share data without exposing PII or violating GDPR/HIPAA/CCPA.

**Data Access**: Give developers and analysts access to realistic data without security reviews.

**Data Augmentation**: Generate more training samples to improve ML model performance.

**Testing**: Create edge cases and stress test scenarios that don't exist in production data.

**Cost**: Avoid expensive data collection or licensing fees.

### Is Genesis open source?

Yes! Genesis is licensed under the MIT License. You can use it freely for commercial and personal projects, modify it, and distribute it.

### What Python versions are supported?

Genesis supports Python 3.8, 3.9, 3.10, 3.11, and 3.12.

---

## Privacy & Security

### Does Genesis guarantee privacy?

Genesis provides strong privacy controls including:

- **Differential Privacy**: Mathematical guarantee that individual records can't be identified (configurable epsilon)
- **K-Anonymity**: Ensure combinations of quasi-identifiers appear at least K times
- **Privacy Attack Testing**: Validate against membership inference and attribute inference attacks

However, no synthetic data generation is 100% risk-free. Always run privacy audits before releasing sensitive data.

### What privacy certifications does Genesis support?

Genesis can generate compliance reports for:
- **GDPR** (EU General Data Protection Regulation)
- **HIPAA** (US Health Insurance Portability and Accountability Act)
- **CCPA** (California Consumer Privacy Act)

Use `PrivacyCertificate` to generate audit-ready compliance documentation.

### Can synthetic data be re-identified?

With proper privacy settings, re-identification risk is extremely low. Genesis measures:
- **Distance to Closest Record (DCR)**: How similar synthetic records are to real records
- **Membership Inference Risk**: Can an attacker determine if someone was in the training data?
- **Attribute Disclosure Risk**: Can sensitive attributes be inferred?

Run `run_privacy_audit()` to quantify these risks before sharing data.

### Should I use differential privacy?

Use differential privacy when:
- Data contains highly sensitive information (health, financial, personal)
- You need mathematical privacy guarantees
- Compliance requires formal privacy proofs

Skip differential privacy when:
- Speed is critical and data isn't highly sensitive
- You need maximum statistical fidelity
- Data is already aggregated or anonymized

```python
# Recommended for sensitive data
generator = SyntheticGenerator(
    privacy={'differential_privacy': {'epsilon': 1.0}}
)
```

---

## Quality & Accuracy

### How accurate is synthetic data?

Genesis typically achieves:
- **Statistical Fidelity**: 90-98% similarity to original distributions
- **ML Utility**: 85-95% of model performance compared to training on real data
- **Correlation Preservation**: 90%+ preservation of feature relationships

Quality depends on:
- Size and complexity of original data
- Generation method used
- Training duration
- Privacy settings (stricter privacy = lower fidelity)

### Which generator should I use?

| Your Data | Recommended | Why |
|-----------|-------------|-----|
| General tabular | `auto` or `ctgan` | Best overall quality |
| Small dataset (under 1000 rows) | `gaussian_copula` | Fast, works well with limited data |
| Complex distributions | `tvae` | Better at capturing multi-modal distributions |
| Time series | `TimeSeriesGenerator` | Preserves temporal patterns |
| Multiple related tables | `MultiTableGenerator` | Maintains referential integrity |

Or just use AutoML:
```python
synthetic = auto_synthesize(data, n_samples=1000)  # Genesis picks the best method
```

### Why is my synthetic data quality low?

Common causes and fixes:

1. **Discrete columns not specified**
   ```python
   # ❌ Wrong
   generator.fit(data)
   
   # ✅ Correct
   generator.fit(data, discrete_columns=['category', 'status', 'region'])
   ```

2. **Not enough training data** - Need 500+ rows for good results

3. **Not enough epochs**
   ```python
   generator = SyntheticGenerator(config={'epochs': 500})  # Increase from default 300
   ```

4. **Rare categories** - Categories appearing fewer than 10 times are hard to learn

5. **Wrong method** - Try `tvae` instead of `ctgan` or vice versa

### Does synthetic data work for ML training?

Yes! Genesis is designed for ML utility. Key metrics:
- **Train-Synthetic-Test-Real (TSTR)**: Models trained on synthetic data, tested on real data
- **Feature Importance Correlation**: Do the same features matter?
- **Prediction Correlation**: Do models make similar predictions?

Typical ML utility is 85-95% of training on real data.

---

## Performance

### How long does training take?

| Data Size | Method | Time (CPU) | Time (GPU) |
|-----------|--------|------------|------------|
| 1K rows | CTGAN | ~1 min | ~20 sec |
| 10K rows | CTGAN | ~5 min | ~1 min |
| 100K rows | CTGAN | ~30 min | ~5 min |
| 1M rows | CTGAN | ~3 hours | ~30 min |

Speed tips:
- Use GPU with `config={'device': 'cuda'}`
- Use `gaussian_copula` for fast iteration
- Sample training data for initial experiments
- Use distributed training for massive datasets

### How much data can Genesis handle?

Genesis has been tested with:
- **Rows**: Up to 10M+ with distributed training
- **Columns**: Up to 500+
- **Categories**: Up to 10K+ unique values per column

For very large datasets, use:
```python
from genesis.distributed import DistributedTrainer

trainer = DistributedTrainer(
    config={'backend': 'ray', 'n_workers': 8}
)
```

### Does Genesis support GPU acceleration?

Yes! Install with PyTorch GPU support:
```bash
pip install genesis-synth[pytorch]
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Then enable GPU:
```python
generator = SyntheticGenerator(config={'device': 'cuda'})
```

---

## Data Types

### What data types does Genesis support?

| Type | Supported | Generator |
|------|-----------|-----------|
| Numeric (int, float) | ✅ | All generators |
| Categorical (string, enum) | ✅ | All generators |
| Boolean | ✅ | All generators |
| Datetime | ✅ | All generators |
| Time series | ✅ | `TimeSeriesGenerator` |
| Text | ✅ | `TextGenerator` |
| Multi-table (relational) | ✅ | `MultiTableGenerator` |
| Images | ⚠️ Experimental | Diffusion-based |
| JSON/nested | ⚠️ Limited | Flatten first |

### Can Genesis generate time series data?

Yes! Use `TimeSeriesGenerator`:
```python
from genesis import TimeSeriesGenerator

generator = TimeSeriesGenerator()
generator.fit(df, sequence_length=50)
sequences = generator.generate(n_sequences=100)
```

Supports: stock prices, sensor data, logs, any temporal data.

### Can Genesis handle relational databases?

Yes! Use `MultiTableGenerator` for multiple related tables:
```python
from genesis import MultiTableGenerator

generator = MultiTableGenerator()
generator.fit(
    tables={'customers': df1, 'orders': df2},
    relationships=[('orders', 'customer_id', 'customers', 'id')]
)
synthetic_db = generator.generate(scale=2.0)
```

Foreign key relationships are maintained automatically.

---

## Integration

### Does Genesis have a CLI?

Yes! Common commands:
```bash
# Generate synthetic data
genesis generate -i data.csv -o synthetic.csv -n 1000

# Evaluate quality
genesis evaluate -r original.csv -s synthetic.csv

# AutoML generation
genesis automl -i data.csv -o synthetic.csv

# Privacy audit
genesis privacy-audit -r original.csv -s synthetic.csv
```

### Does Genesis have a REST API?

Yes! Start the API server:
```bash
pip install genesis-synth[api]
genesis api start --port 8000
```

Then use HTTP endpoints:
```bash
curl -X POST http://localhost:8000/generate \
  -F "file=@data.csv" \
  -F "n_samples=1000"
```

### Can I use Genesis in a pipeline?

Yes! Use the Pipeline API:
```python
from genesis.pipeline import Pipeline, steps

pipeline = Pipeline([
    steps.load_csv('data.csv'),
    steps.fit_generator('ctgan'),
    steps.generate(1000),
    steps.evaluate(),
    steps.save_csv('output.csv')
])

result = pipeline.run()
```

Or define pipelines in YAML and run with `genesis pipeline run pipeline.yaml`.

---

## Troubleshooting

### "CUDA out of memory" error

```python
# Solution 1: Reduce batch size
generator = SyntheticGenerator(config={'batch_size': 256})

# Solution 2: Use CPU
generator = SyntheticGenerator(config={'device': 'cpu'})
```

### "Generator not fitted" error

You must call `fit()` before `generate()`:
```python
generator = SyntheticGenerator()
generator.fit(data)  # Don't forget this!
synthetic = generator.generate(1000)
```

### Generated data has NaN values

Genesis preserves the null pattern from training data. To avoid NaN:
```python
# Option 1: Remove NaN before training
clean_data = data.dropna()
generator.fit(clean_data)

# Option 2: Fill NaN
data_filled = data.fillna(data.median())
generator.fit(data_filled)
```

### More help?

- [Full Troubleshooting Guide](/docs/troubleshooting)
- [GitHub Issues](https://github.com/genesis-synth/genesis/issues)
- [GitHub Discussions](https://github.com/genesis-synth/genesis/discussions)
