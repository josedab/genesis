---
sidebar_position: 106
title: Benchmarks
---

# Benchmarks

Performance and quality benchmarks for Genesis across common datasets and use cases.

## Overview

We benchmark Genesis against:
- **Quality**: How well synthetic data preserves statistical properties
- **Privacy**: How well synthetic data protects individual records
- **Performance**: Training time and generation throughput
- **ML Utility**: How well models trained on synthetic data perform

All benchmarks are reproducible. See [Running Benchmarks](#running-benchmarks) below.

---

## Quality Benchmarks

### Dataset Summary

| Dataset | Rows | Columns | Numeric | Categorical | Description |
|---------|------|---------|---------|-------------|-------------|
| Adult Census | 48,842 | 14 | 6 | 8 | Income prediction |
| Credit Card Fraud | 284,807 | 31 | 30 | 1 | Fraud detection |
| Covertype | 581,012 | 54 | 54 | 0 | Forest cover type |
| Bank Marketing | 45,211 | 17 | 7 | 10 | Marketing campaign |
| Online Retail | 541,909 | 8 | 4 | 4 | E-commerce transactions |

### Quality Scores by Method

Higher is better (0-1 scale).

| Dataset | CTGAN | TVAE | Gaussian Copula | AutoML |
|---------|-------|------|-----------------|--------|
| Adult Census | 0.92 | 0.91 | 0.87 | **0.94** |
| Credit Card Fraud | 0.89 | **0.91** | 0.82 | 0.90 |
| Covertype | 0.85 | 0.86 | 0.79 | **0.88** |
| Bank Marketing | 0.91 | 0.90 | 0.88 | **0.93** |
| Online Retail | 0.88 | 0.87 | 0.84 | **0.90** |

**Key findings:**
- AutoML consistently achieves the best quality by selecting optimal methods
- TVAE performs best on high-dimensional numeric data (Credit Card Fraud)
- Gaussian Copula is fastest but lower quality for complex distributions

### Quality Score Components

For Adult Census dataset with CTGAN:

| Metric | Score | Description |
|--------|-------|-------------|
| **Overall Quality** | 0.92 | Weighted average of all metrics |
| Column Shapes | 0.94 | Distribution similarity per column |
| Column Pair Trends | 0.89 | Correlation preservation |
| Statistical Tests | 0.91 | KS test, chi-squared p-values |
| ML Utility (TSTR) | 0.93 | Train-Synthetic-Test-Real accuracy |

---

## Privacy Benchmarks

### Privacy Scores

Higher is better. Measured with default privacy settings (no differential privacy).

| Dataset | DCR Score | Membership Risk | Attribute Risk | Overall |
|---------|-----------|-----------------|----------------|---------|
| Adult Census | 0.97 | 0.02 | 0.04 | **0.96** |
| Credit Card Fraud | 0.98 | 0.01 | 0.03 | **0.97** |
| Covertype | 0.96 | 0.03 | 0.05 | **0.95** |
| Bank Marketing | 0.97 | 0.02 | 0.04 | **0.96** |

**Metrics explained:**
- **DCR (Distance to Closest Record)**: How far synthetic records are from real records
- **Membership Risk**: Can an attacker determine if someone was in training data?
- **Attribute Risk**: Can sensitive attributes be inferred from synthetic data?

### Privacy vs Quality Trade-off

Adult Census with differential privacy at various epsilon values:

| Epsilon (ε) | Privacy Score | Quality Score | Notes |
|-------------|---------------|---------------|-------|
| No DP | 0.96 | 0.92 | Default |
| 10.0 | 0.97 | 0.91 | Minimal privacy boost |
| 5.0 | 0.98 | 0.89 | Good balance |
| 1.0 | 0.99 | 0.84 | Strong privacy |
| 0.5 | 0.99 | 0.78 | Very strong privacy |
| 0.1 | 1.00 | 0.65 | Maximum privacy, lower utility |

**Recommendation:** Use ε=1.0 to ε=5.0 for sensitive data requiring formal privacy guarantees.

### Privacy Attack Resistance

Results of simulated privacy attacks on Adult Census synthetic data:

| Attack Type | Attack Accuracy | Random Baseline | Risk Level |
|-------------|-----------------|-----------------|------------|
| Membership Inference | 52.1% | 50.0% | ✅ Low |
| Attribute Inference (income) | 54.3% | 50.0% | ✅ Low |
| Singling Out | 0.02% | 0.00% | ✅ Low |
| Linkage Attack | 51.8% | 50.0% | ✅ Low |

Attack accuracy near 50% (random guessing) indicates strong privacy protection.

---

## Performance Benchmarks

### Training Time

Time to train generators (seconds). Measured on NVIDIA RTX 4090 with 300 epochs.

| Dataset | CTGAN | TVAE | Gaussian Copula | AutoML |
|---------|-------|------|-----------------|--------|
| Adult (49K) | 45s | 38s | **3s** | 52s |
| Credit (285K) | 180s | 145s | **12s** | 195s |
| Covertype (581K) | 420s | 350s | **28s** | 450s |
| Bank (45K) | 42s | 35s | **3s** | 48s |
| Retail (542K) | 380s | 310s | **25s** | 410s |

### Generation Throughput

Samples generated per second after training:

| Method | Throughput (samples/sec) |
|--------|--------------------------|
| CTGAN | 15,000 |
| TVAE | 18,000 |
| Gaussian Copula | **125,000** |
| BatchedGenerator (GPU) | 50,000 |

### CPU vs GPU Performance

Training time comparison for Adult Census (300 epochs):

| Hardware | CTGAN | TVAE | Speedup |
|----------|-------|------|---------|
| CPU (Intel i9-13900K) | 180s | 150s | 1x |
| GPU (RTX 3080) | 52s | 45s | 3.5x |
| GPU (RTX 4090) | 45s | 38s | 4x |
| GPU (A100) | 28s | 24s | 6x |

### Memory Usage

Peak memory during training:

| Dataset | CPU Memory | GPU Memory |
|---------|------------|------------|
| Adult (49K) | 1.2 GB | 2.1 GB |
| Credit (285K) | 3.8 GB | 4.5 GB |
| Covertype (581K) | 6.2 GB | 7.8 GB |

**Tip:** For datasets over 500K rows, use `BatchedGenerator` or distributed training.

---

## ML Utility Benchmarks

### Train-Synthetic-Test-Real (TSTR)

Models trained on synthetic data, tested on held-out real data:

| Dataset | Task | Real→Real | Synth→Real | Utility Ratio |
|---------|------|-----------|------------|---------------|
| Adult | Classification | 85.2% | 81.4% | **95.5%** |
| Credit Fraud | Classification | 99.6% | 98.9% | **99.3%** |
| Covertype | Classification | 95.8% | 91.2% | **95.2%** |
| Bank Marketing | Classification | 90.1% | 86.3% | **95.8%** |

**Utility Ratio** = Synth→Real accuracy / Real→Real accuracy

### Feature Importance Correlation

How well does synthetic data preserve feature importance rankings?

| Dataset | Spearman Correlation | Interpretation |
|---------|---------------------|----------------|
| Adult Census | 0.94 | Excellent |
| Credit Card Fraud | 0.91 | Excellent |
| Covertype | 0.87 | Good |
| Bank Marketing | 0.93 | Excellent |

Values above 0.85 indicate synthetic data preserves the relative importance of features.

---

## Comparison with Other Tools

### Quality Comparison

Adult Census dataset, same configuration (CTGAN, 300 epochs):

| Tool | Quality Score | Training Time | Privacy Score |
|------|---------------|---------------|---------------|
| **Genesis** | **0.92** | 45s | **0.96** |
| SDV | 0.90 | 52s | 0.94 |
| Gretel (cloud) | 0.91 | ~60s | 0.95 |

### Feature Comparison Impact on Quality

| Scenario | Genesis | SDV | Notes |
|----------|---------|-----|-------|
| Default settings | 0.92 | 0.90 | Genesis AutoML advantage |
| With DP (ε=1.0) | 0.84 | 0.80 | Built-in DP support |
| Multi-table | 0.89 | 0.88 | Similar performance |
| Time series | 0.87 | 0.86 | Similar performance |

---

## Benchmark Methodology

### Environment

```
Hardware:
- CPU: Intel Core i9-13900K
- GPU: NVIDIA RTX 4090 (24GB)
- RAM: 64GB DDR5
- Storage: NVMe SSD

Software:
- Python 3.11
- Genesis 1.4.0
- PyTorch 2.1.0 + CUDA 12.1
- Ubuntu 22.04
```

### Configuration

Default settings unless noted:

```python
generator = SyntheticGenerator(
    method='ctgan',
    config={
        'epochs': 300,
        'batch_size': 500,
        'generator_lr': 2e-4,
        'discriminator_lr': 2e-4,
    }
)
```

### Quality Measurement

```python
from genesis import QualityEvaluator

evaluator = QualityEvaluator(real_data, synthetic_data)
report = evaluator.evaluate(target_column='target')

# Quality score is weighted average:
# - Column Shapes: 40%
# - Column Pair Trends: 30%
# - ML Utility: 30%
```

### Privacy Measurement

```python
from genesis import run_privacy_audit

report = run_privacy_audit(
    real_data,
    synthetic_data,
    sensitive_columns=['income', 'age'],
    quasi_identifiers=['education', 'occupation', 'zipcode']
)
```

---

## Running Benchmarks

### Quick Benchmark

```python
from genesis.benchmarks import quick_benchmark

results = quick_benchmark(
    data=your_data,
    methods=['ctgan', 'tvae', 'gaussian_copula'],
    n_samples=len(your_data)
)

print(results.summary())
```

### Full Benchmark Suite

```bash
# Clone and setup
git clone https://github.com/genesis-synth/genesis.git
cd genesis

# Install with benchmark dependencies
pip install -e ".[dev]"

# Run benchmarks
python benchmarks/run_all.py --output results/

# Generate report
python benchmarks/generate_report.py results/ --format html
```

### Custom Benchmark

```python
from genesis.benchmarks import Benchmark

benchmark = Benchmark(
    datasets=['adult', 'credit'],
    methods=['ctgan', 'tvae', 'auto'],
    metrics=['quality', 'privacy', 'performance'],
    n_runs=5  # Average over 5 runs
)

results = benchmark.run()
results.to_csv('my_benchmark_results.csv')
results.plot_comparison()
```

---

## Reproducing Results

All benchmark results are reproducible:

```bash
# Download benchmark datasets
genesis benchmark download-data

# Run official benchmarks
genesis benchmark run --suite official --output results/

# Compare with published results
genesis benchmark compare results/ --baseline published
```

---

## Requesting Benchmarks

Need a benchmark for a specific:
- **Dataset**: Open an issue with dataset details
- **Hardware**: We can add cloud GPU results
- **Configuration**: Suggest via GitHub Discussions

Contributions welcome! See [Contributing](/docs/contributing).
