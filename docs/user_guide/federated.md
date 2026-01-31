# Federated Synthetic Data Generation

Genesis supports federated learning for generating synthetic data across distributed data sources without centralizing sensitive data.

## Overview

| Class | Purpose | Use Case |
|-------|---------|----------|
| **FederatedGenerator** | Coordinate federated training | Production multi-site generation |
| **FederatedTrainingSimulator** | Test federated scenarios | Development and testing |
| **SecureAggregator** | DP-protected aggregation | Privacy-critical deployments |
| **DataSite** | Local site operations | Per-site training |

## Quick Start

```python
from genesis.federated import create_federated_generator

# Data at different sites (in practice, each site has its own data)
site_datasets = {
    "hospital_a": df_hospital_a,
    "hospital_b": df_hospital_b,
    "hospital_c": df_hospital_c,
}

# Create and train federated generator
fed_gen = create_federated_generator(site_datasets, method="gaussian_copula")
fed_gen.train(rounds=5)

# Generate synthetic data
synthetic = fed_gen.generate(10000)
```

## FederatedGenerator

The main class for coordinating federated synthetic data generation.

```python
from genesis.federated import FederatedGenerator, DataSite, ModelAggregator

# Create generator
fed_gen = FederatedGenerator(
    aggregator=ModelAggregator(),
    method="gaussian_copula",
)

# Add sites
for name, data in site_datasets.items():
    site = DataSite(name, data)
    fed_gen.add_site(site)

# Train across all sites
result = fed_gen.train(rounds=3)

print(f"Trained on {result.n_sites} sites")
print(f"Total samples: {result.total_samples}")
```

### Generation Strategies

#### Proportional (Default)
Generate samples proportionally to each site's contribution:

```python
synthetic = fed_gen.generate(10000, strategy="proportional")
# If site_a had 60% of data, ~60% of synthetic comes from site_a's distribution
```

#### Uniform
Equal contribution from each site:

```python
synthetic = fed_gen.generate(10000, strategy="uniform")
# Each site contributes equally regardless of original size
```

#### Weighted
Custom weights per site:

```python
synthetic = fed_gen.generate(
    10000, 
    strategy="weighted",
    weights={"site_a": 0.5, "site_b": 0.3, "site_c": 0.2}
)
```

## SecureAggregator

Add differential privacy to federated aggregation:

```python
from genesis.federated import FederatedGenerator, SecureAggregator

# Create secure aggregator with DP
secure_agg = SecureAggregator(
    noise_scale=0.1,      # Noise multiplier for DP
    min_sites=3,          # Minimum sites required
    clip_threshold=1.0,   # Gradient clipping
)

fed_gen = FederatedGenerator(aggregator=secure_agg)
```

### Privacy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `noise_scale` | float | 0.1 | Gaussian noise multiplier |
| `min_sites` | int | 2 | Minimum sites for aggregation |
| `clip_threshold` | float | 1.0 | Clip parameter values |
| `epsilon` | float | None | Target ε for (ε,δ)-DP |
| `delta` | float | 1e-5 | Target δ for (ε,δ)-DP |

## FederatedTrainingSimulator

Test federated scenarios without actual distributed infrastructure:

```python
from genesis.federated import FederatedTrainingSimulator

# Create simulator
simulator = FederatedTrainingSimulator(n_sites=5)

# Setup from single dataset (auto-partitions)
simulator.setup_from_data(full_dataset)

# Simulate federated training
results = simulator.simulate_training(n_rounds=10)

print(f"Training complete across {results['n_sites']} sites")
for round_info in results['rounds']:
    print(f"  Round {round_info['round']}: aggregated {round_info['samples_aggregated']} samples")
```

### Non-IID Partitioning

Simulate realistic non-uniform data distributions:

```python
# Partition by a column (each site gets subset of values)
simulator.setup_non_iid(
    data=full_dataset,
    partition_column="region",  # Sites get different regions
)

# Or with custom heterogeneity
simulator.setup_non_iid(
    data=full_dataset,
    partition_column="category",
    alpha=0.5,  # Dirichlet concentration (lower = more heterogeneous)
)
```

### Simulate Network Conditions

```python
results = simulator.simulate_training(
    n_rounds=10,
    
    # Simulate failures
    site_failure_rate=0.1,  # 10% chance site fails each round
    
    # Simulate stragglers
    straggler_rate=0.2,     # 20% of sites are slow
    straggler_delay=2.0,    # Slow sites take 2x longer
)
```

## DataSite

Represents a single data site in the federation:

```python
from genesis.federated import DataSite, SiteConfig

# Configure site
config = SiteConfig(
    name="hospital_east",
    weight=1.0,           # Relative importance
    privacy_budget=1.0,   # Local DP budget
    min_samples=100,      # Minimum samples to participate
)

# Create site
site = DataSite(
    name="hospital_east",
    data=local_data,
    config=config,
)

# Initialize local model
site.initialize(method="gaussian_copula")

# Train locally and get parameters
local_params = site.train_local()
```

## Complete Workflow

### Production Deployment

```python
from genesis.federated import (
    FederatedGenerator,
    SecureAggregator,
    DataSite,
    SiteConfig,
)

# 1. Configure secure aggregation
aggregator = SecureAggregator(
    noise_scale=0.1,
    min_sites=3,
)

# 2. Create federated generator
fed_gen = FederatedGenerator(
    aggregator=aggregator,
    method="gaussian_copula",
)

# 3. Each site joins (in practice, via secure channels)
for site_name in ["site_a", "site_b", "site_c", "site_d"]:
    config = SiteConfig(name=site_name, privacy_budget=1.0)
    site = DataSite(site_name, load_site_data(site_name), config)
    fed_gen.add_site(site)

# 4. Run federated training
for round_num in range(10):
    result = fed_gen.train(rounds=1)
    print(f"Round {round_num + 1}: {result.total_samples} samples aggregated")
    
    # Optional: evaluate on holdout
    if round_num % 5 == 4:
        quality = evaluate_quality(fed_gen, holdout_data)
        print(f"  Quality score: {quality:.3f}")

# 5. Generate synthetic data
synthetic = fed_gen.generate(
    n_samples=50000,
    strategy="proportional",
)

# 6. Export with provenance
synthetic.to_csv("federated_synthetic.csv", index=False)
fed_gen.save_metadata("federated_metadata.json")
```

### Testing with Simulator

```python
from genesis.federated import FederatedTrainingSimulator
import pandas as pd

# Load test data
data = pd.read_csv("test_data.csv")

# Create simulator
simulator = FederatedTrainingSimulator(n_sites=10)

# Test IID scenario
print("=== IID Scenario ===")
simulator.setup_from_data(data)
iid_results = simulator.simulate_training(n_rounds=5)
iid_synthetic = simulator.generate_synthetic(10000)

# Test Non-IID scenario
print("\n=== Non-IID Scenario ===")
simulator.setup_non_iid(data, partition_column="category", alpha=0.3)
non_iid_results = simulator.simulate_training(n_rounds=10)  # May need more rounds
non_iid_synthetic = simulator.generate_synthetic(10000)

# Compare quality
from genesis.evaluation import QualityEvaluator

print("\nQuality Comparison:")
print(f"  IID:     {QualityEvaluator(data, iid_synthetic).overall_score:.3f}")
print(f"  Non-IID: {QualityEvaluator(data, non_iid_synthetic).overall_score:.3f}")
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FederatedGenerator                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   Aggregator                             │   │
│  │  (ModelAggregator or SecureAggregator with DP)          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                            ▲                                    │
│              ┌─────────────┼─────────────┐                     │
│              │             │             │                      │
│         ┌────┴────┐   ┌────┴────┐   ┌────┴────┐                │
│         │ Site A  │   │ Site B  │   │ Site C  │                │
│         │ (Local  │   │ (Local  │   │ (Local  │                │
│         │  Data)  │   │  Data)  │   │  Data)  │                │
│         └─────────┘   └─────────┘   └─────────┘                │
└─────────────────────────────────────────────────────────────────┘

Training Flow:
1. Sites train locally → extract parameters
2. Parameters sent to aggregator (no raw data)
3. Aggregator combines (with optional DP noise)
4. Global model updated
5. Repeat for N rounds
```

## Best Practices

### 1. Ensure Minimum Site Participation
```python
aggregator = SecureAggregator(min_sites=3)  # Require 3+ sites
```

### 2. Use Differential Privacy for Sensitive Data
```python
aggregator = SecureAggregator(
    noise_scale=0.1,
    epsilon=1.0,  # Strong privacy
)
```

### 3. Handle Heterogeneous Data
```python
# More training rounds for non-IID data
results = fed_gen.train(rounds=20)  # vs 5 for IID

# Or use weighted aggregation
fed_gen.set_site_weights({
    "large_site": 0.5,
    "small_site_1": 0.25,
    "small_site_2": 0.25,
})
```

### 4. Validate Before Deployment
```python
# Test with simulator first
simulator = FederatedTrainingSimulator(n_sites=len(real_sites))
simulator.setup_non_iid(sample_data, partition_column="site_type")
results = simulator.simulate_training(n_rounds=10)

# Check convergence
if results["final_quality"] > 0.8:
    print("Ready for production deployment")
```
