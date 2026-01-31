# ADR-0008: Statistic Aggregation for Federated Learning

## Status

Accepted

## Context

Federated synthetic data generation enables organizations to collaborate on shared synthetic datasets without sharing raw data. Common scenarios:

- **Healthcare**: Multiple hospitals generate synthetic patient data representing combined populations
- **Finance**: Banks collaborate on fraud detection training data
- **Research**: Universities share synthetic research data across institutions

Traditional federated learning approaches aggregate **model weights**:
1. Each site trains a local model
2. Sites send model weights to coordinator
3. Coordinator averages weights (FedAvg)
4. Updated model sent back to sites

This works for neural networks but creates challenges:

- **Framework dependency**: Must serialize PyTorch/TensorFlow models
- **Architecture lock-in**: All sites must use identical model architecture
- **Complexity**: Gradient aggregation, learning rate coordination
- **Privacy concerns**: Model weights can leak training data information

For synthetic data, we have an alternative: aggregate **statistics** instead of weights.

## Decision

We implement federated synthesis using **statistic aggregation**:

```python
from genesis.federated import FederatedGenerator, DataSite

# Each site has local data (never leaves the site)
sites = [
    DataSite("hospital_a", data_a),
    DataSite("hospital_b", data_b),
    DataSite("hospital_c", data_c),
]

# Federated training
fed_gen = FederatedGenerator(sites)
fed_gen.train(rounds=5)

# Generate synthetic data representing all sites
synthetic = fed_gen.generate(10000)
```

The aggregation process:

```python
class ModelAggregator:
    """Aggregates statistics from multiple sites."""
    
    def aggregate(self, site_stats: List[Dict]) -> Dict:
        """Aggregate statistics using weighted averaging."""
        total_samples = sum(s['n_samples'] for s in site_stats)
        
        aggregated = {
            'n_samples': total_samples,
            'columns': {},
        }
        
        # Aggregate per-column statistics
        for col in site_stats[0]['columns']:
            col_stats = [s['columns'][col] for s in site_stats]
            weights = [s['n_samples'] / total_samples for s in site_stats]
            
            aggregated['columns'][col] = {
                'mean': sum(w * s['mean'] for w, s in zip(weights, col_stats)),
                'std': self._aggregate_std(col_stats, weights),
                'min': min(s['min'] for s in col_stats),
                'max': max(s['max'] for s in col_stats),
            }
        
        # Aggregate correlation matrix
        aggregated['correlation'] = self._aggregate_correlations(
            [s['correlation'] for s in site_stats],
            weights
        )
        
        return aggregated
```

What each site shares:

```python
class DataSite:
    def compute_local_stats(self) -> Dict:
        """Compute statistics to share (never raw data)."""
        return {
            'n_samples': len(self.data),
            'columns': {
                col: {
                    'mean': self.data[col].mean(),
                    'std': self.data[col].std(),
                    'min': self.data[col].min(),
                    'max': self.data[col].max(),
                    'n_unique': self.data[col].nunique(),
                    'distribution': self._fit_distribution(col),
                }
                for col in self.data.columns
            },
            'correlation': self.data.corr().values.tolist(),
        }
```

## Consequences

### Positive

- **Framework-agnostic**: no PyTorch/TensorFlow serialization needed
- **Simple protocol**: JSON-serializable statistics
- **Flexible architecture**: sites can use different local generators
- **Privacy-friendly**: statistics reveal less than model weights
- **Debuggable**: can inspect aggregated statistics manually
- **Incremental updates**: new sites can join without full retraining

### Negative

- **Quality ceiling**: statistics aggregation may not capture complex patterns as well as gradient-based methods
- **Distribution assumptions**: aggregation assumes certain distribution properties
- **Limited to Gaussian Copula**: current implementation uses copula-based generation
- **Heterogeneity challenges**: very different site distributions may not aggregate well

### Mitigations

1. **Site weighting**: weight contributions by data size and quality
   ```python
   config = SiteConfig(weight=1.5)  # Give this site 50% more influence
   ```

2. **Per-site privacy budgets**: control how much information each site reveals
   ```python
   config = SiteConfig(privacy_budget=0.5)  # Limit to epsilon=0.5
   ```

3. **Validation**: compare aggregated statistics to expectations
   ```python
   fed_gen.validate_aggregation()  # Check for anomalies
   ```

4. **Future extension**: add gradient-based aggregation for neural generators
   ```python
   fed_gen = FederatedGenerator(sites, method='gradient')  # Future
   ```

## Statistics Shared

| Statistic | Purpose | Privacy Risk |
|-----------|---------|--------------|
| Column means | Center distributions | Low |
| Column stds | Scale distributions | Low |
| Min/max | Bound generation range | Medium* |
| Correlation matrix | Capture dependencies | Low |
| Category frequencies | Reconstruct categoricals | Medium |
| Distribution params | Fit marginals | Low |

*Exact min/max could reveal outliers; we optionally clip to percentiles.

## Protocol Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Hospital A │    │  Hospital B │    │  Hospital C │
│  (1000 pts) │    │  (2000 pts) │    │  (500 pts)  │
└──────┬──────┘    └──────┬──────┘    └──────┬──────┘
       │                  │                  │
       ▼                  ▼                  ▼
   ┌───────┐          ┌───────┐          ┌───────┐
   │ Stats │          │ Stats │          │ Stats │
   │  JSON │          │  JSON │          │  JSON │
   └───┬───┘          └───┬───┘          └───┬───┘
       │                  │                  │
       └──────────────────┼──────────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │ Aggregator  │
                   │  (Weighted  │
                   │  averaging) │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  Combined   │
                   │   Model     │
                   └──────┬──────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  Synthetic  │
                   │    Data     │
                   │ (3500+ pts) │
                   └─────────────┘
```
