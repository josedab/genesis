# ADR-0007: Gaussian Copula as Fallback Generator

## Status

Accepted

## Context

Genesis supports multiple generation methods with different requirements:

| Method | Dependencies | GPU Benefit | Training Time | Quality |
|--------|--------------|-------------|---------------|---------|
| CTGAN | PyTorch/TF | High | Minutes-Hours | Excellent |
| TVAE | PyTorch/TF | High | Minutes-Hours | Excellent |
| TimeGAN | PyTorch/TF | High | Hours | Excellent |
| Gaussian Copula | NumPy/SciPy | None | Seconds | Good |

Users encounter Genesis in diverse environments:
- **Local development**: may not have GPU or PyTorch installed
- **CI/CD pipelines**: lightweight containers without ML frameworks
- **Quick prototyping**: need results in seconds, not minutes
- **Small datasets**: <1000 rows where deep learning is overkill
- **Edge/embedded**: constrained environments

If Genesis requires PyTorch/TensorFlow to function at all, it creates adoption friction.

## Decision

We designate **Gaussian Copula as the fallback generator** that:

1. **Requires only core dependencies** (numpy, scipy, scikit-learn)
2. **Works without GPU** or deep learning frameworks
3. **Trains in seconds** even on modest hardware
4. **Produces reasonable quality** for most use cases

```python
from genesis import SyntheticGenerator

# This works WITHOUT PyTorch/TensorFlow installed
gen = SyntheticGenerator(method='gaussian_copula')
gen.fit(data, discrete_columns=['category'])
synthetic = gen.generate(1000)

# If user requests deep learning without backend:
gen = SyntheticGenerator(method='ctgan')
gen.fit(data)
# Raises: BackendNotAvailableError with install instructions
```

The auto-selection logic:

```python
def _auto_select_method(data: pd.DataFrame, schema: DataSchema) -> GeneratorMethod:
    """Select best method based on data characteristics."""
    n_rows = len(data)
    n_cols = len(data.columns)
    
    # Small datasets: Gaussian Copula is sufficient
    if n_rows < 1000:
        return GeneratorMethod.GAUSSIAN_COPULA
    
    # Check if deep learning backend is available
    from genesis.backends import get_available_backends
    backends = get_available_backends()
    
    if not backends:
        # No DL backend: fall back to Gaussian Copula
        logger.info("No deep learning backend available, using Gaussian Copula")
        return GeneratorMethod.GAUSSIAN_COPULA
    
    # Large datasets with complex relationships: use CTGAN
    if n_rows > 10000 or n_cols > 20:
        return GeneratorMethod.CTGAN
    
    # Medium datasets: TVAE often works well
    return GeneratorMethod.TVAE
```

## Consequences

### Positive

- **Zero-friction onboarding**: `pip install genesis-synth` just works
- **Fast CI/CD**: tests run without GPU or large dependencies
- **Graceful degradation**: library always produces output
- **Quick iteration**: seconds to synthetic data for prototyping
- **Resource efficiency**: no GPU memory allocation for simple tasks
- **Broad compatibility**: works on any Python 3.8+ environment

### Negative

- **Quality gap**: Gaussian Copula doesn't capture complex relationships as well
- **User confusion**: "why does method='auto' give different results on different machines?"
- **Feature limitations**: no conditional generation guidance (rejection only)

### Mitigations

1. **Clear logging** when fallback occurs:
   ```
   INFO: No deep learning backend available, using Gaussian Copula.
   INFO: For higher quality, install: pip install genesis-synth[pytorch]
   ```

2. **Quality report comparison** shows expected ranges per method:
   ```
   Method: gaussian_copula
   Statistical Fidelity: 0.82 (typical: 0.75-0.90)
   Note: Deep learning methods typically achieve 0.85-0.95
   ```

3. **Documentation guidance** on when to upgrade:
   - Dataset > 5000 rows
   - Many categorical columns
   - Complex inter-column relationships
   - Need for conditional generation

## Gaussian Copula Implementation

The generator works by:

1. **Fitting marginal distributions** per column (Gaussian mixture, empirical CDF)
2. **Transforming to uniform** [0, 1] via CDFs
3. **Transforming to normal** via inverse CDF
4. **Fitting correlation matrix** on normalized data
5. **Sampling from multivariate normal** with learned correlations
6. **Inverse transforming** back to original distributions

```python
class GaussianCopulaGenerator(BaseTabularGenerator):
    def _fit_impl(self, data, discrete_columns, progress_callback):
        # 1. Fit marginals
        self._fit_marginals(data)
        
        # 2-3. Transform to normal
        uniform_data = self._to_uniform(data)
        normal_data = stats.norm.ppf(np.clip(uniform_data, 0.001, 0.999))
        
        # 4. Correlation matrix
        self._correlation_matrix = np.corrcoef(normal_data, rowvar=False)
        
    def _generate_impl(self, n_samples, conditions, progress_callback):
        # 5. Sample multivariate normal
        normal_samples = np.random.multivariate_normal(
            mean=np.zeros(n_cols),
            cov=self._correlation_matrix,
            size=n_samples
        )
        
        # 6. Inverse transform
        uniform_samples = stats.norm.cdf(normal_samples)
        return self._from_uniform(uniform_samples)
```

## Performance Comparison

| Dataset Size | Gaussian Copula | CTGAN (GPU) | CTGAN (CPU) |
|--------------|-----------------|-------------|-------------|
| 1,000 rows | 0.5s | 30s | 120s |
| 10,000 rows | 2s | 60s | 300s |
| 100,000 rows | 15s | 180s | 900s |
| 1,000,000 rows | 120s | 600s | 3600s |

*Training time on representative tabular dataset with 20 columns*
