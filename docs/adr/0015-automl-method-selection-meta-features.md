# ADR-0015: AutoML Method Selection via Meta-Feature Analysis

## Status

Accepted

## Context

Genesis supports multiple generation methods, each with different strengths:

| Method | Best For | Weakness |
|--------|----------|----------|
| Gaussian Copula | Correlated numeric data, fast | Struggles with complex categoricals |
| CTGAN | Mixed types, complex distributions | Slow training, mode collapse risk |
| TVAE | Continuous data, stable training | Less flexible than GAN |
| CopulaGAN | Balanced approach | Moderate at everything |

Users frequently ask: "Which method should I use for my data?"

This question requires understanding:
- Data characteristics (types, distributions, correlations)
- Quality requirements (fidelity vs. speed)
- Available compute resources

We wanted to automate this decision while remaining transparent about the selection rationale.

## Decision

We implement **meta-learning based automatic method selection**:

```python
from genesis import auto_synthesize

# One-line automatic synthesis
synthetic = auto_synthesize(df, n_samples=10000)

# With more control
from genesis.automl import AutoMLSynthesizer

automl = AutoMLSynthesizer(prefer_quality=True)
automl.fit(df)
print(f"Selected: {automl.selected_method}")  # "ctgan"
print(f"Confidence: {automl.selection_confidence:.1%}")  # "87.5%"
synthetic = automl.generate(10000)
```

The selection algorithm:

1. **Extract meta-features** from the dataset
2. **Apply decision rules** based on empirical benchmarks
3. **Return method with confidence score**

### Meta-Features Extracted

```python
@dataclass
class DatasetMetaFeatures:
    # Size
    n_rows: int
    n_columns: int
    
    # Type distribution
    n_numeric: int
    n_categorical: int
    n_datetime: int
    
    # Numeric characteristics
    numeric_mean_skewness: float
    numeric_mean_kurtosis: float
    numeric_outlier_ratio: float
    
    # Categorical characteristics
    categorical_mean_cardinality: float
    categorical_max_cardinality: float
    categorical_imbalance_ratio: float
    
    # Correlation structure
    mean_correlation: float
    max_correlation: float
    correlation_clusters: int
    
    # Complexity indicators
    multimodal_columns: int
    highly_skewed_columns: int
    estimated_complexity: float  # 0-1 score
```

### Decision Rules

```python
def select_method(features: DatasetMetaFeatures, prefer_quality: bool) -> Tuple[str, float]:
    # Small datasets: use fast methods
    if features.n_rows < 1000:
        return ("gaussian_copula", 0.9)
    
    # Highly correlated data: copula excels
    if features.mean_correlation > 0.5:
        return ("gaussian_copula", 0.85)
    
    # Complex mixed types: CTGAN
    if features.n_categorical > 0 and features.estimated_complexity > 0.6:
        if prefer_quality:
            return ("ctgan", 0.8)
        return ("tvae", 0.75)  # Faster alternative
    
    # Mostly numeric, moderate complexity: TVAE
    if features.n_numeric / features.n_columns > 0.7:
        return ("tvae", 0.8)
    
    # Default fallback
    return ("gaussian_copula", 0.7)
```

## Consequences

### Positive

- **Lower barrier to entry**: New users get good results without understanding methods
- **Transparent decisions**: Confidence scores and reasoning are accessible
- **Benchmark-informed**: Rules derived from extensive testing across datasets
- **Overridable**: Users can still specify method explicitly
- **Fast feedback**: Meta-feature extraction is O(n) and takes seconds

### Negative

- **Heuristic limitations**: Rules don't cover all edge cases
- **No online learning**: Rules are static, not learned from user feedback
- **Confidence calibration**: Scores are estimates, not probabilities
- **Method evolution**: New methods require manual rule updates

### Future Improvements

- Train ML model on (meta-features, method, quality) tuples
- A/B test methods and learn optimal selection
- User feedback loop to improve recommendations

## Method Selection Matrix

| Condition | Selected Method | Confidence |
|-----------|-----------------|------------|
| n_rows < 1000 | gaussian_copula | 90% |
| mean_correlation > 0.5 | gaussian_copula | 85% |
| High complexity + mixed types | ctgan | 80% |
| Mostly numeric | tvae | 80% |
| High cardinality categoricals | ctgan | 75% |
| Time pressure (prefer_speed=True) | gaussian_copula | 85% |
| Default | gaussian_copula | 70% |

## Examples

```python
# Automatic selection with explanation
from genesis.automl import AutoMLSynthesizer

automl = AutoMLSynthesizer()
automl.fit(df)

# Inspect the decision
print(automl.meta_features)
# DatasetMetaFeatures(n_rows=50000, n_columns=12, n_numeric=8, 
#                     n_categorical=4, mean_correlation=0.23, ...)

print(automl.selection_reasoning)
# "Selected CTGAN: Mixed column types (8 numeric, 4 categorical) 
#  with moderate complexity (0.65). High cardinality in 'city' column."

# Compare methods (for advanced users)
comparison = automl.compare_methods(df, methods=['ctgan', 'tvae', 'gaussian_copula'])
print(comparison)
# Method           Time    Statistical  ML Utility
# ctgan            120s    0.92         0.89
# tvae             45s     0.88         0.85
# gaussian_copula  5s      0.82         0.78
```
