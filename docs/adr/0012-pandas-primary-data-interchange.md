# ADR-0012: Pandas as Primary Data Interchange Format

## Status

Accepted

## Context

Synthetic data generation involves multiple data handoffs:

1. User provides training data → Generator
2. Generator produces synthetic data → User
3. Evaluator compares real vs. synthetic → Report
4. Constraints validate generated data → Filtered output

We needed to choose a standard data format for these interfaces. Options considered:

- **NumPy arrays**: Efficient but lose column names, types, and metadata
- **Python dicts/lists**: Universal but verbose and slow for large data
- **Apache Arrow**: High performance but unfamiliar to most data scientists
- **Polars DataFrames**: Fast but smaller ecosystem, newer library
- **Pandas DataFrames**: Industry standard, rich ecosystem, familiar API

Our target users—data scientists and ML engineers—work with pandas daily. The library is ubiquitous in data science workflows, Jupyter notebooks, and ML pipelines.

## Decision

We adopt **pandas DataFrames as the universal data interchange format**:

```python
from genesis import SyntheticGenerator
import pandas as pd

# Input: pandas DataFrame
real_data = pd.read_csv("customers.csv")

# All generators accept DataFrames
generator = SyntheticGenerator(method='ctgan')
generator.fit(real_data, discrete_columns=['gender', 'state'])

# Output: pandas DataFrame
synthetic_data = generator.generate(n_samples=10000)

# Type guarantee
assert isinstance(synthetic_data, pd.DataFrame)
assert list(synthetic_data.columns) == list(real_data.columns)
```

Key guarantees:

1. **Column preservation**: Output DataFrame has same columns as input (same names, order)
2. **Type inference**: Generator respects original dtypes where possible
3. **Index handling**: Index is reset to RangeIndex on generation (no meaningful index)
4. **Missing values**: NaN handling follows pandas conventions

## Consequences

### Positive

- **Zero learning curve**: Users already know how to manipulate, save, and visualize DataFrames
- **Rich ecosystem**: Immediate compatibility with scikit-learn, matplotlib, seaborn, etc.
- **Metadata preservation**: Column names, categorical types survive round-trips
- **I/O flexibility**: `to_csv()`, `to_parquet()`, `to_sql()` work out of the box
- **Memory efficiency**: pandas 2.0+ with PyArrow backend handles large datasets well

### Negative

- **Performance overhead**: DataFrame creation is slower than raw NumPy arrays
- **Memory usage**: DataFrames have higher memory footprint than arrays
- **Version sensitivity**: pandas API changes between versions can break code
- **Copy semantics**: Defensive copies to prevent mutation add overhead

### Tradeoffs

We prioritize **user experience over raw performance**:

```python
# Internal implementation often uses NumPy for speed
def _generate_impl(self, n_samples, conditions, progress_callback):
    # Fast NumPy operations internally
    raw_samples = self._model.sample(n_samples)  # Returns ndarray
    
    # Convert to DataFrame at API boundary
    return pd.DataFrame(raw_samples, columns=self._column_names)
```

For performance-critical applications, we provide:

- Batch generation via `StreamingGenerator` to reduce memory pressure
- GPU-accelerated generation that batches DataFrame creation
- Optional `_generate_numpy()` method for internal use (not public API)

## Type Handling

| pandas dtype | Generator behavior |
|--------------|-------------------|
| `int64` | Preserved, rounded after generation |
| `float64` | Preserved directly |
| `object` (string) | Treated as categorical |
| `category` | Preserved with original categories |
| `datetime64` | Converted to numeric, reversed on output |
| `bool` | Treated as binary categorical |

## Examples

```python
# DataFrame in, DataFrame out
synthetic = generator.fit_generate(df, n_samples=1000)

# Works with all pandas I/O
synthetic.to_parquet("synthetic_customers.parquet")
synthetic.to_sql("synthetic_customers", engine)

# Immediate visualization
synthetic.hist(figsize=(12, 8))

# Direct sklearn compatibility  
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(synthetic)
```
