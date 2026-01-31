# Quickstart Guide

Get started with Genesis in 5 minutes.

## Generate Your First Synthetic Dataset

```python
import pandas as pd
import numpy as np
from genesis import SyntheticGenerator, QualityEvaluator

# Create or load your data
data = pd.DataFrame({
    'age': np.random.randint(18, 80, 1000),
    'income': np.random.normal(50000, 15000, 1000),
    'city': np.random.choice(['NYC', 'LA', 'Chicago'], 1000),
    'employed': np.random.choice([True, False], 1000)
})

# Create generator (auto-selects best method)
generator = SyntheticGenerator(method='auto')

# Fit on your data
generator.fit(data, discrete_columns=['city', 'employed'])

# Generate synthetic data
synthetic_data = generator.generate(n_samples=1000)

# Evaluate quality
evaluator = QualityEvaluator(data, synthetic_data)
report = evaluator.evaluate()
print(report.summary())
```

## Choose a Specific Generator

```python
# Gaussian Copula (fast, good baseline)
generator = SyntheticGenerator(method='gaussian_copula')

# CTGAN (best for complex data)
generator = SyntheticGenerator(method='ctgan')

# TVAE (balanced)
generator = SyntheticGenerator(method='tvae')
```

## Add Constraints

```python
from genesis import Constraint

constraints = [
    Constraint.positive('income'),
    Constraint.range('age', 18, 100),
]

generator.fit(data, discrete_columns=['city'], constraints=constraints)
```

## Enable Privacy

```python
from genesis import PrivacyConfig

privacy = PrivacyConfig(
    enable_differential_privacy=True,
    epsilon=1.0
)

generator = SyntheticGenerator(method='auto', privacy=privacy)
```

## Use CLI

```bash
# Generate synthetic data
genesis generate -i data.csv -o synthetic.csv -n 1000

# Evaluate quality
genesis evaluate -r data.csv -s synthetic.csv -o report.html -f html

# Analyze data
genesis analyze -i data.csv
```

## Next Steps

- [Tabular Synthesis Guide](user_guide/tabular.md) - Deep dive into tabular generators
- [Privacy Configuration](user_guide/privacy.md) - Configure privacy settings
- [Evaluation Guide](user_guide/evaluation.md) - Understanding quality metrics
