---
sidebar_position: 100
title: Examples
---

# Examples

Real-world examples of Genesis in action.

## Basic Examples

### Customer Data Generation

Generate synthetic customer data for testing:

```python
import pandas as pd
from genesis import SyntheticGenerator, Constraint

# Load real customer data
customers = pd.read_csv('customers.csv')

# Create generator
generator = SyntheticGenerator(method='ctgan')

# Fit with constraints
generator.fit(
    customers,
    discrete_columns=['region', 'segment', 'status'],
    constraints=[
        Constraint.positive('age'),
        Constraint.range('age', 18, 100),
        Constraint.unique('customer_id'),
        Constraint.not_null('email')
    ]
)

# Generate synthetic customers
synthetic = generator.generate(10000)

# Save
synthetic.to_csv('synthetic_customers.csv', index=False)
print(f"Generated {len(synthetic)} synthetic customers")
```

### E-Commerce Transactions

```python
import pandas as pd
from genesis import auto_synthesize

# Load transaction data
transactions = pd.read_csv('transactions.csv')

# AutoML handles everything
synthetic = auto_synthesize(
    transactions,
    n_samples=100000,
    discrete_columns=['product_category', 'payment_method', 'country'],
    mode='quality'
)

# Verify quality
from genesis import QualityEvaluator
report = QualityEvaluator(transactions, synthetic).evaluate()
print(f"Quality Score: {report.overall_score:.1%}")
```

## Privacy-Preserving Generation

### Healthcare Data with Differential Privacy

```python
import pandas as pd
from genesis import SyntheticGenerator, run_privacy_audit

# Load sensitive patient data
patients = pd.read_csv('patients.csv')

# Generate with strong privacy guarantees
generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        'differential_privacy': {
            'epsilon': 1.0,
            'delta': 1e-5
        },
        'k_anonymity': {
            'k': 5,
            'quasi_identifiers': ['age', 'gender', 'zip_code']
        }
    }
)

generator.fit(
    patients,
    discrete_columns=['diagnosis', 'treatment', 'gender']
)

synthetic = generator.generate(len(patients))

# Verify privacy
report = run_privacy_audit(
    patients, 
    synthetic,
    sensitive_columns=['diagnosis', 'treatment'],
    quasi_identifiers=['age', 'gender', 'zip_code']
)

print(f"Privacy Score: {report.overall_score:.1%}")
print(f"Safe to release: {report.is_safe}")
```

### Financial Data with Privacy Testing

```python
from genesis import SyntheticGenerator, run_privacy_audit
from genesis.privacy_attacks import MembershipInferenceAttack

# Generate financial data
generator = SyntheticGenerator(
    method='ctgan',
    privacy={'differential_privacy': {'epsilon': 2.0}}
)
generator.fit(financial_data, discrete_columns=['account_type'])
synthetic = generator.generate(10000)

# Test against membership inference attack
attack = MembershipInferenceAttack()
result = attack.evaluate(financial_data, synthetic)

print(f"Attack accuracy: {result.accuracy:.1%}")
print(f"Risk level: {result.risk_level}")

if result.risk_level == 'low':
    synthetic.to_csv('safe_financial_data.csv', index=False)
```

## Imbalanced Data

### Fraud Detection Dataset

```python
import pandas as pd
from genesis import augment_imbalanced

# Load imbalanced fraud data
fraud_data = pd.read_csv('fraud_transactions.csv')

print("Original distribution:")
print(fraud_data['is_fraud'].value_counts(normalize=True))
# is_fraud
# 0    0.98
# 1    0.02

# Balance the dataset
balanced = augment_imbalanced(
    fraud_data,
    target_column='is_fraud',
    ratio=0.5,  # 50% fraud (up from 2%)
    discrete_columns=['merchant_category', 'card_type']
)

print("\nBalanced distribution:")
print(balanced['is_fraud'].value_counts(normalize=True))
# is_fraud
# 0    0.67
# 1    0.33

# Train better fraud detection model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(balanced.drop('is_fraud', axis=1), balanced['is_fraud'])
```

## Time Series

### Stock Price Simulation

```python
import pandas as pd
from genesis import TimeSeriesGenerator

# Load historical stock data
stocks = pd.read_csv('stock_prices.csv', parse_dates=['date'])
stocks = stocks.set_index('date').sort_index()

# Generate synthetic price sequences
generator = TimeSeriesGenerator(
    config={
        'hidden_dim': 64,
        'n_layers': 2,
        'epochs': 200
    }
)

generator.fit(stocks[['open', 'high', 'low', 'close', 'volume']], sequence_length=50)

# Generate 100 market scenarios
scenarios = generator.generate(n_sequences=100)

# Analyze scenarios
for i, scenario in enumerate(scenarios[:5]):
    returns = scenario['close'].pct_change().mean()
    volatility = scenario['close'].pct_change().std()
    print(f"Scenario {i+1}: return={returns:.2%}, volatility={volatility:.2%}")
```

### Energy Consumption Forecasting

```python
from genesis import TimeSeriesGenerator

# Hourly energy data
energy = pd.read_csv('energy.csv', parse_dates=['timestamp'])
energy = energy.set_index('timestamp')

generator = TimeSeriesGenerator()
generator.fit(energy[['consumption', 'temperature']], sequence_length=168)  # 1 week

# Generate synthetic weeks
synthetic_weeks = generator.generate(n_sequences=52)  # 1 year of scenarios
```

## Conditional Generation

### A/B Test Data

```python
from genesis import ConditionalGenerator

# Generate test groups
generator = ConditionalGenerator(method='ctgan')
generator.fit(users, discrete_columns=['segment', 'device'])

# Control group (existing behavior)
control = generator.generate(
    n_samples=5000,
    conditions={'experiment_group': 'control'}
)

# Treatment group (new feature)
treatment = generator.generate(
    n_samples=5000,
    conditions={
        'experiment_group': 'treatment',
        'conversion_rate': ('>', 0.05)  # Higher expected conversion
    }
)
```

### Stress Testing

```python
# Generate edge cases
extreme_cases = generator.generate(
    n_samples=1000,
    conditions={
        'transaction_amount': ('>', 100000),
        'velocity': ('>', 10),  # 10+ transactions per hour
        'is_international': True
    }
)

# Test system limits
for _, case in extreme_cases.iterrows():
    result = process_transaction(case)
    assert result.status in ['approved', 'rejected', 'review']
```

## Multi-Table (Relational)

### E-Commerce Database

```python
from genesis import MultiTableGenerator

# Load tables
customers = pd.read_csv('customers.csv')
orders = pd.read_csv('orders.csv')
order_items = pd.read_csv('order_items.csv')
products = pd.read_csv('products.csv')

tables = {
    'customers': customers,
    'orders': orders,
    'order_items': order_items,
    'products': products
}

relationships = [
    ('orders', 'customer_id', 'customers', 'customer_id'),
    ('order_items', 'order_id', 'orders', 'order_id'),
    ('order_items', 'product_id', 'products', 'product_id')
]

# Generate complete database
generator = MultiTableGenerator()
generator.fit(tables, relationships)
synthetic_db = generator.generate(scale=2.0)  # 2x the data

# Save all tables
for name, df in synthetic_db.items():
    df.to_csv(f'synthetic_{name}.csv', index=False)
```

## Pipeline Automation

### Production Pipeline

```python
from genesis.pipeline import Pipeline, steps

# Define production pipeline
pipeline = Pipeline([
    # Load and prepare
    steps.load_csv('raw_data.csv'),
    steps.drop_columns(['internal_id', 'created_at']),
    steps.convert_types({'age': 'int', 'income': 'float'}),
    
    # Train generator
    steps.fit_generator(
        method='ctgan',
        discrete_columns=['region', 'segment'],
        config={'epochs': 300}
    ),
    
    # Generate
    steps.generate(n_samples=50000),
    
    # Validate
    steps.evaluate(target_column='churn'),
    steps.privacy_audit(threshold=0.9),
    
    # Save with versioning
    steps.save_versioned(
        repo='./synthetic_repo',
        message='Production run',
        tag='latest'
    )
])

# Run pipeline
result = pipeline.run()

if result.success:
    print(f"✅ Generated {result.outputs['n_samples']} samples")
    print(f"Quality: {result.metrics['quality_score']:.1%}")
    print(f"Privacy: {result.metrics['privacy_score']:.1%}")
else:
    print(f"❌ Pipeline failed: {result.error}")
```

### YAML Pipeline

```yaml
# production_pipeline.yaml
name: daily_synthesis
schedule: "0 2 * * *"  # Run at 2 AM daily

steps:
  - load_csv:
      path: s3://bucket/daily_export.csv
  
  - fit_generator:
      method: ctgan
      discrete_columns: [status, region, category]
      config:
        epochs: 300
        batch_size: 500
  
  - generate:
      n_samples: 100000
  
  - evaluate:
      target_column: outcome
      min_score: 0.8
  
  - privacy_audit:
      sensitive_columns: [income, age]
      threshold: 0.9
  
  - save_versioned:
      repo: s3://bucket/synthetic/
      tag: daily
      
  - notify:
      on_success: slack://channel
      on_failure: email://team@company.com
```

```bash
genesis pipeline run production_pipeline.yaml
```

## Domain-Specific

### Complete Customer Profiles

```python
from genesis import SyntheticGenerator
from genesis.domains import (
    NameGenerator, EmailGenerator, 
    PhoneGenerator, AddressGenerator
)

# Generate behavioral data
generator = SyntheticGenerator(method='ctgan')
generator.fit(customer_behavior, discrete_columns=['segment'])
synthetic = generator.generate(10000)

# Add realistic PII
synthetic['name'] = NameGenerator('en_US').generate(10000)
synthetic['email'] = EmailGenerator().generate_from_names(synthetic['name'])
synthetic['phone'] = PhoneGenerator('en_US').generate(10000)
synthetic['address'] = AddressGenerator('en_US').generate(10000)

# Complete synthetic customer dataset
synthetic.to_csv('complete_synthetic_customers.csv', index=False)
```

## Notebooks

For interactive examples, see our Jupyter notebooks:

- [Basic Generation](https://github.com/genesis/genesis/blob/main/examples/basic_generation.ipynb)
- [Privacy-Preserving Synthesis](https://github.com/genesis/genesis/blob/main/examples/privacy.ipynb)
- [Time Series Generation](https://github.com/genesis/genesis/blob/main/examples/time_series.ipynb)
- [Multi-Table Synthesis](https://github.com/genesis/genesis/blob/main/examples/multi_table.ipynb)
- [AutoML Synthesis](https://github.com/genesis/genesis/blob/main/examples/automl.ipynb)

## Next Steps

- **[Getting Started](/docs/getting-started/quickstart)** - First steps with Genesis
- **[API Reference](/docs/api/reference)** - Complete API documentation
- **[Troubleshooting](/docs/troubleshooting)** - Common issues and solutions
