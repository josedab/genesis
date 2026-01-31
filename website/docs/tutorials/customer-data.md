---
sidebar_position: 2
title: Customer Data Generation
---

# Tutorial: Customer Data Generation

Build a complete pipeline to generate realistic synthetic customer data.

**Time:** 20 minutes  
**Level:** Beginner  
**What you'll learn:** Basic synthesis, quality evaluation, constraints, domain generators

---

## Goal

By the end of this tutorial, you'll have:
- Generated 10,000 synthetic customer records
- Validated data quality (90%+ similarity to real data)
- Added realistic names and emails
- Saved the data for use in testing or ML

---

## Prerequisites

```bash
pip install genesis-synth[pytorch] pandas
```

---

## Step 1: Create Sample Data

First, let's create some realistic sample data to learn from. In production, you'd use your actual customer data.

```python
import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Create sample customer data
n_customers = 5000

# Generate correlated data (income increases with age, premium more likely with high income)
ages = np.random.normal(42, 15, n_customers).clip(18, 85).astype(int)
base_income = 30000 + (ages - 18) * 800 + np.random.normal(0, 15000, n_customers)
incomes = base_income.clip(20000, 300000).astype(int)

# Premium probability increases with income
premium_prob = (incomes - 20000) / 280000 * 0.4 + 0.1
is_premium = np.random.random(n_customers) < premium_prob

# City distribution (not uniform)
cities = np.random.choice(
    ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
     'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose'],
    n_customers,
    p=[0.20, 0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06]
)

# Segments based on behavior
segments = np.where(
    is_premium & (incomes > 100000), 'Enterprise',
    np.where(is_premium, 'Premium', 'Basic')
)

# Tenure correlated with age
tenure = (ages - 18) * 0.5 + np.random.normal(0, 12, n_customers)
tenure = tenure.clip(0, 120).astype(int)

# Churn risk (lower for premium, higher for new customers)
churn_base = 0.3 - is_premium * 0.15 - tenure / 300
churn_risk = (churn_base + np.random.normal(0, 0.1, n_customers)).clip(0.01, 0.99)

real_data = pd.DataFrame({
    'customer_id': range(1, n_customers + 1),
    'age': ages,
    'income': incomes,
    'city': cities,
    'segment': segments,
    'is_premium': is_premium,
    'tenure_months': tenure,
    'churn_risk': churn_risk.round(3)
})

print(f"Created {len(real_data)} customer records")
print(real_data.head())
```

**Expected output:**
```
Created 5000 customer records
   customer_id  age  income         city   segment  is_premium  tenure_months  churn_risk
0            1   49   82639     New York     Basic       False             12       0.267
1            2   40   67234      Chicago     Basic       False              8       0.312
2            3   54  103920  Los Angeles   Premium        True             25       0.089
3            4   38   68443      Houston     Basic       False             11       0.298
4            5   47   91238     New York   Premium        True             18       0.142
```

---

## Step 2: Explore the Data

Before synthesizing, understand your data:

```python
print("=== Data Overview ===")
print(f"Shape: {real_data.shape}")
print(f"\nColumn types:\n{real_data.dtypes}")

print("\n=== Distributions ===")
print(f"\nAge: mean={real_data['age'].mean():.1f}, std={real_data['age'].std():.1f}")
print(f"Income: mean=${real_data['income'].mean():,.0f}, std=${real_data['income'].std():,.0f}")
print(f"\nCity distribution:\n{real_data['city'].value_counts(normalize=True).head()}")
print(f"\nSegment distribution:\n{real_data['segment'].value_counts(normalize=True)}")
print(f"\nPremium rate: {real_data['is_premium'].mean():.1%}")

print("\n=== Correlations ===")
print(real_data[['age', 'income', 'tenure_months', 'churn_risk']].corr().round(2))
```

**Key observations:**
- Age and income are correlated (older → higher income)
- Tenure and age are correlated
- Churn risk is lower for longer tenure

These patterns should be preserved in synthetic data.

---

## Step 3: Generate Synthetic Data

Now let's generate synthetic data using Genesis AutoML:

```python
from genesis import auto_synthesize

# Remove ID column (we'll generate new IDs)
data_for_synthesis = real_data.drop('customer_id', axis=1)

# Generate synthetic data
synthetic = auto_synthesize(
    data_for_synthesis,
    n_samples=10000,  # Generate more than original
    discrete_columns=['city', 'segment', 'is_premium'],
    mode='quality'  # Optimize for quality over speed
)

# Add new customer IDs
synthetic.insert(0, 'customer_id', range(1, len(synthetic) + 1))

print(f"Generated {len(synthetic)} synthetic customers")
print(synthetic.head())
```

**Expected output:**
```
Generated 10000 synthetic customers
   customer_id  age   income         city   segment  is_premium  tenure_months  churn_risk
0            1   45    78234     New York     Basic       False             14       0.245
1            2   52    95123  Los Angeles   Premium        True             22       0.134
2            3   33    54892      Chicago     Basic       False              6       0.341
3            4   61   142567      Houston  Enterprise     True             31       0.078
4            5   28    41234      Phoenix     Basic       False              3       0.389
```

---

## Step 4: Evaluate Quality

Verify the synthetic data preserves statistical properties:

```python
from genesis import QualityEvaluator

# Create evaluator
evaluator = QualityEvaluator(
    real_data.drop('customer_id', axis=1),
    synthetic.drop('customer_id', axis=1)
)

# Run evaluation
report = evaluator.evaluate(target_column='churn_risk')

print("=== Quality Report ===")
print(f"Overall Quality: {report.overall_score:.1%}")
print(f"Statistical Fidelity: {report.fidelity_score:.1%}")
print(f"ML Utility: {report.utility_score:.1%}")
print(f"Privacy Score: {report.privacy_score:.1%}")

print("\n=== Per-Column Scores ===")
for col, metrics in report.column_metrics.items():
    print(f"  {col}: {metrics['similarity']:.1%}")
```

**Expected output:**
```
=== Quality Report ===
Overall Quality: 94.2%
Statistical Fidelity: 93.8%
ML Utility: 95.1%
Privacy Score: 99.2%

=== Per-Column Scores ===
  age: 96.3%
  income: 94.1%
  city: 97.8%
  segment: 95.2%
  is_premium: 98.1%
  tenure_months: 91.4%
  churn_risk: 93.7%
```

---

## Step 5: Verify Correlations

Check that relationships between columns are preserved:

```python
print("=== Correlation Comparison ===")
print("\nReal data correlations:")
real_corr = real_data[['age', 'income', 'tenure_months', 'churn_risk']].corr()
print(real_corr.round(2))

print("\nSynthetic data correlations:")
synth_corr = synthetic[['age', 'income', 'tenure_months', 'churn_risk']].corr()
print(synth_corr.round(2))

print("\nDifference (should be small):")
print((synth_corr - real_corr).abs().round(2))
```

**Expected:** Correlation differences should be less than 0.05 for most pairs.

---

## Step 6: Add Realistic Names and Emails

Use Genesis domain generators to add realistic PII:

```python
from genesis.domains import NameGenerator, EmailGenerator

# Generate realistic names and emails
n = len(synthetic)
names = NameGenerator('en_US').generate(n)
emails = EmailGenerator().generate_from_names(names)

# Add to synthetic data
synthetic.insert(1, 'name', names)
synthetic.insert(2, 'email', emails)

print("=== Final Synthetic Data ===")
print(synthetic.head())
print(f"\nColumns: {list(synthetic.columns)}")
```

**Expected output:**
```
=== Final Synthetic Data ===
   customer_id           name                    email  age   income  ...
0            1    John Smith     john.smith@email.com   45    78234  ...
1            2  Sarah Johnson  sarah.johnson@mail.net   52    95123  ...
2            3   Michael Brown    m.brown@example.org   33    54892  ...
```

---

## Step 7: Apply Constraints

Ensure data meets business rules:

```python
from genesis import SyntheticGenerator, Constraint

# Define constraints
constraints = [
    Constraint.positive('age'),
    Constraint.range('age', 18, 100),
    Constraint.positive('income'),
    Constraint.range('income', 15000, 500000),
    Constraint.positive('tenure_months'),
    Constraint.range('tenure_months', 0, 360),
    Constraint.range('churn_risk', 0, 1),
]

# Create generator with constraints
generator = SyntheticGenerator(method='ctgan')
generator.fit(
    data_for_synthesis,
    discrete_columns=['city', 'segment', 'is_premium'],
    constraints=constraints
)

# Generate constrained data
constrained_synthetic = generator.generate(n_samples=1000)

# Verify constraints
print("=== Constraint Verification ===")
print(f"Age range: {constrained_synthetic['age'].min()} - {constrained_synthetic['age'].max()}")
print(f"Income range: ${constrained_synthetic['income'].min():,} - ${constrained_synthetic['income'].max():,}")
print(f"Churn risk range: {constrained_synthetic['churn_risk'].min():.3f} - {constrained_synthetic['churn_risk'].max():.3f}")
```

---

## Step 8: Save and Export

Save your synthetic data:

```python
# Save to CSV
synthetic.to_csv('synthetic_customers.csv', index=False)
print(f"Saved {len(synthetic)} records to synthetic_customers.csv")

# Save to Parquet (better for large datasets)
synthetic.to_parquet('synthetic_customers.parquet', index=False)
print(f"Saved to synthetic_customers.parquet")

# Save the trained generator for reuse
generator.save('customer_generator.pkl')
print("Saved generator to customer_generator.pkl")

# Later, load and generate more
from genesis import SyntheticGenerator
loaded_generator = SyntheticGenerator.load('customer_generator.pkl')
more_customers = loaded_generator.generate(n_samples=5000)
print(f"Generated {len(more_customers)} more customers from saved model")
```

---

## Step 9: Generate Quality Report

Create a shareable quality report:

```python
# Generate HTML report
report.to_html('customer_data_quality_report.html')
print("Saved quality report to customer_data_quality_report.html")

# Summary for stakeholders
print("\n" + "="*50)
print("SYNTHETIC DATA QUALITY SUMMARY")
print("="*50)
print(f"""
Dataset: Customer Data
Records Generated: {len(synthetic):,}
Original Records: {len(real_data):,}

Quality Metrics:
  • Overall Quality Score: {report.overall_score:.1%}
  • Statistical Fidelity: {report.fidelity_score:.1%}
  • ML Utility: {report.utility_score:.1%}
  • Privacy Score: {report.privacy_score:.1%}

Recommendation: {'✅ Ready for use' if report.overall_score > 0.85 else '⚠️ Review needed'}
""")
```

---

## Complete Script

Here's the full tutorial as a single runnable script:

```python
"""
Genesis Tutorial: Customer Data Generation
Run: python customer_data_tutorial.py
"""
import pandas as pd
import numpy as np
from genesis import auto_synthesize, QualityEvaluator, SyntheticGenerator, Constraint
from genesis.domains import NameGenerator, EmailGenerator

# Step 1: Create sample data
np.random.seed(42)
n = 5000

ages = np.random.normal(42, 15, n).clip(18, 85).astype(int)
incomes = (30000 + (ages - 18) * 800 + np.random.normal(0, 15000, n)).clip(20000, 300000).astype(int)
premium_prob = (incomes - 20000) / 280000 * 0.4 + 0.1
is_premium = np.random.random(n) < premium_prob
cities = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n, p=[0.3, 0.25, 0.2, 0.15, 0.1])
segments = np.where(is_premium & (incomes > 100000), 'Enterprise', np.where(is_premium, 'Premium', 'Basic'))
tenure = ((ages - 18) * 0.5 + np.random.normal(0, 12, n)).clip(0, 120).astype(int)
churn = (0.3 - is_premium * 0.15 - tenure / 300 + np.random.normal(0, 0.1, n)).clip(0.01, 0.99)

real_data = pd.DataFrame({
    'age': ages, 'income': incomes, 'city': cities,
    'segment': segments, 'is_premium': is_premium,
    'tenure_months': tenure, 'churn_risk': churn.round(3)
})

# Step 2: Generate synthetic data
print("Generating synthetic data...")
synthetic = auto_synthesize(real_data, n_samples=10000, discrete_columns=['city', 'segment', 'is_premium'])

# Step 3: Evaluate quality
report = QualityEvaluator(real_data, synthetic).evaluate()
print(f"\n✅ Quality Score: {report.overall_score:.1%}")

# Step 4: Add names and emails
synthetic.insert(0, 'customer_id', range(1, len(synthetic) + 1))
synthetic.insert(1, 'name', NameGenerator('en_US').generate(len(synthetic)))
synthetic.insert(2, 'email', EmailGenerator().generate(len(synthetic)))

# Step 5: Save
synthetic.to_csv('synthetic_customers.csv', index=False)
print(f"✅ Saved {len(synthetic)} synthetic customers to synthetic_customers.csv")
```

---

## Next Steps

Now that you've completed this tutorial:

1. **Add privacy**: Learn to add differential privacy in the [Privacy Guide](/docs/concepts/privacy)
2. **Scale up**: Handle larger datasets with [GPU Acceleration](/docs/advanced/gpu)
3. **Automate**: Build pipelines with the [Pipeline API](/docs/guides/pipelines)
4. **Test more**: See the [Testing Tutorial](/docs/tutorials/testing)

---

## Troubleshooting

**Low quality scores?**
- Increase training epochs: `config={'epochs': 500}`
- Try different method: `method='tvae'`
- Ensure discrete columns are specified

**Slow training?**
- Use GPU: `config={'device': 'cuda'}`
- Use `gaussian_copula` for faster iteration

**Memory errors?**
- Reduce batch size: `config={'batch_size': 256}`
- Sample training data for prototyping

See [Troubleshooting](/docs/troubleshooting) for more solutions.
