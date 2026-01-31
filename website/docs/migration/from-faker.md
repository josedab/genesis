---
sidebar_position: 3
title: From Faker
---

# Migrating from Faker to Genesis

Upgrade from template-based fake data to statistically-learned synthetic data.

## Understanding the Difference

**Faker** generates random data from templates—useful for test fixtures but doesn't preserve real data patterns.

**Genesis** learns from your actual data and generates new samples that maintain statistical properties—essential for ML training and realistic testing.

| Aspect | Faker | Genesis |
|--------|-------|---------|
| **How it works** | Random from templates | Learns from real data |
| **Statistical fidelity** | ❌ None | ✅ High (90%+) |
| **Correlations** | ❌ Columns independent | ✅ Preserves relationships |
| **ML utility** | ❌ Low | ✅ High (85-95% of real) |
| **Use case** | Test fixtures | ML training, realistic testing |

---

## When to Migrate

**Keep using Faker when:**
- You don't have real data to learn from
- You need simple test fixtures (names, emails)
- Realism doesn't matter

**Migrate to Genesis when:**
- You have real data and want realistic synthetic version
- Data will be used for ML training or analytics
- You need to preserve statistical distributions
- Privacy matters (Genesis has formal guarantees)

---

## Installation

```bash
# Keep Faker for simple fixtures (optional)
# pip install faker

# Add Genesis for statistical synthesis
pip install genesis-synth[pytorch]
```

---

## Basic Migration

### Faker: Template-Based

```python
# ❌ Faker (before) - random, no patterns
from faker import Faker
import pandas as pd

fake = Faker()

data = pd.DataFrame([{
    'name': fake.name(),
    'age': fake.random_int(18, 80),
    'income': fake.random_int(30000, 150000),
    'city': fake.city(),
    'is_premium': fake.boolean()
} for _ in range(1000)])

# Problem: No correlations!
# - High income customers are not more likely to be premium
# - Age doesn't correlate with income
# - Cities are random, not from your actual distribution
```

### Genesis: Learn from Real Data

```python
# ✅ Genesis (after) - learns real patterns
from genesis import auto_synthesize
import pandas as pd

# Load your real data
real_data = pd.read_csv('customers.csv')

# Generate synthetic data that preserves patterns
synthetic = auto_synthesize(
    real_data,
    n_samples=1000,
    discrete_columns=['city', 'is_premium']
)

# Result: Correlations preserved!
# - High income → more likely premium (if true in real data)
# - Age-income correlation maintained
# - City distribution matches real data
```

---

## Side-by-Side Comparison

### Customer Generation

```python
# Faker - Random customers
from faker import Faker
fake = Faker()

customers_faker = pd.DataFrame([{
    'customer_id': i,
    'name': fake.name(),
    'email': fake.email(),
    'age': fake.random_int(18, 85),
    'income': fake.random_int(20000, 200000),
    'segment': fake.random_element(['A', 'B', 'C']),
    'lifetime_value': fake.pyfloat(min_value=0, max_value=10000)
} for i in range(1000)])
```

```python
# Genesis - Statistical synthesis
from genesis import SyntheticGenerator, Constraint
from genesis.domains import NameGenerator, EmailGenerator

# Learn from real customer behavior
generator = SyntheticGenerator(method='ctgan')
generator.fit(
    real_customers[['age', 'income', 'segment', 'lifetime_value']],
    discrete_columns=['segment'],
    constraints=[
        Constraint.positive('income'),
        Constraint.positive('lifetime_value')
    ]
)

# Generate synthetic behavioral data
synthetic = generator.generate(1000)

# Add realistic PII (similar to Faker but better)
synthetic['customer_id'] = range(1, len(synthetic) + 1)
synthetic['name'] = NameGenerator('en_US').generate(len(synthetic))
synthetic['email'] = EmailGenerator().generate(len(synthetic))
```

### Transaction Generation

```python
# Faker - Random transactions (no patterns)
transactions_faker = pd.DataFrame([{
    'transaction_id': fake.uuid4(),
    'customer_id': fake.random_int(1, 100),
    'amount': fake.pyfloat(min_value=1, max_value=1000),
    'category': fake.random_element(['Food', 'Travel', 'Shopping']),
    'timestamp': fake.date_time_this_year()
} for _ in range(5000)])
```

```python
# Genesis - Realistic transaction patterns
from genesis import auto_synthesize

# Learn from real transactions
synthetic_transactions = auto_synthesize(
    real_transactions,
    n_samples=5000,
    discrete_columns=['category']
)

# Result preserves:
# - Amount distributions per category
# - Customer purchase frequency patterns
# - Temporal patterns (if time series)
```

---

## Using Genesis Domain Generators

Genesis includes Faker-like generators for realistic PII that doesn't come from real data:

```python
from genesis.domains import (
    NameGenerator,
    EmailGenerator,
    PhoneGenerator,
    AddressGenerator,
    DateGenerator,
    CompanyGenerator
)

n = 1000

# Generate realistic but fake PII
names = NameGenerator('en_US').generate(n)
emails = EmailGenerator().generate(n)  # or .generate_from_names(names)
phones = PhoneGenerator('en_US').generate(n)
addresses = AddressGenerator('en_US').generate(n)
birthdates = DateGenerator().generate(n, start='1940-01-01', end='2005-12-31')
companies = CompanyGenerator('en_US').generate(n)

# Combine with statistically-generated behavioral data
synthetic_customers = generator.generate(n)
synthetic_customers['name'] = names
synthetic_customers['email'] = emails
synthetic_customers['phone'] = phones
```

### Supported Locales

Genesis domain generators support 50+ locales:

```python
# US English
names_us = NameGenerator('en_US').generate(100)

# UK English
names_uk = NameGenerator('en_GB').generate(100)

# German
names_de = NameGenerator('de_DE').generate(100)

# Japanese
names_jp = NameGenerator('ja_JP').generate(100)

# Full list
from genesis.domains import list_locales
print(list_locales())
```

---

## Complete Migration Example

### Before: Pure Faker

```python
from faker import Faker
import pandas as pd

fake = Faker()
Faker.seed(42)

def generate_customer_data(n):
    return pd.DataFrame([{
        'customer_id': i,
        'name': fake.name(),
        'email': fake.email(),
        'age': fake.random_int(18, 80),
        'income': fake.random_int(30000, 150000),
        'city': fake.random_element(['NYC', 'LA', 'Chicago', 'Houston']),
        'segment': fake.random_element(['Basic', 'Premium', 'Enterprise']),
        'tenure_months': fake.random_int(1, 120),
        'churn_risk': fake.pyfloat(min_value=0, max_value=1)
    } for i in range(n)])

test_data = generate_customer_data(1000)
```

### After: Genesis + Domain Generators

```python
from genesis import auto_synthesize, SyntheticGenerator
from genesis.domains import NameGenerator, EmailGenerator
import pandas as pd

# Load real customer data (with PII removed or pseudonymized)
real_data = pd.read_csv('real_customers.csv')
real_behavioral = real_data[['age', 'income', 'city', 'segment', 'tenure_months', 'churn_risk']]

# Generate synthetic behavioral data
synthetic_behavioral = auto_synthesize(
    real_behavioral,
    n_samples=1000,
    discrete_columns=['city', 'segment']
)

# Add synthetic PII
n = len(synthetic_behavioral)
synthetic_behavioral['customer_id'] = range(1, n + 1)
synthetic_behavioral['name'] = NameGenerator('en_US').generate(n)
synthetic_behavioral['email'] = EmailGenerator().generate(n)

# Reorder columns to match original
test_data = synthetic_behavioral[[
    'customer_id', 'name', 'email', 'age', 'income',
    'city', 'segment', 'tenure_months', 'churn_risk'
]]

# Verify quality
from genesis import QualityEvaluator
report = QualityEvaluator(real_behavioral, synthetic_behavioral).evaluate()
print(f"Quality: {report.overall_score:.1%}")  # e.g., "Quality: 94.2%"
```

---

## Hybrid Approach

You can use both together:

```python
from faker import Faker
from genesis import auto_synthesize

fake = Faker()

# Use Genesis for behavioral data (needs patterns)
synthetic_behavior = auto_synthesize(real_data, n_samples=1000)

# Use Faker for non-critical fields
synthetic_behavior['notes'] = [fake.sentence() for _ in range(1000)]
synthetic_behavior['created_by'] = [fake.user_name() for _ in range(1000)]
```

---

## Verifying Your Migration

### Check Distribution Similarity

```python
import matplotlib.pyplot as plt

# Faker: Uniform distribution (random)
faker_income = [fake.random_int(30000, 150000) for _ in range(1000)]

# Genesis: Matches real distribution
genesis_income = synthetic['income']

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].hist(real_data['income'], bins=30, alpha=0.7, label='Real')
axes[1].hist(faker_income, bins=30, alpha=0.7, label='Faker', color='orange')
axes[2].hist(genesis_income, bins=30, alpha=0.7, label='Genesis', color='green')
plt.show()
```

### Check Correlations

```python
# Real data correlation
real_corr = real_data[['age', 'income', 'tenure_months']].corr()

# Faker: Near-zero correlations (random)
faker_corr = faker_data[['age', 'income', 'tenure_months']].corr()

# Genesis: Preserves correlations
genesis_corr = synthetic[['age', 'income', 'tenure_months']].corr()

print("Real correlations:")
print(real_corr)
print("\nGenesis correlations:")
print(genesis_corr)
# Should be similar!
```

### ML Model Test

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Train on synthetic, test on real
X_train = synthetic.drop('churn_risk', axis=1)
y_train = (synthetic['churn_risk'] > 0.5).astype(int)

X_test = real_data.drop('churn_risk', axis=1)
y_test = (real_data['churn_risk'] > 0.5).astype(int)

model = RandomForestClassifier()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print(f"Model trained on synthetic, tested on real: {accuracy:.1%}")
# Genesis: ~85-95% of training on real data
# Faker: Much lower (data is random noise)
```

---

## Migration Checklist

- [ ] Identify which data needs statistical learning vs simple fixtures
- [ ] Install Genesis: `pip install genesis-synth[pytorch]`
- [ ] Prepare real data for learning (remove actual PII)
- [ ] Replace Faker calls with `auto_synthesize()` for behavioral data
- [ ] Use Genesis domain generators for realistic PII
- [ ] Verify quality with `QualityEvaluator`
- [ ] Update tests to use new data generation

---

## Need Help?

- [GitHub Discussions](https://github.com/genesis-synth/genesis/discussions)
- [Examples](/docs/examples) - Working code examples
- [API Reference](/docs/api/reference) - Full API documentation
