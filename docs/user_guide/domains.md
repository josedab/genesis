# Domain-Specific Generators

Genesis provides specialized generators for healthcare, finance, and retail domains, producing realistic synthetic data that respects domain constraints and regulations.

## Overview

Domain-specific generators understand the semantics of their target domain:
- **Healthcare**: Patient records, diagnoses, medications (HIPAA-aware)
- **Finance**: Transactions, accounts, risk profiles
- **Retail**: Customers, orders, products, inventory

```python
from genesis.domains import HealthcareGenerator, FinanceGenerator, RetailGenerator

healthcare = HealthcareGenerator()
patients = healthcare.generate_patient_cohort(1000)
```

## Healthcare Generator

### Patient Cohort Generation

```python
from genesis.domains import HealthcareGenerator

generator = HealthcareGenerator()

# Generate patient cohort
patients = generator.generate_patient_cohort(
    n_patients=1000,
    age_range=(18, 85),
    include_demographics=True,
    include_conditions=True
)

print(patients.columns)
# ['patient_id', 'age', 'gender', 'race', 'ethnicity', 
#  'primary_diagnosis', 'comorbidities', 'admission_date', ...]
```

### Clinical Events

```python
# Generate clinical events for patients
events = generator.generate_clinical_events(
    patient_ids=patients["patient_id"],
    event_types=["lab_result", "medication", "procedure"],
    time_range=("2023-01-01", "2024-01-01"),
    events_per_patient=(5, 20)
)
```

### Realistic Medical Data

```python
# Generate realistic lab results
labs = generator.generate_lab_results(
    n_results=5000,
    tests=["glucose", "hemoglobin", "creatinine"],
    include_abnormal=True,
    abnormal_rate=0.15
)

# Generate medications
meds = generator.generate_medications(
    n_records=3000,
    drug_classes=["analgesic", "antibiotic", "cardiovascular"]
)
```

### HIPAA-Safe Generation

```python
# Generate with differential privacy for HIPAA compliance
generator = HealthcareGenerator(
    privacy_config={
        "enable_differential_privacy": True,
        "epsilon": 0.1,
        "remove_identifiers": True
    }
)

safe_patients = generator.generate_patient_cohort(1000)
```

## Finance Generator

### Transaction Generation

```python
from genesis.domains import FinanceGenerator

generator = FinanceGenerator()

# Generate transactions
transactions = generator.generate_transactions(
    n_transactions=10000,
    accounts=100,
    time_range=("2023-01-01", "2024-01-01"),
    include_fraud=True,
    fraud_rate=0.02  # 2% fraud rate
)

print(transactions.columns)
# ['transaction_id', 'account_id', 'amount', 'timestamp', 
#  'merchant_category', 'is_fraud', 'fraud_type', ...]
```

### Account Portfolios

```python
# Generate customer accounts
accounts = generator.generate_accounts(
    n_accounts=1000,
    account_types=["checking", "savings", "investment"],
    include_risk_profile=True
)

# Generate credit profiles
credit = generator.generate_credit_profiles(
    n_profiles=1000,
    score_range=(300, 850)
)
```

### Realistic Financial Patterns

```python
# Generate with realistic patterns
transactions = generator.generate_transactions(
    n_transactions=50000,
    patterns={
        "seasonal": True,         # Holiday spending spikes
        "weekly_cycle": True,     # Weekday vs weekend
        "salary_deposits": True,  # Monthly income patterns
        "recurring_bills": True   # Regular payments
    }
)
```

### Fraud Scenarios

```python
# Generate specific fraud patterns
fraud_data = generator.generate_fraud_scenarios(
    scenarios=["account_takeover", "card_not_present", "identity_theft"],
    n_per_scenario=100
)
```

## Retail Generator

### Customer and Order Data

```python
from genesis.domains import RetailGenerator

generator = RetailGenerator()

# Generate customers
customers = generator.generate_customers(
    n_customers=5000,
    include_segments=True,
    segments=["premium", "regular", "budget"]
)

# Generate orders
orders = generator.generate_orders(
    n_orders=20000,
    customer_ids=customers["customer_id"],
    time_range=("2023-01-01", "2024-01-01")
)
```

### Product Catalog

```python
# Generate product catalog
products = generator.generate_products(
    n_products=500,
    categories=["electronics", "clothing", "home"],
    include_pricing=True,
    include_inventory=True
)

# Generate order items
items = generator.generate_order_items(
    order_ids=orders["order_id"],
    product_ids=products["product_id"]
)
```

### Customer Behavior

```python
# Generate with realistic shopping patterns
orders = generator.generate_orders(
    n_orders=50000,
    patterns={
        "seasonal": True,        # Holiday shopping
        "customer_loyalty": True, # Repeat customers
        "cart_abandonment": True, # Incomplete sessions
        "promotions": True       # Sale response
    }
)
```

### Complete E-commerce Dataset

```python
# Generate complete interconnected dataset
ecommerce = generator.generate_ecommerce_dataset(
    n_customers=10000,
    n_products=1000,
    n_orders=100000,
    time_range=("2022-01-01", "2024-01-01")
)

# Returns dict with all tables
customers = ecommerce["customers"]
products = ecommerce["products"]
orders = ecommerce["orders"]
order_items = ecommerce["order_items"]
reviews = ecommerce["reviews"]
```

## Custom Constraints

All domain generators support custom constraints:

```python
from genesis.domains import FinanceGenerator

generator = FinanceGenerator()

transactions = generator.generate_transactions(
    n_transactions=10000,
    constraints={
        "amount_min": 0.01,
        "amount_max": 50000,
        "amount_distribution": "lognormal",
        "merchant_categories": ["grocery", "gas", "restaurant"],
        "geographic_region": "US_WEST"
    }
)
```

## Training on Real Data

Use real data to improve realism:

```python
from genesis.domains import HealthcareGenerator

# Initialize with real data patterns
generator = HealthcareGenerator()
generator.fit(real_patient_data)

# Generate synthetic data matching real patterns
synthetic = generator.generate_patient_cohort(
    n_patients=len(real_patient_data) * 2
)
```

## Referential Integrity

Domain generators maintain relationships:

```python
from genesis.domains import RetailGenerator

generator = RetailGenerator()

# Generate connected data
data = generator.generate_connected_dataset()

# Verify relationships
assert set(data["orders"]["customer_id"]).issubset(
    set(data["customers"]["customer_id"])
)
assert set(data["order_items"]["order_id"]).issubset(
    set(data["orders"]["order_id"])
)
```

## Pipeline Integration

```python
from genesis.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .add_node("healthcare", "domain_generate", {
        "domain": "healthcare",
        "type": "patient_cohort",
        "n_samples": 10000
    })
    .add_node("privacy", "privacy_audit", {
        "sensitive_columns": ["diagnosis", "medications"]
    })
    .sink("synthetic_patients.csv")
    .build()
)

pipeline.execute()
```

## CLI Usage

```bash
# Generate healthcare data
genesis domain healthcare -t patient_cohort -n 1000 -o patients.csv

# Generate finance transactions with fraud
genesis domain finance -t transactions -n 10000 --fraud-rate 0.02 -o transactions.csv

# Generate retail orders
genesis domain retail -t orders -n 5000 -o orders.csv
```

## Configuration

Each generator supports detailed configuration:

```python
from genesis.domains import HealthcareGenerator, HealthcareConfig

config = HealthcareConfig(
    locale="en_US",
    include_phi=False,
    code_system="ICD-10",
    date_format="%Y-%m-%d",
    age_distribution="us_census",
    condition_prevalence="cdc_stats"
)

generator = HealthcareGenerator(config=config)
```

## Extending Domains

Create custom domain generators:

```python
from genesis.domains import DomainGenerator

class InsuranceGenerator(DomainGenerator):
    domain = "insurance"
    
    def generate_policies(self, n_policies: int):
        # Custom implementation
        pass
    
    def generate_claims(self, n_claims: int):
        # Custom implementation
        pass
```

## Best Practices

1. **Use domain generators**: They encode domain knowledge you'd otherwise implement manually
2. **Validate referential integrity**: Check foreign key relationships
3. **Apply privacy**: Use differential privacy for sensitive domains
4. **Match real distributions**: Fit on real data when available
5. **Test domain constraints**: Verify generated data meets business rules
