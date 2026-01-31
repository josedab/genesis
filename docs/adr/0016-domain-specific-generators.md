# ADR-0016: Domain-Specific Generators as Specialization Layer

## Status

Accepted

## Context

While Genesis provides general-purpose synthetic data generation, certain industries have specific requirements:

**Healthcare**:
- HIPAA compliance for PHI (Protected Health Information)
- Realistic ICD-10 codes, drug interactions, lab value ranges
- Temporal patterns in clinical events
- Patient cohort demographics matching real populations

**Finance**:
- PCI-DSS considerations for payment data
- Realistic transaction patterns (amounts, frequencies, merchant categories)
- Fraud indicators that match real attack patterns
- Regulatory stress testing scenarios

**Retail**:
- Seasonal purchasing patterns
- Product affinity and basket analysis compatibility
- Customer lifetime value distributions
- Realistic review sentiment and ratings

Building these domain rules into the core generators would:
- Bloat the codebase with domain-specific logic
- Require domain expertise from core maintainers
- Make it harder for users to customize for their specific industry

## Decision

We implement **domain-specific generators as a specialization layer** on top of base generators:

```python
from genesis.domains import HealthcareGenerator, FinanceGenerator, RetailGenerator

# Healthcare: HIPAA-aware patient data
healthcare = HealthcareGenerator()
patients = healthcare.generate_patient_cohort(
    n_patients=1000,
    include_demographics=True,
    include_conditions=True,
    hipaa_safe_mode=True  # Extra privacy controls
)

# Finance: Realistic transactions with fraud
finance = FinanceGenerator()
transactions = finance.generate_transactions(
    n_transactions=10000,
    include_fraud=True,
    fraud_rate=0.02,
    seasonal_patterns=True
)

# Retail: Complete e-commerce dataset
retail = RetailGenerator()
dataset = retail.generate_ecommerce_dataset(
    n_customers=5000,
    n_products=500,
    n_orders=20000
)
# Returns: customers, products, orders, order_items, reviews
```

Architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Domain Generators                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Healthcare  │  │   Finance    │  │    Retail    │      │
│  │  Generator   │  │  Generator   │  │  Generator   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                 │               │
│         ▼                 ▼                 ▼               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              DomainGenerator (Base)                  │   │
│  │  - domain schemas                                    │   │
│  │  - constraint templates                              │   │
│  │  - compliance rules                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                           │                                 │
└───────────────────────────┼─────────────────────────────────┘
                            ▼
                 ┌─────────────────────┐
                 │  SyntheticGenerator │
                 │  (Core Engine)      │
                 └─────────────────────┘
```

## Consequences

### Positive

- **Domain expertise encapsulated**: Healthcare experts maintain healthcare generator
- **Compliance built-in**: HIPAA/PCI-DSS rules embedded in domain generators
- **Realistic by default**: Domain patterns (seasonality, fraud signals) automatic
- **Composable**: Domain generators use core engine, inherit improvements
- **Discoverable**: CLI supports `genesis domain healthcare --help`

### Negative

- **Maintenance surface**: Each domain is essentially a separate product
- **Domain drift**: Real-world patterns change; generators need updates
- **Validation complexity**: Need domain experts to verify output realism
- **Scope creep risk**: Pressure to add more domains indefinitely

### Domain Schemas

Each domain defines schemas with semantic column types:

```python
class HealthcareSchema:
    patient_id: PatientID          # Unique, HIPAA-safe format
    birth_date: BirthDate          # Age distribution, not actual dates
    gender: Gender                 # M/F/Other with realistic distribution
    icd10_code: ICD10Code          # Valid codes with realistic frequency
    lab_value: LabResult           # Within clinically valid ranges
    medication: Medication         # Drug names with interaction awareness
```

### Compliance Modes

```python
# HIPAA Safe Mode
healthcare = HealthcareGenerator(hipaa_safe_mode=True)
# - No exact dates (uses age ranges)
# - No geographic data below state level
# - Suppresses rare conditions (k-anonymity)
# - Adds differential privacy noise

# PCI-DSS Mode
finance = FinanceGenerator(pci_compliant=True)
# - Card numbers are fake but pass Luhn check
# - No real BINs (first 6 digits)
# - CVV always synthetic
```

## Examples

```python
# Healthcare: Clinical trial cohort
from genesis.domains import HealthcareGenerator

gen = HealthcareGenerator()

# Generate patient demographics
patients = gen.generate_patient_cohort(
    n_patients=500,
    age_range=(18, 65),
    conditions=["diabetes", "hypertension"],
    include_labs=True
)

# Generate related clinical events
events = gen.generate_clinical_events(
    patient_ids=patients["patient_id"],
    event_types=["lab_result", "medication", "visit"],
    time_range_days=365
)

# Finance: Fraud detection training data
from genesis.domains import FinanceGenerator

gen = FinanceGenerator()
transactions = gen.generate_transactions(
    n_transactions=100000,
    include_fraud=True,
    fraud_rate=0.01,
    fraud_patterns=["card_testing", "account_takeover", "bust_out"]
)

# Retail: Complete e-commerce dataset
from genesis.domains import RetailGenerator

gen = RetailGenerator()
data = gen.generate_ecommerce_dataset(
    n_customers=10000,
    n_products=1000,
    n_orders=50000,
    include_reviews=True,
    seasonal_patterns=True
)

# Access individual tables
customers = data["customers"]
products = data["products"]
orders = data["orders"]
order_items = data["order_items"]
reviews = data["reviews"]
```

## CLI Integration

```bash
# Generate healthcare data
genesis domain healthcare -t patient_cohort -n 1000 -o patients.csv

# Generate financial transactions
genesis domain finance -t transactions -n 10000 --fraud-rate 0.02 -o txns.csv

# Generate retail dataset
genesis domain retail -t ecommerce --customers 5000 --products 500 -o retail/
```
