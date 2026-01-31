---
sidebar_position: 12
title: Domain Generators
---

# Domain-Specific Generators

Pre-built generators for common data types: names, addresses, emails, phone numbers, and more.

## Quick Start

```python
from genesis.domains import NameGenerator, AddressGenerator, EmailGenerator

# Generate realistic names
names = NameGenerator().generate(100)
print(names[:5])
# ['John Smith', 'Maria Garcia', 'James Wilson', 'Sarah Johnson', 'Michael Brown']

# Generate addresses
addresses = AddressGenerator(locale='en_US').generate(100)

# Generate emails
emails = EmailGenerator().generate(100)
```

## Available Generators

| Generator | Output | Locales |
|-----------|--------|---------|
| `NameGenerator` | Full names | 50+ |
| `AddressGenerator` | Street addresses | 30+ |
| `EmailGenerator` | Email addresses | All |
| `PhoneGenerator` | Phone numbers | 50+ |
| `DateGenerator` | Dates/datetimes | All |
| `SSNGenerator` | Social security numbers | US |
| `CreditCardGenerator` | Credit card numbers | All |
| `CompanyGenerator` | Company names | 20+ |

## Name Generation

```python
from genesis.domains import NameGenerator

gen = NameGenerator(locale='en_US')

# Full names
full_names = gen.generate(100)

# First names only
first_names = gen.generate(100, format='first')

# Last names only
last_names = gen.generate(100, format='last')

# With title
formal_names = gen.generate(100, format='title_full')
# ['Dr. John Smith', 'Ms. Sarah Johnson', ...]
```

### Gender-Specific

```python
# Female names
female_names = gen.generate(100, gender='female')

# Male names
male_names = gen.generate(100, gender='male')

# Mixed (default)
mixed_names = gen.generate(100, gender='any')
```

### International Names

```python
# Japanese names
jp_gen = NameGenerator(locale='ja_JP')
japanese_names = jp_gen.generate(100)

# German names
de_gen = NameGenerator(locale='de_DE')
german_names = de_gen.generate(100)

# Spanish names
es_gen = NameGenerator(locale='es_ES')
spanish_names = es_gen.generate(100)
```

## Address Generation

```python
from genesis.domains import AddressGenerator

gen = AddressGenerator(locale='en_US')

# Full addresses
addresses = gen.generate(100)
# ['123 Main St, New York, NY 10001', ...]

# Components
streets = gen.generate(100, format='street')
cities = gen.generate(100, format='city')
states = gen.generate(100, format='state')
zip_codes = gen.generate(100, format='zipcode')

# Structured output
structured = gen.generate(100, format='dict')
# [{'street': '123 Main St', 'city': 'New York', 'state': 'NY', 'zipcode': '10001'}, ...]
```

### Regional Addresses

```python
# US addresses
us_addr = AddressGenerator(locale='en_US').generate(100)

# UK addresses
uk_addr = AddressGenerator(locale='en_GB').generate(100)

# German addresses
de_addr = AddressGenerator(locale='de_DE').generate(100)
```

## Email Generation

```python
from genesis.domains import EmailGenerator

gen = EmailGenerator()

# Random emails
emails = gen.generate(100)
# ['john.smith@gmail.com', 'maria_garcia@yahoo.com', ...]

# Based on names
emails = gen.generate_from_names(names)
# Matches name format: 'John Smith' -> 'john.smith@domain.com'

# Corporate emails
corp_emails = gen.generate(100, domain='company.com')
# ['j.smith@company.com', 'm.garcia@company.com', ...]

# Specific providers
gmail_only = gen.generate(100, providers=['gmail.com'])
```

## Phone Generation

```python
from genesis.domains import PhoneGenerator

gen = PhoneGenerator(locale='en_US')

# Standard format
phones = gen.generate(100)
# ['(555) 123-4567', ...]

# Different formats
e164 = gen.generate(100, format='e164')     # +15551234567
national = gen.generate(100, format='national')  # (555) 123-4567
digits = gen.generate(100, format='digits')  # 5551234567

# Mobile only
mobiles = gen.generate(100, phone_type='mobile')
```

## Date Generation

```python
from genesis.domains import DateGenerator

gen = DateGenerator()

# Random dates
dates = gen.generate(100)

# Date range
dates = gen.generate(100, 
    start_date='2020-01-01',
    end_date='2024-12-31'
)

# Birthdates (realistic age distribution)
birthdates = gen.generate(100, 
    distribution='age',
    min_age=18,
    max_age=80
)

# Business dates (weekdays only)
business_dates = gen.generate(100, weekdays_only=True)
```

## Credit Card Generation

```python
from genesis.domains import CreditCardGenerator

gen = CreditCardGenerator()

# Generate card numbers (valid format, not real)
cards = gen.generate(100)

# Specific providers
visa_cards = gen.generate(100, provider='visa')
mastercard = gen.generate(100, provider='mastercard')

# Full card details
full_cards = gen.generate(100, format='full')
# [{'number': '4532...', 'expiry': '12/26', 'cvv': '123'}, ...]
```

⚠️ **Note**: Generated card numbers pass Luhn validation but are not real.

## Company Generation

```python
from genesis.domains import CompanyGenerator

gen = CompanyGenerator()

# Company names
companies = gen.generate(100)
# ['Smith & Associates', 'Global Tech Solutions', ...]

# With suffix
with_suffix = gen.generate(100, include_suffix=True)
# ['Smith & Associates LLC', 'Global Tech Solutions Inc', ...]

# Industry-specific
tech_companies = gen.generate(100, industry='technology')
healthcare = gen.generate(100, industry='healthcare')
```

## Composite Generation

Combine multiple generators:

```python
from genesis.domains import CompositeGenerator

gen = CompositeGenerator({
    'name': NameGenerator(locale='en_US'),
    'email': EmailGenerator(link_to='name'),  # Derives from name
    'phone': PhoneGenerator(locale='en_US'),
    'address': AddressGenerator(locale='en_US'),
    'birthdate': DateGenerator(distribution='age', min_age=18, max_age=80)
})

# Generate complete records
records = gen.generate(1000)

# Returns DataFrame with all fields
print(records.columns)
# ['name', 'email', 'phone', 'address', 'birthdate']
```

## Integration with Synthetic Data

Use domain generators to fill specific columns:

```python
from genesis import SyntheticGenerator
from genesis.domains import NameGenerator, EmailGenerator

# Generate tabular data
generator = SyntheticGenerator(method='ctgan')
generator.fit(df)
synthetic = generator.generate(1000)

# Replace identifier columns with domain generators
synthetic['name'] = NameGenerator().generate(1000)
synthetic['email'] = EmailGenerator().generate_from_names(synthetic['name'])
synthetic['phone'] = PhoneGenerator('en_US').generate(1000)
```

## Complete Example

```python
import pandas as pd
from genesis import SyntheticGenerator
from genesis.domains import (
    NameGenerator, EmailGenerator, PhoneGenerator,
    AddressGenerator, DateGenerator
)

# Load original data (with PII)
df = pd.read_csv('customers_with_pii.csv')

# Train generator on non-PII columns
non_pii_cols = ['age', 'income', 'purchase_count', 'segment']
generator = SyntheticGenerator(method='ctgan')
generator.fit(df[non_pii_cols])

# Generate synthetic non-PII data
synthetic = generator.generate(10000)

# Add synthetic PII with domain generators
synthetic['name'] = NameGenerator('en_US').generate(10000)
synthetic['email'] = EmailGenerator().generate_from_names(synthetic['name'])
synthetic['phone'] = PhoneGenerator('en_US').generate(10000)
synthetic['address'] = AddressGenerator('en_US').generate(10000)
synthetic['birthdate'] = DateGenerator().generate(
    10000, 
    distribution='age',
    min_age=18,
    max_age=80
)

# Save complete synthetic dataset
synthetic.to_csv('synthetic_customers.csv', index=False)
```

## CLI Usage

```bash
# Generate standalone domain data
genesis domain names --count 1000 --locale en_US --output names.csv
genesis domain emails --count 1000 --output emails.csv
genesis domain addresses --count 1000 --locale en_GB --output addresses.csv

# Generate composite records
genesis domain composite \
  --config composite_config.yaml \
  --count 10000 \
  --output people.csv
```

## Best Practices

1. **Match locales to your data** - US names for US customers
2. **Link related fields** - Email should match name format
3. **Use for PII replacement** - Not for entire datasets
4. **Validate formats** - Especially for regulated fields
5. **Consider cultural accuracy** - Names should match locale conventions

## Next Steps

- **[Privacy](/docs/concepts/privacy)** - Protecting real data
- **[Tabular Data](/docs/guides/tabular-data)** - Complete generation workflow
- **[Pipelines](/docs/guides/pipelines)** - Automate domain generation
