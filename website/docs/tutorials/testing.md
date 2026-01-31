---
sidebar_position: 5
title: Testing with Synthetic Data
---

# Tutorial: Testing with Synthetic Data

Create realistic test data for software testing and CI/CD pipelines.

**Time:** 25 minutes  
**Level:** Beginner  
**What you'll learn:** Test fixtures, edge cases, CI/CD integration, pytest fixtures

---

## Goal

By the end of this tutorial, you'll have:
- Created a reusable test data generator
- Generated edge cases and boundary conditions
- Integrated synthetic data into pytest
- Set up CI/CD with synthetic test data

---

## Prerequisites

```bash
pip install genesis-synth pandas pytest
```

---

## Why Synthetic Data for Testing?

| Traditional Testing | Synthetic Data Testing |
|---------------------|------------------------|
| Hand-crafted fixtures | Learned from production patterns |
| Limited edge cases | Systematic edge case generation |
| Stale test data | Fresh, realistic data |
| Copy production (risky) | Privacy-safe alternative |
| Hard to maintain | Auto-generated |

---

## Step 1: Define Your Test Schema

First, understand what you're testing. Let's test an order processing system:

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define schema for orders
order_schema = {
    'order_id': 'unique_id',
    'customer_id': 'foreign_key',
    'product_id': 'foreign_key',
    'quantity': 'positive_integer',
    'unit_price': 'positive_float',
    'discount_percent': 'percentage',
    'status': 'categorical',
    'created_at': 'datetime',
    'shipping_country': 'categorical'
}

# Sample real data to learn from (or production sample)
np.random.seed(42)
n_orders = 5000

sample_orders = pd.DataFrame({
    'order_id': range(1, n_orders + 1),
    'customer_id': np.random.randint(1, 1000, n_orders),
    'product_id': np.random.randint(1, 500, n_orders),
    'quantity': np.random.poisson(2, n_orders).clip(1, 100),
    'unit_price': np.random.lognormal(3, 1, n_orders).clip(1, 1000).round(2),
    'discount_percent': np.random.choice([0, 5, 10, 15, 20, 25], n_orders, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]),
    'status': np.random.choice(['pending', 'confirmed', 'shipped', 'delivered', 'cancelled'], n_orders, p=[0.15, 0.2, 0.25, 0.35, 0.05]),
    'created_at': [datetime.now() - timedelta(days=np.random.randint(0, 365)) for _ in range(n_orders)],
    'shipping_country': np.random.choice(['US', 'CA', 'UK', 'DE', 'FR', 'AU'], n_orders, p=[0.5, 0.15, 0.1, 0.1, 0.08, 0.07])
})

# Add calculated field
sample_orders['total_amount'] = (
    sample_orders['quantity'] * sample_orders['unit_price'] * 
    (1 - sample_orders['discount_percent'] / 100)
).round(2)

print("Sample orders:")
print(sample_orders.head())
print(f"\nSchema: {sample_orders.dtypes}")
```

---

## Step 2: Create Test Data Generator

Build a reusable generator class:

```python
from genesis import SyntheticGenerator, Constraint
from genesis.domains import DateGenerator

class OrderTestDataGenerator:
    """Generate realistic test orders."""
    
    def __init__(self, sample_data: pd.DataFrame = None):
        self.generator = SyntheticGenerator(method='ctgan', config={'epochs': 100})
        self.is_fitted = False
        
        if sample_data is not None:
            self.fit(sample_data)
    
    def fit(self, sample_data: pd.DataFrame):
        """Learn patterns from sample data."""
        # Remove IDs (will generate new ones)
        data = sample_data.drop(['order_id', 'created_at', 'total_amount'], axis=1)
        
        self.generator.fit(
            data,
            discrete_columns=['status', 'shipping_country', 'discount_percent'],
            constraints=[
                Constraint.positive('quantity'),
                Constraint.positive('unit_price'),
                Constraint.range('discount_percent', 0, 50),
            ]
        )
        self.is_fitted = True
        return self
    
    def generate(self, n: int = 100, start_id: int = 1) -> pd.DataFrame:
        """Generate n test orders."""
        if not self.is_fitted:
            raise RuntimeError("Generator not fitted. Call fit() first.")
        
        # Generate base data
        orders = self.generator.generate(n_samples=n)
        
        # Add IDs
        orders.insert(0, 'order_id', range(start_id, start_id + n))
        
        # Add timestamps
        orders['created_at'] = DateGenerator().generate(
            n, start='2024-01-01', end='2024-12-31'
        )
        
        # Calculate total
        orders['total_amount'] = (
            orders['quantity'] * orders['unit_price'] * 
            (1 - orders['discount_percent'] / 100)
        ).round(2)
        
        return orders
    
    def generate_edge_cases(self) -> pd.DataFrame:
        """Generate specific edge cases for testing."""
        edge_cases = []
        
        # Minimum quantity
        edge_cases.append({
            'order_id': 'EDGE001',
            'customer_id': 1,
            'product_id': 1,
            'quantity': 1,
            'unit_price': 0.01,
            'discount_percent': 0,
            'status': 'pending',
            'shipping_country': 'US',
        })
        
        # Maximum discount
        edge_cases.append({
            'order_id': 'EDGE002',
            'customer_id': 2,
            'product_id': 2,
            'quantity': 1,
            'unit_price': 100.00,
            'discount_percent': 50,
            'status': 'confirmed',
            'shipping_country': 'CA',
        })
        
        # High quantity order
        edge_cases.append({
            'order_id': 'EDGE003',
            'customer_id': 3,
            'product_id': 3,
            'quantity': 999,
            'unit_price': 10.00,
            'discount_percent': 0,
            'status': 'pending',
            'shipping_country': 'UK',
        })
        
        # Cancelled order
        edge_cases.append({
            'order_id': 'EDGE004',
            'customer_id': 4,
            'product_id': 4,
            'quantity': 5,
            'unit_price': 50.00,
            'discount_percent': 10,
            'status': 'cancelled',
            'shipping_country': 'DE',
        })
        
        df = pd.DataFrame(edge_cases)
        df['created_at'] = datetime.now()
        df['total_amount'] = (
            df['quantity'] * df['unit_price'] * 
            (1 - df['discount_percent'] / 100)
        ).round(2)
        
        return df
    
    def save(self, path: str):
        """Save generator for reuse."""
        self.generator.save(path)
    
    @classmethod
    def load(cls, path: str):
        """Load saved generator."""
        instance = cls()
        instance.generator = SyntheticGenerator.load(path)
        instance.is_fitted = True
        return instance

# Create and train generator
test_gen = OrderTestDataGenerator(sample_orders)

# Generate test data
test_orders = test_gen.generate(100)
print("Generated test orders:")
print(test_orders.head())
```

---

## Step 3: Pytest Integration

Create pytest fixtures for synthetic test data:

```python
# tests/conftest.py
import pytest
import pandas as pd
from pathlib import Path

# Assuming OrderTestDataGenerator is in test_utils.py
# from test_utils import OrderTestDataGenerator

@pytest.fixture(scope="session")
def order_generator():
    """Session-scoped generator (trained once)."""
    # Load pre-trained generator or train from sample
    generator_path = Path("tests/fixtures/order_generator.pkl")
    
    if generator_path.exists():
        return OrderTestDataGenerator.load(str(generator_path))
    else:
        # Train from sample data
        sample = pd.read_csv("tests/fixtures/sample_orders.csv")
        gen = OrderTestDataGenerator(sample)
        gen.save(str(generator_path))
        return gen

@pytest.fixture
def sample_orders(order_generator):
    """Generate fresh sample orders for each test."""
    return order_generator.generate(n=50)

@pytest.fixture
def edge_case_orders(order_generator):
    """Get edge case orders."""
    return order_generator.generate_edge_cases()

@pytest.fixture
def large_order_set(order_generator):
    """Generate large dataset for performance tests."""
    return order_generator.generate(n=10000)
```

---

## Step 4: Write Tests with Synthetic Data

```python
# tests/test_order_processing.py
import pytest
import pandas as pd

# Import your application code
# from myapp.orders import OrderProcessor, calculate_total, validate_order

class TestOrderCalculations:
    """Test order calculations with synthetic data."""
    
    def test_total_calculation(self, sample_orders):
        """Test that totals are calculated correctly."""
        for _, order in sample_orders.iterrows():
            expected = order['quantity'] * order['unit_price'] * (1 - order['discount_percent'] / 100)
            assert abs(order['total_amount'] - expected) < 0.01
    
    def test_discount_applied(self, sample_orders):
        """Test discounts are applied correctly."""
        discounted = sample_orders[sample_orders['discount_percent'] > 0]
        for _, order in discounted.iterrows():
            full_price = order['quantity'] * order['unit_price']
            assert order['total_amount'] < full_price
    
    def test_edge_case_minimum_order(self, edge_case_orders):
        """Test minimum order handling."""
        min_order = edge_case_orders[edge_case_orders['order_id'] == 'EDGE001'].iloc[0]
        assert min_order['total_amount'] == 0.01
    
    def test_edge_case_max_discount(self, edge_case_orders):
        """Test maximum discount handling."""
        max_discount = edge_case_orders[edge_case_orders['order_id'] == 'EDGE002'].iloc[0]
        assert max_discount['total_amount'] == 50.00  # 50% off $100

class TestOrderValidation:
    """Test order validation with realistic data."""
    
    def test_all_orders_have_required_fields(self, sample_orders):
        """Test all required fields are present."""
        required = ['order_id', 'customer_id', 'quantity', 'unit_price', 'status']
        for field in required:
            assert field in sample_orders.columns
            assert sample_orders[field].notna().all()
    
    def test_quantity_is_positive(self, sample_orders):
        """Test all quantities are positive."""
        assert (sample_orders['quantity'] > 0).all()
    
    def test_price_is_positive(self, sample_orders):
        """Test all prices are positive."""
        assert (sample_orders['unit_price'] > 0).all()
    
    def test_discount_in_valid_range(self, sample_orders):
        """Test discounts are within valid range."""
        assert (sample_orders['discount_percent'] >= 0).all()
        assert (sample_orders['discount_percent'] <= 100).all()
    
    def test_status_is_valid(self, sample_orders):
        """Test all statuses are valid."""
        valid_statuses = {'pending', 'confirmed', 'shipped', 'delivered', 'cancelled'}
        assert set(sample_orders['status'].unique()).issubset(valid_statuses)

class TestPerformance:
    """Performance tests with large synthetic datasets."""
    
    def test_process_large_batch(self, large_order_set):
        """Test processing performance with 10k orders."""
        # Your batch processing logic here
        assert len(large_order_set) == 10000
        
        # Example: Test processing time
        import time
        start = time.time()
        
        # Process orders (your logic)
        processed = large_order_set.copy()
        processed['processed'] = True
        
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Processing took too long: {elapsed:.2f}s"
```

Run tests:
```bash
pytest tests/ -v
```

---

## Step 5: Generate Edge Cases Systematically

```python
from genesis import ConditionalGenerator

class EdgeCaseGenerator:
    """Generate systematic edge cases."""
    
    def __init__(self, generator: SyntheticGenerator):
        self.generator = generator
    
    def boundary_values(self, column: str, boundaries: list) -> pd.DataFrame:
        """Generate records at boundary values."""
        edge_cases = []
        
        for boundary in boundaries:
            # Generate records near boundary
            conditions = {column: boundary}
            cases = self.generator.generate(n_samples=5)
            cases[column] = boundary
            edge_cases.append(cases)
        
        return pd.concat(edge_cases, ignore_index=True)
    
    def null_handling(self, columns: list) -> pd.DataFrame:
        """Generate records with null values for testing."""
        base = self.generator.generate(n_samples=len(columns) * 3)
        
        for i, col in enumerate(columns):
            # Set some values to null
            base.loc[i*3:(i+1)*3-1, col] = None
        
        return base
    
    def extreme_combinations(self) -> pd.DataFrame:
        """Generate extreme value combinations."""
        return pd.DataFrame([
            # All minimum values
            {'quantity': 1, 'unit_price': 0.01, 'discount_percent': 0},
            # All maximum reasonable values
            {'quantity': 1000, 'unit_price': 9999.99, 'discount_percent': 50},
            # Mixed extremes
            {'quantity': 1, 'unit_price': 9999.99, 'discount_percent': 50},
            {'quantity': 1000, 'unit_price': 0.01, 'discount_percent': 0},
        ])

# Usage
edge_gen = EdgeCaseGenerator(test_gen.generator)
boundary_cases = edge_gen.boundary_values('quantity', [1, 10, 100, 1000])
```

---

## Step 6: CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/test.yml
name: Tests with Synthetic Data

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install genesis-synth pytest pytest-cov
      
      - name: Generate test data
        run: |
          python scripts/generate_test_data.py
      
      - name: Run tests
        run: |
          pytest tests/ -v --cov=myapp --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Test Data Generation Script

```python
# scripts/generate_test_data.py
"""Generate test data for CI/CD."""
import pandas as pd
from pathlib import Path
from genesis import SyntheticGenerator

def generate_test_fixtures():
    """Generate all test fixtures."""
    fixtures_dir = Path("tests/fixtures")
    fixtures_dir.mkdir(exist_ok=True)
    
    # Load sample data (checked into repo or downloaded)
    sample = pd.read_csv("data/sample_orders.csv")
    
    # Train generator
    gen = SyntheticGenerator(method='ctgan', config={'epochs': 100})
    gen.fit(sample, discrete_columns=['status', 'shipping_country'])
    
    # Generate fixtures
    fixtures = {
        'small': gen.generate(100),
        'medium': gen.generate(1000),
        'large': gen.generate(10000),
    }
    
    # Save fixtures
    for name, data in fixtures.items():
        data.to_csv(fixtures_dir / f"orders_{name}.csv", index=False)
        print(f"Generated {name}: {len(data)} records")
    
    # Save generator
    gen.save(str(fixtures_dir / "order_generator.pkl"))
    print("Saved generator")

if __name__ == "__main__":
    generate_test_fixtures()
```

---

## Step 7: Database Testing

Generate data for database tests:

```python
from genesis import MultiTableGenerator

class DatabaseTestFixtures:
    """Generate related tables for database testing."""
    
    def __init__(self):
        self.generator = None
    
    def fit(self, tables: dict, relationships: list):
        """Learn from sample database."""
        self.generator = MultiTableGenerator()
        self.generator.fit(tables, relationships)
    
    def generate(self, scale: float = 1.0) -> dict:
        """Generate complete database fixture."""
        return self.generator.generate(scale=scale)
    
    def to_sql(self, tables: dict, connection):
        """Insert fixtures into test database."""
        for name, df in tables.items():
            df.to_sql(name, connection, if_exists='replace', index=False)

# Usage
db_fixtures = DatabaseTestFixtures()
db_fixtures.fit(
    tables={
        'customers': customers_df,
        'orders': orders_df,
        'order_items': items_df
    },
    relationships=[
        ('orders', 'customer_id', 'customers', 'id'),
        ('order_items', 'order_id', 'orders', 'id')
    ]
)

# Generate test database
test_db = db_fixtures.generate(scale=0.1)  # 10% of original size

# Insert into test database
import sqlite3
conn = sqlite3.connect(':memory:')
db_fixtures.to_sql(test_db, conn)
```

---

## Complete Test Setup

```python
"""
tests/conftest.py - Complete test configuration
"""
import pytest
import pandas as pd
from pathlib import Path
from genesis import SyntheticGenerator

# Paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
GENERATOR_PATH = FIXTURES_DIR / "order_generator.pkl"
SAMPLE_PATH = FIXTURES_DIR / "sample_orders.csv"

@pytest.fixture(scope="session")
def generator():
    """Load or create test data generator."""
    if GENERATOR_PATH.exists():
        return SyntheticGenerator.load(str(GENERATOR_PATH))
    
    # Create from sample
    sample = pd.read_csv(SAMPLE_PATH)
    gen = SyntheticGenerator(method='gaussian_copula')  # Fast for testing
    gen.fit(sample.drop(['order_id', 'created_at'], axis=1),
            discrete_columns=['status', 'shipping_country'])
    gen.save(str(GENERATOR_PATH))
    return gen

@pytest.fixture
def orders(generator):
    """Generate fresh orders for each test."""
    return generator.generate(50)

@pytest.fixture
def single_order(generator):
    """Generate single order."""
    return generator.generate(1).iloc[0]

@pytest.fixture
def empty_orders():
    """Empty DataFrame with correct schema."""
    return pd.DataFrame(columns=[
        'customer_id', 'product_id', 'quantity', 
        'unit_price', 'discount_percent', 'status', 'shipping_country'
    ])

# Parametrized fixtures for different scenarios
@pytest.fixture(params=['pending', 'confirmed', 'shipped', 'delivered', 'cancelled'])
def order_by_status(request, generator):
    """Generate order with specific status."""
    order = generator.generate(1).iloc[0].to_dict()
    order['status'] = request.param
    return order

@pytest.fixture(params=[0, 10, 25, 50])
def order_by_discount(request, generator):
    """Generate order with specific discount."""
    order = generator.generate(1).iloc[0].to_dict()
    order['discount_percent'] = request.param
    return order
```

---

## Best Practices

1. **Train once, generate many** - Use session-scoped fixtures
2. **Version your generators** - Save trained models to repo/artifacts
3. **Include edge cases** - Don't rely only on random generation
4. **Validate generated data** - Ensure constraints are met
5. **Keep fixtures small** - Generate what you need, not more
6. **Use fast methods for CI** - `gaussian_copula` is faster than `ctgan`

---

## Next Steps

- [Customer Data Tutorial](/docs/tutorials/customer-data) - Full synthesis walkthrough
- [Pipeline API](/docs/guides/pipelines) - Automate test data generation
- [Examples](/docs/examples) - More testing patterns
