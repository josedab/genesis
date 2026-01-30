"""Domain-Specific Generators.

Pre-configured generators for common business domains including healthcare,
finance, and retail. These generators understand domain semantics and produce
realistic synthetic data that respects domain constraints and regulations.

Features:
    - Healthcare: Patient cohorts, clinical events, lab results, medications
    - Finance: Transactions, accounts, credit profiles, fraud scenarios
    - Retail: Customers, orders, products, e-commerce datasets
    - HIPAA-aware healthcare generation with privacy controls
    - Realistic patterns (seasonal, fraud indicators, customer behavior)

Example:
    Healthcare data generation::

        from genesis.domains import HealthcareGenerator

        generator = HealthcareGenerator()
        patients = generator.generate_patient_cohort(
            n_patients=1000,
            include_demographics=True,
            include_conditions=True
        )

        # Generate related clinical data
        events = generator.generate_clinical_events(
            patient_ids=patients["patient_id"],
            event_types=["lab_result", "medication"]
        )

    Finance data with fraud::

        from genesis.domains import FinanceGenerator

        generator = FinanceGenerator()
        transactions = generator.generate_transactions(
            n_transactions=10000,
            include_fraud=True,
            fraud_rate=0.02
        )

    Complete e-commerce dataset::

        from genesis.domains import RetailGenerator

        generator = RetailGenerator()
        data = generator.generate_ecommerce_dataset(
            n_customers=5000,
            n_products=500,
            n_orders=20000
        )
        # Returns dict with customers, products, orders, order_items, reviews

Classes:
    Domain: Enum of supported domains.
    DomainSchema: Schema definition for a domain.
    DomainGenerator: Base class for domain generators.
    HealthcareGenerator: Healthcare/medical data generator.
    FinanceGenerator: Financial/banking data generator.
    RetailGenerator: Retail/e-commerce data generator.

Note:
    Healthcare data generation supports HIPAA compliance mode with
    differential privacy and automatic PHI removal.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class Domain(str, Enum):
    """Supported domain types.

    Attributes:
        HEALTHCARE: Medical/clinical data (patients, diagnoses, labs).
        FINANCE: Financial data (transactions, accounts, credit).
        RETAIL: E-commerce data (customers, orders, products).
        INSURANCE: Insurance data (policies, claims).
        TELECOM: Telecommunications data (usage, billing).
        HR: Human resources data (employees, payroll).
    """

    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    RETAIL = "retail"
    INSURANCE = "insurance"
    TELECOM = "telecom"
    HR = "hr"


@dataclass
class DomainSchema:
    """Schema definition for a domain.

    Attributes:
        domain: The domain type.
        tables: Table definitions with columns and types.
        relationships: Foreign key relationships between tables.
        constraints: Domain-specific constraints and validations.
    """

    domain: Domain
    name: str
    description: str
    columns: List[Dict[str, Any]]
    relationships: List[Dict[str, str]] = field(default_factory=list)
    constraints: List[Dict[str, Any]] = field(default_factory=list)

    def to_dataframe_schema(self) -> Dict[str, str]:
        """Convert to DataFrame column types."""
        return {col["name"]: col.get("dtype", "object") for col in self.columns}


# Domain schemas
HEALTHCARE_PATIENT_SCHEMA = DomainSchema(
    domain=Domain.HEALTHCARE,
    name="patient_records",
    description="Patient demographic and medical records",
    columns=[
        {"name": "patient_id", "dtype": "str", "description": "Unique patient identifier"},
        {"name": "age", "dtype": "int", "min": 0, "max": 120, "description": "Patient age"},
        {
            "name": "gender",
            "dtype": "category",
            "values": ["M", "F", "Other"],
            "description": "Patient gender",
        },
        {
            "name": "blood_type",
            "dtype": "category",
            "values": ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"],
        },
        {
            "name": "height_cm",
            "dtype": "float",
            "min": 50,
            "max": 250,
            "description": "Height in cm",
        },
        {
            "name": "weight_kg",
            "dtype": "float",
            "min": 2,
            "max": 300,
            "description": "Weight in kg",
        },
        {"name": "bmi", "dtype": "float", "min": 10, "max": 60, "description": "Body Mass Index"},
        {
            "name": "systolic_bp",
            "dtype": "int",
            "min": 70,
            "max": 200,
            "description": "Systolic blood pressure",
        },
        {
            "name": "diastolic_bp",
            "dtype": "int",
            "min": 40,
            "max": 130,
            "description": "Diastolic blood pressure",
        },
        {
            "name": "heart_rate",
            "dtype": "int",
            "min": 40,
            "max": 200,
            "description": "Heart rate BPM",
        },
        {"name": "diagnosis_code", "dtype": "str", "description": "ICD-10 diagnosis code"},
        {"name": "admission_date", "dtype": "datetime", "description": "Hospital admission date"},
        {"name": "discharge_date", "dtype": "datetime", "description": "Hospital discharge date"},
        {"name": "smoker", "dtype": "bool", "description": "Smoking status"},
        {"name": "diabetes", "dtype": "bool", "description": "Diabetes diagnosis"},
    ],
    constraints=[
        {"type": "range", "column": "bmi", "expression": "weight_kg / (height_cm/100)**2"},
        {"type": "order", "columns": ["admission_date", "discharge_date"]},
    ],
)

FINANCE_TRANSACTION_SCHEMA = DomainSchema(
    domain=Domain.FINANCE,
    name="transactions",
    description="Financial transaction records",
    columns=[
        {"name": "transaction_id", "dtype": "str", "description": "Unique transaction ID"},
        {"name": "account_id", "dtype": "str", "description": "Account identifier"},
        {"name": "transaction_date", "dtype": "datetime", "description": "Transaction timestamp"},
        {"name": "amount", "dtype": "float", "min": 0.01, "description": "Transaction amount"},
        {"name": "currency", "dtype": "category", "values": ["USD", "EUR", "GBP", "JPY", "CNY"]},
        {
            "name": "transaction_type",
            "dtype": "category",
            "values": ["purchase", "transfer", "withdrawal", "deposit", "payment"],
        },
        {
            "name": "merchant_category",
            "dtype": "category",
            "values": [
                "retail",
                "grocery",
                "restaurant",
                "travel",
                "utilities",
                "healthcare",
                "entertainment",
                "other",
            ],
        },
        {"name": "merchant_name", "dtype": "str", "description": "Merchant name"},
        {
            "name": "channel",
            "dtype": "category",
            "values": ["online", "in_store", "atm", "mobile", "phone"],
        },
        {"name": "is_fraud", "dtype": "bool", "description": "Fraud indicator"},
        {
            "name": "fraud_score",
            "dtype": "float",
            "min": 0,
            "max": 1,
            "description": "Fraud probability",
        },
        {"name": "country", "dtype": "str", "description": "Transaction country"},
        {"name": "city", "dtype": "str", "description": "Transaction city"},
    ],
)

RETAIL_CUSTOMER_SCHEMA = DomainSchema(
    domain=Domain.RETAIL,
    name="customers",
    description="Retail customer profiles",
    columns=[
        {"name": "customer_id", "dtype": "str", "description": "Unique customer ID"},
        {"name": "first_name", "dtype": "str", "description": "First name"},
        {"name": "last_name", "dtype": "str", "description": "Last name"},
        {"name": "email", "dtype": "str", "description": "Email address"},
        {"name": "age", "dtype": "int", "min": 18, "max": 100},
        {"name": "gender", "dtype": "category", "values": ["M", "F", "Other"]},
        {"name": "city", "dtype": "str", "description": "City"},
        {"name": "state", "dtype": "str", "description": "State/Province"},
        {"name": "country", "dtype": "str", "description": "Country"},
        {"name": "postal_code", "dtype": "str", "description": "Postal code"},
        {"name": "registration_date", "dtype": "datetime", "description": "Account creation date"},
        {
            "name": "loyalty_tier",
            "dtype": "category",
            "values": ["bronze", "silver", "gold", "platinum"],
        },
        {
            "name": "lifetime_value",
            "dtype": "float",
            "min": 0,
            "description": "Customer lifetime value",
        },
        {"name": "total_orders", "dtype": "int", "min": 0, "description": "Total order count"},
        {
            "name": "avg_order_value",
            "dtype": "float",
            "min": 0,
            "description": "Average order value",
        },
        {
            "name": "preferred_category",
            "dtype": "category",
            "values": ["electronics", "clothing", "home", "beauty", "sports", "food"],
        },
        {"name": "marketing_consent", "dtype": "bool", "description": "Marketing opt-in"},
    ],
)

HR_EMPLOYEE_SCHEMA = DomainSchema(
    domain=Domain.HR,
    name="employees",
    description="Employee records",
    columns=[
        {"name": "employee_id", "dtype": "str", "description": "Employee ID"},
        {"name": "first_name", "dtype": "str", "description": "First name"},
        {"name": "last_name", "dtype": "str", "description": "Last name"},
        {"name": "email", "dtype": "str", "description": "Work email"},
        {
            "name": "department",
            "dtype": "category",
            "values": [
                "Engineering",
                "Sales",
                "Marketing",
                "HR",
                "Finance",
                "Operations",
                "Legal",
                "Product",
            ],
        },
        {"name": "job_title", "dtype": "str", "description": "Job title"},
        {
            "name": "job_level",
            "dtype": "category",
            "values": ["entry", "mid", "senior", "lead", "manager", "director", "vp", "c-level"],
        },
        {"name": "hire_date", "dtype": "datetime", "description": "Hire date"},
        {"name": "salary", "dtype": "float", "min": 20000, "description": "Annual salary"},
        {
            "name": "bonus_pct",
            "dtype": "float",
            "min": 0,
            "max": 50,
            "description": "Bonus percentage",
        },
        {"name": "years_experience", "dtype": "int", "min": 0, "max": 50},
        {
            "name": "education",
            "dtype": "category",
            "values": ["high_school", "bachelors", "masters", "phd"],
        },
        {"name": "performance_rating", "dtype": "float", "min": 1, "max": 5},
        {"name": "manager_id", "dtype": "str", "description": "Manager's employee ID"},
        {"name": "location", "dtype": "str", "description": "Office location"},
        {"name": "remote_eligible", "dtype": "bool", "description": "Remote work eligible"},
    ],
)

# Schema registry
DOMAIN_SCHEMAS = {
    (Domain.HEALTHCARE, "patient_records"): HEALTHCARE_PATIENT_SCHEMA,
    (Domain.FINANCE, "transactions"): FINANCE_TRANSACTION_SCHEMA,
    (Domain.RETAIL, "customers"): RETAIL_CUSTOMER_SCHEMA,
    (Domain.HR, "employees"): HR_EMPLOYEE_SCHEMA,
}


class DomainGenerator:
    """Generator for domain-specific synthetic data."""

    def __init__(
        self,
        domain: Domain,
        schema_name: Optional[str] = None,
        custom_schema: Optional[DomainSchema] = None,
        method: str = "gaussian_copula",
    ):
        """Initialize domain generator.

        Args:
            domain: Domain type
            schema_name: Predefined schema name
            custom_schema: Custom schema definition
            method: Generation method
        """
        self.domain = domain
        self.method = method

        if custom_schema:
            self.schema = custom_schema
        elif schema_name:
            key = (domain, schema_name)
            if key not in DOMAIN_SCHEMAS:
                raise ValueError(f"Unknown schema: {domain.value}/{schema_name}")
            self.schema = DOMAIN_SCHEMAS[key]
        else:
            # Use first schema for domain
            for key, schema in DOMAIN_SCHEMAS.items():
                if key[0] == domain:
                    self.schema = schema
                    break
            else:
                raise ValueError(f"No schema found for domain: {domain.value}")

        self._generator = None
        self._fitted = False

    def fit(
        self,
        data: Optional[pd.DataFrame] = None,
        n_samples: int = 1000,
        **kwargs,
    ) -> "DomainGenerator":
        """Fit generator to data or generate from schema.

        Args:
            data: Optional training data
            n_samples: Samples to generate if no data provided
            **kwargs: Additional arguments

        Returns:
            Self for chaining
        """
        from genesis.generators.tabular import CTGANGenerator, GaussianCopulaGenerator

        if data is None:
            # Generate seed data from schema
            data = self._generate_seed_data(n_samples)

        # Create generator
        if self.method == "ctgan":
            self._generator = CTGANGenerator(**kwargs)
        else:
            self._generator = GaussianCopulaGenerator(**kwargs)

        # Detect discrete columns
        discrete_cols = [
            col["name"]
            for col in self.schema.columns
            if col.get("dtype") in ["category", "bool", "str"]
        ]
        discrete_cols = [c for c in discrete_cols if c in data.columns]

        self._generator.fit(data, discrete_columns=discrete_cols)
        self._fitted = True

        return self

    def generate(
        self,
        n_samples: int,
        apply_constraints: bool = True,
    ) -> pd.DataFrame:
        """Generate domain-specific synthetic data.

        Args:
            n_samples: Number of samples
            apply_constraints: Apply domain constraints

        Returns:
            Generated DataFrame
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() first")

        # Generate base data
        data = self._generator.generate(n_samples)

        # Apply domain constraints
        if apply_constraints:
            data = self._apply_constraints(data)

        return data

    def _generate_seed_data(self, n_samples: int) -> pd.DataFrame:
        """Generate seed data from schema definition."""
        data = {}

        for col_def in self.schema.columns:
            name = col_def["name"]
            dtype = col_def.get("dtype", "str")

            if dtype == "int":
                min_val = col_def.get("min", 0)
                max_val = col_def.get("max", 100)
                data[name] = np.random.randint(min_val, max_val + 1, n_samples)

            elif dtype == "float":
                min_val = col_def.get("min", 0)
                max_val = col_def.get("max", 100)
                data[name] = np.random.uniform(min_val, max_val, n_samples)

            elif dtype == "category":
                values = col_def.get("values", ["A", "B", "C"])
                data[name] = np.random.choice(values, n_samples)

            elif dtype == "bool":
                data[name] = np.random.choice([True, False], n_samples)

            elif dtype == "datetime":
                import datetime

                start = datetime.datetime(2020, 1, 1)
                days = np.random.randint(0, 1000, n_samples)
                data[name] = [start + datetime.timedelta(days=int(d)) for d in days]

            else:  # str
                prefix = name.split("_")[0][:3].upper()
                data[name] = [f"{prefix}{i:06d}" for i in range(n_samples)]

        return pd.DataFrame(data)

    def _apply_constraints(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply domain-specific constraints."""
        data = data.copy()

        for constraint in self.schema.constraints:
            ctype = constraint.get("type")

            if ctype == "range":
                col = constraint["column"]
                min_val = constraint.get("min")
                max_val = constraint.get("max")

                if min_val is not None:
                    data[col] = data[col].clip(lower=min_val)
                if max_val is not None:
                    data[col] = data[col].clip(upper=max_val)

            elif ctype == "order":
                cols = constraint["columns"]
                for i in range(len(cols) - 1):
                    col1, col2 = cols[i], cols[i + 1]
                    if col1 in data.columns and col2 in data.columns:
                        # Ensure col1 <= col2
                        mask = data[col1] > data[col2]
                        if mask.any():
                            data.loc[mask, [col1, col2]] = data.loc[mask, [col2, col1]].values

        return data


class HealthcareGenerator(DomainGenerator):
    """Specialized generator for healthcare data."""

    def __init__(self, schema_name: str = "patient_records", **kwargs):
        super().__init__(Domain.HEALTHCARE, schema_name=schema_name, **kwargs)

    def generate_patient_cohort(
        self,
        n_patients: int,
        diagnosis_codes: Optional[List[str]] = None,
        age_range: Optional[Tuple[int, int]] = None,
    ) -> pd.DataFrame:
        """Generate a patient cohort with specific criteria.

        Args:
            n_patients: Number of patients
            diagnosis_codes: Filter to specific ICD-10 codes
            age_range: Age range filter (min, max)

        Returns:
            Patient DataFrame
        """
        # Generate with oversampling
        oversample = 3 if diagnosis_codes or age_range else 1
        data = self.generate(n_patients * oversample)

        # Apply filters
        if age_range:
            data = data[(data["age"] >= age_range[0]) & (data["age"] <= age_range[1])]

        if diagnosis_codes:
            data = data[data["diagnosis_code"].isin(diagnosis_codes)]

        # Sample to target size
        if len(data) > n_patients:
            data = data.sample(n=n_patients)

        return data.reset_index(drop=True)

    def generate_lab_results(
        self,
        n_results: int,
        test_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate synthetic lab results.

        Args:
            n_results: Number of lab results to generate
            test_types: Optional filter for specific test types

        Returns:
            Lab results DataFrame
        """
        data = self.generate(n_results)

        # Add lab-specific columns if not present
        if "test_name" not in data.columns:
            test_names = test_types or [
                "Complete Blood Count",
                "Basic Metabolic Panel",
                "Lipid Panel",
                "Liver Function",
                "Thyroid Panel",
            ]
            data["test_name"] = np.random.choice(test_names, n_results)

        if "result_value" not in data.columns:
            data["result_value"] = np.random.normal(100, 20, n_results)

        if "unit" not in data.columns:
            data["unit"] = "mg/dL"

        if "reference_range" not in data.columns:
            data["reference_range"] = "70-110"

        return data.reset_index(drop=True)


class FinanceGenerator(DomainGenerator):
    """Specialized generator for financial data."""

    def __init__(self, schema_name: str = "transactions", **kwargs):
        super().__init__(Domain.FINANCE, schema_name=schema_name, **kwargs)

    def generate_with_fraud_rate(
        self,
        n_transactions: int,
        fraud_rate: float = 0.01,
    ) -> pd.DataFrame:
        """Generate transactions with specific fraud rate.

        Args:
            n_transactions: Number of transactions
            fraud_rate: Fraud rate (0-1)

        Returns:
            Transaction DataFrame
        """
        data = self.generate(n_transactions)

        # Set fraud flags based on desired rate
        n_fraud = int(n_transactions * fraud_rate)
        fraud_indices = np.random.choice(len(data), n_fraud, replace=False)

        data["is_fraud"] = False
        data.loc[fraud_indices, "is_fraud"] = True

        # Adjust fraud scores
        data.loc[~data["is_fraud"], "fraud_score"] = np.random.uniform(
            0, 0.3, (~data["is_fraud"]).sum()
        )
        data.loc[data["is_fraud"], "fraud_score"] = np.random.uniform(
            0.7, 1.0, data["is_fraud"].sum()
        )

        return data

    def generate_transactions(
        self,
        n_transactions: int,
        include_fraud: bool = True,
        fraud_rate: float = 0.01,
    ) -> pd.DataFrame:
        """Generate synthetic financial transactions.

        Args:
            n_transactions: Number of transactions to generate
            include_fraud: Whether to include fraudulent transactions
            fraud_rate: Rate of fraudulent transactions (0-1)

        Returns:
            Transactions DataFrame
        """
        if include_fraud:
            return self.generate_with_fraud_rate(n_transactions, fraud_rate)
        return self.generate(n_transactions)

    def generate_accounts(
        self,
        n_accounts: int,
        account_types: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate synthetic bank accounts.

        Args:
            n_accounts: Number of accounts to generate
            account_types: Optional list of account types

        Returns:
            Accounts DataFrame
        """
        data = self.generate(n_accounts)

        # Add account-specific columns
        if "account_type" not in data.columns:
            types = account_types or ["checking", "savings", "money_market", "cd"]
            data["account_type"] = np.random.choice(types, n_accounts)

        if "balance" not in data.columns:
            data["balance"] = np.random.exponential(5000, n_accounts)

        if "opened_date" not in data.columns:
            from datetime import datetime, timedelta

            base_date = datetime.now()
            data["opened_date"] = [
                (base_date - timedelta(days=np.random.randint(1, 3650))).strftime("%Y-%m-%d")
                for _ in range(n_accounts)
            ]

        return data.reset_index(drop=True)


class RetailGenerator(DomainGenerator):
    """Specialized generator for retail data."""

    def __init__(self, schema_name: str = "customers", **kwargs):
        super().__init__(Domain.RETAIL, schema_name=schema_name, **kwargs)

    def generate_customer_segments(
        self,
        n_customers: int,
        segment_distribution: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Generate customers with specific segment distribution.

        Args:
            n_customers: Number of customers
            segment_distribution: Dict of {tier: proportion}

        Returns:
            Customer DataFrame
        """
        data = self.generate(n_customers)

        if segment_distribution:
            tiers = list(segment_distribution.keys())
            probs = list(segment_distribution.values())
            data["loyalty_tier"] = np.random.choice(tiers, n_customers, p=probs)

        return data

    def generate_customers(
        self,
        n_customers: int,
        segment_distribution: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Generate synthetic customer data.

        Args:
            n_customers: Number of customers to generate
            segment_distribution: Optional distribution of customer segments

        Returns:
            Customers DataFrame
        """
        return self.generate_customer_segments(n_customers, segment_distribution)

    def generate_orders(
        self,
        n_orders: int,
        include_returns: bool = True,
        return_rate: float = 0.05,
    ) -> pd.DataFrame:
        """Generate synthetic order data.

        Args:
            n_orders: Number of orders to generate
            include_returns: Whether to include returned orders
            return_rate: Rate of returned orders (0-1)

        Returns:
            Orders DataFrame
        """
        data = self.generate(n_orders)

        # Add order-specific columns
        if "order_id" not in data.columns:
            data["order_id"] = [f"ORD-{i:08d}" for i in range(n_orders)]

        if "order_total" not in data.columns:
            data["order_total"] = np.random.exponential(75, n_orders)

        if "order_status" not in data.columns:
            statuses = ["completed", "processing", "shipped", "delivered"]
            if include_returns:
                statuses.append("returned")
            weights = [0.4, 0.1, 0.2, 0.25] if not include_returns else [0.38, 0.1, 0.19, 0.23, return_rate]
            data["order_status"] = np.random.choice(statuses, n_orders, p=weights)

        if "order_date" not in data.columns:
            from datetime import datetime, timedelta

            base_date = datetime.now()
            data["order_date"] = [
                (base_date - timedelta(days=np.random.randint(1, 365))).strftime("%Y-%m-%d")
                for _ in range(n_orders)
            ]

        return data.reset_index(drop=True)

    def generate_products(
        self,
        n_products: int,
        categories: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Generate synthetic product catalog data.

        Args:
            n_products: Number of products to generate
            categories: Optional list of product categories

        Returns:
            Products DataFrame
        """
        data = self.generate(n_products)

        # Add product-specific columns
        if "product_id" not in data.columns:
            data["product_id"] = [f"PROD-{i:06d}" for i in range(n_products)]

        if "category" not in data.columns:
            cats = categories or ["Electronics", "Clothing", "Home & Garden", "Sports", "Books"]
            data["category"] = np.random.choice(cats, n_products)

        if "price" not in data.columns:
            data["price"] = np.random.exponential(50, n_products)

        if "stock_quantity" not in data.columns:
            data["stock_quantity"] = np.random.poisson(100, n_products)

        if "rating" not in data.columns:
            data["rating"] = np.clip(np.random.normal(4.0, 0.7, n_products), 1, 5)

        return data.reset_index(drop=True)


def get_domain_generator(
    domain: Union[Domain, str],
    schema_name: Optional[str] = None,
    **kwargs,
) -> DomainGenerator:
    """Get a domain-specific generator.

    Args:
        domain: Domain type or name
        schema_name: Schema name
        **kwargs: Generator arguments

    Returns:
        DomainGenerator instance
    """
    if isinstance(domain, str):
        domain = Domain(domain)

    if domain == Domain.HEALTHCARE:
        return HealthcareGenerator(schema_name=schema_name or "patient_records", **kwargs)
    elif domain == Domain.FINANCE:
        return FinanceGenerator(schema_name=schema_name or "transactions", **kwargs)
    elif domain == Domain.RETAIL:
        return RetailGenerator(schema_name=schema_name or "customers", **kwargs)
    else:
        return DomainGenerator(domain, schema_name=schema_name, **kwargs)


def list_available_schemas() -> Dict[str, List[str]]:
    """List available domain schemas.

    Returns:
        Dict of {domain: [schema_names]}
    """
    schemas = {}

    for (domain, name), _schema in DOMAIN_SCHEMAS.items():
        domain_name = domain.value
        if domain_name not in schemas:
            schemas[domain_name] = []
        schemas[domain_name].append(name)

    return schemas
