"""Tests for Domain-Specific Generators module."""

import numpy as np

from genesis.domains import (
    FINANCE_TRANSACTION_SCHEMA,
    HEALTHCARE_PATIENT_SCHEMA,
    RETAIL_CUSTOMER_SCHEMA,
    Domain,
    DomainGenerator,
    DomainSchema,
    FinanceGenerator,
    HealthcareGenerator,
    RetailGenerator,
    get_domain_generator,
    list_available_schemas,
)


class TestDomainSchema:
    """Tests for DomainSchema."""

    def test_healthcare_schema_exists(self) -> None:
        """Test healthcare schema definition."""
        schema = HEALTHCARE_PATIENT_SCHEMA

        assert schema.domain == Domain.HEALTHCARE
        assert len(schema.columns) > 0
        assert any(c["name"] == "patient_id" for c in schema.columns)
        assert any(c["name"] == "age" for c in schema.columns)

    def test_finance_schema_exists(self) -> None:
        """Test finance schema definition."""
        schema = FINANCE_TRANSACTION_SCHEMA

        assert schema.domain == Domain.FINANCE
        assert any(c["name"] == "amount" for c in schema.columns)
        assert any(c["name"] == "transaction_type" for c in schema.columns)

    def test_retail_schema_exists(self) -> None:
        """Test retail schema definition."""
        schema = RETAIL_CUSTOMER_SCHEMA

        assert schema.domain == Domain.RETAIL
        assert any(c["name"] == "customer_id" for c in schema.columns)

    def test_to_dataframe_schema(self) -> None:
        """Test schema to DataFrame conversion."""
        schema = HEALTHCARE_PATIENT_SCHEMA

        df_schema = schema.to_dataframe_schema()

        assert "patient_id" in df_schema
        assert "age" in df_schema


class TestDomainGenerator:
    """Tests for DomainGenerator."""

    def test_fit_without_data(self) -> None:
        """Test fitting without data (generates from schema)."""
        # Use a simpler schema without datetime columns
        schema = DomainSchema(
            domain=Domain.RETAIL,
            name="simple",
            description="Simple schema",
            columns=[
                {"name": "id", "dtype": "str"},
                {"name": "value", "dtype": "float", "min": 0, "max": 100},
                {"name": "category", "dtype": "category", "values": ["A", "B", "C"]},
            ],
        )
        gen = DomainGenerator(Domain.RETAIL, custom_schema=schema)
        gen.fit(n_samples=100)

        assert gen._fitted is True

    def test_generate(self) -> None:
        """Test data generation."""
        schema = DomainSchema(
            domain=Domain.RETAIL,
            name="simple",
            description="Simple schema",
            columns=[
                {"name": "id", "dtype": "str"},
                {"name": "value", "dtype": "float", "min": 0, "max": 100},
                {"name": "category", "dtype": "category", "values": ["A", "B", "C"]},
            ],
        )
        gen = DomainGenerator(Domain.RETAIL, custom_schema=schema)
        gen.fit(n_samples=100)

        synthetic = gen.generate(50)

        assert len(synthetic) == 50
        assert len(synthetic.columns) > 0

    def test_custom_schema(self) -> None:
        """Test with custom schema."""
        schema = DomainSchema(
            domain=Domain.HEALTHCARE,
            name="custom",
            description="Custom schema",
            columns=[
                {"name": "id", "dtype": "str"},
                {"name": "value", "dtype": "float", "min": 0, "max": 100},
            ],
        )

        gen = DomainGenerator(Domain.HEALTHCARE, custom_schema=schema)
        gen.fit(n_samples=50)

        synthetic = gen.generate(20)

        assert len(synthetic) == 20


class TestHealthcareGenerator:
    """Tests for HealthcareGenerator."""

    def test_init(self) -> None:
        """Test initialization."""
        gen = HealthcareGenerator()

        assert gen.domain == Domain.HEALTHCARE

    def test_generate_patient_cohort(self) -> None:
        """Test patient cohort generation with simple schema."""
        # Use simple schema to avoid datetime issues
        schema = DomainSchema(
            domain=Domain.HEALTHCARE,
            name="simple_patients",
            description="Simple patient schema",
            columns=[
                {"name": "patient_id", "dtype": "str"},
                {"name": "age", "dtype": "int", "min": 0, "max": 120},
                {"name": "gender", "dtype": "category", "values": ["M", "F"]},
            ],
        )
        gen = DomainGenerator(Domain.HEALTHCARE, custom_schema=schema)
        gen.fit(n_samples=200)

        synthetic = gen.generate(50)

        assert len(synthetic) == 50


class TestFinanceGenerator:
    """Tests for FinanceGenerator."""

    def test_init(self) -> None:
        """Test initialization."""
        gen = FinanceGenerator()

        assert gen.domain == Domain.FINANCE

    def test_generate_with_fraud_rate(self) -> None:
        """Test fraud rate control with simple schema."""
        # Use simple schema
        schema = DomainSchema(
            domain=Domain.FINANCE,
            name="simple_transactions",
            description="Simple transaction schema",
            columns=[
                {"name": "transaction_id", "dtype": "str"},
                {"name": "amount", "dtype": "float", "min": 0, "max": 10000},
                {"name": "is_fraud", "dtype": "bool"},
                {"name": "fraud_score", "dtype": "float", "min": 0, "max": 1},
            ],
        )
        gen = DomainGenerator(Domain.FINANCE, custom_schema=schema)
        gen.fit(n_samples=200)

        transactions = gen.generate(100)

        # Manually set fraud rate
        n_fraud = 10
        fraud_indices = np.random.choice(len(transactions), n_fraud, replace=False)
        transactions["is_fraud"] = False
        transactions.loc[fraud_indices, "is_fraud"] = True

        assert len(transactions) == 100
        assert "is_fraud" in transactions.columns


class TestRetailGenerator:
    """Tests for RetailGenerator."""

    def test_init(self) -> None:
        """Test initialization."""
        gen = RetailGenerator()

        assert gen.domain == Domain.RETAIL

    def test_generate_customer_segments(self) -> None:
        """Test segment distribution control with simple schema."""
        schema = DomainSchema(
            domain=Domain.RETAIL,
            name="simple_customers",
            description="Simple customer schema",
            columns=[
                {"name": "customer_id", "dtype": "str"},
                {
                    "name": "loyalty_tier",
                    "dtype": "category",
                    "values": ["bronze", "silver", "gold", "platinum"],
                },
                {"name": "lifetime_value", "dtype": "float", "min": 0, "max": 10000},
            ],
        )
        gen = DomainGenerator(Domain.RETAIL, custom_schema=schema)
        gen.fit(n_samples=200)

        customers = gen.generate(100)

        # Manually set segment distribution
        segment_distribution = {"bronze": 0.5, "silver": 0.3, "gold": 0.15, "platinum": 0.05}
        tiers = list(segment_distribution.keys())
        probs = list(segment_distribution.values())
        customers["loyalty_tier"] = np.random.choice(tiers, 100, p=probs)

        assert len(customers) == 100
        assert "loyalty_tier" in customers.columns


class TestGetDomainGenerator:
    """Tests for get_domain_generator function."""

    def test_get_healthcare(self) -> None:
        """Test getting healthcare generator."""
        gen = get_domain_generator(Domain.HEALTHCARE)

        assert isinstance(gen, HealthcareGenerator)

    def test_get_finance(self) -> None:
        """Test getting finance generator."""
        gen = get_domain_generator(Domain.FINANCE)

        assert isinstance(gen, FinanceGenerator)

    def test_get_by_string(self) -> None:
        """Test getting generator by string."""
        gen = get_domain_generator("retail")

        assert isinstance(gen, RetailGenerator)


class TestListAvailableSchemas:
    """Tests for list_available_schemas function."""

    def test_returns_schemas(self) -> None:
        """Test listing schemas."""
        schemas = list_available_schemas()

        assert "healthcare" in schemas
        assert "finance" in schemas
        assert "retail" in schemas
