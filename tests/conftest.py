"""Pytest configuration and fixtures for Genesis tests."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_numeric_df():
    """Create a sample DataFrame with numeric columns."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n_samples),
            "income": np.random.normal(50000, 15000, n_samples),
            "score": np.random.uniform(0, 100, n_samples),
        }
    )


@pytest.fixture
def sample_mixed_df():
    """Create a sample DataFrame with mixed column types."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "age": np.random.randint(18, 80, n_samples),
            "income": np.random.normal(50000, 15000, n_samples),
            "gender": np.random.choice(["M", "F"], n_samples),
            "city": np.random.choice(["NYC", "LA", "Chicago", "Houston"], n_samples),
            "active": np.random.choice([True, False], n_samples),
        }
    )


@pytest.fixture
def sample_categorical_df():
    """Create a sample DataFrame with categorical columns."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "gender": np.random.choice(["M", "F"], n_samples),
            "education": np.random.choice(["High School", "Bachelor", "Master", "PhD"], n_samples),
            "city": np.random.choice(["NYC", "LA", "Chicago"], n_samples),
            "status": np.random.choice(["Active", "Inactive"], n_samples),
        }
    )


@pytest.fixture
def sample_timeseries_df():
    """Create a sample time series DataFrame."""
    np.random.seed(42)
    n_samples = 200

    # Generate trend + seasonality + noise
    t = np.arange(n_samples)
    trend = 0.1 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 24)
    noise = np.random.normal(0, 2, n_samples)

    return pd.DataFrame(
        {
            "value1": trend + seasonality + noise + 50,
            "value2": np.cumsum(np.random.randn(n_samples)) + 100,
        }
    )


@pytest.fixture
def sample_multitable():
    """Create sample multi-table data with foreign key."""
    np.random.seed(42)

    users = pd.DataFrame(
        {
            "user_id": range(1, 21),
            "name": [f"User_{i}" for i in range(1, 21)],
            "age": np.random.randint(20, 60, 20),
        }
    )

    orders = pd.DataFrame(
        {
            "order_id": range(1, 51),
            "user_id": np.random.choice(range(1, 21), 50),
            "amount": np.random.uniform(10, 500, 50),
            "status": np.random.choice(["pending", "completed", "cancelled"], 50),
        }
    )

    return {"users": users, "orders": orders}


@pytest.fixture
def sample_text_df():
    """Create a sample DataFrame with text data."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be, that is the question.",
        "All that glitters is not gold.",
        "Actions speak louder than words.",
    ] * 20

    return pd.DataFrame(
        {
            "id": range(len(texts)),
            "text": texts,
            "category": np.random.choice(["A", "B", "C"], len(texts)),
        }
    )


@pytest.fixture
def small_df():
    """Create a very small DataFrame for quick tests."""
    return pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            "c": ["x", "y", "x", "y", "x"],
        }
    )
