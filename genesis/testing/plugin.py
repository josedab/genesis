"""pytest plugin for Genesis test data generation.

This module provides pytest plugin hooks and configuration for
automatic synthetic test data generation.

Usage:
    Add to conftest.py:
    >>> from genesis.testing.plugin import configure_genesis_testing
    >>> configure_genesis_testing(seed=42)

    Or use pytest command line:
    >>> pytest --genesis-seed=42
"""

from __future__ import annotations

import os
from typing import Any, Generator, Optional

from genesis.utils.logging import get_logger

logger = get_logger(__name__)

# Global configuration
_config = {
    "seed": None,
    "deterministic": True,
    "default_n_samples": 100,
    "cache_fixtures": True,
}


def configure_genesis_testing(
    seed: Optional[int] = None,
    deterministic: bool = True,
    default_n_samples: int = 100,
    cache_fixtures: bool = True,
) -> None:
    """Configure global settings for Genesis test data generation.

    This should be called in conftest.py before any fixtures are used.

    Args:
        seed: Global random seed for reproducibility
        deterministic: If True, always generate same data for same seed
        default_n_samples: Default number of samples for fixtures
        cache_fixtures: If True, cache fixture data within scope

    Example:
        >>> # In conftest.py
        >>> from genesis.testing.plugin import configure_genesis_testing
        >>> configure_genesis_testing(seed=42, deterministic=True)
    """
    global _config
    _config["seed"] = seed
    _config["deterministic"] = deterministic
    _config["default_n_samples"] = default_n_samples
    _config["cache_fixtures"] = cache_fixtures

    # Set global seed for fixtures module
    from genesis.testing.fixtures import set_global_seed

    if seed is not None:
        set_global_seed(seed)

    logger.info(f"Genesis testing configured with seed={seed}")


def get_config() -> dict:
    """Get current configuration."""
    return _config.copy()


def genesis_seed(seed: int) -> None:
    """Set the global seed for deterministic test data.

    Args:
        seed: Random seed

    Example:
        >>> from genesis.testing.plugin import genesis_seed
        >>> genesis_seed(12345)
    """
    configure_genesis_testing(seed=seed)


# pytest plugin hooks
def pytest_addoption(parser: Any) -> None:
    """Add command line options for Genesis testing."""
    group = parser.getgroup("genesis")
    group.addoption(
        "--genesis-seed",
        action="store",
        type=int,
        default=None,
        help="Random seed for Genesis synthetic data generation",
    )
    group.addoption(
        "--genesis-samples",
        action="store",
        type=int,
        default=100,
        help="Default number of samples for Genesis fixtures",
    )
    group.addoption(
        "--genesis-no-cache",
        action="store_true",
        default=False,
        help="Disable caching of Genesis fixtures",
    )


def pytest_configure(config: Any) -> None:
    """Configure Genesis based on pytest options."""
    seed = config.getoption("--genesis-seed", None)
    n_samples = config.getoption("--genesis-samples", 100)
    no_cache = config.getoption("--genesis-no-cache", False)

    # Also check environment variable
    if seed is None:
        env_seed = os.environ.get("GENESIS_TEST_SEED")
        if env_seed:
            seed = int(env_seed)

    configure_genesis_testing(
        seed=seed,
        default_n_samples=n_samples,
        cache_fixtures=not no_cache,
    )


def pytest_report_header(config: Any) -> str:
    """Add Genesis info to pytest header."""
    seed = _config.get("seed", "random")
    return f"genesis: seed={seed}, samples={_config.get('default_n_samples')}"


# Fixture for accessing generator in tests
try:
    import pytest

    @pytest.fixture(scope="session")
    def genesis_generator() -> Generator:
        """Provide a TestDataGenerator instance."""
        from genesis.testing.generators import TestDataGenerator

        generator = TestDataGenerator(seed=_config.get("seed"))
        yield generator

    @pytest.fixture(scope="session")
    def genesis_config() -> dict:
        """Provide Genesis configuration."""
        return get_config()

except ImportError:
    # pytest not available
    pass


# Convenience function for non-pytest usage
def quick_generate(
    schema: dict,
    n_samples: int = 100,
    seed: Optional[int] = None,
) -> "pd.DataFrame":
    """Quick generation without pytest context.

    Args:
        schema: Schema dictionary
        n_samples: Number of samples
        seed: Random seed

    Returns:
        Generated DataFrame

    Example:
        >>> from genesis.testing.plugin import quick_generate
        >>> df = quick_generate({"name": "name", "age": "int:18-65"}, n_samples=50)
    """
    import pandas as pd

    from genesis.testing.generators import TestDataGenerator

    generator = TestDataGenerator(seed=seed or _config.get("seed"))
    return generator.from_dict(schema, n_samples)


__all__ = [
    "configure_genesis_testing",
    "get_config",
    "genesis_seed",
    "quick_generate",
]
