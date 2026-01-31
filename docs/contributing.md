# Contributing to Genesis

Thank you for your interest in contributing to Genesis! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Development Setup](#development-setup)
- [Running Tests](#running-tests)
- [Code Style](#code-style)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) CUDA-compatible GPU for deep learning generators

### Installation

```bash
# Clone the repository
git clone https://github.com/genesis-synth/genesis.git
cd genesis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Optional Dependencies

```bash
# For PyTorch-based generators
pip install -e ".[dev,pytorch]"

# For TensorFlow-based generators
pip install -e ".[dev,tensorflow]"

# For LLM text generation
pip install -e ".[dev,llm]"

# Everything
pip install -e ".[all]"
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Skip slow tests
pytest tests/ -m "not slow"

# Run specific test file
pytest tests/unit/test_core.py

# Run with coverage
pytest tests/ --cov=genesis --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/
```

## Code Style

This project uses strict code quality tools:

| Tool | Purpose |
|------|---------|
| **Black** | Code formatting |
| **isort** | Import sorting |
| **Ruff** | Fast linting |
| **mypy** | Static type checking |

### Running Code Quality Checks

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Individual tools
black genesis tests
isort genesis tests
ruff check genesis tests
mypy genesis
```

### Style Guidelines

1. **Type Hints**: All public functions must have type hints
2. **Docstrings**: Use Google-style docstrings for all public APIs
3. **Line Length**: Maximum 100 characters
4. **Imports**: Use absolute imports, sorted by isort

Example:

```python
from typing import List, Optional

import pandas as pd

from genesis.core.config import GeneratorConfig


def generate_data(
    data: pd.DataFrame,
    n_samples: int,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate synthetic data from input.

    Args:
        data: Input DataFrame to learn from.
        n_samples: Number of samples to generate.
        columns: Optional list of columns to include.

    Returns:
        DataFrame with synthetic data.

    Raises:
        ValueError: If n_samples is not positive.
    """
    ...
```

## Making Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

Use clear, descriptive commit messages:

```
feat: Add conditional generation support

- Implement sample_conditional method
- Add filtering for categorical conditions
- Add range support for numeric conditions
```

Prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Tests
- `refactor:` - Code refactoring
- `chore:` - Maintenance tasks

## Pull Request Process

1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Make** your changes with tests
4. **Run** the test suite and linting
5. **Push** to your fork
6. **Open** a Pull Request

### PR Checklist

- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] New code has type hints
- [ ] Public APIs have docstrings
- [ ] CHANGELOG.md updated (for user-facing changes)
- [ ] Documentation updated if needed

### Review Process

1. Automated CI checks must pass
2. At least one maintainer review required
3. Address all review comments
4. Squash commits if requested

## Reporting Issues

### Bug Reports

Please include:
- Genesis version (`python -c "import genesis; print(genesis.__version__)"`)
- Python version
- Operating system
- Minimal code to reproduce
- Full error message/traceback

### Feature Requests

Please describe:
- The problem you're trying to solve
- Your proposed solution
- Alternatives you've considered
- Any relevant examples or references

## Questions?

- Check existing [Issues](https://github.com/genesis-synth/genesis/issues)
- Open a [Discussion](https://github.com/genesis-synth/genesis/discussions)
- Read the [Documentation](https://genesis-synth.github.io/genesis)

Thank you for contributing! 🎉
