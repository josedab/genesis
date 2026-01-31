---
sidebar_position: 105
title: Contributing
---

# Contributing to Genesis

Thank you for your interest in contributing to Genesis! This guide will help you get started.

## Ways to Contribute

- **Report bugs** - Found a bug? Open an issue
- **Suggest features** - Have an idea? We'd love to hear it
- **Improve documentation** - Docs can always be better
- **Submit code** - Fix bugs or add features
- **Answer questions** - Help others in discussions

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- (Optional) CUDA for GPU development

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/genesis/genesis.git
cd genesis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Verify Setup

```bash
# Run tests
pytest

# Check code style
ruff check genesis/

# Build docs
cd website && npm install && npm run build
```

## Project Structure

```
genesis/
├── genesis/                # Main package
│   ├── __init__.py
│   ├── core/              # Core functionality
│   ├── generators/        # Generator implementations
│   ├── evaluation/        # Quality metrics
│   ├── privacy/           # Privacy modules
│   ├── cli/               # Command-line interface
│   └── ...
├── tests/                 # Test suite
│   ├── unit/
│   └── integration/
├── docs/                  # Documentation source
├── website/               # Docusaurus site
├── examples/              # Example notebooks
├── benchmarks/            # Performance benchmarks
└── pyproject.toml         # Project configuration
```

## Making Changes

### 1. Create a Branch

```bash
# Update main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feature/my-feature
# or
git checkout -b fix/bug-description
```

### 2. Make Changes

- Write code following our style guide
- Add tests for new functionality
- Update documentation if needed
- Keep commits focused and atomic

### 3. Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_generators.py

# Run with coverage
pytest --cov=genesis --cov-report=html
```

### 4. Check Code Quality

```bash
# Lint code
ruff check genesis/

# Format code
ruff format genesis/

# Type checking
mypy genesis/
```

### 5. Submit Pull Request

```bash
# Push branch
git push origin feature/my-feature
```

Then open a PR on GitHub with:
- Clear title and description
- Link to related issues
- Screenshots/examples if applicable

## Code Style

### Python Style

We follow PEP 8 with some modifications:

```python
# Good
def generate_samples(
    self,
    n_samples: int,
    conditions: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Generate synthetic samples.
    
    Args:
        n_samples: Number of samples to generate.
        conditions: Optional conditions for generation.
        
    Returns:
        DataFrame with synthetic samples.
    """
    ...

# Bad
def generate_samples(self, n_samples, conditions=None):
    # missing type hints and docstring
    ...
```

### Docstring Format

Use Google-style docstrings:

```python
def fit(
    self,
    data: pd.DataFrame,
    discrete_columns: list[str] | None = None,
) -> "SyntheticGenerator":
    """Train the generator on data.
    
    Args:
        data: Training data as a pandas DataFrame.
        discrete_columns: List of categorical column names.
            If None, columns are auto-detected.
    
    Returns:
        The fitted generator instance.
        
    Raises:
        ValueError: If data is empty.
        TypeError: If data is not a DataFrame.
        
    Example:
        >>> generator = SyntheticGenerator()
        >>> generator.fit(df, discrete_columns=['category'])
    """
```

### Test Style

```python
import pytest
import pandas as pd
from genesis import SyntheticGenerator

class TestSyntheticGenerator:
    """Tests for SyntheticGenerator class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for tests."""
        return pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'income': [50000, 60000, 70000, 80000, 90000],
            'status': ['A', 'B', 'A', 'B', 'A']
        })
    
    def test_fit_returns_self(self, sample_data):
        """Test that fit() returns the generator instance."""
        generator = SyntheticGenerator(method='ctgan')
        result = generator.fit(sample_data)
        assert result is generator
    
    def test_generate_correct_shape(self, sample_data):
        """Test that generate() returns correct number of rows."""
        generator = SyntheticGenerator(method='ctgan')
        generator.fit(sample_data)
        synthetic = generator.generate(100)
        assert len(synthetic) == 100
        assert set(synthetic.columns) == set(sample_data.columns)
```

## Documentation

### Writing Docs

Documentation is in `website/docs/`. Use Markdown with:

```markdown
---
sidebar_position: 1
title: My Page
---

# My Page Title

Introduction paragraph.

## Section

Content with code examples:

```python
from genesis import SyntheticGenerator

generator = SyntheticGenerator()
```

:::tip
Helpful tip here.
:::

:::warning
Important warning here.
:::
```

### Building Docs

```bash
cd website
npm install
npm run start  # Development server
npm run build  # Production build
```

## Commit Messages

Use conventional commits:

```
feat: add conditional generation support
fix: handle NaN values in categorical columns
docs: update quickstart guide
test: add tests for time series generator
refactor: simplify constraint validation
chore: update dependencies
```

## Pull Request Process

1. **Before submitting:**
   - All tests pass
   - Code is formatted and linted
   - Documentation is updated
   - Commits are clean and focused

2. **PR description should include:**
   - What changes were made
   - Why the changes were needed
   - How to test the changes
   - Any breaking changes

3. **Review process:**
   - Maintainer reviews code
   - CI checks pass
   - Changes requested are addressed
   - PR is approved and merged

## Release Process

Releases are handled by maintainers:

1. Update version in `genesis/version.py`
2. Update CHANGELOG.md
3. Create release PR
4. After merge, tag release
5. CI builds and publishes to PyPI

## Getting Help

- **Questions:** Use [GitHub Discussions](https://github.com/genesis/genesis/discussions)
- **Bugs:** Open a [GitHub Issue](https://github.com/genesis/genesis/issues)
- **Chat:** Join our [Discord](https://discord.gg/genesis)

## Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/). In summary:

- Be welcoming and inclusive
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Genesis! 🎉
