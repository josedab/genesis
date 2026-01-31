# ADR-0003: Optional Dependencies via Extras

## Status

Accepted

## Context

Genesis is a comprehensive platform with capabilities spanning:
- Tabular data generation (CTGAN, TVAE, Gaussian Copula)
- Time series generation (TimeGAN)
- Text generation (LLM-based)
- Image synthesis (Stable Diffusion)
- REST API serving (FastAPI)
- Experiment tracking (MLflow, W&B)
- Visualization (Plotly, Matplotlib)

Installing all dependencies would require:
- ~2GB for PyTorch
- ~2GB for TensorFlow
- ~1GB for diffusers/transformers
- Plus numerous smaller packages

This creates problems:
- **Installation time**: 10+ minutes on slow connections
- **Disk space**: 5GB+ total footprint
- **Dependency conflicts**: version constraints clash
- **CI/CD burden**: slow builds, large Docker images
- **User frustration**: "I just want to generate some tabular data"

## Decision

We structure dependencies using **Python extras** (PEP 508):

```toml
# pyproject.toml
[project]
dependencies = [
    # Core: always installed (~50MB)
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "tqdm>=4.62.0",
    "pyyaml>=6.0",
    "click>=8.0.0",
    "rich>=12.0.0",
]

[project.optional-dependencies]
pytorch = ["torch>=1.9.0"]
tensorflow = ["tensorflow>=2.6.0"]
llm = ["openai>=1.0.0", "transformers>=4.20.0"]
viz = ["matplotlib>=3.4.0", "plotly>=5.0.0"]
api = ["fastapi>=0.100.0", "uvicorn>=0.22.0", "pydantic>=2.0.0"]
image = ["diffusers>=0.20.0", "Pillow>=9.0.0"]
integrations = ["mlflow>=2.0.0", "wandb>=0.15.0"]
dev = ["pytest>=7.0.0", "black>=22.0.0", "mypy>=0.950", ...]
docs = ["mkdocs>=1.4.0", "mkdocs-material>=8.5.0", ...]
all = ["genesis-synth[pytorch,tensorflow,llm,viz,api,image,integrations,dev,docs]"]
```

Installation options:

```bash
# Minimal install (Gaussian Copula only)
pip install genesis-synth

# With PyTorch for deep learning generators
pip install genesis-synth[pytorch]

# For production API deployment
pip install genesis-synth[pytorch,api]

# Everything (development)
pip install genesis-synth[all]
```

## Consequences

### Positive

- **Fast installation**: core package installs in seconds
- **Small footprint**: 50MB vs 5GB+
- **User choice**: install only what you need
- **Cleaner environments**: fewer dependency conflicts
- **Faster CI**: test matrix can target specific extras
- **Smaller Docker images**: production images are lean

### Negative

- **Import-time errors**: user might call a feature without its extra installed
- **Documentation complexity**: must explain what's needed for each feature
- **Testing burden**: must test all extra combinations

### Mitigations

We use **lazy imports with helpful errors**:

```python
# genesis/api/server.py
try:
    from fastapi import FastAPI
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

def create_app():
    if not HAS_FASTAPI:
        raise ImportError(
            "FastAPI is not installed. Install with: pip install genesis-synth[api]"
        )
    # ... create app
```

Error messages always include the exact install command:

```
ImportError: PyTorch is not installed. 
Install with: pip install genesis-synth[pytorch]
```

## Extras Reference

| Extra | Use Case | Key Packages |
|-------|----------|--------------|
| `pytorch` | Deep learning generators (CTGAN, TVAE, TimeGAN) | torch |
| `tensorflow` | Alternative DL backend | tensorflow |
| `llm` | Text generation, LLM agents | openai, transformers |
| `viz` | Visualization, dashboards | matplotlib, plotly |
| `api` | REST API deployment | fastapi, uvicorn |
| `image` | Image synthesis | diffusers, Pillow |
| `integrations` | Experiment tracking | mlflow, wandb |
| `dev` | Development and testing | pytest, black, mypy |
| `docs` | Documentation building | mkdocs |
| `all` | Everything | All of the above |
