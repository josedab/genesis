# Examples

Interactive Jupyter notebooks demonstrating Genesis capabilities.

## Quick Links

| Notebook | Description | Colab |
|----------|-------------|-------|
| [01_quickstart](01_quickstart.ipynb) | Basic usage and quick start | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genesis-synth/genesis/blob/main/examples/01_quickstart.ipynb) |
| [02_tabular_synthesis](02_tabular_synthesis.ipynb) | CTGAN, TVAE, Gaussian Copula comparison | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genesis-synth/genesis/blob/main/examples/02_tabular_synthesis.ipynb) |
| [03_time_series](03_time_series.ipynb) | Time series data generation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genesis-synth/genesis/blob/main/examples/03_time_series.ipynb) |
| [04_text_generation](04_text_generation.ipynb) | LLM-based text synthesis | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genesis-synth/genesis/blob/main/examples/04_text_generation.ipynb) |
| [05_privacy_config](05_privacy_config.ipynb) | Privacy configuration and DP | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genesis-synth/genesis/blob/main/examples/05_privacy_config.ipynb) |
| [06_healthcare_example](06_healthcare_example.ipynb) | Healthcare synthetic data | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genesis-synth/genesis/blob/main/examples/06_healthcare_example.ipynb) |
| [07_finance_example](07_finance_example.ipynb) | Financial transaction data | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genesis-synth/genesis/blob/main/examples/07_finance_example.ipynb) |
| [08_multitable_example](08_multitable_example.ipynb) | Multi-table relational data | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genesis-synth/genesis/blob/main/examples/08_multitable_example.ipynb) |

## Running Locally

```bash
# Clone the repository
git clone https://github.com/genesis-synth/genesis.git
cd genesis

# Install dependencies
pip install -e ".[all]"

# Start Jupyter
jupyter lab examples/
```

## Running on Colab

Click the "Open in Colab" badge on any notebook. Then run:

```python
!pip install genesis-synth[pytorch]
```
