---
sidebar_position: 4
title: From Gretel
---

# Migrating from Gretel.ai to Genesis

Move from Gretel's cloud platform to self-hosted Genesis.

## Why Migrate?

| Aspect | Gretel.ai | Genesis |
|--------|-----------|---------|
| **Deployment** | Cloud only | Self-hosted anywhere |
| **Data location** | Uploaded to cloud | Stays on your infrastructure |
| **Pricing** | Per-record fees | Free (open source) |
| **Customization** | Limited | Full source code access |
| **Compliance** | SOC 2 certified | You control the audit |

**Migrate to Genesis when:**
- Data cannot leave your environment (regulatory, security)
- You need cost-effective high-volume generation
- You want full control and customization
- You prefer open source transparency

---

## Installation

```bash
pip install genesis-synth[pytorch]
```

---

## API Comparison

### Gretel Synthetics

```python
# ❌ Gretel (before)
from gretel_synthetics.config import LocalConfig
from gretel_synthetics.train import train_rnn

config = LocalConfig(
    max_lines=None,
    epochs=30,
    field_delimiter=",",
    overwrite=True
)

train_rnn(config)

# Requires Gretel cloud for full features
from gretel_client import Gretel
gretel = Gretel(api_key="your-api-key")
project = gretel.get_project(name="my-project")

model = project.create_model_obj(
    model_config="synthetics/default",
    data_source="customers.csv"
)
model.submit_cloud()
model.wait_for_completion()
synthetic = model.generate_data_frame(num_records=1000)
```

```python
# ✅ Genesis (after) - all local, no API keys
from genesis import auto_synthesize
import pandas as pd

# Load data locally
data = pd.read_csv('customers.csv')

# Generate locally - no cloud upload
synthetic = auto_synthesize(data, n_samples=1000)

# Everything stays on your machine
synthetic.to_csv('synthetic_customers.csv', index=False)
```

---

## Feature Mapping

### Data Types

| Gretel | Genesis |
|--------|---------|
| Tabular (ACTGAN) | `SyntheticGenerator(method='ctgan')` |
| Tabular (LSTM) | `SyntheticGenerator(method='tvae')` |
| Time Series | `TimeSeriesGenerator` |
| Text (GPT) | `TextGenerator` with LLM backend |
| Multi-table | `MultiTableGenerator` |

### Configuration

```python
# Gretel model config
model_config = {
    "models": [{
        "synthetics": {
            "data_source": "customers.csv",
            "params": {
                "epochs": 100,
                "batch_size": 64,
            },
            "privacy_filters": {
                "outliers": True,
                "similarity": "high"
            }
        }
    }]
}

# Genesis equivalent
from genesis import SyntheticGenerator

generator = SyntheticGenerator(
    method='ctgan',
    config={
        'epochs': 100,
        'batch_size': 64,
    },
    privacy={
        'suppress_outliers': True,
        'differential_privacy': {'epsilon': 1.0}
    }
)
```

---

## Workflow Migration

### Training

```python
# Gretel
model.submit_cloud()  # Data uploaded to cloud
model.wait_for_completion()

# Genesis - all local
generator.fit(data, discrete_columns=['category', 'status'])
```

### Generation

```python
# Gretel
synthetic = model.generate_data_frame(num_records=1000)

# Genesis
synthetic = generator.generate(n_samples=1000)
```

### Evaluation

```python
# Gretel
report = model.report  # Basic quality metrics

# Genesis - comprehensive evaluation
from genesis import QualityEvaluator, run_privacy_audit

quality = QualityEvaluator(real, synthetic).evaluate()
privacy = run_privacy_audit(real, synthetic)

print(f"Quality: {quality.overall_score:.1%}")
print(f"Privacy: {privacy.overall_score:.1%}")
print(f"Safe to release: {privacy.is_safe}")
```

---

## Privacy Features

### Gretel Privacy Filters

```python
# Gretel
model_config = {
    "models": [{
        "synthetics": {
            "privacy_filters": {
                "outliers": True,
                "similarity": "high"
            }
        }
    }]
}
```

### Genesis Privacy (More Options)

```python
# Genesis - more granular control
from genesis import SyntheticGenerator

generator = SyntheticGenerator(
    method='ctgan',
    privacy={
        # Differential privacy
        'differential_privacy': {
            'epsilon': 1.0,
            'delta': 1e-5
        },
        # K-anonymity
        'k_anonymity': {
            'k': 5,
            'quasi_identifiers': ['age', 'zipcode', 'gender']
        },
        # Outlier suppression
        'suppress_outliers': True,
        'min_category_count': 10
    }
)

# Privacy attack testing (not available in Gretel basic)
from genesis import run_privacy_audit
from genesis.privacy_attacks import MembershipInferenceAttack

audit = run_privacy_audit(real, synthetic)
attack = MembershipInferenceAttack().evaluate(real, synthetic)

print(f"Membership attack accuracy: {attack.accuracy:.1%}")
print(f"Risk level: {attack.risk_level}")
```

---

## Complete Migration Example

### Before: Gretel Cloud

```python
from gretel_client import Gretel
import pandas as pd

# Initialize (requires API key)
gretel = Gretel(api_key="grtu_abc123...")
project = gretel.get_project(name="customer-synthesis")

# Upload data to cloud
model = project.create_model_obj(
    model_config="synthetics/default",
    data_source="customers.csv"
)

# Train in cloud
model.submit_cloud()
model.wait_for_completion()  # Minutes to hours

# Generate
synthetic = model.generate_data_frame(num_records=10000)

# Download results
synthetic.to_csv("synthetic_customers.csv", index=False)
```

### After: Genesis Local

```python
from genesis import auto_synthesize, QualityEvaluator, run_privacy_audit
import pandas as pd

# Load data locally (never leaves your machine)
data = pd.read_csv("customers.csv")

# Train locally
synthetic = auto_synthesize(
    data,
    n_samples=10000,
    discrete_columns=['segment', 'region', 'status'],
    privacy={'differential_privacy': {'epsilon': 1.0}},
    mode='quality'
)

# Evaluate locally
quality = QualityEvaluator(data, synthetic).evaluate()
privacy = run_privacy_audit(data, synthetic)

print(f"Quality Score: {quality.overall_score:.1%}")
print(f"Privacy Score: {privacy.overall_score:.1%}")

# Save locally
synthetic.to_csv("synthetic_customers.csv", index=False)
```

---

## Handling Gretel-Specific Features

### Gretel Transforms (Data Prep)

```python
# Gretel transforms
from gretel_client.transformers import DataTransformPipeline

pipeline = DataTransformPipeline([
    ("fake_names", FakeTransform(columns=["name"])),
    ("hash_emails", HashTransform(columns=["email"])),
])
transformed = pipeline.transform(data)

# Genesis equivalent
from genesis.domains import NameGenerator, EmailGenerator

# Remove real PII first
data_clean = data.drop(['name', 'email'], axis=1)

# Synthesize behavioral data
synthetic = auto_synthesize(data_clean, n_samples=len(data))

# Add fake PII
synthetic['name'] = NameGenerator('en_US').generate(len(synthetic))
synthetic['email'] = EmailGenerator().generate(len(synthetic))
```

### Gretel Classify (PII Detection)

```python
# Gretel
from gretel_client import Gretel
gretel = Gretel()
classified = gretel.classify(data)

# Genesis (manual or use third-party)
# Genesis focuses on synthesis, use dedicated PII tools:
# - Microsoft Presidio
# - AWS Comprehend
# - spaCy NER

# Or simple heuristics:
pii_columns = []
for col in data.columns:
    if data[col].dtype == 'object':
        sample = data[col].dropna().iloc[0] if len(data[col].dropna()) > 0 else ''
        if '@' in str(sample):
            pii_columns.append(col)  # Likely email
        elif str(sample).replace('-', '').isdigit() and len(str(sample)) > 8:
            pii_columns.append(col)  # Likely phone/SSN
```

---

## Cost Comparison

### Gretel Pricing (Approximate)

| Volume | Gretel Cost | Genesis Cost |
|--------|-------------|--------------|
| 10K records | ~$1-5 | $0 |
| 100K records | ~$10-50 | $0 |
| 1M records | ~$100-500 | $0 |
| 10M records | ~$1000+ | $0 |

Genesis is free and open source. Your only costs are compute (which you'd pay anyway with Gretel) and your time.

### Compute Requirements

Genesis runs on standard hardware:

```
Minimum:
- 4GB RAM
- 2 CPU cores
- No GPU required (optional)

Recommended:
- 16GB RAM
- 4+ CPU cores
- NVIDIA GPU (10x faster training)
```

---

## Deployment Options

### Local Development

```bash
pip install genesis-synth[pytorch]
```

### Docker

```dockerfile
FROM python:3.11
RUN pip install genesis-synth[pytorch]
COPY . /app
WORKDIR /app
CMD ["python", "synthesize.py"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: genesis-synthesis
spec:
  replicas: 1
  template:
    spec:
      containers:
      - name: genesis
        image: your-registry/genesis:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

### REST API

```bash
# Start Genesis API server
pip install genesis-synth[api]
genesis api start --port 8000

# Use from any language
curl -X POST http://localhost:8000/synthesize \
  -F "file=@customers.csv" \
  -F "n_samples=1000"
```

---

## Migration Checklist

- [ ] Export data from Gretel projects (if needed)
- [ ] Install Genesis: `pip install genesis-synth[pytorch]`
- [ ] Replace Gretel API calls with Genesis equivalents
- [ ] Configure privacy settings
- [ ] Test quality on sample data
- [ ] Set up local/on-premise deployment
- [ ] Update CI/CD pipelines
- [ ] Cancel Gretel subscription

---

## Need Help?

- [GitHub Discussions](https://github.com/genesis-synth/genesis/discussions)
- [API Reference](/docs/api/reference)
- [Deployment Guide](/docs/advanced/distributed)
