---
sidebar_position: 103
title: Why Genesis?
---

# Why Genesis?

A detailed comparison of Genesis with other synthetic data tools.

## The Landscape

Synthetic data generation tools fall into three categories:

| Category | Examples | Trade-offs |
|----------|----------|------------|
| **Open Source Libraries** | Genesis, SDV, Faker | Full control, self-hosted, requires expertise |
| **Cloud Platforms** | Gretel.ai, Mostly AI, Tonic | Managed service, higher cost, data leaves your environment |
| **Fake Data Generators** | Faker, Mimesis | Fast but no statistical learning, just random templates |

Genesis sits in the first category—a full-featured open source library you can run anywhere.

## Genesis vs. SDV (Synthetic Data Vault)

[SDV](https://sdv.dev/) is the most similar open source alternative.

| Feature | Genesis | SDV |
|---------|---------|-----|
| **License** | MIT | BSL 1.1 (Business Source License) |
| **Core Methods** | CTGAN, TVAE, Gaussian Copula | CTGAN, TVAE, Gaussian Copula, CopulaGAN |
| **AutoML** | ✅ Automatic method selection | ❌ Manual selection |
| **Differential Privacy** | ✅ Built-in with configurable epsilon | ⚠️ Separate package (DP-CTGAN) |
| **Privacy Attack Testing** | ✅ Membership inference, attribute inference | ❌ Not included |
| **Time Series** | ✅ TimeGAN | ✅ PAR, TimeGAN |
| **Multi-Table** | ✅ With referential integrity | ✅ HMA, CTGAN |
| **Text Generation** | ✅ LLM-based | ❌ Not included |
| **Dataset Versioning** | ✅ Git-like versioning | ❌ Not included |
| **Drift Detection** | ✅ Built-in | ❌ Not included |
| **Pipeline API** | ✅ Python and YAML | ❌ Not included |
| **REST API** | ✅ Built-in | ⚠️ Community extensions |
| **Distributed Training** | ✅ Ray, Dask | ❌ Not included |
| **Compliance Reports** | ✅ GDPR, HIPAA, CCPA | ❌ Not included |

**When to choose SDV:**
- You need CopulaGAN specifically
- You're already using SDV in production
- You prefer their API style

**When to choose Genesis:**
- You need MIT license for commercial use
- Privacy is critical (built-in DP, attack testing)
- You want AutoML for automatic optimization
- You need enterprise features (versioning, pipelines, APIs)

### Code Comparison

```python
# SDV
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data)
synthesizer = CTGANSynthesizer(metadata)
synthesizer.fit(data)
synthetic = synthesizer.sample(num_rows=1000)

# Genesis - simpler API
from genesis import auto_synthesize
synthetic = auto_synthesize(data, n_samples=1000)
```

---

## Genesis vs. Gretel.ai

[Gretel.ai](https://gretel.ai/) is a commercial cloud platform.

| Feature | Genesis | Gretel.ai |
|---------|---------|-----------|
| **Deployment** | Self-hosted anywhere | Cloud-only (data leaves your environment) |
| **Pricing** | Free (open source) | $0.10-0.50 per 1K records |
| **Privacy** | Data stays local | Data uploaded to cloud |
| **Methods** | CTGAN, TVAE, Copula, TimeGAN | ACTGAN, LSTM, GPT |
| **Text Generation** | ✅ | ✅ (stronger with GPT) |
| **Support** | Community + GitHub | Paid enterprise support |
| **Compliance Certs** | ✅ Self-generated | ✅ SOC 2, managed |

**When to choose Gretel:**
- You need managed infrastructure
- Enterprise support is required
- Budget isn't a constraint
- You're comfortable with cloud data processing

**When to choose Genesis:**
- Data cannot leave your environment (regulatory, security)
- You need cost-effective high-volume generation
- You want full control and customization
- You prefer open source transparency

---

## Genesis vs. Faker

[Faker](https://faker.readthedocs.io/) generates fake data from templates.

| Feature | Genesis | Faker |
|---------|---------|-------|
| **How it works** | Learns from your data | Random from templates |
| **Statistical Fidelity** | ✅ Preserves distributions | ❌ No learning |
| **Correlation Preservation** | ✅ Maintains relationships | ❌ Columns are independent |
| **ML Utility** | ✅ High (85-95% of real) | ❌ Low (random noise) |
| **Privacy** | ✅ Formal guarantees | N/A (no real data used) |
| **Speed** | Slower (requires training) | Very fast |
| **Use Case** | Replace real data | Generate test fixtures |

**When to choose Faker:**
- You don't have real data to learn from
- You need simple test fixtures (names, emails, addresses)
- Speed is more important than realism

**When to choose Genesis:**
- You have real data and want realistic synthetic version
- You need to preserve statistical properties
- Synthetic data will be used for ML training
- Privacy guarantees matter

### Using Them Together

Genesis actually includes domain generators similar to Faker:

```python
from genesis import SyntheticGenerator
from genesis.domains import NameGenerator, EmailGenerator

# Generate statistical patterns from real data
generator = SyntheticGenerator()
generator.fit(real_data, discrete_columns=['segment'])
synthetic = generator.generate(10000)

# Add realistic PII columns (not from real data)
synthetic['name'] = NameGenerator('en_US').generate(10000)
synthetic['email'] = EmailGenerator().generate(10000)
```

---

## Genesis vs. Mostly AI

[Mostly AI](https://mostly.ai/) is another commercial platform.

| Feature | Genesis | Mostly AI |
|---------|---------|-----------|
| **Deployment** | Self-hosted | Cloud + on-premise (enterprise) |
| **Pricing** | Free | Free tier + enterprise |
| **UI** | CLI + API | Web UI |
| **AutoML** | ✅ | ✅ |
| **Privacy Testing** | ✅ | ✅ |

**When to choose Mostly AI:**
- You prefer a visual UI over code
- You're exploring synthetic data and want free tier
- Enterprise with on-premise deployment

**When to choose Genesis:**
- You need full programmatic control
- You're integrating into existing ML pipelines
- Open source is a requirement

---

## Feature Matrix

| Feature | Genesis | SDV | Gretel | Faker |
|---------|:-------:|:---:|:------:|:-----:|
| Open Source | ✅ MIT | ⚠️ BSL | ❌ | ✅ MIT |
| Self-Hosted | ✅ | ✅ | ❌ | ✅ |
| Statistical Learning | ✅ | ✅ | ✅ | ❌ |
| AutoML | ✅ | ❌ | ✅ | N/A |
| Differential Privacy | ✅ | ⚠️ | ✅ | N/A |
| Privacy Attack Testing | ✅ | ❌ | ✅ | N/A |
| Time Series | ✅ | ✅ | ✅ | ❌ |
| Multi-Table | ✅ | ✅ | ✅ | ❌ |
| Text Generation | ✅ | ❌ | ✅ | ⚠️ |
| Dataset Versioning | ✅ | ❌ | ❌ | N/A |
| Drift Detection | ✅ | ❌ | ❌ | N/A |
| Pipeline API | ✅ | ❌ | ✅ | N/A |
| REST API | ✅ | ⚠️ | ✅ | N/A |
| GPU Acceleration | ✅ | ✅ | ✅ | N/A |
| Distributed Training | ✅ | ❌ | ✅ | N/A |
| Compliance Reports | ✅ | ❌ | ✅ | N/A |
| Domain Generators | ✅ | ❌ | ❌ | ✅ |

---

## Migration Guides

### Migrating from SDV

```python
# Before (SDV)
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)
metadata.update_column('category', sdtype='categorical')

synth = CTGANSynthesizer(metadata, epochs=300)
synth.fit(df)
synthetic = synth.sample(1000)

# After (Genesis)
from genesis import SyntheticGenerator

synth = SyntheticGenerator(method='ctgan', config={'epochs': 300})
synth.fit(df, discrete_columns=['category'])
synthetic = synth.generate(1000)
```

### Migrating from Faker

```python
# Before (Faker) - no statistical learning
from faker import Faker
fake = Faker()

data = [{
    'name': fake.name(),
    'age': fake.random_int(18, 80),
    'city': fake.city()
} for _ in range(1000)]

# After (Genesis) - learns from real data
from genesis import auto_synthesize

# Learn from your actual data distributions
synthetic = auto_synthesize(real_data, n_samples=1000)
```

---

## Benchmarks

Performance on common datasets (lower is better for time, higher for quality):

| Dataset | Metric | Genesis | SDV |
|---------|--------|---------|-----|
| Adult Census | Quality Score | **0.94** | 0.92 |
| Adult Census | Training Time | **45s** | 52s |
| Credit Card Fraud | Quality Score | **0.91** | 0.89 |
| Credit Card Fraud | Training Time | 120s | **115s** |
| Covertype | Quality Score | **0.88** | 0.85 |
| Covertype | Training Time | **180s** | 210s |

*Benchmarks run on NVIDIA RTX 3080, CTGAN with 300 epochs. Your results may vary.*

---

## Summary

**Choose Genesis if you need:**
- ✅ MIT licensed open source
- ✅ Privacy-first with formal guarantees
- ✅ AutoML for automatic optimization
- ✅ Enterprise features (versioning, pipelines, drift detection)
- ✅ Self-hosted deployment
- ✅ Active development and community

**Try Genesis now:**
```bash
pip install genesis-synth
```

```python
from genesis import auto_synthesize
synthetic = auto_synthesize(your_data, n_samples=1000)
```
