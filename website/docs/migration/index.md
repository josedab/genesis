---
sidebar_position: 1
title: Overview
---

# Migration Guides

Step-by-step guides to migrate from other synthetic data tools to Genesis.

## Why Migrate?

Genesis offers several advantages over other tools:

| Feature | Benefit |
|---------|---------|
| **MIT License** | Full commercial freedom, no BSL restrictions |
| **AutoML** | Automatic method selection for optimal results |
| **Built-in Privacy** | Differential privacy and attack testing included |
| **Enterprise Features** | Versioning, pipelines, drift detection, REST API |
| **Active Development** | Regular releases with new features |

## Migration Guides

<div className="row">
  <div className="col col--4">
    <div className="card">
      <div className="card__header">
        <h3>From SDV</h3>
      </div>
      <div className="card__body">
        <p>Migrate from Synthetic Data Vault (SDV) to Genesis.</p>
      </div>
      <div className="card__footer">
        <a className="button button--primary button--block" href="/docs/migration/from-sdv">Migration Guide</a>
      </div>
    </div>
  </div>
  <div className="col col--4">
    <div className="card">
      <div className="card__header">
        <h3>From Faker</h3>
      </div>
      <div className="card__body">
        <p>Upgrade from Faker templates to statistical learning.</p>
      </div>
      <div className="card__footer">
        <a className="button button--primary button--block" href="/docs/migration/from-faker">Migration Guide</a>
      </div>
    </div>
  </div>
  <div className="col col--4">
    <div className="card">
      <div className="card__header">
        <h3>From Gretel</h3>
      </div>
      <div className="card__body">
        <p>Move from Gretel.ai cloud to self-hosted Genesis.</p>
      </div>
      <div className="card__footer">
        <a className="button button--primary button--block" href="/docs/migration/from-gretel">Migration Guide</a>
      </div>
    </div>
  </div>
</div>

## Quick Comparison

```python
# SDV (before)
from sdv.single_table import CTGANSynthesizer
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df)
synth = CTGANSynthesizer(metadata)
synth.fit(df)
synthetic = synth.sample(1000)

# Genesis (after) - simpler!
from genesis import auto_synthesize
synthetic = auto_synthesize(df, n_samples=1000)
```

## Compatibility

Genesis can read models and data from:
- SDV metadata files
- Pandas DataFrames (CSV, Parquet, Excel)
- SQL databases
- JSON/JSONL files

## Getting Help

Migration questions? 
- [GitHub Discussions](https://github.com/genesis-synth/genesis/discussions)
- [Open an Issue](https://github.com/genesis-synth/genesis/issues)
