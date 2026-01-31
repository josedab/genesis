---
sidebar_position: 10
title: Dataset Versioning
---

# Dataset Versioning

Track, compare, and manage versions of your synthetic datasets with content-addressable storage.

## Quick Start

```python
from genesis.versioning import DatasetRepository

# Initialize repository
repo = DatasetRepository('./datasets')

# Save a dataset version
version_id = repo.save(
    df,
    message="Initial customer dataset",
    tags=['production', 'v1.0']
)

print(f"Saved as: {version_id}")
```

## Core Concepts

### Content-Addressable Storage

Each dataset version is identified by its content hash:

```python
# Same data = same ID (deduplication)
id1 = repo.save(df, message="First save")
id2 = repo.save(df, message="Duplicate save")

assert id1 == id2  # No duplicate storage!
```

### Version Metadata

```python
version = repo.get_version(version_id)

print(f"ID: {version.id}")
print(f"Created: {version.created_at}")
print(f"Message: {version.message}")
print(f"Tags: {version.tags}")
print(f"Schema: {version.schema}")
print(f"Row count: {version.row_count}")
print(f"Size: {version.size_bytes}")
```

## Saving Datasets

### Basic Save

```python
version_id = repo.save(df, message="Customer data Q1 2024")
```

### With Tags

```python
version_id = repo.save(
    df,
    message="Production dataset for ML model",
    tags=['production', 'ml-ready', 'v2.1']
)
```

### With Metadata

```python
version_id = repo.save(
    df,
    message="Training data",
    tags=['training'],
    metadata={
        'source': 'data_pipeline_v3',
        'model_version': '2.1.0',
        'quality_score': 0.95
    }
)
```

## Loading Datasets

### By Version ID

```python
df = repo.load(version_id)
```

### By Tag

```python
# Load latest with tag
df = repo.load(tag='production')

# Load specific version with tag
df = repo.load(tag='v2.1')
```

### Latest Version

```python
df = repo.load_latest()
```

## Listing Versions

```python
# All versions
versions = repo.list_versions()
for v in versions:
    print(f"{v.id[:8]} | {v.created_at} | {v.message}")

# Filter by tag
production_versions = repo.list_versions(tag='production')

# Filter by date
recent = repo.list_versions(since='2024-01-01')
```

## Comparing Versions

```python
from genesis.versioning import compare_versions

# Compare two versions
diff = repo.compare(version_id_1, version_id_2)

print(f"Schema changes: {diff.schema_changes}")
print(f"Row count: {diff.row_count_change}")
print(f"Statistical drift: {diff.drift_score}")
print(f"Changed columns: {diff.changed_columns}")
```

### Detailed Comparison

```python
diff = repo.compare(v1, v2, detailed=True)

# Per-column changes
for col, changes in diff.column_changes.items():
    print(f"{col}:")
    print(f"  Mean: {changes['mean_before']} → {changes['mean_after']}")
    print(f"  Std: {changes['std_before']} → {changes['std_after']}")
```

## Branching

Create parallel version lines:

```python
# Create a branch
repo.create_branch('experiment')

# Switch to branch
repo.checkout('experiment')

# Save to branch
version_id = repo.save(df, message="Experimental changes")

# Switch back to main
repo.checkout('main')
```

## Rollback

Revert to a previous version:

```python
# Rollback to specific version
repo.rollback(version_id)

# Rollback by steps
repo.rollback(steps=2)  # Go back 2 versions

# Rollback to tag
repo.rollback(tag='v1.0')
```

## Tagging

```python
# Add tag to existing version
repo.tag(version_id, 'production')

# Add multiple tags
repo.tag(version_id, ['stable', 'v2.0', 'approved'])

# Remove tag
repo.untag(version_id, 'experimental')

# List tags
all_tags = repo.list_tags()
```

## Garbage Collection

Remove unreferenced versions:

```python
# Preview what would be deleted
orphans = repo.gc(dry_run=True)
print(f"Would delete: {len(orphans)} versions")

# Actually delete
deleted = repo.gc()
print(f"Deleted: {len(deleted)} versions")
```

## Repository Information

```python
info = repo.info()

print(f"Total versions: {info['version_count']}")
print(f"Total size: {info['total_size_mb']:.1f} MB")
print(f"Unique datasets: {info['unique_datasets']}")
print(f"Storage saved by dedup: {info['dedup_savings_mb']:.1f} MB")
```

## Complete Example

```python
import pandas as pd
from genesis import SyntheticGenerator
from genesis.versioning import DatasetRepository

# Setup
repo = DatasetRepository('./synthetic_data_repo')
real_data = pd.read_csv('customers.csv')

# Generate synthetic data
generator = SyntheticGenerator(method='ctgan')
generator.fit(real_data)
synthetic_v1 = generator.generate(10000)

# Save initial version
v1_id = repo.save(
    synthetic_v1,
    message="Initial synthetic customer data",
    tags=['v1.0', 'production'],
    metadata={
        'generator': 'ctgan',
        'training_rows': len(real_data),
        'quality_score': 0.92
    }
)
print(f"Saved v1: {v1_id}")

# Later: generate improved version
generator = SyntheticGenerator(method='ctgan', config={'epochs': 500})
generator.fit(real_data)
synthetic_v2 = generator.generate(10000)

# Save new version
v2_id = repo.save(
    synthetic_v2,
    message="Improved with more training epochs",
    tags=['v2.0'],
    metadata={
        'generator': 'ctgan',
        'epochs': 500,
        'quality_score': 0.95
    }
)

# Compare versions
diff = repo.compare(v1_id, v2_id)
print(f"\nVersion comparison:")
print(f"  Drift score: {diff.drift_score:.3f}")
print(f"  Quality improvement: {diff.metadata_diff.get('quality_score')}")

# Promote v2 to production
repo.tag(v2_id, 'production')
repo.untag(v1_id, 'production')

# List production history
print("\nProduction versions:")
for v in repo.list_versions(tag='production'):
    print(f"  {v.id[:8]} | {v.created_at}")

# Load current production
production_data = repo.load(tag='production')
```

## CLI Usage

```bash
# Initialize repository
genesis version init ./datasets

# Save a dataset
genesis version save data.csv --message "Initial version" --tag v1.0

# List versions
genesis version list

# Load a version
genesis version load abc123 --output loaded_data.csv

# Compare versions
genesis version compare abc123 def456

# Tag a version
genesis version tag abc123 production

# Rollback
genesis version rollback --tag v1.0
```

## Integration with Pipelines

```python
from genesis import Pipeline
from genesis.versioning import VersionedOutput

pipeline = Pipeline([
    ('generate', SyntheticGenerator(method='ctgan')),
    ('evaluate', QualityEvaluator()),
    ('version', VersionedOutput(
        repo='./datasets',
        tag_on_success='latest',
        metadata_from='evaluate'
    ))
])

result = pipeline.run(real_data, n_samples=10000)
print(f"Saved version: {result['version_id']}")
```

## Best Practices

1. **Use meaningful messages** - Describe what changed and why
2. **Tag releases** - Use semantic versioning (v1.0, v1.1, v2.0)
3. **Track metadata** - Quality scores, generation params, source info
4. **Compare before promoting** - Check drift and quality
5. **Periodic garbage collection** - Clean up old versions

## Storage Backends

### Local File System (Default)

```python
repo = DatasetRepository('./datasets')
```

### S3-Compatible

```python
repo = DatasetRepository(
    's3://my-bucket/datasets',
    aws_access_key_id='...',
    aws_secret_access_key='...'
)
```

### Custom Backend

```python
from genesis.versioning import StorageBackend

class MyBackend(StorageBackend):
    def save(self, data, key): ...
    def load(self, key): ...
    def delete(self, key): ...
    def exists(self, key): ...

repo = DatasetRepository(backend=MyBackend())
```

## Next Steps

- **[Drift Detection](/docs/guides/drift-detection)** - Compare version distributions
- **[Pipelines](/docs/guides/pipelines)** - Automated versioning workflows
- **[Evaluation](/docs/concepts/evaluation)** - Track quality across versions
