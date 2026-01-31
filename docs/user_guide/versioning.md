# Dataset Versioning

Genesis provides Git-like versioning for synthetic datasets, enabling reproducible data generation pipelines with full history tracking.

## Overview

Dataset versioning allows you to:
- Track changes to synthetic datasets over time
- Branch and merge dataset variations
- Tag releases for reproducibility
- Compare versions and roll back changes

```python
from genesis.versioning import DatasetRepository

repo = DatasetRepository("./my_datasets")
repo.commit(df, message="Initial synthetic dataset")
```

## Components

| Component | Purpose |
|-----------|---------|
| **DatasetRepository** | Git-like repository for datasets |
| **VersionedGenerator** | Generator with automatic versioning |
| **DatasetDiff** | Compare two dataset versions |

## Quick Start

### Creating a Repository

```python
from genesis.versioning import DatasetRepository

# Create new repository
repo = DatasetRepository.init("./data_repo")

# Or open existing
repo = DatasetRepository("./data_repo")
```

### Basic Workflow

```python
# Generate and commit
synthetic_v1 = generator.generate(1000)
repo.commit(synthetic_v1, message="v1: Initial generation")

# Modify and commit again
synthetic_v2 = generator.generate(1000, privacy={"epsilon": 0.5})
repo.commit(synthetic_v2, message="v2: Added privacy")

# View history
for commit in repo.log():
    print(f"{commit.hash[:8]} - {commit.message}")
```

## Commits

### Creating Commits

```python
# Basic commit
repo.commit(df, message="Generated 1000 samples")

# With metadata
repo.commit(
    df,
    message="Production release",
    metadata={
        "generator": "ctgan",
        "epochs": 300,
        "privacy_epsilon": 1.0,
        "author": "data-team"
    }
)
```

### Viewing Commits

```python
# Get all commits
commits = repo.log()
for c in commits:
    print(f"""
Hash: {c.hash}
Date: {c.timestamp}
Message: {c.message}
Rows: {c.metadata.get('n_rows')}
""")

# Get specific commit
commit = repo.get_commit("abc123")
df = commit.load_data()
```

## Branches

Work with multiple variations of your data:

```python
# Create branch
repo.branch("experiment")

# Switch to branch
repo.checkout("experiment")

# Make changes on branch
modified_df = synthetic_df[synthetic_df["age"] > 21]
repo.commit(modified_df, message="Adults only subset")

# View branches
for branch in repo.branches():
    print(f"{'*' if branch.is_current else ' '} {branch.name}")

# Merge back to main
repo.checkout("main")
repo.merge("experiment")
```

### Branch Strategies

```python
# Feature branches for experiments
repo.branch("high-privacy")
repo.checkout("high-privacy")
# ... generate with epsilon=0.1
repo.commit(df, message="High privacy variant")

# Release branches
repo.checkout("main")
repo.branch("release-v1.0")
repo.tag("v1.0.0")
```

## Tags

Mark important versions:

```python
# Create tag
repo.tag("v1.0", message="Production release")

# List tags
for tag in repo.tags():
    print(f"{tag.name}: {tag.commit_hash[:8]} - {tag.message}")

# Checkout tag
repo.checkout("v1.0")
df = repo.get_current_data()
```

## Comparing Versions

```python
from genesis.versioning import DatasetDiff

# Compare two commits
diff = repo.diff("abc123", "def456")

print(f"Rows added: {diff.rows_added}")
print(f"Rows removed: {diff.rows_removed}")
print(f"Columns changed: {diff.columns_changed}")

# Statistical comparison
for col, changes in diff.column_changes.items():
    print(f"""
Column: {col}
  Mean change: {changes['mean_diff']:+.3f}
  Std change: {changes['std_diff']:+.3f}
""")

# Compare with current
diff = repo.diff("v1.0", "HEAD")
```

## Auto-Versioning with VersionedGenerator

```python
from genesis.versioning import VersionedGenerator

generator = VersionedGenerator(
    method="ctgan",
    repository="./data_repo"
)

generator.fit(training_data)

# Each generate() auto-commits
v1 = generator.generate(1000, message="Batch 1")
v2 = generator.generate(1000, message="Batch 2")

# View history
for commit in generator.repo.log():
    print(commit.message)
```

## Reproducibility

```python
# Get exact reproduction of a version
repo.checkout("v1.0")
df = repo.get_current_data()

# Include generation config in commits
config = {
    "method": "ctgan",
    "epochs": 300,
    "batch_size": 500,
    "random_seed": 42
}

repo.commit(
    df,
    message="Reproducible generation",
    metadata={"config": config}
)

# Later: reproduce exactly
commit = repo.get_commit_by_tag("v1.0")
config = commit.metadata["config"]
```

## Storage

### Storage Backends

```python
# Local filesystem (default)
repo = DatasetRepository("./data_repo")

# Custom storage path
repo = DatasetRepository(
    path="./data_repo",
    storage_format="parquet"  # or "csv"
)
```

### Content-Addressable Storage

Datasets are stored by content hash, enabling:
- Deduplication of identical datasets
- Efficient storage of similar versions
- Integrity verification

```python
# Check storage stats
stats = repo.storage_stats()
print(f"Total commits: {stats['n_commits']}")
print(f"Unique datasets: {stats['n_unique']}")
print(f"Storage used: {stats['storage_mb']:.1f} MB")
print(f"Deduplication ratio: {stats['dedup_ratio']:.1f}x")
```

## Pipeline Integration

```python
from genesis.pipeline import PipelineBuilder

pipeline = (
    PipelineBuilder()
    .source("training_data.csv")
    .add_node("generate", "synthesize", {
        "method": "ctgan",
        "n_samples": 10000
    })
    .add_node("version", "commit", {
        "repository": "./data_repo",
        "message": "Pipeline generated data"
    })
    .sink("latest_synthetic.csv")
    .build()
)

pipeline.execute()
```

## CLI Usage

```bash
# Initialize repository
genesis version init ./data_repo

# Commit data
genesis version commit -r ./data_repo -d synthetic.csv -m "Initial commit"

# View history
genesis version log -r ./data_repo

# Create tag
genesis version tag -r ./data_repo v1.0 -m "First release"

# Checkout version
genesis version checkout -r ./data_repo v1.0 -o output.csv

# Compare versions
genesis version diff -r ./data_repo HEAD~1 HEAD
```

## Best Practices

1. **Commit frequently**: Version after each significant generation
2. **Use meaningful messages**: Describe what changed and why
3. **Tag releases**: Mark versions used in production
4. **Include metadata**: Store generation parameters for reproducibility
5. **Branch for experiments**: Keep main clean, experiment on branches
6. **Review before merge**: Compare diffs before merging branches

## Example: Production Workflow

```python
from genesis.versioning import DatasetRepository, VersionedGenerator
from genesis import QualityEvaluator

repo = DatasetRepository.init("./production_data")
generator = VersionedGenerator(method="ctgan", repository="./production_data")

# Initial training
generator.fit(real_data)

# Development cycle
repo.branch("dev")
repo.checkout("dev")

for iteration in range(5):
    synthetic = generator.generate(10000)
    
    # Evaluate
    evaluator = QualityEvaluator(real_data, synthetic)
    report = evaluator.evaluate()
    
    repo.commit(
        synthetic,
        message=f"Iteration {iteration}: score={report.overall_score:.1%}",
        metadata={"quality_score": report.overall_score}
    )

# Find best version
best_commit = max(
    repo.log(branch="dev"),
    key=lambda c: c.metadata.get("quality_score", 0)
)

# Merge best to main
repo.checkout("main")
repo.cherry_pick(best_commit.hash)
repo.tag("production-v1", message="Best quality version")

# Export for use
repo.checkout("production-v1")
production_data = repo.get_current_data()
production_data.to_csv("production_synthetic.csv", index=False)
```
