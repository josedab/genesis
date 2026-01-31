# ADR-0017: Git-Like Dataset Versioning System

## Status

Accepted

## Context

Synthetic data generation pipelines face versioning challenges:

1. **Reproducibility**: "Which version of synthetic data trained model v2.3?"
2. **Rollback**: "The new synthetic data degraded model performance—revert to last week's"
3. **Experimentation**: "Try different generation parameters without losing the original"
4. **Collaboration**: "Share a specific dataset version with the compliance team"
5. **Audit**: "Show the history of changes to this synthetic dataset"

Traditional approaches have limitations:

- **File naming** (`data_v1.csv`, `data_v2_final_FINAL.csv`): Unscalable, no metadata
- **Git LFS**: Works but heavyweight, not designed for DataFrames
- **Database snapshots**: Requires infrastructure, not portable
- **Cloud versioning** (S3 versions): Vendor-specific, no branching

We wanted a git-like experience purpose-built for synthetic data.

## Decision

We implement a **git-inspired versioning system** with commits, branches, tags, and diffs:

```python
from genesis.versioning import DatasetRepository

# Initialize repository
repo = DatasetRepository.init("./my_data_repo")

# Commit versions
repo.commit(df_v1, message="Initial synthetic dataset")
repo.commit(df_v2, message="Added 5000 more samples")
repo.commit(df_v3, message="Fixed age distribution")

# View history
for commit in repo.log():
    print(f"{commit.hash[:8]} - {commit.message}")
# a1b2c3d4 - Fixed age distribution
# e5f6g7h8 - Added 5000 more samples
# i9j0k1l2 - Initial synthetic dataset

# Tag for release
repo.tag("v1.0", message="Production release")

# Compare versions
diff = repo.diff("v1.0", "HEAD")
print(f"Rows added: {diff.rows_added}")
print(f"Schema changes: {diff.columns_added}")

# Branch for experimentation
repo.branch("experiment-ctgan")
repo.checkout("experiment-ctgan")
repo.commit(experimental_df, message="Try CTGAN instead of TVAE")

# Checkout specific version
old_data = repo.checkout("v1.0")
```

### Storage Architecture

```
my_data_repo/
├── .genesis/
│   ├── config.json          # Repository configuration
│   ├── HEAD                  # Current branch reference
│   ├── refs/
│   │   ├── heads/
│   │   │   ├── main         # Branch pointers
│   │   │   └── experiment
│   │   └── tags/
│   │       └── v1.0         # Tag pointers
│   ├── objects/             # Content-addressable storage
│   │   ├── a1/
│   │   │   └── b2c3d4...    # Commit object
│   │   └── e5/
│   │       └── f6g7h8...    # Data blob (parquet)
│   └── commits/
│       └── index.json       # Commit index for fast lookup
```

### Content-Addressable Storage

Identical datasets share storage via content hashing:

```python
def _compute_data_hash(self, df: pd.DataFrame) -> str:
    """SHA-256 of DataFrame content."""
    content = df.to_csv(index=False).encode()
    return hashlib.sha256(content).hexdigest()
```

## Consequences

### Positive

- **Familiar workflow**: Git users immediately understand the model
- **Space efficient**: Identical data deduplicated via content addressing
- **Portable**: Repository is a directory, easy to zip/share
- **Queryable**: Commits have metadata (generator config, quality scores)
- **Atomic operations**: Commits are all-or-nothing

### Negative

- **Storage overhead**: Each version stored fully (no delta compression yet)
- **Large datasets**: Multi-GB datasets make commits slow
- **No distributed sync**: Unlike git, no push/pull to remotes (yet)
- **Learning curve**: Another versioning system to learn

### DatasetVersion Structure

```python
@dataclass
class DatasetVersion:
    version_id: str           # Content hash
    parent_id: Optional[str]  # Previous version
    message: str              # Commit message
    timestamp: str            # ISO 8601
    metadata: Dict[str, Any]  # Generator config, quality scores
    data_hash: str            # Hash of actual data
    n_rows: int
    n_columns: int
    columns: List[str]
    tags: List[str]
```

### Diff Output

```python
@dataclass
class DatasetDiff:
    version_a: str
    version_b: str
    rows_added: int
    rows_removed: int
    columns_added: List[str]
    columns_removed: List[str]
    columns_modified: List[str]
    statistical_changes: Dict[str, Dict[str, float]]
    # e.g., {"age": {"mean_delta": 2.3, "std_delta": -0.5}}
```

## Auto-Versioning Generator

For automatic versioning on every generation:

```python
from genesis.versioning import VersionedGenerator

generator = VersionedGenerator(
    method="ctgan",
    repository="./data_repo"
)
generator.fit(training_data)

# Each generate() auto-commits
synthetic = generator.generate(1000, message="Batch generation for model training")
# Automatically committed to repository

# Later: reproduce exact dataset
old_synthetic = generator.checkout_generate("a1b2c3d4")
```

## Examples

```python
# Initialize and populate
from genesis.versioning import DatasetRepository

repo = DatasetRepository.init("./synthetic_customers")

# Development workflow
repo.commit(df1, message="Initial generation with Gaussian Copula")
repo.commit(df2, message="Switch to CTGAN for better categoricals")
repo.commit(df3, message="Increase epsilon for more privacy")

# Tag stable version
repo.tag("v1.0-beta", message="Ready for QA testing")

# Experiment without affecting main
repo.branch("high-privacy")
repo.checkout("high-privacy")
repo.commit(df_private, message="epsilon=0.1 for maximum privacy")

# Compare branches
diff = repo.diff("main", "high-privacy")
print(diff.statistical_changes)

# Merge if satisfied
repo.checkout("main")
repo.merge("high-privacy")

# Export specific version for compliance
v1_data = repo.checkout("v1.0-beta")
v1_data.to_parquet("compliance_submission.parquet")
```

## CLI Integration

```bash
# Initialize repository
genesis version init ./my_repo

# Commit synthetic data
genesis version commit ./data.csv -m "Initial dataset"

# View history
genesis version log

# Create tag
genesis version tag v1.0 -m "Production release"

# Diff versions
genesis version diff v1.0 HEAD

# Checkout version
genesis version checkout v1.0 -o ./output.csv
```
