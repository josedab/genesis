# Data Lineage & Provenance

Genesis provides blockchain-style data lineage tracking for complete audit trails of synthetic data generation.

## Overview

| Class | Purpose | Use Case |
|-------|---------|----------|
| **DataLineage** | Basic provenance tracking | Simple audit requirements |
| **LineageChain** | Immutable blockchain-style chain | Regulatory compliance, tamper detection |
| **LineageManifest** | Export/import provenance | Sharing and archiving |

## Quick Start

```python
from genesis.lineage import DataLineage

# Create lineage tracker
lineage = DataLineage()

# Record source data
lineage.record_source(
    name="customer_data",
    data=original_df,
    description="Customer records from CRM export",
)

# Record transformations
lineage.record_transformation(
    name="privacy_filter",
    input_name="customer_data",
    description="Removed PII columns",
)

# Record generation
lineage.record_generation(
    name="synthetic_customers",
    method="ctgan",
    n_samples=10000,
    parameters={"epochs": 300, "batch_size": 500},
)

# Export manifest
manifest = lineage.create_manifest()
manifest.save("lineage_manifest.json")
```

## LineageChain

Blockchain-style immutable audit trail with cryptographic verification.

```python
from genesis.lineage import LineageChain

# Create chain
chain = LineageChain()

# Add source data block
chain.add_source_block(
    data=original_df,
    name="training_data",
    metadata={
        "source": "database_export",
        "export_date": "2026-01-28",
        "row_count": len(original_df),
    }
)

# Add transformation blocks
chain.add_transform_block(
    operation="normalize",
    input_hash=chain.latest_hash,
    parameters={"columns": ["age", "income"]},
)

chain.add_transform_block(
    operation="encode_categorical",
    input_hash=chain.latest_hash,
    parameters={"columns": ["region", "status"]},
)

# Add generation block
chain.add_generation_block(
    method="ctgan",
    parameters={"epochs": 300},
    n_samples=10000,
    quality_metrics={"fidelity": 0.92, "privacy": 0.95},
)

# Verify chain integrity
is_valid = chain.verify()
print(f"Chain valid: {is_valid}")
```

### Block Types

#### Source Block
Records original data ingestion:

```python
chain.add_source_block(
    data=df,
    name="raw_data",
    metadata={
        "format": "csv",
        "encoding": "utf-8",
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
    }
)
```

#### Transform Block
Records data transformations:

```python
chain.add_transform_block(
    operation="filter_rows",
    input_hash=chain.latest_hash,
    parameters={
        "condition": "age >= 18",
        "rows_before": 10000,
        "rows_after": 8500,
    }
)
```

#### Generation Block
Records synthetic data generation:

```python
chain.add_generation_block(
    method="gaussian_copula",
    parameters={
        "epochs": 100,
        "batch_size": 500,
    },
    n_samples=50000,
    quality_metrics={
        "statistical_fidelity": 0.94,
        "ml_utility": 0.91,
        "privacy_score": 0.98,
    }
)
```

#### Quality Check Block
Records quality assessments:

```python
chain.add_quality_check(
    metrics={
        "ks_statistic_mean": 0.05,
        "correlation_diff": 0.03,
        "ml_accuracy_ratio": 0.97,
    },
    passed=True,
    threshold={"ml_accuracy_ratio": 0.90},
)
```

### Chain Verification

```python
# Verify entire chain
if chain.verify():
    print("✓ Chain integrity verified")
else:
    print("✗ Chain has been tampered with!")

# Get detailed verification
for i, block in enumerate(chain.blocks):
    is_valid = block.verify()
    print(f"Block {i} ({block.block_type}): {'✓' if is_valid else '✗'}")
```

### Export and Import

```python
# Export chain
chain.export("lineage_chain.json")

# Import and verify
loaded_chain = LineageChain.load("lineage_chain.json")
if loaded_chain.verify():
    print("Loaded chain is valid")
```

### Audit Trail

```python
# Get human-readable audit trail
trail = chain.get_audit_trail()

for entry in trail:
    print(f"[{entry['timestamp']}] {entry['block_type']}: {entry['description']}")
```

Output:
```
[2026-01-28T10:00:00] source: Ingested training_data (10000 rows)
[2026-01-28T10:05:00] transform: Applied normalize to columns
[2026-01-28T10:06:00] transform: Applied encode_categorical to columns
[2026-01-28T10:30:00] generation: Generated 50000 samples using ctgan
[2026-01-28T10:35:00] quality_check: Quality check passed
```

## DataLineage (Simple API)

For simpler use cases without blockchain verification:

```python
from genesis.lineage import DataLineage, SourceMetadata

# Create lineage tracker
lineage = DataLineage()

# Record with automatic metadata extraction
lineage.record_source(
    name="sales_data",
    data=sales_df,
    description="Q4 2025 sales records",
    tags=["sales", "quarterly", "2025"],
)

# Access recorded metadata
source = lineage.sources["sales_data"]
print(f"Columns: {source.columns}")
print(f"Row count: {source.n_rows}")
print(f"Column stats: {source.column_stats}")
```

### Manifest Export

```python
# Create exportable manifest
manifest = lineage.create_manifest()

# Save in multiple formats
manifest.save("lineage.json")           # JSON
manifest.save("lineage.yaml")           # YAML
manifest.to_dict()                       # Dictionary

# Load manifest
from genesis.lineage import LineageManifest
loaded = LineageManifest.load("lineage.json")
```

## Integration with Generators

```python
from genesis import SyntheticGenerator
from genesis.lineage import LineageChain

# Create lineage-tracked generator
chain = LineageChain()

# Record source
chain.add_source_block(data=training_data, name="input")

# Create and train generator
generator = SyntheticGenerator(method="ctgan")
generator.fit(training_data)

# Record training
chain.add_transform_block(
    operation="train_generator",
    input_hash=chain.latest_hash,
    parameters=generator.get_params(),
)

# Generate synthetic data
synthetic = generator.generate(10000)

# Record generation with quality metrics
report = generator.quality_report(training_data, synthetic)
chain.add_generation_block(
    method="ctgan",
    parameters=generator.get_params(),
    n_samples=10000,
    quality_metrics=report.to_dict(),
)

# Add quality check
chain.add_quality_check(
    metrics=report.to_dict(),
    passed=report.overall_score > 0.8,
    threshold={"overall_score": 0.8},
)

# Save everything
synthetic.to_csv("synthetic_data.csv", index=False)
chain.export("synthetic_data_lineage.json")
```

## Compliance Use Cases

### GDPR Compliance

```python
chain = LineageChain()

# Document data minimization
chain.add_source_block(
    data=raw_data,
    name="raw_customer_data",
    metadata={
        "legal_basis": "legitimate_interest",
        "data_controller": "ACME Corp",
        "retention_period": "3 years",
    }
)

# Document PII removal
chain.add_transform_block(
    operation="remove_pii",
    input_hash=chain.latest_hash,
    parameters={
        "removed_columns": ["name", "email", "phone", "ssn"],
        "reason": "GDPR data minimization",
    }
)

# Document synthetic generation
chain.add_generation_block(
    method="ctgan",
    parameters={"epochs": 300},
    n_samples=50000,
    quality_metrics={"privacy_score": 0.99},
)

# Export for compliance documentation
chain.export("gdpr_compliance_lineage.json")
```

### Healthcare (HIPAA)

```python
chain.add_source_block(
    data=patient_data,
    name="patient_records",
    metadata={
        "hipaa_category": "PHI",
        "deidentification_method": "safe_harbor",
        "covered_entity": "Hospital XYZ",
    }
)

chain.add_transform_block(
    operation="deidentify",
    input_hash=chain.latest_hash,
    parameters={
        "method": "safe_harbor",
        "removed_identifiers": [
            "name", "address", "dates", "phone", "fax", 
            "email", "ssn", "mrn", "account_numbers"
        ],
    }
)
```

## Block Structure

Each block in the chain contains:

```python
{
    "block_id": "uuid-string",
    "block_type": "source|transform|generation|quality_check",
    "timestamp": "2026-01-28T10:00:00Z",
    "previous_hash": "sha256-of-previous-block",
    "data_hash": "sha256-of-block-content",
    "content": {
        # Block-specific content
    },
    "metadata": {
        # Optional metadata
    }
}
```

## Best Practices

### 1. Record Everything
```python
# Don't skip transformations
chain.add_transform_block(operation="drop_na", ...)
chain.add_transform_block(operation="normalize", ...)
chain.add_transform_block(operation="encode", ...)
```

### 2. Include Quality Metrics
```python
chain.add_generation_block(
    method="ctgan",
    quality_metrics={
        "statistical_fidelity": 0.94,
        "ml_utility": 0.91,
        "privacy_score": 0.98,
        "coverage": 0.96,
    }
)
```

### 3. Verify Before Sharing
```python
# Always verify before export
assert chain.verify(), "Chain integrity compromised!"
chain.export("lineage.json")
```

### 4. Store Lineage with Data
```python
# Keep lineage alongside synthetic data
synthetic.to_parquet("data/synthetic_v1.parquet")
chain.export("data/synthetic_v1_lineage.json")
```
