# ADR-0013: Blockchain-Style Immutable Lineage Tracking

## Status

Accepted

## Context

Enterprise users of synthetic data face strict compliance requirements:

- **GDPR Article 35**: Data Protection Impact Assessments require documenting data processing
- **HIPAA**: Healthcare data requires audit trails showing data transformations
- **SOC 2**: Requires demonstrable data integrity controls
- **Internal governance**: Organizations need to trace synthetic data back to source

Traditional logging approaches have limitations:

- Log files can be modified or deleted
- Database records can be updated
- File-based manifests lack integrity verification
- No standard format for cross-system verification

We needed a provenance system that:

1. **Immutable**: Past records cannot be modified
2. **Verifiable**: Integrity can be checked cryptographically  
3. **Portable**: Export to standard formats for external audit
4. **Lightweight**: No external dependencies (no actual blockchain)

## Decision

We implement a **blockchain-inspired lineage chain** where each operation creates a cryptographically-linked block:

```python
from genesis.lineage import LineageChain

chain = LineageChain()

# Each operation adds a block
chain.add_source(real_data, "customers_v1")
chain.add_generation(generator, synthetic_data)
chain.add_transformation("filter", {"condition": "age > 18"}, filtered_data)
chain.add_quality_check(quality_report, passed=True)

# Verify integrity
assert chain.verify()  # Checks all hashes

# Export for compliance
chain.export("audit_trail.json")
```

Block structure:

```python
@dataclass
class LineageBlock:
    block_id: str           # UUID
    previous_hash: str      # SHA-256 of previous block
    timestamp: str          # ISO 8601
    action: str             # "source", "generation", "transformation", "quality"
    data_hash: str          # SHA-256 of data at this point
    metadata: Dict[str, Any]
    block_hash: str         # SHA-256 of this block's contents
```

Chain verification:

```python
def verify(self) -> bool:
    for i, block in enumerate(self._blocks):
        # Verify block's own hash
        if not block.verify():
            return False
        
        # Verify chain linkage
        if i > 0 and block.previous_hash != self._blocks[i-1].block_hash:
            return False
    
    return True
```

## Consequences

### Positive

- **Tamper-evident**: Any modification breaks the hash chain
- **Self-contained**: No external blockchain infrastructure required
- **Auditable**: JSON export provides human-readable audit trail
- **Compliance-ready**: Maps directly to GDPR/HIPAA documentation requirements
- **Lightweight**: Pure Python, ~500 lines, no dependencies beyond hashlib

### Negative

- **Not distributed**: Unlike true blockchain, doesn't have consensus mechanism
- **Single point of failure**: Chain file can be deleted (though tampering is detectable)
- **Storage growth**: Full chain kept in memory and on disk
- **No rollback**: Immutable by design; mistakes require new chain

### Design Choices

**Why not actual blockchain?**
- Overkill for single-organization use
- Adds infrastructure complexity (nodes, consensus)
- Most users need audit trail, not decentralization

**Why SHA-256?**
- Industry standard for integrity verification
- Fast enough for data science workflows
- Recognized by compliance frameworks

**Why JSON export?**
- Human-readable for auditors
- Easy to parse programmatically
- Standard format across tools

## SBOM-Style Export

For supply chain transparency, we support SBOM-like format:

```python
manifest = lineage.create_manifest()
sbom = manifest.to_sbom_format()

# Output follows CycloneDX-inspired structure
{
    "bomFormat": "GenesisDataBOM",
    "specVersion": "1.0",
    "components": [
        {"type": "data", "name": "source-data", "hashes": [...]},
        {"type": "data", "name": "synthetic-data", "hashes": [...]}
    ],
    "dependencies": [
        {"ref": "synthetic-data", "dependsOn": ["source-data"]}
    ]
}
```

## Examples

```python
# Simple lineage tracking
from genesis.lineage import DataLineage

lineage = DataLineage(creator="data-team", purpose="ML training")
lineage.record_source(real_data, "customers_q4_2024")
lineage.record_generation(generator, synthetic_data, execution_time=45.2)
lineage.record_quality(quality_report)

manifest = lineage.create_manifest()
manifest.save("synthetic_customers_manifest.json")

# Full chain with verification
chain = LineageChain()
chain.add_source(df, "input")
chain.add_generation(gen, synthetic)

# Later: verify nothing was tampered
loaded_chain = LineageChain.load("audit_trail.json")
if not loaded_chain.verify():
    raise SecurityError("Lineage chain has been tampered with!")
```
