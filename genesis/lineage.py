"""Data lineage and provenance tracking for synthetic data.

This module provides comprehensive tracking of:
- Source data characteristics
- Generation parameters and configuration
- Quality metrics
- Audit trail for compliance

Example:
    >>> from genesis.lineage import DataLineage
    >>>
    >>> lineage = DataLineage()
    >>> lineage.record_source(real_data, "customer_data_v1")
    >>> lineage.record_generation(generator, synthetic_data)
    >>> manifest = lineage.create_manifest()
    >>> manifest.save("synthetic_data_manifest.json")
"""

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)

# Chunk size for streaming hash computation (1MB)
HASH_CHUNK_SIZE = 1024 * 1024


def _compute_dataframe_hash(df: pd.DataFrame, chunk_size: int = HASH_CHUNK_SIZE) -> str:
    """Compute SHA-256 hash of DataFrame using streaming approach.

    Uses chunked processing to avoid memory issues with large DataFrames.
    Processes the DataFrame in row chunks, converting each chunk to CSV
    and updating the hash incrementally.

    Args:
        df: DataFrame to hash
        chunk_size: Approximate chunk size in bytes for processing

    Returns:
        SHA-256 hash hex string
    """
    hasher = hashlib.sha256()

    # Estimate rows per chunk based on average row size
    if len(df) == 0:
        return hasher.hexdigest()

    # Sample first 100 rows to estimate row size
    sample_size = min(100, len(df))
    sample_bytes = df.head(sample_size).to_csv(index=False).encode()
    avg_row_size = len(sample_bytes) / sample_size if sample_size > 0 else 100
    rows_per_chunk = max(1, int(chunk_size / avg_row_size))

    # Process DataFrame in chunks
    for start_idx in range(0, len(df), rows_per_chunk):
        end_idx = min(start_idx + rows_per_chunk, len(df))
        chunk = df.iloc[start_idx:end_idx]

        # Include header only for first chunk
        include_header = start_idx == 0
        chunk_bytes = chunk.to_csv(index=False, header=include_header).encode()
        hasher.update(chunk_bytes)

    return hasher.hexdigest()


@dataclass
class SourceMetadata:
    """Metadata about source/training data."""

    source_id: str
    name: str
    n_rows: int
    n_columns: int
    columns: List[Dict[str, Any]]
    hash_sha256: str
    created_at: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> "SourceMetadata":
        """Create metadata from a DataFrame."""
        # Compute hash of data using streaming approach for memory efficiency
        data_hash = _compute_dataframe_hash(df)

        # Column info
        columns = []
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "n_unique": int(df[col].nunique()),
                "null_count": int(df[col].isna().sum()),
            }
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info["min"] = float(df[col].min()) if not df[col].isna().all() else None
                col_info["max"] = float(df[col].max()) if not df[col].isna().all() else None
                col_info["mean"] = float(df[col].mean()) if not df[col].isna().all() else None
            columns.append(col_info)

        return cls(
            source_id=str(uuid.uuid4()),
            name=name,
            n_rows=len(df),
            n_columns=len(df.columns),
            columns=columns,
            hash_sha256=data_hash,
            created_at=datetime.utcnow().isoformat(),
            description=description,
            tags=tags or [],
        )


@dataclass
class GenerationRecord:
    """Record of a synthetic data generation run."""

    generation_id: str
    generator_type: str
    generator_config: Dict[str, Any]
    privacy_config: Dict[str, Any]
    source_id: str
    n_samples: int
    conditions: Optional[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    execution_time_seconds: float
    created_at: str
    random_seed: Optional[int] = None

    @classmethod
    def from_generator(
        cls,
        generator: Any,
        source_id: str,
        n_samples: int,
        execution_time: float,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> "GenerationRecord":
        """Create record from generator."""
        params = generator.get_parameters()

        constraints = []
        if hasattr(generator, "_constraints"):
            for c in generator._constraints.constraints:
                constraints.append(
                    {
                        "type": c.__class__.__name__,
                        "column": getattr(c, "column", None),
                    }
                )

        return cls(
            generation_id=str(uuid.uuid4()),
            generator_type=generator.__class__.__name__,
            generator_config=params.get("config", {}),
            privacy_config=params.get("privacy", {}),
            source_id=source_id,
            n_samples=n_samples,
            conditions=conditions,
            constraints=constraints,
            execution_time_seconds=execution_time,
            created_at=datetime.utcnow().isoformat(),
            random_seed=params.get("config", {}).get("random_seed"),
        )


@dataclass
class QualityRecord:
    """Record of quality evaluation results."""

    evaluation_id: str
    generation_id: str
    overall_score: float
    statistical_metrics: Dict[str, float]
    ml_utility_metrics: Dict[str, float]
    privacy_metrics: Dict[str, float]
    created_at: str

    @classmethod
    def from_report(
        cls,
        report: Any,
        generation_id: str,
    ) -> "QualityRecord":
        """Create record from quality report."""
        metrics = report.to_dict()

        return cls(
            evaluation_id=str(uuid.uuid4()),
            generation_id=generation_id,
            overall_score=metrics.get("overall_score", 0.0),
            statistical_metrics=metrics.get("statistical", {}),
            ml_utility_metrics=metrics.get("ml_utility", {}),
            privacy_metrics=metrics.get("privacy", {}),
            created_at=datetime.utcnow().isoformat(),
        )


@dataclass
class DataManifest:
    """Complete manifest for a synthetic dataset.

    This provides full provenance information for compliance and audit purposes.
    """

    manifest_version: str = "1.0.0"
    manifest_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Provenance
    source: Optional[SourceMetadata] = None
    generation: Optional[GenerationRecord] = None
    quality: Optional[QualityRecord] = None

    # Synthetic data info
    synthetic_hash: Optional[str] = None
    synthetic_n_rows: Optional[int] = None

    # Audit
    creator: Optional[str] = None
    purpose: Optional[str] = None
    retention_days: Optional[int] = None

    def save(self, path: Union[str, Path]) -> None:
        """Save manifest to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info(f"Saved manifest to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DataManifest":
        """Load manifest from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "manifest_version": self.manifest_version,
            "manifest_id": self.manifest_id,
            "created_at": self.created_at,
            "source": asdict(self.source) if self.source else None,
            "generation": asdict(self.generation) if self.generation else None,
            "quality": asdict(self.quality) if self.quality else None,
            "synthetic_hash": self.synthetic_hash,
            "synthetic_n_rows": self.synthetic_n_rows,
            "creator": self.creator,
            "purpose": self.purpose,
            "retention_days": self.retention_days,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataManifest":
        """Create from dictionary."""
        manifest = cls(
            manifest_version=data.get("manifest_version", "1.0.0"),
            manifest_id=data.get("manifest_id", str(uuid.uuid4())),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            synthetic_hash=data.get("synthetic_hash"),
            synthetic_n_rows=data.get("synthetic_n_rows"),
            creator=data.get("creator"),
            purpose=data.get("purpose"),
            retention_days=data.get("retention_days"),
        )

        if data.get("source"):
            manifest.source = SourceMetadata(**data["source"])
        if data.get("generation"):
            manifest.generation = GenerationRecord(**data["generation"])
        if data.get("quality"):
            manifest.quality = QualityRecord(**data["quality"])

        return manifest

    def to_sbom_format(self) -> Dict[str, Any]:
        """Export in SBOM-like format for supply chain transparency.

        Based on CycloneDX format adapted for data.
        """
        return {
            "bomFormat": "GenesisDataBOM",
            "specVersion": "1.0",
            "version": 1,
            "metadata": {
                "timestamp": self.created_at,
                "tools": [
                    {
                        "vendor": "Genesis",
                        "name": "synthetic-data-generator",
                        "version": self.manifest_version,
                    }
                ],
                "component": {
                    "type": "data",
                    "name": self.source.name if self.source else "synthetic-dataset",
                    "version": self.manifest_id[:8],
                },
            },
            "components": [
                {
                    "type": "data",
                    "name": "source-data",
                    "hashes": [
                        {
                            "alg": "SHA-256",
                            "content": self.source.hash_sha256 if self.source else None,
                        }
                    ],
                    "properties": [
                        {"name": "n_rows", "value": str(self.source.n_rows if self.source else 0)},
                        {
                            "name": "n_columns",
                            "value": str(self.source.n_columns if self.source else 0),
                        },
                    ],
                },
                {
                    "type": "data",
                    "name": "synthetic-data",
                    "hashes": [
                        {
                            "alg": "SHA-256",
                            "content": self.synthetic_hash,
                        }
                    ],
                    "properties": [
                        {"name": "n_rows", "value": str(self.synthetic_n_rows or 0)},
                        {
                            "name": "generator",
                            "value": self.generation.generator_type if self.generation else None,
                        },
                    ],
                },
            ],
            "dependencies": [
                {
                    "ref": "synthetic-data",
                    "dependsOn": ["source-data"],
                }
            ],
        }


class DataLineage:
    """Track lineage and provenance of synthetic data generation.

    Example:
        >>> lineage = DataLineage(creator="data-team", purpose="ML training")
        >>> lineage.record_source(real_data, "customers_v1")
        >>>
        >>> generator.fit(real_data)
        >>> start = time.time()
        >>> synthetic = generator.generate(10000)
        >>>
        >>> lineage.record_generation(generator, synthetic, time.time() - start)
        >>> lineage.record_quality(quality_report)
        >>>
        >>> manifest = lineage.create_manifest()
        >>> manifest.save("customers_synthetic_manifest.json")
    """

    def __init__(
        self,
        creator: Optional[str] = None,
        purpose: Optional[str] = None,
        retention_days: Optional[int] = None,
    ) -> None:
        """Initialize lineage tracker.

        Args:
            creator: Who created this synthetic data
            purpose: Why the synthetic data was created
            retention_days: Data retention policy in days
        """
        self.creator = creator
        self.purpose = purpose
        self.retention_days = retention_days

        self._source: Optional[SourceMetadata] = None
        self._generation: Optional[GenerationRecord] = None
        self._quality: Optional[QualityRecord] = None
        self._synthetic_hash: Optional[str] = None
        self._synthetic_n_rows: Optional[int] = None

    def record_source(
        self,
        data: pd.DataFrame,
        name: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Record source data metadata.

        Args:
            data: Source DataFrame
            name: Name/identifier for the source
            description: Optional description
            tags: Optional tags

        Returns:
            Source ID
        """
        self._source = SourceMetadata.from_dataframe(data, name, description, tags)
        logger.info(f"Recorded source: {name} ({self._source.source_id})")
        return self._source.source_id

    def record_generation(
        self,
        generator: Any,
        synthetic_data: pd.DataFrame,
        execution_time: float,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record generation run.

        Args:
            generator: The generator that was used
            synthetic_data: Generated data
            execution_time: Time taken in seconds
            conditions: Optional generation conditions

        Returns:
            Generation ID
        """
        if self._source is None:
            raise ValueError("Must record source before generation")

        self._generation = GenerationRecord.from_generator(
            generator=generator,
            source_id=self._source.source_id,
            n_samples=len(synthetic_data),
            execution_time=execution_time,
            conditions=conditions,
        )

        # Record synthetic data hash using streaming approach for memory efficiency
        self._synthetic_hash = _compute_dataframe_hash(synthetic_data)
        self._synthetic_n_rows = len(synthetic_data)

        logger.info(f"Recorded generation: {self._generation.generation_id}")
        return self._generation.generation_id

    def record_quality(self, report: Any) -> str:
        """Record quality evaluation results.

        Args:
            report: QualityReport instance

        Returns:
            Evaluation ID
        """
        if self._generation is None:
            raise ValueError("Must record generation before quality")

        self._quality = QualityRecord.from_report(
            report=report,
            generation_id=self._generation.generation_id,
        )

        logger.info(f"Recorded quality: {self._quality.evaluation_id}")
        return self._quality.evaluation_id

    def create_manifest(self) -> DataManifest:
        """Create a complete data manifest.

        Returns:
            DataManifest with all recorded information
        """
        return DataManifest(
            source=self._source,
            generation=self._generation,
            quality=self._quality,
            synthetic_hash=self._synthetic_hash,
            synthetic_n_rows=self._synthetic_n_rows,
            creator=self.creator,
            purpose=self.purpose,
            retention_days=self.retention_days,
        )

    def reset(self) -> None:
        """Reset all tracked information."""
        self._source = None
        self._generation = None
        self._quality = None
        self._synthetic_hash = None
        self._synthetic_n_rows = None


def create_manifest_for_dataset(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    generator: Any,
    source_name: str,
    execution_time: float,
    quality_report: Optional[Any] = None,
    creator: Optional[str] = None,
    purpose: Optional[str] = None,
) -> DataManifest:
    """Convenience function to create a complete manifest.

    Args:
        real_data: Original training data
        synthetic_data: Generated synthetic data
        generator: Generator that was used
        source_name: Name for the source data
        execution_time: Generation time in seconds
        quality_report: Optional quality report
        creator: Who created the data
        purpose: Why it was created

    Returns:
        Complete DataManifest
    """
    lineage = DataLineage(creator=creator, purpose=purpose)
    lineage.record_source(real_data, source_name)
    lineage.record_generation(generator, synthetic_data, execution_time)

    if quality_report:
        lineage.record_quality(quality_report)

    return lineage.create_manifest()


@dataclass
class LineageBlock:
    """A block in the lineage chain (blockchain-style)."""

    block_id: str
    previous_hash: str
    timestamp: str
    action: str  # "source", "generation", "quality", "transformation"
    data_hash: str
    metadata: Dict[str, Any]
    block_hash: str = ""

    def __post_init__(self) -> None:
        """Compute block hash if not provided."""
        if not self.block_hash:
            self.block_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute hash of this block."""
        content = f"{self.block_id}{self.previous_hash}{self.timestamp}{self.action}{self.data_hash}{json.dumps(self.metadata, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()

    def verify(self) -> bool:
        """Verify the block hash is valid."""
        return self.block_hash == self._compute_hash()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "block_id": self.block_id,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "action": self.action,
            "data_hash": self.data_hash,
            "metadata": self.metadata,
            "block_hash": self.block_hash,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LineageBlock":
        """Create from dictionary."""
        return cls(**data)


class LineageChain:
    """Blockchain-style chain for immutable lineage tracking.

    Each operation (source, generation, transformation) creates a new
    block that references the previous block's hash, creating a tamper-proof
    audit trail.

    Example:
        >>> chain = LineageChain()
        >>> chain.add_source(real_data, "customers_v1")
        >>> chain.add_generation(generator, synthetic_data)
        >>> chain.add_transformation("filter", {"condition": "age > 18"})
        >>>
        >>> # Verify integrity
        >>> assert chain.verify()
        >>>
        >>> # Export for compliance
        >>> chain.export("lineage_chain.json")
    """

    GENESIS_HASH = "0" * 64  # Genesis block previous hash

    def __init__(self) -> None:
        """Initialize empty chain."""
        self._blocks: List[LineageBlock] = []

    def _create_block(
        self,
        action: str,
        data_hash: str,
        metadata: Dict[str, Any],
    ) -> LineageBlock:
        """Create a new block."""
        previous_hash = self._blocks[-1].block_hash if self._blocks else self.GENESIS_HASH

        block = LineageBlock(
            block_id=str(uuid.uuid4()),
            previous_hash=previous_hash,
            timestamp=datetime.utcnow().isoformat(),
            action=action,
            data_hash=data_hash,
            metadata=metadata,
        )

        return block

    def add_source(
        self,
        data: pd.DataFrame,
        name: str,
        description: Optional[str] = None,
    ) -> str:
        """Add a source data block.

        Args:
            data: Source DataFrame
            name: Name of the data source
            description: Optional description

        Returns:
            Block hash
        """
        data_bytes = data.to_csv(index=False).encode()
        data_hash = hashlib.sha256(data_bytes).hexdigest()

        metadata = {
            "name": name,
            "description": description,
            "n_rows": len(data),
            "n_columns": len(data.columns),
            "columns": list(data.columns),
        }

        block = self._create_block("source", data_hash, metadata)
        self._blocks.append(block)

        logger.info(f"Added source block: {block.block_id}")
        return block.block_hash

    def add_generation(
        self,
        generator: Any,
        synthetic_data: pd.DataFrame,
        conditions: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add a generation block.

        Args:
            generator: Generator that was used
            synthetic_data: Generated data
            conditions: Optional generation conditions

        Returns:
            Block hash
        """
        data_bytes = synthetic_data.to_csv(index=False).encode()
        data_hash = hashlib.sha256(data_bytes).hexdigest()

        params = generator.get_parameters()

        metadata = {
            "generator_type": generator.__class__.__name__,
            "method": params.get("config", {}).get("method"),
            "n_samples": len(synthetic_data),
            "conditions": conditions,
            "config_hash": hashlib.sha256(
                json.dumps(params.get("config", {}), sort_keys=True).encode()
            ).hexdigest()[:16],
        }

        block = self._create_block("generation", data_hash, metadata)
        self._blocks.append(block)

        logger.info(f"Added generation block: {block.block_id}")
        return block.block_hash

    def add_transformation(
        self,
        transform_type: str,
        params: Dict[str, Any],
        output_data: pd.DataFrame,
    ) -> str:
        """Add a transformation block.

        Args:
            transform_type: Type of transformation
            params: Transformation parameters
            output_data: Resulting data

        Returns:
            Block hash
        """
        data_bytes = output_data.to_csv(index=False).encode()
        data_hash = hashlib.sha256(data_bytes).hexdigest()

        metadata = {
            "transform_type": transform_type,
            "params": params,
            "n_rows": len(output_data),
        }

        block = self._create_block("transformation", data_hash, metadata)
        self._blocks.append(block)

        logger.info(f"Added transformation block: {block.block_id}")
        return block.block_hash

    def add_quality_check(
        self,
        report: Any,
        passed: bool,
    ) -> str:
        """Add a quality check block.

        Args:
            report: Quality report
            passed: Whether quality check passed

        Returns:
            Block hash
        """
        metrics = report.to_dict() if hasattr(report, "to_dict") else {}

        metadata = {
            "passed": passed,
            "overall_score": metrics.get("overall_score", 0),
            "metrics_hash": hashlib.sha256(
                json.dumps(metrics, sort_keys=True, default=str).encode()
            ).hexdigest()[:16],
        }

        # Quality checks reference the previous data hash
        data_hash = self._blocks[-1].data_hash if self._blocks else ""

        block = self._create_block("quality", data_hash, metadata)
        self._blocks.append(block)

        logger.info(f"Added quality block: {block.block_id}")
        return block.block_hash

    def verify(self) -> bool:
        """Verify the integrity of the entire chain.

        Returns:
            True if chain is valid, False otherwise
        """
        if not self._blocks:
            return True

        # Check genesis block
        if self._blocks[0].previous_hash != self.GENESIS_HASH:
            logger.error("Genesis block has invalid previous hash")
            return False

        # Check each block
        for i, block in enumerate(self._blocks):
            # Verify block hash
            if not block.verify():
                logger.error(f"Block {i} has invalid hash")
                return False

            # Verify chain linkage
            if i > 0 and block.previous_hash != self._blocks[i - 1].block_hash:
                logger.error(f"Block {i} has incorrect previous hash")
                return False

        logger.info(f"Chain verified: {len(self._blocks)} blocks")
        return True

    def export(self, path: Union[str, Path]) -> None:
        """Export chain to JSON file.

        Args:
            path: Output file path
        """
        path = Path(path)

        data = {
            "chain_version": "1.0",
            "created_at": datetime.utcnow().isoformat(),
            "n_blocks": len(self._blocks),
            "blocks": [b.to_dict() for b in self._blocks],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported chain to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "LineageChain":
        """Load chain from JSON file.

        Args:
            path: Input file path

        Returns:
            LineageChain instance
        """
        path = Path(path)

        with open(path) as f:
            data = json.load(f)

        chain = cls()
        chain._blocks = [LineageBlock.from_dict(b) for b in data["blocks"]]

        if not chain.verify():
            raise ValueError("Loaded chain failed verification")

        return chain

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get human-readable audit trail.

        Returns:
            List of audit events
        """
        trail = []

        for block in self._blocks:
            event = {
                "timestamp": block.timestamp,
                "action": block.action,
                "block_hash": block.block_hash[:16] + "...",
            }
            event.update(block.metadata)
            trail.append(event)

        return trail

    def __len__(self) -> int:
        return len(self._blocks)

    def __repr__(self) -> str:
        return f"LineageChain(blocks={len(self._blocks)}, verified={self.verify()})"
