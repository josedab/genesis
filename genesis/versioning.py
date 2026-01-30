"""Synthetic Data Versioning.

Git-like versioning system for synthetic datasets with commit, branch,
merge, tag, and diff capabilities. Enables reproducible data generation
pipelines with full history tracking.

Features:
    - Commit datasets with messages and metadata
    - Branch and merge dataset variations
    - Tag important versions for releases
    - Diff between versions (rows, columns, statistics)
    - Content-addressable storage for deduplication

Example:
    Basic versioning workflow::

        from genesis.versioning import DatasetRepository

        # Initialize repository
        repo = DatasetRepository.init("./my_data_repo")

        # Commit versions
        repo.commit(df_v1, message="Initial dataset")
        repo.commit(df_v2, message="Added more samples")

        # View history
        for commit in repo.log():
            print(f"{commit.hash[:8]} - {commit.message}")

        # Tag for release
        repo.tag("v1.0", message="Production release")

        # Compare versions
        diff = repo.diff("v1.0", "HEAD")
        print(f"Rows added: {diff.rows_added}")

    Auto-versioning with VersionedGenerator::

        from genesis.versioning import VersionedGenerator

        generator = VersionedGenerator(
            method="ctgan",
            repository="./data_repo"
        )
        generator.fit(training_data)

        # Each generate() auto-commits
        synthetic = generator.generate(1000, message="Batch generation")

Classes:
    DatasetRepository: Git-like repository for datasets.
    DatasetVersion (Commit): Immutable dataset snapshot.
    DatasetDiff: Comparison between two versions.
    VersionedGenerator: Generator with automatic versioning.
    Branch: Branch reference.
    Tag: Tag reference.

Note:
    Storage uses content-addressable design where identical datasets
    share storage, providing automatic deduplication.
"""

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


@dataclass
class DatasetVersion:
    """A version (commit) of a synthetic dataset.

    Attributes:
        version_id: Unique hash identifying this version.
        parent_id: Parent version's hash (None for initial commit).
        message: Human-readable commit message.
        timestamp: ISO format timestamp of commit.
        metadata: Additional metadata (generator config, quality scores, etc.).
        data_hash: Content hash of the dataset.
        n_rows: Number of rows in this version.
        n_columns: Number of columns in this version.
    """

    version_id: str
    parent_id: Optional[str]
    message: str
    timestamp: str
    metadata: Dict[str, Any]
    data_hash: str
    n_rows: int
    n_columns: int
    columns: List[str]
    tags: List[str] = field(default_factory=list)

    @property
    def hash(self) -> str:
        """Alias for version_id for git-like CLI compatibility."""
        return self.version_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "parent_id": self.parent_id,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "data_hash": self.data_hash,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": self.columns,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetVersion":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class DatasetDiff:
    """Difference between two dataset versions."""

    source_version: str
    target_version: str
    rows_added: int
    rows_removed: int
    columns_added: List[str]
    columns_removed: List[str]
    schema_changes: Dict[str, Dict[str, str]]
    statistical_changes: Dict[str, Dict[str, float]]

    @property
    def columns_changed(self) -> int:
        """Total number of columns changed (added + removed)."""
        return len(self.columns_added) + len(self.columns_removed)

    def summary(self) -> str:
        """Get human-readable summary."""
        lines = [
            f"Diff: {self.source_version[:8]} → {self.target_version[:8]}",
            f"Rows: +{self.rows_added} / -{self.rows_removed}",
        ]

        if self.columns_added:
            lines.append(f"Columns added: {', '.join(self.columns_added)}")
        if self.columns_removed:
            lines.append(f"Columns removed: {', '.join(self.columns_removed)}")

        return "\n".join(lines)


@dataclass
class Branch:
    """A branch in the version history."""

    name: str
    head: str  # Version ID
    created_at: str
    description: str = ""


class DatasetRepository:
    """Repository for versioned synthetic datasets."""

    def __init__(self, path: Union[str, Path]):
        """Initialize repository.

        Args:
            path: Path to repository directory
        """
        self.path = Path(path)
        self._versions_dir = self.path / "versions"
        self._data_dir = self.path / "data"
        self._index_file = self.path / "index.json"

        self._index: Dict[str, Any] = {
            "versions": {},
            "branches": {"main": None},
            "current_branch": "main",
            "tags": {},
        }
        self._current_ref: Optional[str] = None

        self._load_or_init()

    @classmethod
    def init(cls, path: Union[str, Path]) -> "DatasetRepository":
        """Initialize a new repository at the given path.

        Args:
            path: Path to repository directory

        Returns:
            Initialized DatasetRepository
        """
        return cls(path)

    def _load_or_init(self) -> None:
        """Load existing repository or initialize new one."""
        if self.path.exists() and self._index_file.exists():
            with open(self._index_file) as f:
                self._index = json.load(f)
        else:
            self._init_repo()

    def _init_repo(self) -> None:
        """Initialize new repository."""
        self.path.mkdir(parents=True, exist_ok=True)
        self._versions_dir.mkdir(exist_ok=True)
        self._data_dir.mkdir(exist_ok=True)
        self._save_index()

    def _save_index(self) -> None:
        """Save index to disk."""
        with open(self._index_file, "w") as f:
            json.dump(self._index, f, indent=2)

    def _compute_hash(self, data: pd.DataFrame) -> str:
        """Compute hash of DataFrame."""
        # Use pandas hash for content-addressable storage
        content = data.to_csv(index=False).encode()
        return hashlib.sha256(content).hexdigest()

    def _generate_version_id(self) -> str:
        """Generate unique version ID."""
        import uuid

        return uuid.uuid4().hex

    def commit(
        self,
        data: pd.DataFrame,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
    ) -> DatasetVersion:
        """Commit a new version of the dataset.

        Args:
            data: DataFrame to commit
            message: Commit message
            metadata: Optional metadata
            tags: Optional tags

        Returns:
            DatasetVersion created
        """
        version_id = self._generate_version_id()
        data_hash = self._compute_hash(data)

        # Get parent (current HEAD)
        current_branch = self._index["current_branch"]
        parent_id = self._index["branches"].get(current_branch)

        # Create version
        version = DatasetVersion(
            version_id=version_id,
            parent_id=parent_id,
            message=message,
            timestamp=datetime.now().isoformat(),
            metadata=metadata or {},
            data_hash=data_hash,
            n_rows=len(data),
            n_columns=len(data.columns),
            columns=list(data.columns),
            tags=tags or [],
        )

        # Save data
        data_path = self._data_dir / f"{data_hash}.parquet"
        if not data_path.exists():
            data.to_parquet(data_path, index=False)

        # Save version metadata
        version_path = self._versions_dir / f"{version_id}.json"
        with open(version_path, "w") as f:
            json.dump(version.to_dict(), f, indent=2)

        # Update index
        self._index["versions"][version_id] = version.to_dict()
        self._index["branches"][current_branch] = version_id

        # Add tags
        for tag in tags or []:
            self._index["tags"][tag] = version_id

        self._save_index()

        return version

    def checkout(
        self,
        ref: str,
    ) -> pd.DataFrame:
        """Checkout a version by ID, tag, or branch name.

        Args:
            ref: Version ID, tag name, or branch name

        Returns:
            DataFrame at that version
        """
        # Resolve ref to version ID
        version_id = self._resolve_ref(ref)

        if version_id is None:
            raise ValueError(f"Unknown reference: {ref}")

        self._current_ref = version_id
        version = self._get_version(version_id)

        # Load data
        data_path = self._data_dir / f"{version.data_hash}.parquet"
        return pd.read_parquet(data_path)

    def get_current_data(self) -> pd.DataFrame:
        """Get the data at the currently checked out version.

        Returns:
            DataFrame at the current HEAD or checked out version

        Raises:
            ValueError: If no version has been checked out
        """
        # Use the last checked out ref, or HEAD of current branch
        ref = self._current_ref
        if ref is None:
            branch = self._index["current_branch"]
            ref = self._index["branches"].get(branch)

        if ref is None:
            raise ValueError("No data available - repository is empty or no version checked out")

        version = self._get_version(ref)
        data_path = self._data_dir / f"{version.data_hash}.parquet"
        return pd.read_parquet(data_path)

    def _resolve_ref(self, ref: str) -> Optional[str]:
        """Resolve a reference to a version ID."""
        # Check if it's a version ID
        if ref in self._index["versions"]:
            return ref

        # Check if it's a tag
        if ref in self._index["tags"]:
            return self._index["tags"][ref]

        # Check if it's a branch
        if ref in self._index["branches"]:
            return self._index["branches"][ref]

        # Try partial version ID match
        for vid in self._index["versions"]:
            if vid.startswith(ref):
                return vid

        return None

    def _get_version(self, version_id: str) -> DatasetVersion:
        """Get version by ID."""
        if version_id not in self._index["versions"]:
            raise ValueError(f"Unknown version: {version_id}")

        return DatasetVersion.from_dict(self._index["versions"][version_id])

    def log(
        self,
        n: int = 10,
        branch: Optional[str] = None,
    ) -> List[DatasetVersion]:
        """Get version history.

        Args:
            n: Maximum versions to return
            branch: Branch to show (default: current)

        Returns:
            List of versions, most recent first
        """
        branch = branch or self._index["current_branch"]
        head_id = self._index["branches"].get(branch)

        if head_id is None:
            return []

        versions = []
        current_id = head_id

        while current_id and len(versions) < n:
            version = self._get_version(current_id)
            versions.append(version)
            current_id = version.parent_id

        return versions

    def diff(
        self,
        source_ref: str,
        target_ref: str,
    ) -> DatasetDiff:
        """Compute diff between two versions.

        Args:
            source_ref: Source version reference
            target_ref: Target version reference

        Returns:
            DatasetDiff
        """
        source_id = self._resolve_ref(source_ref)
        target_id = self._resolve_ref(target_ref)

        if source_id is None or target_id is None:
            raise ValueError("Invalid version references")

        source_version = self._get_version(source_id)
        target_version = self._get_version(target_id)

        # Load data
        source_data = self.checkout(source_id)
        target_data = self.checkout(target_id)

        # Compute diff
        columns_added = list(set(target_version.columns) - set(source_version.columns))
        columns_removed = list(set(source_version.columns) - set(target_version.columns))

        # Statistical changes for common columns
        common_cols = set(source_version.columns) & set(target_version.columns)
        statistical_changes = {}

        for col in common_cols:
            if pd.api.types.is_numeric_dtype(source_data[col]):
                source_mean = source_data[col].mean()
                target_mean = target_data[col].mean()
                source_std = source_data[col].std()
                target_std = target_data[col].std()

                if source_mean != 0:
                    mean_change = (target_mean - source_mean) / abs(source_mean)
                else:
                    mean_change = target_mean - source_mean

                statistical_changes[col] = {
                    "mean_change_pct": mean_change * 100,
                    "std_change": target_std - source_std,
                }

        return DatasetDiff(
            source_version=source_id,
            target_version=target_id,
            rows_added=max(0, target_version.n_rows - source_version.n_rows),
            rows_removed=max(0, source_version.n_rows - target_version.n_rows),
            columns_added=columns_added,
            columns_removed=columns_removed,
            schema_changes={},
            statistical_changes=statistical_changes,
        )

    def branch(
        self,
        name: str,
        from_ref: Optional[str] = None,
        description: str = "",
    ) -> Branch:
        """Create a new branch.

        Args:
            name: Branch name
            from_ref: Start point (default: current HEAD)
            description: Branch description

        Returns:
            Created Branch
        """
        if name in self._index["branches"]:
            raise ValueError(f"Branch '{name}' already exists")

        if from_ref:
            head = self._resolve_ref(from_ref)
        else:
            current_branch = self._index["current_branch"]
            head = self._index["branches"].get(current_branch)

        self._index["branches"][name] = head
        self._save_index()

        return Branch(
            name=name,
            head=head or "",
            created_at=datetime.now().isoformat(),
            description=description,
        )

    def switch_branch(self, name: str) -> None:
        """Switch to a different branch.

        Args:
            name: Branch name
        """
        if name not in self._index["branches"]:
            raise ValueError(f"Branch '{name}' not found")

        self._index["current_branch"] = name
        self._save_index()

    def merge(
        self,
        source_branch: str,
        strategy: str = "combine",
    ) -> DatasetVersion:
        """Merge another branch into current branch.

        Args:
            source_branch: Branch to merge from
            strategy: Merge strategy ("combine", "source", "target")

        Returns:
            New merged version
        """
        if source_branch not in self._index["branches"]:
            raise ValueError(f"Branch '{source_branch}' not found")

        source_head = self._index["branches"][source_branch]
        target_branch = self._index["current_branch"]
        target_head = self._index["branches"][target_branch]

        if source_head is None:
            raise ValueError(f"Branch '{source_branch}' has no commits")

        # Load data from both branches
        source_data = self.checkout(source_head)

        if target_head:
            target_data = self.checkout(target_head)

            if strategy == "combine":
                # Combine rows from both
                merged = pd.concat([target_data, source_data], ignore_index=True)
                merged = merged.drop_duplicates()
            elif strategy == "source":
                merged = source_data
            else:  # target
                merged = target_data
        else:
            merged = source_data

        # Commit merge
        return self.commit(
            merged,
            message=f"Merge branch '{source_branch}' into '{target_branch}'",
            metadata={
                "merge": True,
                "source_branch": source_branch,
                "source_version": source_head,
            },
        )

    def tag(self, name: str, ref: Optional[str] = None) -> None:
        """Create a tag.

        Args:
            name: Tag name
            ref: Version to tag (default: current HEAD)
        """
        if ref:
            version_id = self._resolve_ref(ref)
        else:
            current_branch = self._index["current_branch"]
            version_id = self._index["branches"].get(current_branch)

        if version_id is None:
            raise ValueError("No version to tag")

        self._index["tags"][name] = version_id
        self._save_index()

    def list_branches(self) -> List[Branch]:
        """List all branches."""
        branches = []
        for name, head in self._index["branches"].items():
            branches.append(
                Branch(
                    name=name,
                    head=head or "",
                    created_at="",
                )
            )
        return branches

    def list_tags(self) -> Dict[str, str]:
        """List all tags."""
        return self._index["tags"].copy()

    @property
    def head(self) -> Optional[DatasetVersion]:
        """Get current HEAD version."""
        current_branch = self._index["current_branch"]
        head_id = self._index["branches"].get(current_branch)

        if head_id:
            return self._get_version(head_id)
        return None

    @property
    def current_branch(self) -> str:
        """Get current branch name."""
        return self._index["current_branch"]


class VersionedGenerator:
    """Generator with built-in versioning."""

    def __init__(
        self,
        generator,
        repo_path: Union[str, Path],
    ):
        """Initialize versioned generator.

        Args:
            generator: Base generator
            repo_path: Path to version repository
        """
        self.generator = generator
        self.repo = DatasetRepository(repo_path)
        self._fitted = False

    def fit(self, data: pd.DataFrame, **kwargs) -> "VersionedGenerator":
        """Fit generator."""
        self.generator.fit(data, **kwargs)
        self._fitted = True
        return self

    def generate(
        self,
        n_samples: int,
        message: str = "Generated synthetic data",
        auto_commit: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Generate and optionally commit synthetic data.

        Args:
            n_samples: Number of samples
            message: Commit message
            auto_commit: Whether to auto-commit
            **kwargs: Additional generate arguments

        Returns:
            Generated DataFrame
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() first")

        synthetic = self.generator.generate(n_samples, **kwargs)

        if auto_commit:
            self.repo.commit(
                synthetic,
                message=message,
                metadata={
                    "n_samples": n_samples,
                    "generator": type(self.generator).__name__,
                },
            )

        return synthetic

    def rollback(self, ref: str) -> pd.DataFrame:
        """Rollback to a previous version.

        Args:
            ref: Version reference

        Returns:
            Data from that version
        """
        return self.repo.checkout(ref)

    def history(self, n: int = 10) -> List[DatasetVersion]:
        """Get generation history."""
        return self.repo.log(n=n)


def init_repository(path: Union[str, Path]) -> DatasetRepository:
    """Initialize a new dataset repository.

    Args:
        path: Repository path

    Returns:
        DatasetRepository
    """
    return DatasetRepository(path)


def clone_repository(
    source_path: Union[str, Path],
    dest_path: Union[str, Path],
) -> DatasetRepository:
    """Clone a repository.

    Args:
        source_path: Source repository path
        dest_path: Destination path

    Returns:
        Cloned DatasetRepository
    """
    source = Path(source_path)
    dest = Path(dest_path)

    if dest.exists():
        raise ValueError(f"Destination already exists: {dest}")

    shutil.copytree(source, dest)
    return DatasetRepository(dest)
