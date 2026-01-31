"""Tests for Synthetic Data Versioning module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from genesis.versioning import (
    DatasetRepository,
    DatasetVersion,
    init_repository,
)


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create sample data."""
    np.random.seed(42)
    return pd.DataFrame(
        {
            "a": np.random.randint(0, 100, 100),
            "b": np.random.normal(0, 1, 100),
            "c": np.random.choice(["X", "Y", "Z"], 100),
        }
    )


@pytest.fixture
def modified_data() -> pd.DataFrame:
    """Create modified data."""
    np.random.seed(43)
    return pd.DataFrame(
        {
            "a": np.random.randint(0, 100, 150),
            "b": np.random.normal(5, 2, 150),
            "c": np.random.choice(["X", "Y", "Z"], 150),
        }
    )


@pytest.fixture
def temp_repo_path(tmp_path: Path) -> Path:
    """Create temporary repository path."""
    return tmp_path / "test_repo"


class TestDatasetVersion:
    """Tests for DatasetVersion."""

    def test_to_dict(self) -> None:
        """Test serialization."""
        version = DatasetVersion(
            version_id="abc123",
            parent_id=None,
            message="Initial commit",
            timestamp="2026-01-28T10:00:00",
            metadata={"key": "value"},
            data_hash="hash123",
            n_rows=100,
            n_columns=3,
            columns=["a", "b", "c"],
        )

        d = version.to_dict()

        assert d["version_id"] == "abc123"
        assert d["message"] == "Initial commit"
        assert d["n_rows"] == 100

    def test_from_dict(self) -> None:
        """Test deserialization."""
        data = {
            "version_id": "abc123",
            "parent_id": None,
            "message": "Test",
            "timestamp": "2026-01-28T10:00:00",
            "metadata": {},
            "data_hash": "hash",
            "n_rows": 50,
            "n_columns": 2,
            "columns": ["x", "y"],
            "tags": [],
        }

        version = DatasetVersion.from_dict(data)

        assert version.version_id == "abc123"
        assert version.n_rows == 50


class TestDatasetRepository:
    """Tests for DatasetRepository."""

    def test_init_creates_dirs(self, temp_repo_path: Path) -> None:
        """Test repository initialization."""
        repo = DatasetRepository(temp_repo_path)

        assert temp_repo_path.exists()
        assert (temp_repo_path / "versions").exists()
        assert (temp_repo_path / "data").exists()

    def test_commit(self, temp_repo_path: Path, sample_data: pd.DataFrame) -> None:
        """Test committing data."""
        repo = DatasetRepository(temp_repo_path)

        version = repo.commit(sample_data, "Initial commit")

        assert version.message == "Initial commit"
        assert version.n_rows == 100
        assert version.n_columns == 3

    def test_checkout(self, temp_repo_path: Path, sample_data: pd.DataFrame) -> None:
        """Test checking out data."""
        repo = DatasetRepository(temp_repo_path)
        version = repo.commit(sample_data, "Test")

        retrieved = repo.checkout(version.version_id)

        assert len(retrieved) == len(sample_data)
        assert list(retrieved.columns) == list(sample_data.columns)

    def test_log(
        self, temp_repo_path: Path, sample_data: pd.DataFrame, modified_data: pd.DataFrame
    ) -> None:
        """Test version log."""
        repo = DatasetRepository(temp_repo_path)
        repo.commit(sample_data, "First")
        repo.commit(modified_data, "Second")

        history = repo.log(n=10)

        assert len(history) == 2
        assert history[0].message == "Second"
        assert history[1].message == "First"

    def test_branch(self, temp_repo_path: Path, sample_data: pd.DataFrame) -> None:
        """Test creating branches."""
        repo = DatasetRepository(temp_repo_path)
        repo.commit(sample_data, "Initial")

        branch = repo.branch("feature", description="Feature branch")

        assert branch.name == "feature"
        assert "feature" in [b.name for b in repo.list_branches()]

    def test_switch_branch(self, temp_repo_path: Path, sample_data: pd.DataFrame) -> None:
        """Test switching branches."""
        repo = DatasetRepository(temp_repo_path)
        repo.commit(sample_data, "Initial")
        repo.branch("feature")

        repo.switch_branch("feature")

        assert repo.current_branch == "feature"

    def test_tag(self, temp_repo_path: Path, sample_data: pd.DataFrame) -> None:
        """Test tagging versions."""
        repo = DatasetRepository(temp_repo_path)
        version = repo.commit(sample_data, "Release")

        repo.tag("v1.0")

        tags = repo.list_tags()
        assert "v1.0" in tags
        assert tags["v1.0"] == version.version_id

    def test_checkout_by_tag(self, temp_repo_path: Path, sample_data: pd.DataFrame) -> None:
        """Test checking out by tag."""
        repo = DatasetRepository(temp_repo_path)
        repo.commit(sample_data, "Release")
        repo.tag("v1.0")

        data = repo.checkout("v1.0")

        assert len(data) == len(sample_data)


class TestDatasetDiff:
    """Tests for DatasetDiff."""

    def test_diff(
        self, temp_repo_path: Path, sample_data: pd.DataFrame, modified_data: pd.DataFrame
    ) -> None:
        """Test computing diff."""
        repo = DatasetRepository(temp_repo_path)
        v1 = repo.commit(sample_data, "First")
        v2 = repo.commit(modified_data, "Second")

        diff = repo.diff(v1.version_id, v2.version_id)

        assert diff.rows_added == 50  # 150 - 100
        assert len(diff.statistical_changes) > 0

    def test_diff_summary(
        self, temp_repo_path: Path, sample_data: pd.DataFrame, modified_data: pd.DataFrame
    ) -> None:
        """Test diff summary."""
        repo = DatasetRepository(temp_repo_path)
        v1 = repo.commit(sample_data, "First")
        v2 = repo.commit(modified_data, "Second")

        diff = repo.diff(v1.version_id, v2.version_id)
        summary = diff.summary()

        assert "Diff" in summary
        assert "Rows" in summary


class TestInitRepository:
    """Tests for init_repository function."""

    def test_creates_repo(self, temp_repo_path: Path) -> None:
        """Test repository initialization function."""
        repo = init_repository(temp_repo_path)

        assert isinstance(repo, DatasetRepository)
        assert temp_repo_path.exists()
