"""Synthetic Data Marketplace backend.

This module provides the backend infrastructure for a synthetic data
marketplace, including dataset registry, discovery, and licensing.

Example:
    >>> from genesis.marketplace import Marketplace, DatasetListing
    >>>
    >>> marketplace = Marketplace()
    >>>
    >>> # Upload a dataset
    >>> listing = marketplace.create_listing(
    ...     name="Healthcare Demographics",
    ...     data=synthetic_df,
    ...     price=0,  # Free
    ...     license="MIT",
    ... )
    >>>
    >>> # Search datasets
    >>> results = marketplace.search("healthcare")
"""

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class LicenseType(Enum):
    """Available license types for datasets."""

    MIT = "mit"
    APACHE2 = "apache-2.0"
    CC_BY = "cc-by-4.0"
    CC_BY_SA = "cc-by-sa-4.0"
    CC_BY_NC = "cc-by-nc-4.0"
    CC0 = "cc0-1.0"
    PROPRIETARY = "proprietary"
    CUSTOM = "custom"


class DatasetCategory(Enum):
    """Dataset categories."""

    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"
    TELECOMMUNICATIONS = "telecommunications"
    TRANSPORTATION = "transportation"
    ENERGY = "energy"
    EDUCATION = "education"
    GOVERNMENT = "government"
    GENERAL = "general"


class DatasetStatus(Enum):
    """Dataset listing status."""

    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    PUBLISHED = "published"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


@dataclass
class QualityMetrics:
    """Quality metrics for a dataset."""

    statistical_fidelity: float = 0.0
    ml_utility: float = 0.0
    privacy_score: float = 0.0
    overall_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statistical_fidelity": self.statistical_fidelity,
            "ml_utility": self.ml_utility,
            "privacy_score": self.privacy_score,
            "overall_score": self.overall_score,
        }


@dataclass
class ProvenanceInfo:
    """Provenance information for a dataset."""

    generator_method: str
    generator_version: str
    generation_date: str
    source_description: str = ""
    training_samples: int = 0
    generation_parameters: Dict[str, Any] = field(default_factory=dict)
    privacy_mechanisms: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generator_method": self.generator_method,
            "generator_version": self.generator_version,
            "generation_date": self.generation_date,
            "source_description": self.source_description,
            "training_samples": self.training_samples,
            "generation_parameters": self.generation_parameters,
            "privacy_mechanisms": self.privacy_mechanisms,
        }


@dataclass
class UserProfile:
    """User profile in the marketplace."""

    user_id: str
    username: str
    email: str
    organization: str = ""
    verified: bool = False
    created_at: str = ""
    datasets_published: int = 0
    datasets_downloaded: int = 0
    reputation_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "organization": self.organization,
            "verified": self.verified,
            "datasets_published": self.datasets_published,
            "reputation_score": self.reputation_score,
        }


@dataclass
class Review:
    """User review for a dataset."""

    review_id: str
    dataset_id: str
    user_id: str
    rating: int  # 1-5
    comment: str
    created_at: str
    helpful_votes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "review_id": self.review_id,
            "dataset_id": self.dataset_id,
            "user_id": self.user_id,
            "rating": self.rating,
            "comment": self.comment,
            "created_at": self.created_at,
            "helpful_votes": self.helpful_votes,
        }


@dataclass
class DatasetListing:
    """A dataset listing in the marketplace."""

    dataset_id: str
    name: str
    description: str
    owner_id: str

    # Metadata
    category: DatasetCategory = DatasetCategory.GENERAL
    tags: List[str] = field(default_factory=list)
    status: DatasetStatus = DatasetStatus.DRAFT

    # Data info
    n_rows: int = 0
    n_columns: int = 0
    columns: List[Dict[str, Any]] = field(default_factory=list)
    file_size_bytes: int = 0
    data_hash: str = ""

    # Quality
    quality_metrics: Optional[QualityMetrics] = None
    provenance: Optional[ProvenanceInfo] = None

    # Licensing
    license_type: LicenseType = LicenseType.CC_BY
    custom_license: str = ""
    price: float = 0.0  # 0 = free
    currency: str = "USD"

    # Stats
    downloads: int = 0
    views: int = 0
    rating: float = 0.0
    n_reviews: int = 0

    # Timestamps
    created_at: str = ""
    updated_at: str = ""
    published_at: Optional[str] = None

    # Sample data
    sample_data: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "owner_id": self.owner_id,
            "category": self.category.value,
            "tags": self.tags,
            "status": self.status.value,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "columns": self.columns,
            "file_size_bytes": self.file_size_bytes,
            "data_hash": self.data_hash,
            "quality_metrics": self.quality_metrics.to_dict() if self.quality_metrics else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
            "license_type": self.license_type.value,
            "price": self.price,
            "currency": self.currency,
            "downloads": self.downloads,
            "views": self.views,
            "rating": self.rating,
            "n_reviews": self.n_reviews,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "published_at": self.published_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetListing":
        quality_metrics = None
        if data.get("quality_metrics"):
            qm = data["quality_metrics"]
            quality_metrics = QualityMetrics(**qm)

        provenance = None
        if data.get("provenance"):
            provenance = ProvenanceInfo(**data["provenance"])

        return cls(
            dataset_id=data["dataset_id"],
            name=data["name"],
            description=data["description"],
            owner_id=data["owner_id"],
            category=DatasetCategory(data.get("category", "general")),
            tags=data.get("tags", []),
            status=DatasetStatus(data.get("status", "draft")),
            n_rows=data.get("n_rows", 0),
            n_columns=data.get("n_columns", 0),
            columns=data.get("columns", []),
            file_size_bytes=data.get("file_size_bytes", 0),
            data_hash=data.get("data_hash", ""),
            quality_metrics=quality_metrics,
            provenance=provenance,
            license_type=LicenseType(data.get("license_type", "cc-by-4.0")),
            price=data.get("price", 0.0),
            downloads=data.get("downloads", 0),
            views=data.get("views", 0),
            rating=data.get("rating", 0.0),
            n_reviews=data.get("n_reviews", 0),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            published_at=data.get("published_at"),
        )


@dataclass
class SearchResult:
    """Search result from marketplace query."""

    listings: List[DatasetListing]
    total_count: int
    page: int
    per_page: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "listings": [listing.to_dict() for listing in self.listings],
            "total_count": self.total_count,
            "page": self.page,
            "per_page": self.per_page,
        }


class DatasetStorage:
    """Storage backend for dataset files."""

    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, dataset_id: str, data: pd.DataFrame) -> str:
        """Save dataset to storage.

        Returns:
            Data hash
        """
        file_path = self.base_path / f"{dataset_id}.parquet"
        data.to_parquet(file_path, index=False)

        # Compute hash
        data_bytes = data.to_csv(index=False).encode()
        data_hash = hashlib.sha256(data_bytes).hexdigest()

        return data_hash

    def load(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """Load dataset from storage."""
        file_path = self.base_path / f"{dataset_id}.parquet"
        if file_path.exists():
            return pd.read_parquet(file_path)
        return None

    def delete(self, dataset_id: str) -> bool:
        """Delete dataset from storage."""
        file_path = self.base_path / f"{dataset_id}.parquet"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def get_size(self, dataset_id: str) -> int:
        """Get file size in bytes."""
        file_path = self.base_path / f"{dataset_id}.parquet"
        if file_path.exists():
            return file_path.stat().st_size
        return 0


class Marketplace:
    """Synthetic Data Marketplace backend.

    Provides functionality for:
    - Dataset listing and discovery
    - Quality verification
    - Provenance tracking
    - License management
    """

    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
    ):
        """Initialize marketplace.

        Args:
            storage_path: Path for dataset storage
        """
        self.storage_path = (
            Path(storage_path) if storage_path else Path.home() / ".genesis" / "marketplace"
        )
        self.storage = DatasetStorage(self.storage_path / "datasets")

        self._listings: Dict[str, DatasetListing] = {}
        self._users: Dict[str, UserProfile] = {}
        self._reviews: Dict[str, List[Review]] = {}

    def create_listing(
        self,
        name: str,
        description: str,
        data: pd.DataFrame,
        owner_id: str,
        category: DatasetCategory = DatasetCategory.GENERAL,
        tags: Optional[List[str]] = None,
        license_type: LicenseType = LicenseType.CC_BY,
        price: float = 0.0,
        provenance: Optional[ProvenanceInfo] = None,
    ) -> DatasetListing:
        """Create a new dataset listing.

        Args:
            name: Dataset name
            description: Dataset description
            data: The synthetic data
            owner_id: Owner user ID
            category: Dataset category
            tags: Searchable tags
            license_type: License type
            price: Price (0 = free)
            provenance: Provenance information

        Returns:
            Created DatasetListing
        """
        dataset_id = str(uuid.uuid4())[:12]

        # Save data
        data_hash = self.storage.save(dataset_id, data)
        file_size = self.storage.get_size(dataset_id)

        # Extract column info
        columns = []
        for col in data.columns:
            col_info = {
                "name": col,
                "dtype": str(data[col].dtype),
                "n_unique": int(data[col].nunique()),
                "null_count": int(data[col].isna().sum()),
            }
            columns.append(col_info)

        # Create sample (first 5 rows)
        sample_data = data.head(5).to_dict("records")

        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(data)

        # Create listing
        listing = DatasetListing(
            dataset_id=dataset_id,
            name=name,
            description=description,
            owner_id=owner_id,
            category=category,
            tags=tags or [],
            status=DatasetStatus.DRAFT,
            n_rows=len(data),
            n_columns=len(data.columns),
            columns=columns,
            file_size_bytes=file_size,
            data_hash=data_hash,
            quality_metrics=quality_metrics,
            provenance=provenance,
            license_type=license_type,
            price=price,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            sample_data=sample_data,
        )

        self._listings[dataset_id] = listing
        logger.info(f"Created listing: {name} ({dataset_id})")

        return listing

    def get_listing(self, dataset_id: str) -> Optional[DatasetListing]:
        """Get a listing by ID."""
        listing = self._listings.get(dataset_id)
        if listing:
            listing.views += 1
        return listing

    def update_listing(
        self,
        dataset_id: str,
        updates: Dict[str, Any],
    ) -> bool:
        """Update a listing."""
        listing = self._listings.get(dataset_id)
        if not listing:
            return False

        for key, value in updates.items():
            if hasattr(listing, key):
                setattr(listing, key, value)

        listing.updated_at = datetime.now().isoformat()
        return True

    def publish_listing(self, dataset_id: str) -> bool:
        """Publish a listing (make it visible)."""
        listing = self._listings.get(dataset_id)
        if not listing:
            return False

        # Run quality checks
        if not self._verify_quality(listing):
            logger.warning(f"Quality check failed for {dataset_id}")
            return False

        listing.status = DatasetStatus.PUBLISHED
        listing.published_at = datetime.now().isoformat()
        listing.updated_at = datetime.now().isoformat()

        logger.info(f"Published listing: {listing.name}")
        return True

    def search(
        self,
        query: str = "",
        category: Optional[DatasetCategory] = None,
        tags: Optional[List[str]] = None,
        min_quality: float = 0.0,
        max_price: Optional[float] = None,
        sort_by: str = "downloads",
        page: int = 1,
        per_page: int = 20,
    ) -> SearchResult:
        """Search for datasets.

        Args:
            query: Text query
            category: Filter by category
            tags: Filter by tags
            min_quality: Minimum quality score
            max_price: Maximum price (None = any)
            sort_by: Sort field (downloads, rating, created_at)
            page: Page number
            per_page: Results per page

        Returns:
            SearchResult with matching listings
        """
        # Filter listings
        results = []

        for listing in self._listings.values():
            # Only show published listings
            if listing.status != DatasetStatus.PUBLISHED:
                continue

            # Text search
            if query:
                query_lower = query.lower()
                if (
                    query_lower not in listing.name.lower()
                    and query_lower not in listing.description.lower()
                    and not any(query_lower in tag.lower() for tag in listing.tags)
                ):
                    continue

            # Category filter
            if category and listing.category != category:
                continue

            # Tags filter
            if tags:
                if not any(tag in listing.tags for tag in tags):
                    continue

            # Quality filter
            if listing.quality_metrics:
                if listing.quality_metrics.overall_score < min_quality:
                    continue

            # Price filter
            if max_price is not None and listing.price > max_price:
                continue

            results.append(listing)

        # Sort
        if sort_by == "downloads":
            results.sort(key=lambda x: x.downloads, reverse=True)
        elif sort_by == "rating":
            results.sort(key=lambda x: x.rating, reverse=True)
        elif sort_by == "created_at":
            results.sort(key=lambda x: x.created_at, reverse=True)
        elif sort_by == "price":
            results.sort(key=lambda x: x.price)

        # Pagination
        total_count = len(results)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_results = results[start_idx:end_idx]

        return SearchResult(
            listings=page_results,
            total_count=total_count,
            page=page,
            per_page=per_page,
        )

    def download(
        self,
        dataset_id: str,
        user_id: str,
    ) -> Optional[pd.DataFrame]:
        """Download a dataset.

        Args:
            dataset_id: Dataset ID
            user_id: Downloading user ID

        Returns:
            Dataset as DataFrame
        """
        listing = self._listings.get(dataset_id)
        if not listing or listing.status != DatasetStatus.PUBLISHED:
            return None

        # Check license/payment (simplified)
        if listing.price > 0:
            logger.warning(f"Dataset {dataset_id} requires payment")
            # In real implementation, verify payment

        # Load data
        data = self.storage.load(dataset_id)
        if data is not None:
            listing.downloads += 1
            listing.updated_at = datetime.now().isoformat()

        return data

    def add_review(
        self,
        dataset_id: str,
        user_id: str,
        rating: int,
        comment: str,
    ) -> Optional[Review]:
        """Add a review to a dataset.

        Args:
            dataset_id: Dataset ID
            user_id: Reviewing user ID
            rating: Rating (1-5)
            comment: Review comment

        Returns:
            Created Review
        """
        listing = self._listings.get(dataset_id)
        if not listing:
            return None

        review = Review(
            review_id=str(uuid.uuid4())[:8],
            dataset_id=dataset_id,
            user_id=user_id,
            rating=min(5, max(1, rating)),
            comment=comment,
            created_at=datetime.now().isoformat(),
        )

        if dataset_id not in self._reviews:
            self._reviews[dataset_id] = []
        self._reviews[dataset_id].append(review)

        # Update listing rating
        reviews = self._reviews[dataset_id]
        listing.rating = sum(r.rating for r in reviews) / len(reviews)
        listing.n_reviews = len(reviews)

        return review

    def get_reviews(self, dataset_id: str) -> List[Review]:
        """Get reviews for a dataset."""
        return self._reviews.get(dataset_id, [])

    def verify_provenance(
        self,
        dataset_id: str,
        claimed_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Verify dataset provenance.

        Returns:
            Verification result
        """
        listing = self._listings.get(dataset_id)
        if not listing:
            return {"verified": False, "error": "Dataset not found"}

        result = {
            "verified": True,
            "dataset_id": dataset_id,
            "data_hash": listing.data_hash,
            "provenance": listing.provenance.to_dict() if listing.provenance else None,
            "checks": [],
        }

        # Check data integrity
        data = self.storage.load(dataset_id)
        if data is not None:
            current_hash = hashlib.sha256(data.to_csv(index=False).encode()).hexdigest()

            if current_hash == listing.data_hash:
                result["checks"].append(
                    {
                        "name": "data_integrity",
                        "passed": True,
                        "message": "Data hash matches",
                    }
                )
            else:
                result["verified"] = False
                result["checks"].append(
                    {
                        "name": "data_integrity",
                        "passed": False,
                        "message": "Data hash mismatch",
                    }
                )

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get marketplace statistics."""
        published = [
            listing for listing in self._listings.values()
            if listing.status == DatasetStatus.PUBLISHED
        ]

        return {
            "total_datasets": len(published),
            "total_downloads": sum(listing.downloads for listing in published),
            "total_rows": sum(listing.n_rows for listing in published),
            "categories": {
                cat.value: sum(1 for listing in published if listing.category == cat)
                for cat in DatasetCategory
            },
            "avg_quality_score": (
                np.mean([
                    listing.quality_metrics.overall_score
                    for listing in published if listing.quality_metrics
                ])
                if published
                else 0
            ),
        }

    def _compute_quality_metrics(self, data: pd.DataFrame) -> QualityMetrics:
        """Compute quality metrics for a dataset."""
        # Simplified quality assessment
        # In production, would use full evaluation suite

        # Check for issues
        null_rate = data.isna().sum().sum() / (len(data) * len(data.columns))
        duplicate_rate = data.duplicated().sum() / len(data)

        # Compute scores
        statistical = max(0, 1 - null_rate * 2 - duplicate_rate)
        ml_utility = 0.8  # Would need holdout data to compute
        privacy = 0.9  # Would need privacy analysis

        overall = statistical * 0.4 + ml_utility * 0.3 + privacy * 0.3

        return QualityMetrics(
            statistical_fidelity=round(statistical, 3),
            ml_utility=round(ml_utility, 3),
            privacy_score=round(privacy, 3),
            overall_score=round(overall, 3),
        )

    def _verify_quality(self, listing: DatasetListing) -> bool:
        """Verify listing meets quality standards."""
        if not listing.quality_metrics:
            return False

        # Minimum thresholds
        if listing.quality_metrics.overall_score < 0.5:
            return False
        if listing.quality_metrics.privacy_score < 0.7:
            return False
        if listing.n_rows < 100:
            return False

        return True


__all__ = [
    # Main classes
    "Marketplace",
    "DatasetListing",
    "SearchResult",
    # Support classes
    "QualityMetrics",
    "ProvenanceInfo",
    "UserProfile",
    "Review",
    "DatasetStorage",
    # Types
    "LicenseType",
    "DatasetCategory",
    "DatasetStatus",
]
