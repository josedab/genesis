"""Synthetic Data Marketplace backend.

This module provides the backend infrastructure for a synthetic data
marketplace, including dataset registry, discovery, licensing, versioning,
and purchase flow.

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
    >>> # Search datasets with advanced filtering
    >>> results = marketplace.advanced_search(
    ...     SearchFilters(
    ...         query="healthcare",
    ...         min_rows=1000,
    ...         license_types=[LicenseType.MIT, LicenseType.CC_BY],
    ...     )
    ... )
    >>>
    >>> # Create a new version
    >>> marketplace.create_version(listing.dataset_id, new_data, "Added more columns")
    >>>
    >>> # Purchase flow
    >>> purchase = marketplace.initiate_purchase(listing.dataset_id, user_id)
    >>> marketplace.complete_purchase(purchase.purchase_id, "stripe_payment_123")
"""

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

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


class PurchaseStatus(Enum):
    """Purchase transaction status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    EXPIRED = "expired"


class PaymentMethod(Enum):
    """Supported payment methods."""

    STRIPE = "stripe"
    PAYPAL = "paypal"
    CREDITS = "credits"
    FREE = "free"


class VerificationLevel(Enum):
    """Dataset verification levels."""

    UNVERIFIED = "unverified"
    BASIC = "basic"
    ENHANCED = "enhanced"
    CERTIFIED = "certified"


@dataclass
class SearchFilters:
    """Advanced search filters for marketplace."""

    query: str = ""
    category: Optional[DatasetCategory] = None
    categories: Optional[List[DatasetCategory]] = None
    tags: Optional[List[str]] = None
    license_types: Optional[List[LicenseType]] = None
    min_quality: float = 0.0
    max_price: Optional[float] = None
    min_price: float = 0.0
    free_only: bool = False
    min_rows: Optional[int] = None
    max_rows: Optional[int] = None
    min_columns: Optional[int] = None
    max_columns: Optional[int] = None
    min_rating: float = 0.0
    min_downloads: int = 0
    owner_id: Optional[str] = None
    verified_only: bool = False
    verification_level: Optional[VerificationLevel] = None
    created_after: Optional[str] = None
    created_before: Optional[str] = None
    has_provenance: bool = False
    column_types: Optional[List[str]] = None
    column_names: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "category": self.category.value if self.category else None,
            "categories": [c.value for c in self.categories] if self.categories else None,
            "tags": self.tags,
            "license_types": [l.value for l in self.license_types] if self.license_types else None,
            "min_quality": self.min_quality,
            "max_price": self.max_price,
            "min_price": self.min_price,
            "free_only": self.free_only,
            "min_rows": self.min_rows,
            "max_rows": self.max_rows,
            "min_columns": self.min_columns,
            "max_columns": self.max_columns,
            "min_rating": self.min_rating,
            "min_downloads": self.min_downloads,
            "owner_id": self.owner_id,
            "verified_only": self.verified_only,
            "created_after": self.created_after,
            "created_before": self.created_before,
            "has_provenance": self.has_provenance,
        }


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
class DatasetVersion:
    """A version of a dataset."""

    version_id: str
    dataset_id: str
    version_number: int
    version_tag: str  # e.g., "1.0.0", "2.0.0"
    data_hash: str
    n_rows: int
    n_columns: int
    file_size_bytes: int
    changelog: str
    created_at: str
    created_by: str
    is_current: bool = True
    quality_metrics: Optional[QualityMetrics] = None
    columns: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "dataset_id": self.dataset_id,
            "version_number": self.version_number,
            "version_tag": self.version_tag,
            "data_hash": self.data_hash,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "file_size_bytes": self.file_size_bytes,
            "changelog": self.changelog,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "is_current": self.is_current,
            "quality_metrics": self.quality_metrics.to_dict() if self.quality_metrics else None,
        }


@dataclass
class Purchase:
    """A purchase transaction."""

    purchase_id: str
    dataset_id: str
    user_id: str
    version_id: Optional[str]
    amount: float
    currency: str
    status: PurchaseStatus
    payment_method: PaymentMethod
    payment_reference: Optional[str] = None
    created_at: str = ""
    completed_at: Optional[str] = None
    expires_at: Optional[str] = None
    license_key: Optional[str] = None
    download_count: int = 0
    max_downloads: int = -1  # -1 = unlimited

    def to_dict(self) -> Dict[str, Any]:
        return {
            "purchase_id": self.purchase_id,
            "dataset_id": self.dataset_id,
            "user_id": self.user_id,
            "version_id": self.version_id,
            "amount": self.amount,
            "currency": self.currency,
            "status": self.status.value,
            "payment_method": self.payment_method.value,
            "payment_reference": self.payment_reference,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "license_key": self.license_key,
            "download_count": self.download_count,
        }


@dataclass
class UsageAnalytics:
    """Analytics for a dataset."""

    dataset_id: str
    views_total: int = 0
    views_30d: int = 0
    downloads_total: int = 0
    downloads_30d: int = 0
    revenue_total: float = 0.0
    revenue_30d: float = 0.0
    unique_downloaders: int = 0
    avg_rating: float = 0.0
    review_count: int = 0
    geographic_distribution: Dict[str, int] = field(default_factory=dict)
    referral_sources: Dict[str, int] = field(default_factory=dict)
    daily_views: List[Dict[str, Any]] = field(default_factory=list)
    daily_downloads: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "views_total": self.views_total,
            "views_30d": self.views_30d,
            "downloads_total": self.downloads_total,
            "downloads_30d": self.downloads_30d,
            "revenue_total": self.revenue_total,
            "revenue_30d": self.revenue_30d,
            "unique_downloaders": self.unique_downloaders,
            "avg_rating": self.avg_rating,
            "review_count": self.review_count,
        }


@dataclass
class QualityVerification:
    """Quality verification result."""

    verification_id: str
    dataset_id: str
    level: VerificationLevel
    verified_at: str
    verified_by: str  # "automated" or user_id
    expires_at: Optional[str] = None
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    score: float = 0.0
    certificate_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "dataset_id": self.dataset_id,
            "level": self.level.value,
            "verified_at": self.verified_at,
            "verified_by": self.verified_by,
            "expires_at": self.expires_at,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "score": self.score,
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

    # Versioning
    current_version: Optional[str] = None
    version_count: int = 1

    # Verification
    verification_level: VerificationLevel = VerificationLevel.UNVERIFIED
    verified_at: Optional[str] = None

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
        self._versions: Dict[str, List[DatasetVersion]] = {}
        self._purchases: Dict[str, Dict[str, Purchase]] = {}
        self._verifications: Dict[str, QualityVerification] = {}

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

    # ==================== Enhanced Features ====================

    def advanced_search(
        self,
        filters: SearchFilters,
        sort_by: str = "relevance",
        sort_order: str = "desc",
        page: int = 1,
        per_page: int = 20,
    ) -> SearchResult:
        """Advanced search with comprehensive filtering.

        Args:
            filters: Search filter criteria
            sort_by: Sort field (relevance, downloads, rating, created_at, price, rows)
            sort_order: Sort order (asc, desc)
            page: Page number
            per_page: Results per page

        Returns:
            SearchResult with matching listings
        """
        results = []
        query_terms = filters.query.lower().split() if filters.query else []

        for listing in self._listings.values():
            # Only show published listings
            if listing.status != DatasetStatus.PUBLISHED:
                continue

            # Calculate relevance score
            relevance = 0.0

            # Text search with relevance scoring
            if query_terms:
                name_lower = listing.name.lower()
                desc_lower = listing.description.lower()
                tags_lower = [t.lower() for t in listing.tags]

                matched = False
                for term in query_terms:
                    if term in name_lower:
                        relevance += 3.0
                        matched = True
                    if term in desc_lower:
                        relevance += 1.0
                        matched = True
                    if any(term in tag for tag in tags_lower):
                        relevance += 2.0
                        matched = True

                if not matched:
                    continue

            # Category filter
            if filters.category and listing.category != filters.category:
                continue

            # Multiple categories filter
            if filters.categories and listing.category not in filters.categories:
                continue

            # Tags filter
            if filters.tags:
                if not any(tag in listing.tags for tag in filters.tags):
                    continue

            # License filter
            if filters.license_types and listing.license_type not in filters.license_types:
                continue

            # Quality filter
            if listing.quality_metrics:
                if listing.quality_metrics.overall_score < filters.min_quality:
                    continue

            # Price filters
            if filters.free_only and listing.price > 0:
                continue
            if filters.min_price and listing.price < filters.min_price:
                continue
            if filters.max_price is not None and listing.price > filters.max_price:
                continue

            # Size filters
            if filters.min_rows and listing.n_rows < filters.min_rows:
                continue
            if filters.max_rows and listing.n_rows > filters.max_rows:
                continue
            if filters.min_columns and listing.n_columns < filters.min_columns:
                continue
            if filters.max_columns and listing.n_columns > filters.max_columns:
                continue

            # Rating filter
            if listing.rating < filters.min_rating:
                continue

            # Downloads filter
            if listing.downloads < filters.min_downloads:
                continue

            # Owner filter
            if filters.owner_id and listing.owner_id != filters.owner_id:
                continue

            # Verification filter
            if filters.verified_only and listing.verification_level == VerificationLevel.UNVERIFIED:
                continue
            if filters.verification_level and listing.verification_level != filters.verification_level:
                continue

            # Date filters
            if filters.created_after:
                if listing.created_at < filters.created_after:
                    continue
            if filters.created_before:
                if listing.created_at > filters.created_before:
                    continue

            # Provenance filter
            if filters.has_provenance and not listing.provenance:
                continue

            # Column filters
            if filters.column_names:
                listing_cols = [c["name"] for c in listing.columns]
                if not all(col in listing_cols for col in filters.column_names):
                    continue

            if filters.column_types:
                listing_types = [c.get("dtype", "") for c in listing.columns]
                if not any(t in filters.column_types for t in listing_types):
                    continue

            results.append((listing, relevance))

        # Sort results
        reverse = sort_order == "desc"

        if sort_by == "relevance":
            results.sort(key=lambda x: x[1], reverse=True)
        elif sort_by == "downloads":
            results.sort(key=lambda x: x[0].downloads, reverse=reverse)
        elif sort_by == "rating":
            results.sort(key=lambda x: x[0].rating, reverse=reverse)
        elif sort_by == "created_at":
            results.sort(key=lambda x: x[0].created_at, reverse=reverse)
        elif sort_by == "price":
            results.sort(key=lambda x: x[0].price, reverse=reverse)
        elif sort_by == "rows":
            results.sort(key=lambda x: x[0].n_rows, reverse=reverse)

        # Pagination
        total_count = len(results)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_results = [r[0] for r in results[start_idx:end_idx]]

        return SearchResult(
            listings=page_results,
            total_count=total_count,
            page=page,
            per_page=per_page,
        )

    # ==================== Versioning ====================

    def create_version(
        self,
        dataset_id: str,
        data: pd.DataFrame,
        changelog: str,
        user_id: str,
        version_tag: Optional[str] = None,
    ) -> Optional[DatasetVersion]:
        """Create a new version of a dataset.

        Args:
            dataset_id: Dataset ID
            data: New version data
            changelog: Description of changes
            user_id: User creating the version
            version_tag: Optional semantic version tag

        Returns:
            Created DatasetVersion
        """
        listing = self._listings.get(dataset_id)
        if not listing:
            return None

        if listing.owner_id != user_id:
            logger.warning(f"User {user_id} is not owner of dataset {dataset_id}")
            return None

        # Get next version number
        versions = self._versions.get(dataset_id, [])
        next_version = len(versions) + 1

        if not version_tag:
            version_tag = f"{next_version}.0.0"

        # Create version ID
        version_id = f"{dataset_id}_v{next_version}"

        # Save data
        data_hash = self.storage.save(version_id, data)
        file_size = self.storage.get_size(version_id)

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

        # Compute quality metrics
        quality_metrics = self._compute_quality_metrics(data)

        # Mark previous version as not current
        for v in versions:
            v.is_current = False

        # Create version
        version = DatasetVersion(
            version_id=version_id,
            dataset_id=dataset_id,
            version_number=next_version,
            version_tag=version_tag,
            data_hash=data_hash,
            n_rows=len(data),
            n_columns=len(data.columns),
            file_size_bytes=file_size,
            changelog=changelog,
            created_at=datetime.now().isoformat(),
            created_by=user_id,
            is_current=True,
            quality_metrics=quality_metrics,
            columns=columns,
        )

        if dataset_id not in self._versions:
            self._versions[dataset_id] = []
        self._versions[dataset_id].append(version)

        # Update listing
        listing.current_version = version_id
        listing.version_count = next_version
        listing.n_rows = len(data)
        listing.n_columns = len(data.columns)
        listing.columns = columns
        listing.data_hash = data_hash
        listing.file_size_bytes = file_size
        listing.quality_metrics = quality_metrics
        listing.updated_at = datetime.now().isoformat()

        logger.info(f"Created version {version_tag} for dataset {dataset_id}")

        return version

    def get_versions(self, dataset_id: str) -> List[DatasetVersion]:
        """Get all versions of a dataset.

        Args:
            dataset_id: Dataset ID

        Returns:
            List of versions (newest first)
        """
        versions = self._versions.get(dataset_id, [])
        return sorted(versions, key=lambda v: v.version_number, reverse=True)

    def get_version(self, version_id: str) -> Optional[DatasetVersion]:
        """Get a specific version.

        Args:
            version_id: Version ID

        Returns:
            DatasetVersion or None
        """
        dataset_id = version_id.split("_v")[0]
        versions = self._versions.get(dataset_id, [])

        for version in versions:
            if version.version_id == version_id:
                return version
        return None

    def download_version(
        self,
        version_id: str,
        user_id: str,
    ) -> Optional[pd.DataFrame]:
        """Download a specific version of a dataset.

        Args:
            version_id: Version ID
            user_id: Downloading user ID

        Returns:
            DataFrame or None
        """
        version = self.get_version(version_id)
        if not version:
            return None

        # Check purchase for paid datasets
        listing = self._listings.get(version.dataset_id)
        if listing and listing.price > 0:
            if not self._verify_purchase(version.dataset_id, user_id):
                logger.warning(f"User {user_id} has not purchased dataset {version.dataset_id}")
                return None

        return self.storage.load(version_id)

    def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str,
    ) -> Dict[str, Any]:
        """Compare two versions of a dataset.

        Args:
            version_id_1: First version ID
            version_id_2: Second version ID

        Returns:
            Comparison result
        """
        v1 = self.get_version(version_id_1)
        v2 = self.get_version(version_id_2)

        if not v1 or not v2:
            return {"error": "Version not found"}

        if v1.dataset_id != v2.dataset_id:
            return {"error": "Versions are from different datasets"}

        # Column comparison
        v1_cols = {c["name"]: c for c in v1.columns}
        v2_cols = {c["name"]: c for c in v2.columns}

        added_cols = [c for c in v2_cols if c not in v1_cols]
        removed_cols = [c for c in v1_cols if c not in v2_cols]
        changed_cols = []

        for col in v1_cols:
            if col in v2_cols:
                if v1_cols[col]["dtype"] != v2_cols[col]["dtype"]:
                    changed_cols.append({
                        "name": col,
                        "old_dtype": v1_cols[col]["dtype"],
                        "new_dtype": v2_cols[col]["dtype"],
                    })

        return {
            "version_1": v1.to_dict(),
            "version_2": v2.to_dict(),
            "schema_changes": {
                "columns_added": added_cols,
                "columns_removed": removed_cols,
                "columns_changed": changed_cols,
            },
            "size_changes": {
                "rows_delta": v2.n_rows - v1.n_rows,
                "columns_delta": v2.n_columns - v1.n_columns,
                "size_delta_bytes": v2.file_size_bytes - v1.file_size_bytes,
            },
            "quality_changes": {
                "old_score": v1.quality_metrics.overall_score if v1.quality_metrics else None,
                "new_score": v2.quality_metrics.overall_score if v2.quality_metrics else None,
            },
        }

    # ==================== Purchase Flow ====================

    def initiate_purchase(
        self,
        dataset_id: str,
        user_id: str,
        payment_method: PaymentMethod = PaymentMethod.STRIPE,
        version_id: Optional[str] = None,
    ) -> Optional[Purchase]:
        """Initiate a purchase transaction.

        Args:
            dataset_id: Dataset ID
            user_id: Purchasing user ID
            payment_method: Payment method
            version_id: Specific version to purchase

        Returns:
            Purchase object
        """
        listing = self._listings.get(dataset_id)
        if not listing or listing.status != DatasetStatus.PUBLISHED:
            return None

        # Check if already purchased
        if self._verify_purchase(dataset_id, user_id):
            logger.info(f"User {user_id} already purchased dataset {dataset_id}")
            return None

        # Free datasets
        if listing.price == 0:
            payment_method = PaymentMethod.FREE

        purchase_id = str(uuid.uuid4())[:12]
        expires_at = (datetime.now() + timedelta(hours=24)).isoformat()

        purchase = Purchase(
            purchase_id=purchase_id,
            dataset_id=dataset_id,
            user_id=user_id,
            version_id=version_id or listing.current_version,
            amount=listing.price,
            currency=listing.currency,
            status=PurchaseStatus.PENDING if listing.price > 0 else PurchaseStatus.COMPLETED,
            payment_method=payment_method,
            created_at=datetime.now().isoformat(),
            expires_at=expires_at,
        )

        # Auto-complete free purchases
        if listing.price == 0:
            purchase.completed_at = datetime.now().isoformat()
            purchase.license_key = str(uuid.uuid4())

        if dataset_id not in self._purchases:
            self._purchases[dataset_id] = {}
        self._purchases[dataset_id][user_id] = purchase

        logger.info(f"Initiated purchase {purchase_id} for dataset {dataset_id}")

        return purchase

    def complete_purchase(
        self,
        purchase_id: str,
        payment_reference: str,
    ) -> Optional[Purchase]:
        """Complete a purchase transaction.

        Args:
            purchase_id: Purchase ID
            payment_reference: Payment provider reference

        Returns:
            Updated Purchase
        """
        # Find purchase
        purchase = None
        for dataset_purchases in self._purchases.values():
            for user_purchase in dataset_purchases.values():
                if user_purchase.purchase_id == purchase_id:
                    purchase = user_purchase
                    break

        if not purchase:
            return None

        if purchase.status != PurchaseStatus.PENDING:
            return None

        # Check expiration
        if purchase.expires_at and purchase.expires_at < datetime.now().isoformat():
            purchase.status = PurchaseStatus.EXPIRED
            return purchase

        # Complete purchase
        purchase.status = PurchaseStatus.COMPLETED
        purchase.payment_reference = payment_reference
        purchase.completed_at = datetime.now().isoformat()
        purchase.license_key = str(uuid.uuid4())

        # Update listing stats
        listing = self._listings.get(purchase.dataset_id)
        if listing:
            listing.downloads += 1

        logger.info(f"Completed purchase {purchase_id}")

        return purchase

    def get_purchase(self, dataset_id: str, user_id: str) -> Optional[Purchase]:
        """Get a user's purchase for a dataset.

        Args:
            dataset_id: Dataset ID
            user_id: User ID

        Returns:
            Purchase or None
        """
        dataset_purchases = self._purchases.get(dataset_id, {})
        return dataset_purchases.get(user_id)

    def get_user_purchases(self, user_id: str) -> List[Purchase]:
        """Get all purchases for a user.

        Args:
            user_id: User ID

        Returns:
            List of purchases
        """
        purchases = []
        for dataset_purchases in self._purchases.values():
            if user_id in dataset_purchases:
                purchases.append(dataset_purchases[user_id])
        return purchases

    def _verify_purchase(self, dataset_id: str, user_id: str) -> bool:
        """Check if user has valid purchase for dataset."""
        purchase = self.get_purchase(dataset_id, user_id)
        if not purchase:
            return False
        return purchase.status == PurchaseStatus.COMPLETED

    # ==================== Quality Verification ====================

    def verify_quality(
        self,
        dataset_id: str,
        level: VerificationLevel = VerificationLevel.BASIC,
    ) -> Optional[QualityVerification]:
        """Run quality verification on a dataset.

        Args:
            dataset_id: Dataset ID
            level: Verification level

        Returns:
            QualityVerification result
        """
        listing = self._listings.get(dataset_id)
        if not listing:
            return None

        data = self.storage.load(dataset_id)
        if data is None:
            return None

        checks_passed = []
        checks_failed = []
        score = 0.0

        # Basic checks
        if level in [VerificationLevel.BASIC, VerificationLevel.ENHANCED, VerificationLevel.CERTIFIED]:
            # Row count check
            if listing.n_rows >= 100:
                checks_passed.append("minimum_rows")
                score += 0.1
            else:
                checks_failed.append("minimum_rows")

            # Null rate check
            null_rate = data.isna().sum().sum() / (len(data) * len(data.columns))
            if null_rate < 0.1:
                checks_passed.append("low_null_rate")
                score += 0.1
            else:
                checks_failed.append("low_null_rate")

            # Duplicate check
            dup_rate = data.duplicated().sum() / len(data)
            if dup_rate < 0.05:
                checks_passed.append("low_duplicate_rate")
                score += 0.1
            else:
                checks_failed.append("low_duplicate_rate")

        # Enhanced checks
        if level in [VerificationLevel.ENHANCED, VerificationLevel.CERTIFIED]:
            # Statistical validity
            if listing.quality_metrics and listing.quality_metrics.statistical_fidelity > 0.8:
                checks_passed.append("statistical_fidelity")
                score += 0.2
            else:
                checks_failed.append("statistical_fidelity")

            # Provenance
            if listing.provenance:
                checks_passed.append("has_provenance")
                score += 0.1
            else:
                checks_failed.append("has_provenance")

        # Certified checks
        if level == VerificationLevel.CERTIFIED:
            # Privacy
            if listing.quality_metrics and listing.quality_metrics.privacy_score > 0.9:
                checks_passed.append("high_privacy")
                score += 0.2
            else:
                checks_failed.append("high_privacy")

            # ML utility
            if listing.quality_metrics and listing.quality_metrics.ml_utility > 0.85:
                checks_passed.append("high_ml_utility")
                score += 0.2
            else:
                checks_failed.append("high_ml_utility")

        # Determine verification level achieved
        achieved_level = VerificationLevel.UNVERIFIED
        if len(checks_failed) == 0 and level == VerificationLevel.BASIC:
            achieved_level = VerificationLevel.BASIC
        elif len(checks_failed) <= 1 and level == VerificationLevel.ENHANCED:
            achieved_level = VerificationLevel.ENHANCED
        elif len(checks_failed) == 0 and level == VerificationLevel.CERTIFIED:
            achieved_level = VerificationLevel.CERTIFIED

        verification = QualityVerification(
            verification_id=str(uuid.uuid4())[:8],
            dataset_id=dataset_id,
            level=achieved_level,
            verified_at=datetime.now().isoformat(),
            verified_by="automated",
            expires_at=(datetime.now() + timedelta(days=365)).isoformat(),
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            score=score,
        )

        # Update listing
        listing.verification_level = achieved_level
        listing.verified_at = datetime.now().isoformat()

        self._verifications[dataset_id] = verification

        logger.info(f"Verified dataset {dataset_id} at level {achieved_level.value}")

        return verification

    def get_verification(self, dataset_id: str) -> Optional[QualityVerification]:
        """Get verification status for a dataset.

        Args:
            dataset_id: Dataset ID

        Returns:
            QualityVerification or None
        """
        return self._verifications.get(dataset_id)

    # ==================== Analytics ====================

    def get_analytics(self, dataset_id: str) -> Optional[UsageAnalytics]:
        """Get usage analytics for a dataset.

        Args:
            dataset_id: Dataset ID

        Returns:
            UsageAnalytics
        """
        listing = self._listings.get(dataset_id)
        if not listing:
            return None

        # Calculate revenue
        revenue_total = 0.0
        unique_downloaders = set()

        for user_id, purchase in self._purchases.get(dataset_id, {}).items():
            if purchase.status == PurchaseStatus.COMPLETED:
                revenue_total += purchase.amount
                unique_downloaders.add(user_id)

        return UsageAnalytics(
            dataset_id=dataset_id,
            views_total=listing.views,
            downloads_total=listing.downloads,
            revenue_total=revenue_total,
            unique_downloaders=len(unique_downloaders),
            avg_rating=listing.rating,
            review_count=listing.n_reviews,
        )

    def get_seller_analytics(self, owner_id: str) -> Dict[str, Any]:
        """Get analytics for a seller.

        Args:
            owner_id: Owner user ID

        Returns:
            Aggregated analytics
        """
        listings = [l for l in self._listings.values() if l.owner_id == owner_id]

        total_revenue = 0.0
        total_downloads = 0
        total_views = 0

        for listing in listings:
            total_views += listing.views
            total_downloads += listing.downloads

            for purchase in self._purchases.get(listing.dataset_id, {}).values():
                if purchase.status == PurchaseStatus.COMPLETED:
                    total_revenue += purchase.amount

        return {
            "owner_id": owner_id,
            "total_listings": len(listings),
            "published_listings": sum(1 for l in listings if l.status == DatasetStatus.PUBLISHED),
            "total_views": total_views,
            "total_downloads": total_downloads,
            "total_revenue": total_revenue,
            "avg_rating": np.mean([l.rating for l in listings if l.rating > 0]) if listings else 0,
        }

    # ==================== Recommendations ====================

    def get_recommendations(
        self,
        user_id: str,
        n_recommendations: int = 10,
    ) -> List[DatasetListing]:
        """Get personalized dataset recommendations.

        Args:
            user_id: User ID
            n_recommendations: Number of recommendations

        Returns:
            List of recommended datasets
        """
        # Get user's purchase history
        purchased_ids = set()
        purchased_categories = []
        purchased_tags = []

        for dataset_id, user_purchases in self._purchases.items():
            if user_id in user_purchases:
                purchase = user_purchases[user_id]
                if purchase.status == PurchaseStatus.COMPLETED:
                    purchased_ids.add(dataset_id)
                    listing = self._listings.get(dataset_id)
                    if listing:
                        purchased_categories.append(listing.category)
                        purchased_tags.extend(listing.tags)

        # Score unpurchased datasets
        scored = []

        for listing in self._listings.values():
            if listing.status != DatasetStatus.PUBLISHED:
                continue
            if listing.dataset_id in purchased_ids:
                continue

            score = 0.0

            # Category similarity
            if listing.category in purchased_categories:
                score += 2.0

            # Tag overlap
            tag_overlap = len(set(listing.tags) & set(purchased_tags))
            score += tag_overlap * 0.5

            # Popularity
            score += min(listing.downloads / 100, 2.0)

            # Rating
            score += listing.rating * 0.5

            # Quality
            if listing.quality_metrics:
                score += listing.quality_metrics.overall_score

            scored.append((listing, score))

        # Sort by score
        scored.sort(key=lambda x: x[1], reverse=True)

        return [s[0] for s in scored[:n_recommendations]]

    def get_similar_datasets(
        self,
        dataset_id: str,
        n_similar: int = 5,
    ) -> List[DatasetListing]:
        """Get datasets similar to a given dataset.

        Args:
            dataset_id: Dataset ID
            n_similar: Number of similar datasets

        Returns:
            List of similar datasets
        """
        reference = self._listings.get(dataset_id)
        if not reference:
            return []

        scored = []

        for listing in self._listings.values():
            if listing.status != DatasetStatus.PUBLISHED:
                continue
            if listing.dataset_id == dataset_id:
                continue

            score = 0.0

            # Same category
            if listing.category == reference.category:
                score += 3.0

            # Tag overlap
            tag_overlap = len(set(listing.tags) & set(reference.tags))
            score += tag_overlap

            # Similar size
            row_diff = abs(listing.n_rows - reference.n_rows)
            score += max(0, 1 - row_diff / 10000)

            # Similar column count
            col_diff = abs(listing.n_columns - reference.n_columns)
            score += max(0, 1 - col_diff / 10)

            scored.append((listing, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [s[0] for s in scored[:n_similar]]


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
    # New classes
    "SearchFilters",
    "DatasetVersion",
    "Purchase",
    "UsageAnalytics",
    "QualityVerification",
    # Types
    "LicenseType",
    "DatasetCategory",
    "DatasetStatus",
    "PurchaseStatus",
    "PaymentMethod",
    "VerificationLevel",
]
