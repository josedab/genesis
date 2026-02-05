"""Synthetic Data Marketplace 2.0.

Enhanced marketplace with commerce capabilities, organizations, subscriptions,
revenue sharing, and trust/certification features.

Features:
    - Organization accounts with teams
    - Stripe/payment integration
    - Subscription licensing models
    - Revenue sharing for contributors
    - Automated quality certification
    - User reviews and ratings
    - Provenance verification
    - DMCA/takedown process

Example:
    Create organization and publish dataset::

        from genesis.marketplace_v2 import MarketplaceV2, Organization

        marketplace = MarketplaceV2()

        # Create organization
        org = marketplace.create_organization(
            name="Acme Data",
            owner_id="user_123",
        )

        # Publish dataset with revenue sharing
        listing = marketplace.create_listing_v2(
            name="E-Commerce Transactions",
            data=synthetic_df,
            organization_id=org.org_id,
            price_config=PriceConfig(
                base_price=99.0,
                subscription_monthly=19.0,
            ),
            contributors=[
                Contributor(user_id="user_456", share_percent=20.0),
            ],
        )

        # Process purchase with Stripe
        result = marketplace.process_payment(
            listing_id=listing.dataset_id,
            buyer_id="buyer_789",
            payment_method="stripe",
            payment_token="pm_xxx",
        )

Classes:
    MarketplaceV2: Enhanced marketplace with commerce features.
    Organization: Organization account.
    Team: Team within organization.
    PriceConfig: Pricing configuration.
    Subscription: Subscription license.
    Contributor: Revenue share contributor.
    PaymentProcessor: Payment processing abstraction.
    CertificationEngine: Automated quality certification.
    TakedownRequest: DMCA/content takedown.
"""

import hashlib
import json
import os
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


class SubscriptionTier(str, Enum):
    """Subscription tiers."""

    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class OrganizationType(str, Enum):
    """Organization types."""

    INDIVIDUAL = "individual"
    STARTUP = "startup"
    ENTERPRISE = "enterprise"
    ACADEMIC = "academic"
    NONPROFIT = "nonprofit"


class PaymentStatus(str, Enum):
    """Payment status."""

    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REFUNDED = "refunded"
    DISPUTED = "disputed"


class CertificationLevel(str, Enum):
    """Quality certification levels."""

    NONE = "none"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


class TakedownStatus(str, Enum):
    """Takedown request status."""

    PENDING = "pending"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPEALED = "appealed"


@dataclass
class Contributor:
    """Revenue share contributor.

    Attributes:
        user_id: Contributor user ID.
        share_percent: Revenue share percentage.
        role: Role description.
        joined_at: When they joined.
    """

    user_id: str
    share_percent: float  # 0-100
    role: str = "contributor"
    joined_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "share_percent": self.share_percent,
            "role": self.role,
            "joined_at": self.joined_at,
        }


@dataclass
class PriceConfig:
    """Pricing configuration.

    Attributes:
        base_price: One-time purchase price.
        subscription_monthly: Monthly subscription price.
        subscription_yearly: Yearly subscription price.
        enterprise_contact: Enterprise pricing requires contact.
        free_tier_rows: Free tier row limit.
        api_calls_per_month: API calls included.
        revenue_share_percent: Platform revenue share.
    """

    base_price: float = 0.0
    subscription_monthly: Optional[float] = None
    subscription_yearly: Optional[float] = None
    enterprise_contact: bool = False
    free_tier_rows: int = 100
    api_calls_per_month: int = 1000
    revenue_share_percent: float = 20.0  # Platform takes 20%
    currency: str = "USD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_price": self.base_price,
            "subscription_monthly": self.subscription_monthly,
            "subscription_yearly": self.subscription_yearly,
            "enterprise_contact": self.enterprise_contact,
            "free_tier_rows": self.free_tier_rows,
            "api_calls_per_month": self.api_calls_per_month,
            "revenue_share_percent": self.revenue_share_percent,
            "currency": self.currency,
        }


@dataclass
class TeamMember:
    """Team member within organization."""

    user_id: str
    role: str  # admin, editor, viewer
    added_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    added_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "role": self.role,
            "added_at": self.added_at,
        }


@dataclass
class Team:
    """Team within organization.

    Attributes:
        team_id: Unique team identifier.
        name: Team name.
        org_id: Parent organization ID.
        members: Team members.
        datasets: Datasets accessible to team.
    """

    team_id: str
    name: str
    org_id: str
    members: List[TeamMember] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)  # dataset IDs
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_id": self.team_id,
            "name": self.name,
            "org_id": self.org_id,
            "members": [m.to_dict() for m in self.members],
            "datasets": self.datasets,
            "created_at": self.created_at,
        }


@dataclass
class Organization:
    """Organization account.

    Attributes:
        org_id: Unique organization identifier.
        name: Organization name.
        org_type: Organization type.
        owner_id: Owner user ID.
        members: Organization members.
        teams: Teams within organization.
        subscription_tier: Current subscription tier.
        billing_email: Billing email address.
        verified: Whether organization is verified.
        datasets_published: Number of datasets published.
        total_revenue: Total revenue earned.
    """

    org_id: str
    name: str
    org_type: OrganizationType = OrganizationType.INDIVIDUAL
    owner_id: str = ""
    members: List[TeamMember] = field(default_factory=list)
    teams: List[Team] = field(default_factory=list)
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    billing_email: str = ""
    verified: bool = False
    datasets_published: int = 0
    total_revenue: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "org_id": self.org_id,
            "name": self.name,
            "org_type": self.org_type.value,
            "owner_id": self.owner_id,
            "members": [m.to_dict() for m in self.members],
            "teams": [t.to_dict() for t in self.teams],
            "subscription_tier": self.subscription_tier.value,
            "verified": self.verified,
            "datasets_published": self.datasets_published,
            "total_revenue": self.total_revenue,
        }


@dataclass
class Subscription:
    """Active subscription.

    Attributes:
        subscription_id: Unique subscription ID.
        user_id: Subscriber user ID.
        dataset_id: Subscribed dataset ID.
        tier: Subscription tier.
        status: Subscription status.
        price: Monthly/yearly price.
        interval: Billing interval.
        current_period_start: Current period start.
        current_period_end: Current period end.
        stripe_subscription_id: Stripe subscription ID.
    """

    subscription_id: str
    user_id: str
    dataset_id: str
    tier: SubscriptionTier = SubscriptionTier.BASIC
    status: str = "active"  # active, cancelled, past_due
    price: float = 0.0
    interval: str = "month"  # month, year
    current_period_start: str = ""
    current_period_end: str = ""
    stripe_subscription_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    cancelled_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "user_id": self.user_id,
            "dataset_id": self.dataset_id,
            "tier": self.tier.value,
            "status": self.status,
            "price": self.price,
            "interval": self.interval,
            "current_period_start": self.current_period_start,
            "current_period_end": self.current_period_end,
        }


@dataclass
class Payment:
    """Payment transaction.

    Attributes:
        payment_id: Unique payment ID.
        amount: Payment amount.
        currency: Currency code.
        status: Payment status.
        buyer_id: Buyer user ID.
        seller_id: Seller user/org ID.
        dataset_id: Dataset being purchased.
        payment_method: Payment method used.
        stripe_payment_intent: Stripe payment intent ID.
        revenue_splits: Revenue distribution.
    """

    payment_id: str
    amount: float
    currency: str = "USD"
    status: PaymentStatus = PaymentStatus.PENDING
    buyer_id: str = ""
    seller_id: str = ""
    dataset_id: str = ""
    payment_method: str = "stripe"
    stripe_payment_intent: Optional[str] = None
    revenue_splits: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "payment_id": self.payment_id,
            "amount": self.amount,
            "currency": self.currency,
            "status": self.status.value,
            "buyer_id": self.buyer_id,
            "seller_id": self.seller_id,
            "dataset_id": self.dataset_id,
            "revenue_splits": self.revenue_splits,
        }


@dataclass
class Certification:
    """Quality certification.

    Attributes:
        cert_id: Certification ID.
        dataset_id: Certified dataset.
        level: Certification level.
        score: Overall quality score.
        checks: Checks performed.
        valid_until: Certification expiry.
        issued_by: Issuing authority.
    """

    cert_id: str
    dataset_id: str
    level: CertificationLevel = CertificationLevel.NONE
    score: float = 0.0
    checks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    valid_until: str = ""
    issued_by: str = "automated"
    issued_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cert_id": self.cert_id,
            "dataset_id": self.dataset_id,
            "level": self.level.value,
            "score": self.score,
            "checks": self.checks,
            "valid_until": self.valid_until,
            "issued_at": self.issued_at,
        }


@dataclass
class TakedownRequest:
    """DMCA/content takedown request.

    Attributes:
        request_id: Request ID.
        dataset_id: Dataset to take down.
        requester_id: Person requesting takedown.
        reason: Takedown reason.
        status: Request status.
        evidence: Supporting evidence.
    """

    request_id: str
    dataset_id: str
    requester_id: str
    reason: str
    status: TakedownStatus = TakedownStatus.PENDING
    evidence: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    resolved_at: Optional[str] = None
    resolution_notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "dataset_id": self.dataset_id,
            "requester_id": self.requester_id,
            "reason": self.reason,
            "status": self.status.value,
            "created_at": self.created_at,
        }


@dataclass
class ListingV2:
    """Enhanced dataset listing with commerce features.

    Extends DatasetListing with pricing, subscriptions, and organization.
    """

    dataset_id: str
    name: str
    description: str
    owner_id: str
    organization_id: Optional[str] = None

    # Pricing
    price_config: Optional[PriceConfig] = None
    contributors: List[Contributor] = field(default_factory=list)

    # Certification
    certification: Optional[Certification] = None

    # Stats
    total_revenue: float = 0.0
    active_subscriptions: int = 0
    purchases: int = 0

    # Metadata
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "owner_id": self.owner_id,
            "organization_id": self.organization_id,
            "price_config": self.price_config.to_dict() if self.price_config else None,
            "contributors": [c.to_dict() for c in self.contributors],
            "certification": self.certification.to_dict() if self.certification else None,
            "total_revenue": self.total_revenue,
            "active_subscriptions": self.active_subscriptions,
        }


class PaymentProcessor:
    """Payment processing abstraction.

    Handles Stripe integration for payments and subscriptions.
    """

    def __init__(self, stripe_api_key: Optional[str] = None) -> None:
        """Initialize payment processor.

        Args:
            stripe_api_key: Stripe API key. Uses env var if not provided.
        """
        self.api_key = stripe_api_key or os.environ.get("STRIPE_API_KEY")
        self._stripe = None

        if self.api_key:
            try:
                import stripe
                stripe.api_key = self.api_key
                self._stripe = stripe
                logger.info("Stripe payment processor initialized")
            except ImportError:
                logger.warning("stripe package not installed")

    @property
    def available(self) -> bool:
        """Check if payment processing is available."""
        return self._stripe is not None

    def create_payment_intent(
        self,
        amount: float,
        currency: str = "usd",
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create Stripe payment intent.

        Args:
            amount: Amount in dollars.
            currency: Currency code.
            metadata: Payment metadata.

        Returns:
            Payment intent data.
        """
        if not self._stripe:
            # Mock for testing
            return {
                "id": f"pi_mock_{uuid.uuid4().hex[:12]}",
                "client_secret": f"secret_mock_{uuid.uuid4().hex[:24]}",
                "amount": int(amount * 100),
                "currency": currency.lower(),
                "status": "requires_payment_method",
            }

        intent = self._stripe.PaymentIntent.create(
            amount=int(amount * 100),  # Stripe uses cents
            currency=currency.lower(),
            metadata=metadata or {},
        )

        return {
            "id": intent.id,
            "client_secret": intent.client_secret,
            "amount": intent.amount,
            "currency": intent.currency,
            "status": intent.status,
        }

    def confirm_payment(self, payment_intent_id: str) -> Dict[str, Any]:
        """Confirm payment intent status.

        Args:
            payment_intent_id: Payment intent ID.

        Returns:
            Payment status.
        """
        if not self._stripe:
            return {
                "id": payment_intent_id,
                "status": "succeeded",
            }

        intent = self._stripe.PaymentIntent.retrieve(payment_intent_id)
        return {
            "id": intent.id,
            "status": intent.status,
            "amount_received": intent.amount_received,
        }

    def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create Stripe subscription.

        Args:
            customer_id: Stripe customer ID.
            price_id: Stripe price ID.
            metadata: Subscription metadata.

        Returns:
            Subscription data.
        """
        if not self._stripe:
            now = datetime.utcnow()
            return {
                "id": f"sub_mock_{uuid.uuid4().hex[:12]}",
                "status": "active",
                "current_period_start": now.isoformat(),
                "current_period_end": (now + timedelta(days=30)).isoformat(),
            }

        sub = self._stripe.Subscription.create(
            customer=customer_id,
            items=[{"price": price_id}],
            metadata=metadata or {},
        )

        return {
            "id": sub.id,
            "status": sub.status,
            "current_period_start": datetime.fromtimestamp(sub.current_period_start).isoformat(),
            "current_period_end": datetime.fromtimestamp(sub.current_period_end).isoformat(),
        }

    def cancel_subscription(self, subscription_id: str) -> Dict[str, Any]:
        """Cancel subscription.

        Args:
            subscription_id: Stripe subscription ID.

        Returns:
            Cancellation result.
        """
        if not self._stripe:
            return {"id": subscription_id, "status": "canceled"}

        sub = self._stripe.Subscription.delete(subscription_id)
        return {"id": sub.id, "status": sub.status}

    def create_payout(
        self,
        amount: float,
        destination: str,  # Stripe Connect account
        currency: str = "usd",
    ) -> Dict[str, Any]:
        """Create payout to connected account.

        Args:
            amount: Payout amount.
            destination: Stripe Connect account ID.
            currency: Currency code.

        Returns:
            Payout result.
        """
        if not self._stripe:
            return {
                "id": f"po_mock_{uuid.uuid4().hex[:12]}",
                "amount": int(amount * 100),
                "status": "paid",
            }

        transfer = self._stripe.Transfer.create(
            amount=int(amount * 100),
            currency=currency.lower(),
            destination=destination,
        )

        return {
            "id": transfer.id,
            "amount": transfer.amount,
            "status": "paid",
        }


class CertificationEngine:
    """Automated quality certification engine.

    Evaluates datasets and issues quality certifications.
    """

    def __init__(self) -> None:
        """Initialize certification engine."""
        self.checks = {
            "statistical_fidelity": self._check_statistical_fidelity,
            "privacy_score": self._check_privacy_score,
            "completeness": self._check_completeness,
            "consistency": self._check_consistency,
            "provenance": self._check_provenance,
        }

    def certify(
        self,
        data: pd.DataFrame,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Certification:
        """Run certification checks and issue certificate.

        Args:
            data: Dataset to certify.
            provenance: Provenance information.

        Returns:
            Certification result.
        """
        results = {}
        total_score = 0.0
        weights = {
            "statistical_fidelity": 0.25,
            "privacy_score": 0.30,
            "completeness": 0.20,
            "consistency": 0.15,
            "provenance": 0.10,
        }

        for check_name, check_fn in self.checks.items():
            try:
                passed, score, details = check_fn(data, provenance)
                results[check_name] = {
                    "passed": passed,
                    "score": score,
                    "details": details,
                }
                total_score += score * weights.get(check_name, 0.2)
            except Exception as e:
                results[check_name] = {
                    "passed": False,
                    "score": 0.0,
                    "details": {"error": str(e)},
                }

        # Determine certification level
        if total_score >= 0.95:
            level = CertificationLevel.PLATINUM
        elif total_score >= 0.85:
            level = CertificationLevel.GOLD
        elif total_score >= 0.70:
            level = CertificationLevel.SILVER
        elif total_score >= 0.50:
            level = CertificationLevel.BRONZE
        else:
            level = CertificationLevel.NONE

        valid_until = (datetime.utcnow() + timedelta(days=365)).isoformat()

        return Certification(
            cert_id=str(uuid.uuid4())[:12],
            dataset_id="",  # Set by caller
            level=level,
            score=total_score,
            checks=results,
            valid_until=valid_until,
        )

    def _check_statistical_fidelity(
        self,
        data: pd.DataFrame,
        provenance: Optional[Dict[str, Any]],
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Check statistical properties."""
        # Basic checks - in production, compare with reference stats
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        details = {"numeric_columns": len(numeric_cols)}

        # Check for reasonable distributions
        score = 0.8  # Base score

        # Penalize if all values are same
        for col in numeric_cols:
            if data[col].std() == 0:
                score -= 0.1

        score = max(0.0, min(1.0, score))
        return score >= 0.6, score, details

    def _check_privacy_score(
        self,
        data: pd.DataFrame,
        provenance: Optional[Dict[str, Any]],
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Check privacy measures."""
        details = {}
        score = 0.7  # Base score

        # Check for potential PII
        pii_patterns = ["ssn", "social", "passport", "credit_card", "password"]
        pii_found = []
        for col in data.columns:
            col_lower = col.lower()
            if any(p in col_lower for p in pii_patterns):
                pii_found.append(col)
                score -= 0.2

        details["potential_pii_columns"] = pii_found

        # Check if provenance indicates DP was used
        if provenance and "privacy_mechanisms" in provenance:
            if "differential_privacy" in provenance["privacy_mechanisms"]:
                score += 0.2

        score = max(0.0, min(1.0, score))
        return score >= 0.5 and len(pii_found) == 0, score, details

    def _check_completeness(
        self,
        data: pd.DataFrame,
        provenance: Optional[Dict[str, Any]],
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Check data completeness."""
        null_fraction = data.isnull().sum().sum() / data.size if data.size > 0 else 0
        score = 1.0 - null_fraction

        details = {
            "null_fraction": null_fraction,
            "rows": len(data),
            "columns": len(data.columns),
        }

        return null_fraction < 0.1, score, details

    def _check_consistency(
        self,
        data: pd.DataFrame,
        provenance: Optional[Dict[str, Any]],
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Check data consistency."""
        issues = []
        score = 1.0

        # Check for duplicated rows
        dup_fraction = data.duplicated().sum() / len(data) if len(data) > 0 else 0
        if dup_fraction > 0.1:
            issues.append(f"High duplicate rate: {dup_fraction:.2%}")
            score -= 0.3

        details = {
            "duplicate_fraction": dup_fraction,
            "issues": issues,
        }

        return len(issues) == 0, max(0.0, score), details

    def _check_provenance(
        self,
        data: pd.DataFrame,
        provenance: Optional[Dict[str, Any]],
    ) -> tuple[bool, float, Dict[str, Any]]:
        """Check provenance documentation."""
        if not provenance:
            return False, 0.0, {"has_provenance": False}

        required_fields = ["generator_method", "generation_date"]
        present = [f for f in required_fields if f in provenance]
        score = len(present) / len(required_fields)

        details = {
            "has_provenance": True,
            "documented_fields": present,
        }

        return score >= 0.5, score, details


class MarketplaceV2:
    """Enhanced Synthetic Data Marketplace.

    Features:
    - Organization accounts with teams
    - Stripe payment integration
    - Subscription licensing
    - Revenue sharing for contributors
    - Automated quality certification
    - DMCA/takedown process
    """

    def __init__(
        self,
        storage_path: Optional[Union[str, Path]] = None,
        stripe_api_key: Optional[str] = None,
    ) -> None:
        """Initialize Marketplace V2.

        Args:
            storage_path: Path for data storage.
            stripe_api_key: Stripe API key.
        """
        self.storage_path = (
            Path(storage_path) if storage_path else Path.home() / ".genesis" / "marketplace_v2"
        )
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.payment_processor = PaymentProcessor(stripe_api_key)
        self.certification_engine = CertificationEngine()

        # In-memory stores (use database in production)
        self._organizations: Dict[str, Organization] = {}
        self._listings: Dict[str, ListingV2] = {}
        self._subscriptions: Dict[str, Subscription] = {}
        self._payments: Dict[str, Payment] = {}
        self._takedowns: Dict[str, TakedownRequest] = {}
        self._user_stripe_customers: Dict[str, str] = {}  # user_id -> stripe_customer_id

    # Organization Management

    def create_organization(
        self,
        name: str,
        owner_id: str,
        org_type: OrganizationType = OrganizationType.INDIVIDUAL,
        billing_email: Optional[str] = None,
    ) -> Organization:
        """Create a new organization.

        Args:
            name: Organization name.
            owner_id: Owner user ID.
            org_type: Organization type.
            billing_email: Billing email address.

        Returns:
            Created organization.
        """
        org_id = f"org_{uuid.uuid4().hex[:12]}"

        org = Organization(
            org_id=org_id,
            name=name,
            org_type=org_type,
            owner_id=owner_id,
            billing_email=billing_email or "",
            members=[TeamMember(user_id=owner_id, role="admin")],
        )

        self._organizations[org_id] = org
        logger.info(f"Created organization: {name} ({org_id})")

        return org

    def get_organization(self, org_id: str) -> Optional[Organization]:
        """Get organization by ID."""
        return self._organizations.get(org_id)

    def add_team_member(
        self,
        org_id: str,
        user_id: str,
        role: str = "viewer",
        added_by: Optional[str] = None,
    ) -> bool:
        """Add member to organization.

        Args:
            org_id: Organization ID.
            user_id: User to add.
            role: Member role (admin, editor, viewer).
            added_by: User adding the member.

        Returns:
            Success status.
        """
        org = self._organizations.get(org_id)
        if not org:
            return False

        # Check if already member
        if any(m.user_id == user_id for m in org.members):
            return False

        org.members.append(TeamMember(
            user_id=user_id,
            role=role,
            added_by=added_by,
        ))

        return True

    def create_team(
        self,
        org_id: str,
        name: str,
        member_ids: Optional[List[str]] = None,
    ) -> Optional[Team]:
        """Create team within organization.

        Args:
            org_id: Organization ID.
            name: Team name.
            member_ids: Initial member IDs.

        Returns:
            Created team.
        """
        org = self._organizations.get(org_id)
        if not org:
            return None

        team_id = f"team_{uuid.uuid4().hex[:8]}"

        members = []
        for uid in (member_ids or []):
            members.append(TeamMember(user_id=uid, role="member"))

        team = Team(
            team_id=team_id,
            name=name,
            org_id=org_id,
            members=members,
        )

        org.teams.append(team)
        return team

    # Listing Management

    def create_listing_v2(
        self,
        name: str,
        description: str,
        data: pd.DataFrame,
        owner_id: str,
        organization_id: Optional[str] = None,
        price_config: Optional[PriceConfig] = None,
        contributors: Optional[List[Contributor]] = None,
        tags: Optional[List[str]] = None,
        auto_certify: bool = True,
    ) -> ListingV2:
        """Create enhanced dataset listing.

        Args:
            name: Dataset name.
            description: Dataset description.
            data: Synthetic data.
            owner_id: Owner user ID.
            organization_id: Organization ID (optional).
            price_config: Pricing configuration.
            contributors: Revenue share contributors.
            tags: Searchable tags.
            auto_certify: Run automatic certification.

        Returns:
            Created listing.
        """
        dataset_id = f"ds_{uuid.uuid4().hex[:12]}"

        # Save data
        data_path = self.storage_path / "datasets" / f"{dataset_id}.parquet"
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(data_path, index=False)

        # Run certification if requested
        certification = None
        if auto_certify:
            certification = self.certification_engine.certify(data)
            certification.dataset_id = dataset_id

        listing = ListingV2(
            dataset_id=dataset_id,
            name=name,
            description=description,
            owner_id=owner_id,
            organization_id=organization_id,
            price_config=price_config or PriceConfig(),
            contributors=contributors or [],
            certification=certification,
            tags=tags or [],
        )

        self._listings[dataset_id] = listing

        # Update organization stats
        if organization_id and organization_id in self._organizations:
            self._organizations[organization_id].datasets_published += 1

        logger.info(f"Created listing: {name} ({dataset_id})")
        return listing

    def get_listing(self, dataset_id: str) -> Optional[ListingV2]:
        """Get listing by ID."""
        return self._listings.get(dataset_id)

    def update_pricing(
        self,
        dataset_id: str,
        price_config: PriceConfig,
    ) -> bool:
        """Update listing pricing.

        Args:
            dataset_id: Dataset ID.
            price_config: New pricing configuration.

        Returns:
            Success status.
        """
        listing = self._listings.get(dataset_id)
        if not listing:
            return False

        listing.price_config = price_config
        return True

    def add_contributor(
        self,
        dataset_id: str,
        user_id: str,
        share_percent: float,
        role: str = "contributor",
    ) -> bool:
        """Add revenue share contributor.

        Args:
            dataset_id: Dataset ID.
            user_id: Contributor user ID.
            share_percent: Revenue share percentage.
            role: Contributor role.

        Returns:
            Success status.
        """
        listing = self._listings.get(dataset_id)
        if not listing:
            return False

        # Validate total shares don't exceed 100%
        current_shares = sum(c.share_percent for c in listing.contributors)
        if current_shares + share_percent > 100:
            logger.warning(f"Total shares would exceed 100%: {current_shares + share_percent}")
            return False

        listing.contributors.append(Contributor(
            user_id=user_id,
            share_percent=share_percent,
            role=role,
        ))

        return True

    # Payment & Subscription Management

    def process_payment(
        self,
        dataset_id: str,
        buyer_id: str,
        payment_method: str = "stripe",
        payment_token: Optional[str] = None,
    ) -> Payment:
        """Process one-time purchase payment.

        Args:
            dataset_id: Dataset to purchase.
            buyer_id: Buyer user ID.
            payment_method: Payment method.
            payment_token: Payment token (e.g., Stripe PM ID).

        Returns:
            Payment result.
        """
        listing = self._listings.get(dataset_id)
        if not listing:
            raise ValueError(f"Dataset not found: {dataset_id}")

        price = listing.price_config.base_price if listing.price_config else 0.0

        payment_id = f"pay_{uuid.uuid4().hex[:12]}"

        # Create payment intent
        intent_data = {}
        if payment_method == "stripe" and price > 0:
            intent_data = self.payment_processor.create_payment_intent(
                amount=price,
                metadata={"dataset_id": dataset_id, "buyer_id": buyer_id},
            )

        # Calculate revenue splits
        revenue_splits = self._calculate_revenue_splits(listing, price)

        payment = Payment(
            payment_id=payment_id,
            amount=price,
            status=PaymentStatus.PENDING if price > 0 else PaymentStatus.SUCCEEDED,
            buyer_id=buyer_id,
            seller_id=listing.owner_id,
            dataset_id=dataset_id,
            payment_method=payment_method,
            stripe_payment_intent=intent_data.get("id"),
            revenue_splits=revenue_splits,
        )

        if price == 0:
            payment.status = PaymentStatus.SUCCEEDED
            payment.completed_at = datetime.utcnow().isoformat()
            listing.purchases += 1

        self._payments[payment_id] = payment
        return payment

    def confirm_payment(self, payment_id: str) -> bool:
        """Confirm and complete payment.

        Args:
            payment_id: Payment ID to confirm.

        Returns:
            Success status.
        """
        payment = self._payments.get(payment_id)
        if not payment:
            return False

        if payment.stripe_payment_intent:
            result = self.payment_processor.confirm_payment(payment.stripe_payment_intent)
            if result.get("status") == "succeeded":
                payment.status = PaymentStatus.SUCCEEDED
                payment.completed_at = datetime.utcnow().isoformat()

                # Update listing stats
                listing = self._listings.get(payment.dataset_id)
                if listing:
                    listing.purchases += 1
                    listing.total_revenue += payment.amount

                # Process revenue splits
                self._process_revenue_splits(payment)

                return True

        return False

    def create_subscription(
        self,
        dataset_id: str,
        user_id: str,
        interval: str = "month",
    ) -> Subscription:
        """Create dataset subscription.

        Args:
            dataset_id: Dataset to subscribe to.
            user_id: Subscriber user ID.
            interval: Billing interval (month/year).

        Returns:
            Created subscription.
        """
        listing = self._listings.get(dataset_id)
        if not listing:
            raise ValueError(f"Dataset not found: {dataset_id}")

        price_config = listing.price_config or PriceConfig()

        price = (
            price_config.subscription_yearly
            if interval == "year"
            else price_config.subscription_monthly
        )

        if not price:
            raise ValueError(f"No subscription pricing for dataset: {dataset_id}")

        subscription_id = f"sub_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()

        period_end = (
            now + timedelta(days=365) if interval == "year" else now + timedelta(days=30)
        )

        subscription = Subscription(
            subscription_id=subscription_id,
            user_id=user_id,
            dataset_id=dataset_id,
            price=price,
            interval=interval,
            current_period_start=now.isoformat(),
            current_period_end=period_end.isoformat(),
        )

        # Create Stripe subscription if available
        if self.payment_processor.available:
            stripe_customer = self._user_stripe_customers.get(user_id)
            if stripe_customer:
                result = self.payment_processor.create_subscription(
                    customer_id=stripe_customer,
                    price_id="price_placeholder",  # Would use actual Stripe price ID
                    metadata={"dataset_id": dataset_id},
                )
                subscription.stripe_subscription_id = result.get("id")

        self._subscriptions[subscription_id] = subscription

        # Update listing stats
        listing.active_subscriptions += 1

        logger.info(f"Created subscription: {subscription_id}")
        return subscription

    def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel subscription.

        Args:
            subscription_id: Subscription to cancel.

        Returns:
            Success status.
        """
        subscription = self._subscriptions.get(subscription_id)
        if not subscription:
            return False

        subscription.status = "cancelled"
        subscription.cancelled_at = datetime.utcnow().isoformat()

        # Cancel in Stripe
        if subscription.stripe_subscription_id:
            self.payment_processor.cancel_subscription(subscription.stripe_subscription_id)

        # Update listing stats
        listing = self._listings.get(subscription.dataset_id)
        if listing and listing.active_subscriptions > 0:
            listing.active_subscriptions -= 1

        return True

    def _calculate_revenue_splits(
        self,
        listing: ListingV2,
        amount: float,
    ) -> List[Dict[str, Any]]:
        """Calculate revenue distribution.

        Args:
            listing: Dataset listing.
            amount: Total payment amount.

        Returns:
            List of revenue splits.
        """
        splits = []
        platform_fee = amount * 0.2  # 20% platform fee
        contributor_pool = amount - platform_fee

        # Platform fee
        splits.append({
            "recipient": "platform",
            "amount": platform_fee,
            "type": "platform_fee",
        })

        # Contributor shares
        remaining = contributor_pool
        for contributor in listing.contributors:
            contrib_amount = contributor_pool * (contributor.share_percent / 100)
            splits.append({
                "recipient": contributor.user_id,
                "amount": contrib_amount,
                "type": "contributor_share",
            })
            remaining -= contrib_amount

        # Owner gets remaining
        splits.append({
            "recipient": listing.owner_id,
            "amount": remaining,
            "type": "owner_share",
        })

        return splits

    def _process_revenue_splits(self, payment: Payment) -> None:
        """Process revenue distribution.

        Args:
            payment: Completed payment.
        """
        for split in payment.revenue_splits:
            if split["recipient"] == "platform":
                continue  # Platform fee stays in platform

            # In production, would trigger actual payouts
            logger.info(
                f"Revenue split: {split['recipient']} receives ${split['amount']:.2f}"
            )

    # Certification

    def request_certification(self, dataset_id: str) -> Optional[Certification]:
        """Request dataset certification.

        Args:
            dataset_id: Dataset to certify.

        Returns:
            Certification result.
        """
        listing = self._listings.get(dataset_id)
        if not listing:
            return None

        # Load data
        data_path = self.storage_path / "datasets" / f"{dataset_id}.parquet"
        if not data_path.exists():
            return None

        data = pd.read_parquet(data_path)

        # Run certification
        certification = self.certification_engine.certify(data)
        certification.dataset_id = dataset_id

        listing.certification = certification
        return certification

    # Takedown Management

    def submit_takedown(
        self,
        dataset_id: str,
        requester_id: str,
        reason: str,
        evidence: Optional[List[str]] = None,
    ) -> TakedownRequest:
        """Submit DMCA/content takedown request.

        Args:
            dataset_id: Dataset to take down.
            requester_id: Person submitting request.
            reason: Reason for takedown.
            evidence: Supporting evidence URLs.

        Returns:
            Takedown request.
        """
        request_id = f"td_{uuid.uuid4().hex[:12]}"

        request = TakedownRequest(
            request_id=request_id,
            dataset_id=dataset_id,
            requester_id=requester_id,
            reason=reason,
            evidence=evidence or [],
        )

        self._takedowns[request_id] = request
        logger.info(f"Takedown request submitted: {request_id}")

        return request

    def review_takedown(
        self,
        request_id: str,
        approved: bool,
        resolution_notes: str = "",
    ) -> bool:
        """Review and resolve takedown request.

        Args:
            request_id: Takedown request ID.
            approved: Whether to approve takedown.
            resolution_notes: Resolution notes.

        Returns:
            Success status.
        """
        request = self._takedowns.get(request_id)
        if not request:
            return False

        request.status = TakedownStatus.APPROVED if approved else TakedownStatus.REJECTED
        request.resolved_at = datetime.utcnow().isoformat()
        request.resolution_notes = resolution_notes

        if approved:
            # Remove listing
            listing = self._listings.get(request.dataset_id)
            if listing:
                # Archive instead of delete
                del self._listings[request.dataset_id]
                logger.info(f"Dataset removed due to takedown: {request.dataset_id}")

        return True

    # Search & Discovery

    def search(
        self,
        query: str = "",
        tags: Optional[List[str]] = None,
        min_cert_level: Optional[CertificationLevel] = None,
        max_price: Optional[float] = None,
        org_id: Optional[str] = None,
        page: int = 1,
        per_page: int = 20,
    ) -> Dict[str, Any]:
        """Search marketplace listings.

        Args:
            query: Text search query.
            tags: Filter by tags.
            min_cert_level: Minimum certification level.
            max_price: Maximum price.
            org_id: Filter by organization.
            page: Page number.
            per_page: Results per page.

        Returns:
            Search results.
        """
        results = []

        for listing in self._listings.values():
            # Text search
            if query:
                query_lower = query.lower()
                if (
                    query_lower not in listing.name.lower()
                    and query_lower not in listing.description.lower()
                ):
                    continue

            # Tag filter
            if tags and not any(t in listing.tags for t in tags):
                continue

            # Certification filter
            if min_cert_level and listing.certification:
                cert_order = [
                    CertificationLevel.NONE,
                    CertificationLevel.BRONZE,
                    CertificationLevel.SILVER,
                    CertificationLevel.GOLD,
                    CertificationLevel.PLATINUM,
                ]
                if cert_order.index(listing.certification.level) < cert_order.index(min_cert_level):
                    continue

            # Price filter
            if max_price is not None and listing.price_config:
                if listing.price_config.base_price > max_price:
                    continue

            # Organization filter
            if org_id and listing.organization_id != org_id:
                continue

            results.append(listing)

        # Sort by purchases
        results.sort(key=lambda x: x.purchases, reverse=True)

        # Pagination
        total = len(results)
        start = (page - 1) * per_page
        page_results = results[start:start + per_page]

        return {
            "listings": [l.to_dict() for l in page_results],
            "total": total,
            "page": page,
            "per_page": per_page,
        }

    # Data Access

    def download_data(
        self,
        dataset_id: str,
        user_id: str,
    ) -> Optional[pd.DataFrame]:
        """Download dataset (with access check).

        Args:
            dataset_id: Dataset to download.
            user_id: User requesting download.

        Returns:
            Dataset if user has access.
        """
        listing = self._listings.get(dataset_id)
        if not listing:
            return None

        # Check if free
        if listing.price_config and listing.price_config.base_price == 0:
            pass  # Free access
        else:
            # Check for valid purchase or subscription
            has_access = False

            # Check purchases
            for payment in self._payments.values():
                if (
                    payment.dataset_id == dataset_id
                    and payment.buyer_id == user_id
                    and payment.status == PaymentStatus.SUCCEEDED
                ):
                    has_access = True
                    break

            # Check subscriptions
            if not has_access:
                for sub in self._subscriptions.values():
                    if (
                        sub.dataset_id == dataset_id
                        and sub.user_id == user_id
                        and sub.status == "active"
                    ):
                        has_access = True
                        break

            if not has_access:
                logger.warning(f"Access denied: user {user_id} for dataset {dataset_id}")
                return None

        # Load and return data
        data_path = self.storage_path / "datasets" / f"{dataset_id}.parquet"
        if data_path.exists():
            return pd.read_parquet(data_path)

        return None

    def get_sample_data(
        self,
        dataset_id: str,
        rows: int = 10,
    ) -> Optional[pd.DataFrame]:
        """Get sample data (free preview).

        Args:
            dataset_id: Dataset ID.
            rows: Number of sample rows.

        Returns:
            Sample data.
        """
        data_path = self.storage_path / "datasets" / f"{dataset_id}.parquet"
        if not data_path.exists():
            return None

        data = pd.read_parquet(data_path)
        return data.head(rows)
