"""Multi-Tenant SaaS Backend for Genesis.

This module provides infrastructure for running Genesis as a multi-tenant
SaaS platform, including:

- Organization/tenant isolation
- API key management and authentication
- Usage metering and rate limiting
- Job queue and scheduling
- Billing integration hooks

Example:
    >>> from genesis.saas import TenantManager, APIKeyManager, UsageMeter
    >>>
    >>> tenant_mgr = TenantManager(database_url="postgresql://...")
    >>> tenant = tenant_mgr.create_tenant("Acme Corp", plan="pro")
    >>>
    >>> key_mgr = APIKeyManager(tenant_mgr)
    >>> api_key = key_mgr.create_key(tenant.id, scopes=["generate", "read"])
    >>>
    >>> meter = UsageMeter(tenant_mgr)
    >>> meter.record_usage(tenant.id, "rows_generated", 10000)
"""

import hashlib
import hmac
import secrets
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from genesis.core.exceptions import ConfigurationError, GenesisError
from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class Plan(Enum):
    """Available subscription plans."""

    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class PlanLimits:
    """Limits for a subscription plan."""

    rows_per_month: int
    concurrent_jobs: int
    retention_days: int
    api_keys: int
    advanced_features: bool
    support_level: str


PLAN_LIMITS: Dict[Plan, PlanLimits] = {
    Plan.FREE: PlanLimits(
        rows_per_month=10_000,
        concurrent_jobs=1,
        retention_days=7,
        api_keys=1,
        advanced_features=False,
        support_level="community",
    ),
    Plan.STARTER: PlanLimits(
        rows_per_month=1_000_000,
        concurrent_jobs=3,
        retention_days=30,
        api_keys=5,
        advanced_features=False,
        support_level="email",
    ),
    Plan.PRO: PlanLimits(
        rows_per_month=50_000_000,
        concurrent_jobs=10,
        retention_days=90,
        api_keys=20,
        advanced_features=True,
        support_level="priority",
    ),
    Plan.ENTERPRISE: PlanLimits(
        rows_per_month=-1,  # Unlimited
        concurrent_jobs=100,
        retention_days=365,
        api_keys=-1,  # Unlimited
        advanced_features=True,
        support_level="dedicated",
    ),
}


@dataclass
class Tenant:
    """Represents a tenant/organization."""

    id: str
    name: str
    plan: Plan
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    billing_email: Optional[str] = None


@dataclass
class APIKey:
    """API key for authentication."""

    key_id: str
    tenant_id: str
    key_hash: str
    name: str
    scopes: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True


@dataclass
class UsageRecord:
    """Record of resource usage."""

    tenant_id: str
    metric: str
    value: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class StorageBackend(ABC):
    """Abstract storage backend for tenant data."""

    @abstractmethod
    def save_tenant(self, tenant: Tenant) -> None:
        pass

    @abstractmethod
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        pass

    @abstractmethod
    def list_tenants(self) -> List[Tenant]:
        pass

    @abstractmethod
    def delete_tenant(self, tenant_id: str) -> None:
        pass

    @abstractmethod
    def save_api_key(self, key: APIKey) -> None:
        pass

    @abstractmethod
    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        pass

    @abstractmethod
    def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        pass

    @abstractmethod
    def list_api_keys(self, tenant_id: str) -> List[APIKey]:
        pass

    @abstractmethod
    def delete_api_key(self, key_id: str) -> None:
        pass

    @abstractmethod
    def save_usage(self, record: UsageRecord) -> None:
        pass

    @abstractmethod
    def get_usage(
        self,
        tenant_id: str,
        metric: str,
        start: datetime,
        end: datetime,
    ) -> List[UsageRecord]:
        pass


class InMemoryStorage(StorageBackend):
    """In-memory storage for development/testing."""

    def __init__(self) -> None:
        self._tenants: Dict[str, Tenant] = {}
        self._api_keys: Dict[str, APIKey] = {}
        self._usage: List[UsageRecord] = []
        self._lock = threading.Lock()

    def save_tenant(self, tenant: Tenant) -> None:
        with self._lock:
            self._tenants[tenant.id] = tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        return self._tenants.get(tenant_id)

    def list_tenants(self) -> List[Tenant]:
        return list(self._tenants.values())

    def delete_tenant(self, tenant_id: str) -> None:
        with self._lock:
            self._tenants.pop(tenant_id, None)

    def save_api_key(self, key: APIKey) -> None:
        with self._lock:
            self._api_keys[key.key_id] = key

    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        return self._api_keys.get(key_id)

    def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        for key in self._api_keys.values():
            if key.key_hash == key_hash:
                return key
        return None

    def list_api_keys(self, tenant_id: str) -> List[APIKey]:
        return [k for k in self._api_keys.values() if k.tenant_id == tenant_id]

    def delete_api_key(self, key_id: str) -> None:
        with self._lock:
            self._api_keys.pop(key_id, None)

    def save_usage(self, record: UsageRecord) -> None:
        with self._lock:
            self._usage.append(record)

    def get_usage(
        self,
        tenant_id: str,
        metric: str,
        start: datetime,
        end: datetime,
    ) -> List[UsageRecord]:
        return [
            r for r in self._usage
            if r.tenant_id == tenant_id
            and r.metric == metric
            and start <= r.timestamp <= end
        ]


class TenantManager:
    """Manages tenant/organization lifecycle.

    Example:
        >>> manager = TenantManager()
        >>> tenant = manager.create_tenant("Acme Corp", plan=Plan.PRO)
        >>> manager.update_plan(tenant.id, Plan.ENTERPRISE)
    """

    def __init__(self, storage: Optional[StorageBackend] = None) -> None:
        """Initialize tenant manager.

        Args:
            storage: Storage backend (defaults to in-memory)
        """
        self.storage = storage or InMemoryStorage()

    def create_tenant(
        self,
        name: str,
        plan: Plan = Plan.FREE,
        billing_email: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tenant:
        """Create a new tenant.

        Args:
            name: Organization name
            plan: Subscription plan
            billing_email: Email for billing
            metadata: Additional metadata

        Returns:
            Created Tenant
        """
        tenant_id = f"ten_{secrets.token_hex(12)}"

        tenant = Tenant(
            id=tenant_id,
            name=name,
            plan=plan,
            billing_email=billing_email,
            metadata=metadata or {},
        )

        self.storage.save_tenant(tenant)
        logger.info(f"Created tenant {tenant_id}: {name}")

        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID."""
        return self.storage.get_tenant(tenant_id)

    def update_tenant(
        self,
        tenant_id: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tenant]:
        """Update tenant details."""
        tenant = self.storage.get_tenant(tenant_id)
        if not tenant:
            return None

        if name:
            tenant.name = name
        if metadata:
            tenant.metadata.update(metadata)

        self.storage.save_tenant(tenant)
        return tenant

    def update_plan(self, tenant_id: str, new_plan: Plan) -> Optional[Tenant]:
        """Update tenant's subscription plan."""
        tenant = self.storage.get_tenant(tenant_id)
        if not tenant:
            return None

        old_plan = tenant.plan
        tenant.plan = new_plan
        self.storage.save_tenant(tenant)

        logger.info(
            f"Tenant {tenant_id} plan changed: {old_plan.value} -> {new_plan.value}"
        )
        return tenant

    def deactivate_tenant(self, tenant_id: str) -> bool:
        """Deactivate a tenant (soft delete)."""
        tenant = self.storage.get_tenant(tenant_id)
        if not tenant:
            return False

        tenant.is_active = False
        self.storage.save_tenant(tenant)
        logger.info(f"Deactivated tenant {tenant_id}")
        return True

    def get_limits(self, tenant_id: str) -> Optional[PlanLimits]:
        """Get plan limits for a tenant."""
        tenant = self.storage.get_tenant(tenant_id)
        if not tenant:
            return None
        return PLAN_LIMITS.get(tenant.plan)


class APIKeyManager:
    """Manages API keys for authentication.

    Example:
        >>> manager = APIKeyManager(tenant_manager)
        >>> raw_key, key_obj = manager.create_key(
        ...     tenant_id="ten_xxx",
        ...     name="Production Key",
        ...     scopes=["generate", "read"]
        ... )
        >>> print(f"Store this key: {raw_key}")
    """

    KEY_PREFIX = "gns_"

    def __init__(self, tenant_manager: TenantManager) -> None:
        self.tenant_manager = tenant_manager
        self.storage = tenant_manager.storage

    def create_key(
        self,
        tenant_id: str,
        name: str = "default",
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
    ) -> Tuple[str, APIKey]:
        """Create a new API key.

        Args:
            tenant_id: Tenant this key belongs to
            name: Human-readable name
            scopes: Allowed scopes (e.g., ['generate', 'read'])
            expires_in_days: Days until expiration (None = never)

        Returns:
            Tuple of (raw_key, APIKey object)
            Note: raw_key is only returned once, store it securely!
        """
        tenant = self.tenant_manager.get_tenant(tenant_id)
        if not tenant:
            raise GenesisError(f"Tenant {tenant_id} not found")

        # Check key limit
        limits = self.tenant_manager.get_limits(tenant_id)
        existing_keys = self.storage.list_api_keys(tenant_id)

        if limits and limits.api_keys > 0 and len(existing_keys) >= limits.api_keys:
            raise GenesisError(
                f"API key limit reached ({limits.api_keys}). "
                "Upgrade your plan for more keys."
            )

        # Generate key
        key_id = f"key_{secrets.token_hex(8)}"
        raw_key = f"{self.KEY_PREFIX}{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)

        api_key = APIKey(
            key_id=key_id,
            tenant_id=tenant_id,
            key_hash=key_hash,
            name=name,
            scopes=scopes or ["generate", "read"],
            expires_at=expires_at,
        )

        self.storage.save_api_key(api_key)
        logger.info(f"Created API key {key_id} for tenant {tenant_id}")

        return raw_key, api_key

    def validate_key(
        self,
        raw_key: str,
        required_scope: Optional[str] = None,
    ) -> Tuple[bool, Optional[APIKey], Optional[str]]:
        """Validate an API key.

        Args:
            raw_key: Raw API key to validate
            required_scope: Required scope (optional)

        Returns:
            Tuple of (is_valid, APIKey or None, error_message or None)
        """
        if not raw_key.startswith(self.KEY_PREFIX):
            return False, None, "Invalid key format"

        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        api_key = self.storage.get_api_key_by_hash(key_hash)

        if not api_key:
            return False, None, "Key not found"

        if not api_key.is_active:
            return False, None, "Key is inactive"

        if api_key.expires_at and api_key.expires_at < datetime.utcnow():
            return False, None, "Key has expired"

        # Check tenant is active
        tenant = self.tenant_manager.get_tenant(api_key.tenant_id)
        if not tenant or not tenant.is_active:
            return False, None, "Tenant is inactive"

        # Check scope
        if required_scope and required_scope not in api_key.scopes:
            return False, None, f"Key lacks required scope: {required_scope}"

        # Update last used
        api_key.last_used_at = datetime.utcnow()
        self.storage.save_api_key(api_key)

        return True, api_key, None

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        api_key = self.storage.get_api_key(key_id)
        if not api_key:
            return False

        api_key.is_active = False
        self.storage.save_api_key(api_key)
        logger.info(f"Revoked API key {key_id}")
        return True

    def list_keys(self, tenant_id: str) -> List[APIKey]:
        """List all API keys for a tenant."""
        return self.storage.list_api_keys(tenant_id)


class UsageMeter:
    """Tracks and meters resource usage.

    Example:
        >>> meter = UsageMeter(tenant_manager)
        >>> meter.record_usage(tenant_id, "rows_generated", 10000)
        >>> usage = meter.get_monthly_usage(tenant_id, "rows_generated")
        >>> print(f"Used: {usage} rows this month")
    """

    def __init__(self, tenant_manager: TenantManager) -> None:
        self.tenant_manager = tenant_manager
        self.storage = tenant_manager.storage
        self._cache: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._cache_lock = threading.Lock()

    def record_usage(
        self,
        tenant_id: str,
        metric: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record usage for a tenant.

        Args:
            tenant_id: Tenant ID
            metric: Metric name (e.g., 'rows_generated')
            value: Usage value
            metadata: Additional context
        """
        record = UsageRecord(
            tenant_id=tenant_id,
            metric=metric,
            value=value,
            metadata=metadata or {},
        )

        self.storage.save_usage(record)

        # Update cache
        with self._cache_lock:
            self._cache[tenant_id][metric] += value

    def get_usage(
        self,
        tenant_id: str,
        metric: str,
        start: datetime,
        end: datetime,
    ) -> float:
        """Get total usage for a time period."""
        records = self.storage.get_usage(tenant_id, metric, start, end)
        return sum(r.value for r in records)

    def get_monthly_usage(
        self,
        tenant_id: str,
        metric: str,
    ) -> float:
        """Get usage for current month."""
        now = datetime.utcnow()
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return self.get_usage(tenant_id, metric, start, now)

    def check_limit(
        self,
        tenant_id: str,
        metric: str,
        requested: float,
    ) -> Tuple[bool, Optional[str]]:
        """Check if usage would exceed limits.

        Args:
            tenant_id: Tenant ID
            metric: Metric to check
            requested: Requested usage

        Returns:
            Tuple of (is_allowed, error_message)
        """
        limits = self.tenant_manager.get_limits(tenant_id)
        if not limits:
            return False, "Tenant not found"

        if metric == "rows_generated":
            limit = limits.rows_per_month
            if limit < 0:  # Unlimited
                return True, None

            current = self.get_monthly_usage(tenant_id, metric)
            if current + requested > limit:
                return False, (
                    f"Monthly limit exceeded. "
                    f"Used: {int(current):,}, Requested: {int(requested):,}, "
                    f"Limit: {limit:,}"
                )

        return True, None


class RateLimiter:
    """Rate limiter for API requests.

    Implements token bucket algorithm with per-tenant limits.

    Example:
        >>> limiter = RateLimiter()
        >>> if limiter.allow(tenant_id, "api_requests"):
        ...     process_request()
        ... else:
        ...     raise RateLimitExceeded()
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _get_bucket(self, tenant_id: str, resource: str) -> Dict[str, Any]:
        """Get or create token bucket for tenant/resource."""
        key = f"{tenant_id}:{resource}"

        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = {
                    "tokens": self.burst_size,
                    "last_update": time.time(),
                }
            return self._buckets[key]

    def allow(
        self,
        tenant_id: str,
        resource: str = "default",
        cost: int = 1,
    ) -> bool:
        """Check if request is allowed under rate limit.

        Args:
            tenant_id: Tenant ID
            resource: Resource type
            cost: Request cost in tokens

        Returns:
            True if allowed, False if rate limited
        """
        bucket = self._get_bucket(tenant_id, resource)

        with self._lock:
            now = time.time()
            elapsed = now - bucket["last_update"]

            # Refill tokens
            refill = elapsed * (self.requests_per_minute / 60.0)
            bucket["tokens"] = min(self.burst_size, bucket["tokens"] + refill)
            bucket["last_update"] = now

            # Check and consume
            if bucket["tokens"] >= cost:
                bucket["tokens"] -= cost
                return True

            return False

    def get_wait_time(self, tenant_id: str, resource: str = "default") -> float:
        """Get time to wait before next request is allowed."""
        bucket = self._get_bucket(tenant_id, resource)

        with self._lock:
            if bucket["tokens"] >= 1:
                return 0.0

            tokens_needed = 1 - bucket["tokens"]
            return tokens_needed * (60.0 / self.requests_per_minute)


@dataclass
class Job:
    """Background job for generation tasks."""

    id: str
    tenant_id: str
    job_type: str
    config: Dict[str, Any]
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class JobQueue:
    """Simple in-memory job queue for background tasks.

    For production, replace with Celery, RQ, or similar.

    Example:
        >>> queue = JobQueue()
        >>> job = queue.enqueue(
        ...     tenant_id,
        ...     "generate",
        ...     {"table": "users", "rows": 10000}
        ... )
        >>> status = queue.get_status(job.id)
    """

    def __init__(self, tenant_manager: TenantManager) -> None:
        self.tenant_manager = tenant_manager
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def enqueue(
        self,
        tenant_id: str,
        job_type: str,
        config: Dict[str, Any],
    ) -> Job:
        """Enqueue a new job.

        Args:
            tenant_id: Tenant ID
            job_type: Type of job
            config: Job configuration

        Returns:
            Created Job
        """
        # Check concurrent job limit
        limits = self.tenant_manager.get_limits(tenant_id)
        if limits:
            active_jobs = [
                j for j in self._jobs.values()
                if j.tenant_id == tenant_id and j.status in ("pending", "running")
            ]
            if len(active_jobs) >= limits.concurrent_jobs:
                raise GenesisError(
                    f"Concurrent job limit reached ({limits.concurrent_jobs}). "
                    "Wait for existing jobs to complete or upgrade your plan."
                )

        job_id = f"job_{secrets.token_hex(12)}"

        job = Job(
            id=job_id,
            tenant_id=tenant_id,
            job_type=job_type,
            config=config,
        )

        with self._lock:
            self._jobs[job_id] = job

        logger.info(f"Enqueued job {job_id} for tenant {tenant_id}")
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        tenant_id: str,
        status: Optional[str] = None,
    ) -> List[Job]:
        """List jobs for a tenant."""
        jobs = [j for j in self._jobs.values() if j.tenant_id == tenant_id]
        if status:
            jobs = [j for j in jobs if j.status == status]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def update_status(
        self,
        job_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[Job]:
        """Update job status."""
        job = self._jobs.get(job_id)
        if not job:
            return None

        with self._lock:
            job.status = status

            if status == "running":
                job.started_at = datetime.utcnow()
            elif status in ("completed", "failed"):
                job.completed_at = datetime.utcnow()

            if result:
                job.result = result
            if error:
                job.error = error

        return job


class TenantContext:
    """Context manager for tenant-scoped operations.

    Example:
        >>> with TenantContext(tenant_id, key_manager, meter) as ctx:
        ...     ctx.check_rate_limit()
        ...     ctx.check_usage_limit("rows_generated", 10000)
        ...     result = generate_data()
        ...     ctx.record_usage("rows_generated", 10000)
    """

    def __init__(
        self,
        tenant_id: str,
        key_manager: APIKeyManager,
        meter: UsageMeter,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.key_manager = key_manager
        self.meter = meter
        self.rate_limiter = rate_limiter or RateLimiter()
        self._tenant: Optional[Tenant] = None

    def __enter__(self) -> "TenantContext":
        self._tenant = self.key_manager.tenant_manager.get_tenant(self.tenant_id)
        if not self._tenant:
            raise GenesisError(f"Tenant {self.tenant_id} not found")
        if not self._tenant.is_active:
            raise GenesisError(f"Tenant {self.tenant_id} is inactive")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    @property
    def tenant(self) -> Tenant:
        """Get current tenant."""
        if not self._tenant:
            raise GenesisError("Context not entered")
        return self._tenant

    @property
    def limits(self) -> PlanLimits:
        """Get plan limits."""
        limits = self.key_manager.tenant_manager.get_limits(self.tenant_id)
        if not limits:
            raise GenesisError("No limits found")
        return limits

    def check_rate_limit(self, resource: str = "api_requests") -> None:
        """Check rate limit, raise if exceeded."""
        if not self.rate_limiter.allow(self.tenant_id, resource):
            wait_time = self.rate_limiter.get_wait_time(self.tenant_id, resource)
            raise GenesisError(
                f"Rate limit exceeded. Retry after {wait_time:.1f} seconds."
            )

    def check_usage_limit(self, metric: str, requested: float) -> None:
        """Check usage limit, raise if exceeded."""
        allowed, error = self.meter.check_limit(self.tenant_id, metric, requested)
        if not allowed:
            raise GenesisError(error)

    def record_usage(
        self,
        metric: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record usage."""
        self.meter.record_usage(self.tenant_id, metric, value, metadata)


def create_saas_infrastructure(
    storage: Optional[StorageBackend] = None,
) -> Dict[str, Any]:
    """Create complete SaaS infrastructure.

    Returns:
        Dictionary with all managers and utilities
    """
    tenant_manager = TenantManager(storage)
    key_manager = APIKeyManager(tenant_manager)
    meter = UsageMeter(tenant_manager)
    rate_limiter = RateLimiter()
    job_queue = JobQueue(tenant_manager)

    return {
        "tenant_manager": tenant_manager,
        "key_manager": key_manager,
        "usage_meter": meter,
        "rate_limiter": rate_limiter,
        "job_queue": job_queue,
    }
