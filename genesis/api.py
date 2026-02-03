"""Real-Time Synthetic Data API SaaS Backend.

This module provides a hosted API service for synthetic data generation
with multi-tenant isolation, streaming generation, usage metering, and
enterprise features.

Example:
    >>> from genesis.api import SyntheticAPI, APIConfig
    >>>
    >>> # Create API server
    >>> api = SyntheticAPI(config=APIConfig(
    ...     rate_limit_per_minute=100,
    ...     max_samples_per_request=10000,
    ... ))
    >>>
    >>> # Run the server
    >>> api.run(host="0.0.0.0", port=8080)
    >>>
    >>> # Or use programmatically
    >>> from genesis.api import APIClient
    >>>
    >>> client = APIClient("https://api.genesis.example.com", api_key="your_key")
    >>> synthetic = client.generate(
    ...     generator_id="gen_123",
    ...     n_samples=1000,
    ...     streaming=True,
    ... )
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from genesis.core.base import SyntheticGenerator
from genesis.core.config import GeneratorConfig, PrivacyConfig
from genesis.utils.logging import get_logger

logger = get_logger(__name__)

# Optional imports for web framework
try:
    from fastapi import (
        BackgroundTasks,
        Depends,
        FastAPI,
        Header,
        HTTPException,
        Query,
        Request,
        Response,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class TierType(Enum):
    """API subscription tiers."""

    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class UsageMetric(Enum):
    """Types of usage metrics."""

    API_CALLS = "api_calls"
    SAMPLES_GENERATED = "samples_generated"
    STORAGE_BYTES = "storage_bytes"
    COMPUTE_SECONDS = "compute_seconds"
    BANDWIDTH_BYTES = "bandwidth_bytes"


@dataclass
class TierLimits:
    """Rate limits and quotas for a subscription tier."""

    tier: TierType
    rate_limit_per_minute: int
    rate_limit_per_hour: int
    daily_api_calls: int
    daily_samples: int
    max_samples_per_request: int
    max_concurrent_jobs: int
    storage_bytes: int
    allowed_methods: List[str]
    priority: int  # Higher = more priority
    support_level: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "rate_limit_per_hour": self.rate_limit_per_hour,
            "daily_api_calls": self.daily_api_calls,
            "daily_samples": self.daily_samples,
            "max_samples_per_request": self.max_samples_per_request,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "storage_bytes": self.storage_bytes,
            "allowed_methods": self.allowed_methods,
        }


# Default tier configurations
TIER_LIMITS: Dict[TierType, TierLimits] = {
    TierType.FREE: TierLimits(
        tier=TierType.FREE,
        rate_limit_per_minute=10,
        rate_limit_per_hour=100,
        daily_api_calls=1000,
        daily_samples=10000,
        max_samples_per_request=1000,
        max_concurrent_jobs=1,
        storage_bytes=100 * 1024 * 1024,  # 100 MB
        allowed_methods=["gaussian_copula"],
        priority=1,
        support_level="community",
    ),
    TierType.STARTER: TierLimits(
        tier=TierType.STARTER,
        rate_limit_per_minute=60,
        rate_limit_per_hour=500,
        daily_api_calls=10000,
        daily_samples=100000,
        max_samples_per_request=10000,
        max_concurrent_jobs=3,
        storage_bytes=1 * 1024 * 1024 * 1024,  # 1 GB
        allowed_methods=["gaussian_copula", "ctgan"],
        priority=2,
        support_level="email",
    ),
    TierType.PROFESSIONAL: TierLimits(
        tier=TierType.PROFESSIONAL,
        rate_limit_per_minute=300,
        rate_limit_per_hour=3000,
        daily_api_calls=100000,
        daily_samples=1000000,
        max_samples_per_request=100000,
        max_concurrent_jobs=10,
        storage_bytes=10 * 1024 * 1024 * 1024,  # 10 GB
        allowed_methods=["gaussian_copula", "ctgan", "tvae", "auto"],
        priority=3,
        support_level="priority",
    ),
    TierType.ENTERPRISE: TierLimits(
        tier=TierType.ENTERPRISE,
        rate_limit_per_minute=1000,
        rate_limit_per_hour=10000,
        daily_api_calls=-1,  # Unlimited
        daily_samples=-1,
        max_samples_per_request=1000000,
        max_concurrent_jobs=50,
        storage_bytes=100 * 1024 * 1024 * 1024,  # 100 GB
        allowed_methods=["*"],  # All methods
        priority=4,
        support_level="dedicated",
    ),
}


@dataclass
class APIKey:
    """API key for authentication."""

    key_id: str
    key_hash: str  # Hashed version of the key
    tenant_id: str
    name: str
    tier: TierType
    created_at: str
    expires_at: Optional[str] = None
    last_used_at: Optional[str] = None
    is_active: bool = True
    allowed_ips: List[str] = field(default_factory=list)
    scopes: List[str] = field(default_factory=lambda: ["read", "write"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key_id": self.key_id,
            "tenant_id": self.tenant_id,
            "name": self.name,
            "tier": self.tier.value,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "is_active": self.is_active,
            "scopes": self.scopes,
        }


@dataclass
class Tenant:
    """A tenant (organization) in the multi-tenant system."""

    tenant_id: str
    name: str
    tier: TierType
    created_at: str
    contact_email: str
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name,
            "tier": self.tier.value,
            "created_at": self.created_at,
            "is_active": self.is_active,
        }


@dataclass
class GeneratorRecord:
    """A registered generator in the API."""

    generator_id: str
    tenant_id: str
    name: str
    description: str
    method: str
    columns: List[Dict[str, Any]]
    created_at: str
    updated_at: str
    is_fitted: bool = False
    n_training_rows: int = 0
    storage_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generator_id": self.generator_id,
            "name": self.name,
            "description": self.description,
            "method": self.method,
            "columns": self.columns,
            "is_fitted": self.is_fitted,
            "n_training_rows": self.n_training_rows,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class GenerationJob:
    """A generation job in the queue."""

    job_id: str
    tenant_id: str
    generator_id: str
    n_samples: int
    status: str  # 'queued', 'running', 'completed', 'failed'
    priority: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    result_path: Optional[str] = None
    progress: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "generator_id": self.generator_id,
            "n_samples": self.n_samples,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class UsageRecord:
    """Usage record for billing and metering."""

    record_id: str
    tenant_id: str
    metric: UsageMetric
    value: float
    timestamp: str
    job_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIConfig:
    """Configuration for the API server."""

    # Rate limiting
    global_rate_limit: int = 10000
    enable_rate_limiting: bool = True

    # Security
    require_api_key: bool = True
    api_key_header: str = "X-API-Key"
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])

    # Generation
    default_batch_size: int = 1000
    max_streaming_chunk_size: int = 100
    job_timeout_seconds: int = 3600

    # Storage
    storage_path: str = "./api_storage"
    max_upload_size_bytes: int = 100 * 1024 * 1024  # 100 MB

    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_rate_limit": self.global_rate_limit,
            "enable_rate_limiting": self.enable_rate_limiting,
            "require_api_key": self.require_api_key,
            "enable_cors": self.enable_cors,
            "default_batch_size": self.default_batch_size,
            "max_upload_size_bytes": self.max_upload_size_bytes,
        }


class RateLimiter:
    """Token bucket rate limiter."""

    def __init__(self):
        self._buckets: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"tokens": 0, "last_update": time.time()}
        )
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def acquire(
        self,
        key: str,
        rate_limit: int,
        window_seconds: int = 60,
    ) -> bool:
        """Try to acquire a token from the bucket.

        Args:
            key: Rate limit key (e.g., tenant_id)
            rate_limit: Maximum requests per window
            window_seconds: Time window in seconds

        Returns:
            True if token acquired, False if rate limited
        """
        async with self._locks[key]:
            bucket = self._buckets[key]
            now = time.time()

            # Refill tokens based on elapsed time
            elapsed = now - bucket["last_update"]
            refill_rate = rate_limit / window_seconds
            bucket["tokens"] = min(rate_limit, bucket["tokens"] + elapsed * refill_rate)
            bucket["last_update"] = now

            if bucket["tokens"] >= 1:
                bucket["tokens"] -= 1
                return True

            return False

    def get_remaining(self, key: str, rate_limit: int) -> int:
        """Get remaining tokens for a key."""
        bucket = self._buckets.get(key)
        if not bucket:
            return rate_limit
        return max(0, int(bucket["tokens"]))


class UsageMeter:
    """Tracks and meters API usage."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._usage: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._daily_usage: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._records: List[UsageRecord] = []

    def record(
        self,
        tenant_id: str,
        metric: UsageMetric,
        value: float,
        job_id: Optional[str] = None,
    ) -> None:
        """Record a usage metric.

        Args:
            tenant_id: Tenant ID
            metric: Type of metric
            value: Metric value
            job_id: Optional associated job ID
        """
        record = UsageRecord(
            record_id=str(uuid.uuid4())[:8],
            tenant_id=tenant_id,
            metric=metric,
            value=value,
            timestamp=datetime.utcnow().isoformat(),
            job_id=job_id,
        )
        self._records.append(record)

        # Update aggregates
        self._usage[tenant_id][metric.value] += value

        # Daily tracking
        today = datetime.utcnow().strftime("%Y-%m-%d")
        self._daily_usage[f"{tenant_id}:{today}"][metric.value] += value

    def get_usage(self, tenant_id: str) -> Dict[str, float]:
        """Get total usage for a tenant."""
        return dict(self._usage[tenant_id])

    def get_daily_usage(self, tenant_id: str, date: Optional[str] = None) -> Dict[str, float]:
        """Get daily usage for a tenant."""
        date = date or datetime.utcnow().strftime("%Y-%m-%d")
        return dict(self._daily_usage[f"{tenant_id}:{date}"])

    def check_quota(
        self,
        tenant_id: str,
        metric: UsageMetric,
        limits: TierLimits,
        requested: float = 1,
    ) -> Tuple[bool, str]:
        """Check if a request would exceed quotas.

        Returns:
            (allowed, reason)
        """
        daily = self.get_daily_usage(tenant_id)

        if metric == UsageMetric.API_CALLS:
            limit = limits.daily_api_calls
            current = daily.get(metric.value, 0)
        elif metric == UsageMetric.SAMPLES_GENERATED:
            limit = limits.daily_samples
            current = daily.get(metric.value, 0)
        else:
            return True, ""

        if limit == -1:  # Unlimited
            return True, ""

        if current + requested > limit:
            return False, f"Daily {metric.value} quota exceeded ({current}/{limit})"

        return True, ""


class TenantManager:
    """Manages tenants and API keys."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._tenants: Dict[str, Tenant] = {}
        self._api_keys: Dict[str, APIKey] = {}
        self._key_to_tenant: Dict[str, str] = {}

    def create_tenant(
        self,
        name: str,
        tier: TierType,
        contact_email: str,
    ) -> Tenant:
        """Create a new tenant."""
        tenant_id = f"ten_{uuid.uuid4().hex[:12]}"

        tenant = Tenant(
            tenant_id=tenant_id,
            name=name,
            tier=tier,
            contact_email=contact_email,
            created_at=datetime.utcnow().isoformat(),
        )

        self._tenants[tenant_id] = tenant
        logger.info(f"Created tenant: {name} ({tenant_id})")

        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID."""
        return self._tenants.get(tenant_id)

    def create_api_key(
        self,
        tenant_id: str,
        name: str,
        scopes: Optional[List[str]] = None,
    ) -> Tuple[str, APIKey]:
        """Create a new API key for a tenant.

        Returns:
            (raw_key, APIKey) - raw_key is only returned once
        """
        tenant = self._tenants.get(tenant_id)
        if not tenant:
            raise ValueError(f"Tenant not found: {tenant_id}")

        # Generate secure key
        raw_key = f"sk_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        key_id = f"key_{uuid.uuid4().hex[:12]}"

        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            tenant_id=tenant_id,
            name=name,
            tier=tenant.tier,
            created_at=datetime.utcnow().isoformat(),
            scopes=scopes or ["read", "write"],
        )

        self._api_keys[key_id] = api_key
        self._key_to_tenant[key_hash] = tenant_id

        logger.info(f"Created API key for tenant {tenant_id}: {key_id}")

        return raw_key, api_key

    def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """Validate an API key and return the associated APIKey object."""
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        for api_key in self._api_keys.values():
            if api_key.key_hash == key_hash:
                if not api_key.is_active:
                    return None
                if api_key.expires_at and api_key.expires_at < datetime.utcnow().isoformat():
                    return None

                # Update last used
                api_key.last_used_at = datetime.utcnow().isoformat()

                return api_key

        return None

    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        api_key = self._api_keys.get(key_id)
        if api_key:
            api_key.is_active = False
            return True
        return False

    def get_tenant_limits(self, tenant_id: str) -> Optional[TierLimits]:
        """Get rate limits for a tenant."""
        tenant = self._tenants.get(tenant_id)
        if tenant:
            return TIER_LIMITS.get(tenant.tier)
        return None


class GeneratorRegistry:
    """Registry for managing generators."""

    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self._generators: Dict[str, GeneratorRecord] = {}
        self._fitted_generators: Dict[str, SyntheticGenerator] = {}

    def register(
        self,
        tenant_id: str,
        name: str,
        description: str,
        method: str,
        columns: List[Dict[str, Any]],
    ) -> GeneratorRecord:
        """Register a new generator."""
        generator_id = f"gen_{uuid.uuid4().hex[:12]}"

        record = GeneratorRecord(
            generator_id=generator_id,
            tenant_id=tenant_id,
            name=name,
            description=description,
            method=method,
            columns=columns,
            created_at=datetime.utcnow().isoformat(),
            updated_at=datetime.utcnow().isoformat(),
        )

        self._generators[generator_id] = record
        logger.info(f"Registered generator: {name} ({generator_id})")

        return record

    def get(self, generator_id: str) -> Optional[GeneratorRecord]:
        """Get a generator record."""
        return self._generators.get(generator_id)

    def get_for_tenant(self, tenant_id: str) -> List[GeneratorRecord]:
        """Get all generators for a tenant."""
        return [g for g in self._generators.values() if g.tenant_id == tenant_id]

    def fit(
        self,
        generator_id: str,
        data: pd.DataFrame,
    ) -> bool:
        """Fit a generator on data."""
        record = self._generators.get(generator_id)
        if not record:
            return False

        # Create and fit generator
        generator = SyntheticGenerator(method=record.method)
        generator.fit(data)

        # Store fitted generator
        self._fitted_generators[generator_id] = generator

        # Update record
        record.is_fitted = True
        record.n_training_rows = len(data)
        record.updated_at = datetime.utcnow().isoformat()

        logger.info(f"Fitted generator {generator_id} on {len(data)} rows")

        return True

    def generate(
        self,
        generator_id: str,
        n_samples: int,
    ) -> Optional[pd.DataFrame]:
        """Generate synthetic data."""
        generator = self._fitted_generators.get(generator_id)
        if not generator:
            return None

        return generator.generate(n_samples=n_samples)

    async def generate_streaming(
        self,
        generator_id: str,
        n_samples: int,
        chunk_size: int = 100,
    ) -> AsyncIterator[pd.DataFrame]:
        """Generate synthetic data in streaming chunks."""
        generator = self._fitted_generators.get(generator_id)
        if not generator:
            return

        remaining = n_samples
        while remaining > 0:
            batch_size = min(chunk_size, remaining)
            chunk = generator.generate(n_samples=batch_size)
            yield chunk
            remaining -= batch_size
            await asyncio.sleep(0)  # Yield control


class JobQueue:
    """Asynchronous job queue for generation tasks."""

    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self._jobs: Dict[str, GenerationJob] = {}
        self._queue: asyncio.PriorityQueue = None
        self._workers: List[asyncio.Task] = []
        self._running = False

    async def start(self) -> None:
        """Start the job queue workers."""
        self._queue = asyncio.PriorityQueue()
        self._running = True

        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self._workers.append(worker)

        logger.info(f"Started job queue with {self.max_workers} workers")

    async def stop(self) -> None:
        """Stop the job queue."""
        self._running = False

        for worker in self._workers:
            worker.cancel()

        self._workers = []

    async def submit(
        self,
        tenant_id: str,
        generator_id: str,
        n_samples: int,
        priority: int = 1,
    ) -> GenerationJob:
        """Submit a generation job."""
        job_id = f"job_{uuid.uuid4().hex[:12]}"

        job = GenerationJob(
            job_id=job_id,
            tenant_id=tenant_id,
            generator_id=generator_id,
            n_samples=n_samples,
            status="queued",
            priority=priority,
            created_at=datetime.utcnow().isoformat(),
        )

        self._jobs[job_id] = job

        # Add to priority queue (negative priority for max-heap behavior)
        await self._queue.put((-priority, time.time(), job_id))

        logger.info(f"Submitted job {job_id} for generator {generator_id}")

        return job

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def get_tenant_jobs(self, tenant_id: str) -> List[GenerationJob]:
        """Get all jobs for a tenant."""
        return [j for j in self._jobs.values() if j.tenant_id == tenant_id]

    async def _worker(self, worker_id: int) -> None:
        """Worker coroutine that processes jobs."""
        while self._running:
            try:
                _, _, job_id = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0,
                )

                job = self._jobs.get(job_id)
                if not job:
                    continue

                await self._process_job(job)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    async def _process_job(self, job: GenerationJob) -> None:
        """Process a single job."""
        job.status = "running"
        job.started_at = datetime.utcnow().isoformat()

        try:
            # Get generator and generate
            # Note: In production, would use the GeneratorRegistry
            job.progress = 0.5
            await asyncio.sleep(0.1)  # Simulate work

            job.progress = 1.0
            job.status = "completed"
            job.completed_at = datetime.utcnow().isoformat()

            logger.info(f"Completed job {job.job_id}")

        except Exception as e:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.utcnow().isoformat()
            logger.error(f"Job {job.job_id} failed: {e}")


class SyntheticAPI:
    """Real-time Synthetic Data API Server."""

    def __init__(self, config: Optional[APIConfig] = None):
        """Initialize the API server.

        Args:
            config: API configuration
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI required: pip install fastapi uvicorn")

        self.config = config or APIConfig()
        self.storage_path = Path(self.config.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.tenant_manager = TenantManager(self.storage_path)
        self.generator_registry = GeneratorRegistry(self.storage_path)
        self.rate_limiter = RateLimiter()
        self.usage_meter = UsageMeter(self.storage_path)
        self.job_queue = JobQueue()

        # Create FastAPI app
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create the FastAPI application."""

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.job_queue.start()
            yield
            # Shutdown
            await self.job_queue.stop()

        app = FastAPI(
            title="Genesis Synthetic Data API",
            description="Real-time synthetic data generation API",
            version="1.0.0",
            lifespan=lifespan,
        )

        # CORS middleware
        if self.config.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.allowed_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Request models
        class RegisterGeneratorRequest(BaseModel):
            name: str
            description: str = ""
            method: str = "auto"
            columns: List[Dict[str, Any]] = []

        class FitGeneratorRequest(BaseModel):
            data: List[Dict[str, Any]]

        class GenerateRequest(BaseModel):
            n_samples: int = 1000
            streaming: bool = False

        class CreateTenantRequest(BaseModel):
            name: str
            tier: str = "free"
            contact_email: str

        class CreateAPIKeyRequest(BaseModel):
            name: str
            scopes: List[str] = ["read", "write"]

        # Dependency to get API key
        async def get_api_key(
            x_api_key: str = Header(None, alias=self.config.api_key_header),
        ) -> APIKey:
            if not self.config.require_api_key:
                # Return a default key for testing
                return APIKey(
                    key_id="test",
                    key_hash="",
                    tenant_id="test",
                    name="Test Key",
                    tier=TierType.FREE,
                    created_at=datetime.utcnow().isoformat(),
                )

            if not x_api_key:
                raise HTTPException(status_code=401, detail="API key required")

            api_key = self.tenant_manager.validate_api_key(x_api_key)
            if not api_key:
                raise HTTPException(status_code=401, detail="Invalid API key")

            return api_key

        # Dependency for rate limiting
        async def check_rate_limit(api_key: APIKey = Depends(get_api_key)):
            limits = TIER_LIMITS.get(api_key.tier)
            if not limits:
                return api_key

            allowed = await self.rate_limiter.acquire(
                api_key.tenant_id,
                limits.rate_limit_per_minute,
            )

            if not allowed:
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": "60"},
                )

            return api_key

        # ==================== Health ====================

        @app.get("/health")
        async def health():
            return {"status": "healthy", "version": "1.0.0"}

        # ==================== Tenant Management ====================

        @app.post("/admin/tenants")
        async def create_tenant(request: CreateTenantRequest):
            tenant = self.tenant_manager.create_tenant(
                name=request.name,
                tier=TierType(request.tier),
                contact_email=request.contact_email,
            )
            return tenant.to_dict()

        @app.get("/admin/tenants/{tenant_id}")
        async def get_tenant(tenant_id: str):
            tenant = self.tenant_manager.get_tenant(tenant_id)
            if not tenant:
                raise HTTPException(status_code=404, detail="Tenant not found")
            return tenant.to_dict()

        @app.post("/admin/tenants/{tenant_id}/api-keys")
        async def create_api_key_endpoint(tenant_id: str, request: CreateAPIKeyRequest):
            try:
                raw_key, api_key = self.tenant_manager.create_api_key(
                    tenant_id=tenant_id,
                    name=request.name,
                    scopes=request.scopes,
                )
                return {
                    "api_key": raw_key,  # Only returned once!
                    "key_info": api_key.to_dict(),
                }
            except ValueError as e:
                raise HTTPException(status_code=404, detail=str(e))

        # ==================== Generator Management ====================

        @app.get("/generators")
        async def list_generators(api_key: APIKey = Depends(check_rate_limit)):
            generators = self.generator_registry.get_for_tenant(api_key.tenant_id)
            return {"generators": [g.to_dict() for g in generators]}

        @app.post("/generators")
        async def register_generator(
            request: RegisterGeneratorRequest,
            api_key: APIKey = Depends(check_rate_limit),
        ):
            # Check quota
            allowed, reason = self.usage_meter.check_quota(
                api_key.tenant_id,
                UsageMetric.API_CALLS,
                TIER_LIMITS[api_key.tier],
            )
            if not allowed:
                raise HTTPException(status_code=429, detail=reason)

            record = self.generator_registry.register(
                tenant_id=api_key.tenant_id,
                name=request.name,
                description=request.description,
                method=request.method,
                columns=request.columns,
            )

            self.usage_meter.record(
                api_key.tenant_id,
                UsageMetric.API_CALLS,
                1,
            )

            return record.to_dict()

        @app.get("/generators/{generator_id}")
        async def get_generator(
            generator_id: str,
            api_key: APIKey = Depends(check_rate_limit),
        ):
            record = self.generator_registry.get(generator_id)
            if not record or record.tenant_id != api_key.tenant_id:
                raise HTTPException(status_code=404, detail="Generator not found")
            return record.to_dict()

        @app.post("/generators/{generator_id}/fit")
        async def fit_generator(
            generator_id: str,
            request: FitGeneratorRequest,
            api_key: APIKey = Depends(check_rate_limit),
        ):
            record = self.generator_registry.get(generator_id)
            if not record or record.tenant_id != api_key.tenant_id:
                raise HTTPException(status_code=404, detail="Generator not found")

            # Convert to DataFrame
            data = pd.DataFrame(request.data)

            # Fit generator
            success = self.generator_registry.fit(generator_id, data)
            if not success:
                raise HTTPException(status_code=500, detail="Failed to fit generator")

            self.usage_meter.record(
                api_key.tenant_id,
                UsageMetric.API_CALLS,
                1,
            )

            return {"status": "fitted", "n_rows": len(data)}

        # ==================== Generation ====================

        @app.post("/generators/{generator_id}/generate")
        async def generate(
            generator_id: str,
            request: GenerateRequest,
            api_key: APIKey = Depends(check_rate_limit),
        ):
            record = self.generator_registry.get(generator_id)
            if not record or record.tenant_id != api_key.tenant_id:
                raise HTTPException(status_code=404, detail="Generator not found")

            if not record.is_fitted:
                raise HTTPException(status_code=400, detail="Generator not fitted")

            limits = TIER_LIMITS[api_key.tier]

            # Check limits
            if request.n_samples > limits.max_samples_per_request:
                raise HTTPException(
                    status_code=400,
                    detail=f"Exceeds max samples per request ({limits.max_samples_per_request})",
                )

            # Check quota
            allowed, reason = self.usage_meter.check_quota(
                api_key.tenant_id,
                UsageMetric.SAMPLES_GENERATED,
                limits,
                request.n_samples,
            )
            if not allowed:
                raise HTTPException(status_code=429, detail=reason)

            if request.streaming:
                # Streaming response
                async def generate_stream():
                    async for chunk in self.generator_registry.generate_streaming(
                        generator_id,
                        request.n_samples,
                        self.config.max_streaming_chunk_size,
                    ):
                        yield chunk.to_json(orient="records") + "\n"

                return StreamingResponse(
                    generate_stream(),
                    media_type="application/x-ndjson",
                )
            else:
                # Regular response
                data = self.generator_registry.generate(generator_id, request.n_samples)
                if data is None:
                    raise HTTPException(status_code=500, detail="Generation failed")

                self.usage_meter.record(
                    api_key.tenant_id,
                    UsageMetric.SAMPLES_GENERATED,
                    len(data),
                )

                return {
                    "data": data.to_dict(orient="records"),
                    "n_samples": len(data),
                }

        # ==================== Job Queue ====================

        @app.post("/generators/{generator_id}/jobs")
        async def submit_job(
            generator_id: str,
            request: GenerateRequest,
            api_key: APIKey = Depends(check_rate_limit),
        ):
            record = self.generator_registry.get(generator_id)
            if not record or record.tenant_id != api_key.tenant_id:
                raise HTTPException(status_code=404, detail="Generator not found")

            limits = TIER_LIMITS[api_key.tier]

            # Check concurrent jobs
            tenant_jobs = self.job_queue.get_tenant_jobs(api_key.tenant_id)
            active_jobs = [j for j in tenant_jobs if j.status in ["queued", "running"]]

            if len(active_jobs) >= limits.max_concurrent_jobs:
                raise HTTPException(
                    status_code=429,
                    detail=f"Max concurrent jobs reached ({limits.max_concurrent_jobs})",
                )

            job = await self.job_queue.submit(
                tenant_id=api_key.tenant_id,
                generator_id=generator_id,
                n_samples=request.n_samples,
                priority=limits.priority,
            )

            return job.to_dict()

        @app.get("/jobs/{job_id}")
        async def get_job(
            job_id: str,
            api_key: APIKey = Depends(check_rate_limit),
        ):
            job = self.job_queue.get_job(job_id)
            if not job or job.tenant_id != api_key.tenant_id:
                raise HTTPException(status_code=404, detail="Job not found")
            return job.to_dict()

        @app.get("/jobs")
        async def list_jobs(api_key: APIKey = Depends(check_rate_limit)):
            jobs = self.job_queue.get_tenant_jobs(api_key.tenant_id)
            return {"jobs": [j.to_dict() for j in jobs]}

        # ==================== Usage & Billing ====================

        @app.get("/usage")
        async def get_usage(api_key: APIKey = Depends(check_rate_limit)):
            usage = self.usage_meter.get_usage(api_key.tenant_id)
            daily = self.usage_meter.get_daily_usage(api_key.tenant_id)
            limits = TIER_LIMITS[api_key.tier]

            return {
                "total_usage": usage,
                "daily_usage": daily,
                "limits": limits.to_dict(),
                "remaining": {
                    "daily_api_calls": max(0, limits.daily_api_calls - daily.get("api_calls", 0))
                    if limits.daily_api_calls != -1
                    else -1,
                    "daily_samples": max(0, limits.daily_samples - daily.get("samples_generated", 0))
                    if limits.daily_samples != -1
                    else -1,
                },
            }

        return app

    def run(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Run the API server.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        import uvicorn

        uvicorn.run(self.app, host=host, port=port)


class APIClient:
    """Client for the Synthetic Data API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: int = 30,
    ):
        """Initialize the API client.

        Args:
            base_url: API base URL
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"X-API-Key": api_key},
            timeout=timeout,
        )

    def list_generators(self) -> List[Dict[str, Any]]:
        """List all generators."""
        response = self._client.get("/generators")
        response.raise_for_status()
        return response.json()["generators"]

    def register_generator(
        self,
        name: str,
        method: str = "auto",
        description: str = "",
        columns: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Register a new generator."""
        response = self._client.post(
            "/generators",
            json={
                "name": name,
                "method": method,
                "description": description,
                "columns": columns or [],
            },
        )
        response.raise_for_status()
        return response.json()

    def fit_generator(
        self,
        generator_id: str,
        data: Union[pd.DataFrame, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Fit a generator on training data."""
        if isinstance(data, pd.DataFrame):
            data = data.to_dict(orient="records")

        response = self._client.post(
            f"/generators/{generator_id}/fit",
            json={"data": data},
        )
        response.raise_for_status()
        return response.json()

    def generate(
        self,
        generator_id: str,
        n_samples: int = 1000,
        streaming: bool = False,
    ) -> Union[pd.DataFrame, AsyncIterator[pd.DataFrame]]:
        """Generate synthetic data."""
        if streaming:
            return self._generate_streaming(generator_id, n_samples)

        response = self._client.post(
            f"/generators/{generator_id}/generate",
            json={"n_samples": n_samples, "streaming": False},
        )
        response.raise_for_status()
        data = response.json()["data"]
        return pd.DataFrame(data)

    def _generate_streaming(
        self,
        generator_id: str,
        n_samples: int,
    ) -> AsyncIterator[pd.DataFrame]:
        """Generate synthetic data with streaming."""
        with self._client.stream(
            "POST",
            f"/generators/{generator_id}/generate",
            json={"n_samples": n_samples, "streaming": True},
        ) as response:
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    yield pd.DataFrame(data)

    def submit_job(
        self,
        generator_id: str,
        n_samples: int,
    ) -> Dict[str, Any]:
        """Submit an async generation job."""
        response = self._client.post(
            f"/generators/{generator_id}/jobs",
            json={"n_samples": n_samples},
        )
        response.raise_for_status()
        return response.json()

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Get job status."""
        response = self._client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        return response.json()

    def get_usage(self) -> Dict[str, Any]:
        """Get usage statistics."""
        response = self._client.get("/usage")
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """Close the client."""
        self._client.close()


__all__ = [
    # Main classes
    "SyntheticAPI",
    "APIClient",
    "APIConfig",
    # Tenant management
    "TenantManager",
    "Tenant",
    "APIKey",
    # Generation
    "GeneratorRegistry",
    "GeneratorRecord",
    "JobQueue",
    "GenerationJob",
    # Rate limiting & usage
    "RateLimiter",
    "UsageMeter",
    "UsageRecord",
    # Types
    "TierType",
    "TierLimits",
    "UsageMetric",
    "TIER_LIMITS",
]
