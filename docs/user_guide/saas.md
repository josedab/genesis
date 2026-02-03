# Multi-Tenant SaaS Backend

Genesis provides enterprise-grade multi-tenant infrastructure for deploying synthetic data generation as a service.

## Overview

| Component | Purpose |
|-----------|---------|
| **TenantManager** | Tenant lifecycle and configuration |
| **APIKeyManager** | API key generation and validation |
| **UsageMeter** | Usage tracking and limits |
| **RateLimiter** | Request rate limiting |
| **JobQueue** | Async job processing |

## Tenant Management

Create and manage tenants with isolated configuration:

```python
from genesis.saas import TenantManager, TenantTier

manager = TenantManager()

# Create a tenant
tenant = manager.create_tenant(
    name="Acme Corp",
    email="admin@acme.com",
    tier=TenantTier.PROFESSIONAL,
)

print(f"Tenant ID: {tenant.id}")
print(f"Created: {tenant.created_at}")

# Update tenant tier
manager.update_tenant(tenant.id, tier=TenantTier.ENTERPRISE)

# Get tenant info
tenant = manager.get_tenant(tenant.id)
print(f"Tier: {tenant.tier}")
print(f"Config: {tenant.config}")
```

### Tenant Tiers

| Tier | Records/Month | API Rate | Concurrent Jobs | Features |
|------|--------------|----------|-----------------|----------|
| **Free** | 10,000 | 10/min | 1 | Basic generation |
| **Professional** | 1,000,000 | 100/min | 5 | + Streaming, Connectors |
| **Enterprise** | Unlimited | 1000/min | 50 | + SLA, Priority Support |

### Custom Configuration

```python
tenant = manager.create_tenant(
    name="Custom Corp",
    email="admin@custom.com",
    tier=TenantTier.ENTERPRISE,
    config={
        "allowed_methods": ["gaussian_copula", "ctgan", "tvae"],
        "max_columns": 500,
        "custom_models_enabled": True,
        "data_retention_days": 90,
    }
)
```

## API Key Management

Generate and manage API keys for authentication:

```python
from genesis.saas import APIKeyManager

api_keys = APIKeyManager()

# Create API key
key = api_keys.create_key(
    tenant_id=tenant.id,
    name="Production API Key",
    scopes=["generate", "read", "write"],
    expires_in_days=365,
)

print(f"API Key: {key.key}")  # Only shown once
print(f"Key ID: {key.id}")
print(f"Expires: {key.expires_at}")

# Validate key
result = api_keys.validate_key(key.key)
if result.valid:
    print(f"Tenant: {result.tenant_id}")
    print(f"Scopes: {result.scopes}")
else:
    print(f"Invalid: {result.error}")

# Revoke key
api_keys.revoke_key(key.id)
```

### Key Scopes

| Scope | Permissions |
|-------|-------------|
| `generate` | Create synthetic data |
| `read` | Read datasets, jobs |
| `write` | Create/update datasets |
| `admin` | Tenant administration |
| `billing` | View/manage billing |

### Key Rotation

```python
# Rotate key (creates new key, schedules old for deletion)
new_key = api_keys.rotate_key(
    key_id=key.id,
    grace_period_hours=24,  # Old key works for 24h
)
```

## Usage Metering

Track and limit usage across tenants:

```python
from genesis.saas import UsageMeter

meter = UsageMeter()

# Record usage
meter.record_usage(
    tenant_id=tenant.id,
    metric="records_generated",
    value=50000,
)

# Check current usage
usage = meter.get_usage(
    tenant_id=tenant.id,
    metric="records_generated",
    period="month",
)

print(f"Used: {usage.current}")
print(f"Limit: {usage.limit}")
print(f"Remaining: {usage.remaining}")
print(f"Reset: {usage.reset_at}")

# Check if limit reached
if meter.is_limit_reached(tenant.id, "records_generated"):
    raise QuotaExceededError("Monthly record limit reached")
```

### Usage Metrics

| Metric | Description | Reset |
|--------|-------------|-------|
| `records_generated` | Total records generated | Monthly |
| `api_calls` | API requests made | Daily |
| `storage_bytes` | Storage used | Never |
| `compute_minutes` | Processing time | Monthly |

### Usage Reports

```python
# Get detailed report
report = meter.get_report(
    tenant_id=tenant.id,
    start_date=datetime(2026, 1, 1),
    end_date=datetime(2026, 1, 31),
)

for metric in report.metrics:
    print(f"{metric.name}: {metric.total}")
    for day in metric.daily_breakdown:
        print(f"  {day.date}: {day.value}")
```

## Rate Limiting

Protect services with configurable rate limits:

```python
from genesis.saas import RateLimiter, RateLimitConfig

limiter = RateLimiter()

# Configure rate limits by tier
limiter.configure(
    TenantTier.FREE,
    RateLimitConfig(requests_per_minute=10, burst=20)
)
limiter.configure(
    TenantTier.PROFESSIONAL,
    RateLimitConfig(requests_per_minute=100, burst=200)
)
limiter.configure(
    TenantTier.ENTERPRISE,
    RateLimitConfig(requests_per_minute=1000, burst=2000)
)

# Check rate limit
result = limiter.check(tenant.id, tenant.tier)

if result.allowed:
    # Process request
    process_request()
else:
    print(f"Rate limited. Retry after: {result.retry_after}s")
    raise RateLimitError(retry_after=result.retry_after)
```

### Token Bucket Algorithm

Rate limiting uses a token bucket algorithm:

```python
config = RateLimitConfig(
    requests_per_minute=100,  # Sustained rate
    burst=200,                # Max burst capacity
    refill_rate=1.67,         # Tokens per second (100/60)
)
```

### Per-Endpoint Limits

```python
# Different limits for different endpoints
limiter.configure_endpoint(
    endpoint="/api/generate",
    tier=TenantTier.PROFESSIONAL,
    config=RateLimitConfig(requests_per_minute=50, burst=100)
)

result = limiter.check(
    tenant_id=tenant.id,
    tier=tenant.tier,
    endpoint="/api/generate"
)
```

## Job Queue

Process generation jobs asynchronously:

```python
from genesis.saas import JobQueue, Job, JobStatus

queue = JobQueue(redis_url="redis://localhost:6379")

# Submit job
job = queue.submit(
    tenant_id=tenant.id,
    job_type="generate",
    params={
        "method": "gaussian_copula",
        "num_rows": 1000000,
        "schema": {...},
    },
    priority=5,  # Higher = more urgent
)

print(f"Job ID: {job.id}")
print(f"Status: {job.status}")

# Check job status
job = queue.get_job(job.id)
print(f"Status: {job.status}")
print(f"Progress: {job.progress}%")

# Wait for completion
job = queue.wait_for_completion(job.id, timeout=300)

if job.status == JobStatus.COMPLETED:
    print(f"Result: {job.result}")
elif job.status == JobStatus.FAILED:
    print(f"Error: {job.error}")
```

### Job Statuses

| Status | Description |
|--------|-------------|
| `PENDING` | Waiting in queue |
| `RUNNING` | Currently processing |
| `COMPLETED` | Successfully finished |
| `FAILED` | Error occurred |
| `CANCELLED` | User cancelled |
| `TIMEOUT` | Exceeded time limit |

### Job Priorities

```python
# Enterprise gets priority
queue.submit(
    tenant_id=enterprise_tenant.id,
    job_type="generate",
    params={...},
    priority=10,  # Processed first
)

queue.submit(
    tenant_id=free_tenant.id,
    job_type="generate",
    params={...},
    priority=1,   # Processed after higher priority
)
```

### Cancel Jobs

```python
# Cancel a pending/running job
queue.cancel(job.id)

# Cancel all jobs for a tenant
queue.cancel_all(tenant_id=tenant.id)
```

## Complete Example

```python
from genesis.saas import (
    TenantManager, APIKeyManager, UsageMeter,
    RateLimiter, JobQueue, TenantTier
)

# Initialize components
tenants = TenantManager()
api_keys = APIKeyManager()
meter = UsageMeter()
limiter = RateLimiter()
queue = JobQueue(redis_url="redis://localhost:6379")

# Create tenant
tenant = tenants.create_tenant(
    name="Acme Corp",
    email="admin@acme.com",
    tier=TenantTier.PROFESSIONAL,
)

# Create API key
key = api_keys.create_key(
    tenant_id=tenant.id,
    name="Production Key",
    scopes=["generate", "read"],
)

# API request handler
def handle_generate_request(api_key: str, params: dict):
    # Validate API key
    auth = api_keys.validate_key(api_key)
    if not auth.valid:
        raise AuthenticationError(auth.error)
    
    tenant = tenants.get_tenant(auth.tenant_id)
    
    # Check rate limit
    rate_check = limiter.check(tenant.id, tenant.tier)
    if not rate_check.allowed:
        raise RateLimitError(retry_after=rate_check.retry_after)
    
    # Check usage quota
    if meter.is_limit_reached(tenant.id, "records_generated"):
        raise QuotaExceededError("Monthly limit reached")
    
    # Submit job
    job = queue.submit(
        tenant_id=tenant.id,
        job_type="generate",
        params=params,
    )
    
    return {"job_id": job.id, "status": job.status}

# Use the API
result = handle_generate_request(
    api_key=key.key,
    params={"method": "gaussian_copula", "num_rows": 10000}
)
```

## Configuration Reference

### TenantManager

| Parameter | Type | Description |
|-----------|------|-------------|
| `storage_backend` | str | Storage backend (memory, redis, postgres) |
| `encryption_key` | str | Key for encrypting sensitive config |

### APIKeyManager

| Parameter | Type | Description |
|-----------|------|-------------|
| `key_length` | int | Length of generated keys (default: 32) |
| `hash_algorithm` | str | Hashing algorithm (default: sha256) |
| `storage_backend` | str | Storage backend |

### UsageMeter

| Parameter | Type | Description |
|-----------|------|-------------|
| `storage_backend` | str | Storage backend |
| `aggregation_interval` | int | Aggregation interval in seconds |

### RateLimiter

| Parameter | Type | Description |
|-----------|------|-------------|
| `storage_backend` | str | Backend for token storage |
| `default_config` | RateLimitConfig | Default rate limit |

### JobQueue

| Parameter | Type | Description |
|-----------|------|-------------|
| `redis_url` | str | Redis connection URL |
| `max_retries` | int | Max job retries (default: 3) |
| `job_timeout` | int | Job timeout in seconds |
| `worker_concurrency` | int | Concurrent workers |
