"""Tests for multi-tenant SaaS backend."""

from datetime import datetime, timedelta

import pytest

from genesis.saas import (
    APIKey,
    APIKeyManager,
    InMemoryStorage,
    Job,
    JobQueue,
    Plan,
    PlanLimits,
    RateLimiter,
    Tenant,
    TenantContext,
    TenantManager,
    UsageMeter,
    UsageRecord,
    create_saas_infrastructure,
    PLAN_LIMITS,
)
from genesis.core.exceptions import GenesisError


class TestPlanLimits:
    """Tests for plan limits."""

    def test_free_plan_limits(self) -> None:
        """Test free plan limits."""
        limits = PLAN_LIMITS[Plan.FREE]

        assert limits.rows_per_month == 10_000
        assert limits.concurrent_jobs == 1
        assert limits.api_keys == 1
        assert limits.advanced_features is False

    def test_enterprise_plan_unlimited(self) -> None:
        """Test enterprise plan has unlimited features."""
        limits = PLAN_LIMITS[Plan.ENTERPRISE]

        assert limits.rows_per_month == -1  # Unlimited
        assert limits.api_keys == -1  # Unlimited


class TestTenant:
    """Tests for Tenant dataclass."""

    def test_default_values(self) -> None:
        """Test default tenant values."""
        tenant = Tenant(
            id="ten_123",
            name="Test Corp",
            plan=Plan.FREE,
        )

        assert tenant.is_active is True
        assert tenant.metadata == {}
        assert tenant.billing_email is None


class TestTenantManager:
    """Tests for TenantManager."""

    @pytest.fixture
    def manager(self) -> TenantManager:
        """Create tenant manager fixture."""
        return TenantManager()

    def test_create_tenant(self, manager: TenantManager) -> None:
        """Test creating a tenant."""
        tenant = manager.create_tenant(
            name="Acme Corp",
            plan=Plan.PRO,
            billing_email="billing@acme.com",
        )

        assert tenant.name == "Acme Corp"
        assert tenant.plan == Plan.PRO
        assert tenant.billing_email == "billing@acme.com"
        assert tenant.id.startswith("ten_")

    def test_get_tenant(self, manager: TenantManager) -> None:
        """Test retrieving a tenant."""
        created = manager.create_tenant("Test Corp")
        retrieved = manager.get_tenant(created.id)

        assert retrieved is not None
        assert retrieved.name == "Test Corp"

    def test_get_nonexistent_tenant(self, manager: TenantManager) -> None:
        """Test retrieving nonexistent tenant."""
        result = manager.get_tenant("nonexistent")
        assert result is None

    def test_update_tenant(self, manager: TenantManager) -> None:
        """Test updating a tenant."""
        tenant = manager.create_tenant("Original Name")
        updated = manager.update_tenant(tenant.id, name="New Name")

        assert updated is not None
        assert updated.name == "New Name"

    def test_update_plan(self, manager: TenantManager) -> None:
        """Test updating tenant plan."""
        tenant = manager.create_tenant("Test Corp", plan=Plan.FREE)
        updated = manager.update_plan(tenant.id, Plan.PRO)

        assert updated is not None
        assert updated.plan == Plan.PRO

    def test_deactivate_tenant(self, manager: TenantManager) -> None:
        """Test deactivating a tenant."""
        tenant = manager.create_tenant("Test Corp")
        result = manager.deactivate_tenant(tenant.id)

        assert result is True

        retrieved = manager.get_tenant(tenant.id)
        assert retrieved.is_active is False

    def test_get_limits(self, manager: TenantManager) -> None:
        """Test getting plan limits."""
        tenant = manager.create_tenant("Test Corp", plan=Plan.PRO)
        limits = manager.get_limits(tenant.id)

        assert limits is not None
        assert limits.rows_per_month == PLAN_LIMITS[Plan.PRO].rows_per_month


class TestAPIKeyManager:
    """Tests for APIKeyManager."""

    @pytest.fixture
    def setup(self) -> tuple:
        """Create managers and tenant fixture."""
        tenant_mgr = TenantManager()
        key_mgr = APIKeyManager(tenant_mgr)
        tenant = tenant_mgr.create_tenant("Test Corp", plan=Plan.PRO)
        return key_mgr, tenant

    def test_create_key(self, setup: tuple) -> None:
        """Test creating an API key."""
        key_mgr, tenant = setup

        raw_key, api_key = key_mgr.create_key(
            tenant.id,
            name="Production",
            scopes=["generate", "read"],
        )

        assert raw_key.startswith("gns_")
        assert api_key.name == "Production"
        assert "generate" in api_key.scopes
        assert api_key.is_active is True

    def test_validate_valid_key(self, setup: tuple) -> None:
        """Test validating a valid key."""
        key_mgr, tenant = setup
        raw_key, _ = key_mgr.create_key(tenant.id)

        is_valid, api_key, error = key_mgr.validate_key(raw_key)

        assert is_valid is True
        assert api_key is not None
        assert error is None

    def test_validate_invalid_key(self, setup: tuple) -> None:
        """Test validating an invalid key."""
        key_mgr, _ = setup

        is_valid, api_key, error = key_mgr.validate_key("invalid_key")

        assert is_valid is False
        assert api_key is None
        assert "Invalid key format" in error

    def test_validate_nonexistent_key(self, setup: tuple) -> None:
        """Test validating nonexistent key."""
        key_mgr, _ = setup

        is_valid, api_key, error = key_mgr.validate_key("gns_nonexistent123456789")

        assert is_valid is False
        assert "Key not found" in error

    def test_validate_checks_scope(self, setup: tuple) -> None:
        """Test scope checking during validation."""
        key_mgr, tenant = setup
        raw_key, _ = key_mgr.create_key(tenant.id, scopes=["read"])

        is_valid, _, error = key_mgr.validate_key(raw_key, required_scope="generate")

        assert is_valid is False
        assert "lacks required scope" in error

    def test_revoke_key(self, setup: tuple) -> None:
        """Test revoking an API key."""
        key_mgr, tenant = setup
        _, api_key = key_mgr.create_key(tenant.id)

        result = key_mgr.revoke_key(api_key.key_id)

        assert result is True

        is_valid, _, error = key_mgr.validate_key(api_key.key_hash)
        assert is_valid is False

    def test_key_limit_enforced(self) -> None:
        """Test API key limit enforcement."""
        tenant_mgr = TenantManager()
        key_mgr = APIKeyManager(tenant_mgr)
        tenant = tenant_mgr.create_tenant("Test Corp", plan=Plan.FREE)

        # Create first key (should succeed)
        key_mgr.create_key(tenant.id)

        # Try to create second key (should fail for FREE plan)
        with pytest.raises(GenesisError, match="API key limit reached"):
            key_mgr.create_key(tenant.id)


class TestUsageMeter:
    """Tests for UsageMeter."""

    @pytest.fixture
    def setup(self) -> tuple:
        """Create meter and tenant fixture."""
        tenant_mgr = TenantManager()
        meter = UsageMeter(tenant_mgr)
        tenant = tenant_mgr.create_tenant("Test Corp", plan=Plan.PRO)
        return meter, tenant

    def test_record_usage(self, setup: tuple) -> None:
        """Test recording usage."""
        meter, tenant = setup

        meter.record_usage(tenant.id, "rows_generated", 1000)
        meter.record_usage(tenant.id, "rows_generated", 500)

        usage = meter.get_monthly_usage(tenant.id, "rows_generated")
        assert usage == 1500

    def test_check_limit_allowed(self, setup: tuple) -> None:
        """Test usage within limit."""
        meter, tenant = setup

        allowed, error = meter.check_limit(tenant.id, "rows_generated", 1000)

        assert allowed is True
        assert error is None

    def test_check_limit_exceeded(self) -> None:
        """Test usage exceeding limit."""
        tenant_mgr = TenantManager()
        meter = UsageMeter(tenant_mgr)
        tenant = tenant_mgr.create_tenant("Test Corp", plan=Plan.FREE)

        # Record usage near limit
        meter.record_usage(tenant.id, "rows_generated", 9500)

        # Check if additional usage would exceed
        allowed, error = meter.check_limit(tenant.id, "rows_generated", 1000)

        assert allowed is False
        assert "limit exceeded" in error


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_allows_within_limit(self) -> None:
        """Test allowing requests within limit."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)

        # Should allow burst
        for _ in range(10):
            assert limiter.allow("tenant_1") is True

    def test_blocks_over_limit(self) -> None:
        """Test blocking requests over limit."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)

        # Exhaust burst
        for _ in range(5):
            limiter.allow("tenant_1")

        # Next request should be blocked
        assert limiter.allow("tenant_1") is False

    def test_different_tenants_independent(self) -> None:
        """Test different tenants have independent limits."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)

        # Exhaust tenant_1
        for _ in range(5):
            limiter.allow("tenant_1")

        # tenant_2 should still have quota
        assert limiter.allow("tenant_2") is True

    def test_get_wait_time(self) -> None:
        """Test wait time calculation."""
        limiter = RateLimiter(requests_per_minute=60, burst_size=1)

        # Exhaust burst
        limiter.allow("tenant_1")

        wait_time = limiter.get_wait_time("tenant_1")
        assert wait_time > 0


class TestJobQueue:
    """Tests for JobQueue."""

    @pytest.fixture
    def setup(self) -> tuple:
        """Create queue and tenant fixture."""
        tenant_mgr = TenantManager()
        queue = JobQueue(tenant_mgr)
        tenant = tenant_mgr.create_tenant("Test Corp", plan=Plan.PRO)
        return queue, tenant

    def test_enqueue_job(self, setup: tuple) -> None:
        """Test enqueueing a job."""
        queue, tenant = setup

        job = queue.enqueue(
            tenant.id,
            "generate",
            {"table": "users", "rows": 10000},
        )

        assert job.id.startswith("job_")
        assert job.status == "pending"
        assert job.job_type == "generate"

    def test_get_job(self, setup: tuple) -> None:
        """Test retrieving a job."""
        queue, tenant = setup
        created = queue.enqueue(tenant.id, "generate", {})

        retrieved = queue.get_job(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id

    def test_update_status(self, setup: tuple) -> None:
        """Test updating job status."""
        queue, tenant = setup
        job = queue.enqueue(tenant.id, "generate", {})

        queue.update_status(job.id, "running")
        assert queue.get_job(job.id).status == "running"

        queue.update_status(job.id, "completed", result={"rows": 10000})
        completed = queue.get_job(job.id)

        assert completed.status == "completed"
        assert completed.result == {"rows": 10000}

    def test_concurrent_job_limit(self) -> None:
        """Test concurrent job limit enforcement."""
        tenant_mgr = TenantManager()
        queue = JobQueue(tenant_mgr)
        tenant = tenant_mgr.create_tenant("Test Corp", plan=Plan.FREE)

        # Create first job
        queue.enqueue(tenant.id, "generate", {})

        # Second job should fail (FREE plan has 1 concurrent job)
        with pytest.raises(GenesisError, match="Concurrent job limit"):
            queue.enqueue(tenant.id, "generate", {})


class TestTenantContext:
    """Tests for TenantContext."""

    def test_context_manager(self) -> None:
        """Test context manager usage."""
        infra = create_saas_infrastructure()
        tenant = infra["tenant_manager"].create_tenant("Test Corp")

        with TenantContext(
            tenant.id,
            infra["key_manager"],
            infra["usage_meter"],
        ) as ctx:
            assert ctx.tenant.name == "Test Corp"

    def test_raises_for_inactive_tenant(self) -> None:
        """Test raises error for inactive tenant."""
        infra = create_saas_infrastructure()
        tenant = infra["tenant_manager"].create_tenant("Test Corp")
        infra["tenant_manager"].deactivate_tenant(tenant.id)

        with pytest.raises(GenesisError, match="inactive"):
            with TenantContext(
                tenant.id,
                infra["key_manager"],
                infra["usage_meter"],
            ):
                pass


class TestCreateSaasInfrastructure:
    """Tests for create_saas_infrastructure."""

    def test_returns_all_components(self) -> None:
        """Test that all components are returned."""
        infra = create_saas_infrastructure()

        assert "tenant_manager" in infra
        assert "key_manager" in infra
        assert "usage_meter" in infra
        assert "rate_limiter" in infra
        assert "job_queue" in infra

    def test_components_are_connected(self) -> None:
        """Test that components share storage."""
        infra = create_saas_infrastructure()

        # Create tenant via manager
        tenant = infra["tenant_manager"].create_tenant("Test")

        # Should be accessible via key manager's tenant manager
        assert infra["key_manager"].tenant_manager.get_tenant(tenant.id) is not None
