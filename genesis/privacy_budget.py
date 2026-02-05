"""Privacy Budget Orchestrator.

Unified privacy budget management across multiple synthetic datasets with
epsilon tracking, audit trails, alerts, and composition theorem support.

Features:
    - Global epsilon budget registry
    - Per-dataset and per-query epsilon accounting
    - Sequential and parallel composition theorems
    - Budget threshold alerts
    - Immutable audit log
    - Budget consumption dashboard
    - Forecasting for budget depletion

Example:
    Basic budget management::

        from genesis.privacy_budget import PrivacyBudgetOrchestrator, BudgetConfig

        orchestrator = PrivacyBudgetOrchestrator(
            total_budget=10.0,  # Total epsilon
            alert_threshold=0.8,  # Alert at 80% usage
        )

        # Allocate budget for a dataset
        allocation = orchestrator.allocate("customer_data", epsilon=1.0)

        # Track usage
        orchestrator.consume("customer_data", epsilon=0.5, operation="generation")

        # Check remaining
        print(orchestrator.remaining("customer_data"))  # 0.5

    With composition::

        from genesis.privacy_budget import CompositionTheorem

        # Sequential composition
        total = orchestrator.compute_composition(
            epsilons=[0.5, 0.3, 0.2],
            theorem=CompositionTheorem.SEQUENTIAL
        )

Classes:
    PrivacyBudgetOrchestrator: Main budget manager.
    BudgetAllocation: Budget allocated to a dataset.
    BudgetConsumption: Record of budget consumption.
    CompositionTheorem: Privacy composition theorems.
    BudgetAuditLog: Immutable audit trail.
    BudgetAlert: Alert configuration and handling.

Note:
    Privacy budgets are a key mechanism for differential privacy.
    Once depleted, no further queries should be allowed on that data.
"""

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import uuid

import numpy as np

from genesis.utils.logging import get_logger

logger = get_logger(__name__)


class CompositionTheorem(str, Enum):
    """Privacy composition theorems."""

    SEQUENTIAL = "sequential"  # Simple sum
    PARALLEL = "parallel"  # Maximum
    ADVANCED = "advanced"  # Advanced composition (sublinear)
    ZCDP = "zcdp"  # Zero-Concentrated DP


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class BudgetStatus(str, Enum):
    """Budget status."""

    ACTIVE = "active"
    DEPLETED = "depleted"
    SUSPENDED = "suspended"
    RESERVED = "reserved"


@dataclass
class BudgetConfig:
    """Configuration for privacy budget.

    Attributes:
        total_epsilon: Total privacy budget (epsilon).
        total_delta: Total delta for (ε,δ)-DP.
        alert_threshold: Fraction at which to alert.
        hard_limit: Enforce hard limit on budget.
        audit_enabled: Enable audit logging.
        composition_theorem: Default composition theorem.
    """

    total_epsilon: float = 10.0
    total_delta: float = 1e-5
    alert_threshold: float = 0.8
    hard_limit: bool = True
    audit_enabled: bool = True
    composition_theorem: CompositionTheorem = CompositionTheorem.SEQUENTIAL


@dataclass
class BudgetAllocation:
    """Budget allocated to a dataset or task.

    Attributes:
        allocation_id: Unique identifier.
        dataset_id: Dataset this is allocated to.
        epsilon: Allocated epsilon budget.
        delta: Allocated delta budget.
        consumed_epsilon: Epsilon consumed so far.
        consumed_delta: Delta consumed so far.
        status: Current status.
        created_at: When allocated.
        expires_at: Optional expiration.
        metadata: Additional metadata.
    """

    allocation_id: str
    dataset_id: str
    epsilon: float
    delta: float = 1e-6
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    status: BudgetStatus = BudgetStatus.ACTIVE
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_epsilon(self) -> float:
        return max(0, self.epsilon - self.consumed_epsilon)

    @property
    def remaining_delta(self) -> float:
        return max(0, self.delta - self.consumed_delta)

    @property
    def usage_fraction(self) -> float:
        return self.consumed_epsilon / self.epsilon if self.epsilon > 0 else 1.0

    @property
    def is_depleted(self) -> bool:
        return self.remaining_epsilon <= 0 or self.status == BudgetStatus.DEPLETED

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allocation_id": self.allocation_id,
            "dataset_id": self.dataset_id,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "consumed_epsilon": self.consumed_epsilon,
            "consumed_delta": self.consumed_delta,
            "remaining_epsilon": self.remaining_epsilon,
            "remaining_delta": self.remaining_delta,
            "usage_fraction": self.usage_fraction,
            "status": self.status.value,
            "created_at": self.created_at,
        }


@dataclass
class BudgetConsumption:
    """Record of budget consumption.

    Attributes:
        consumption_id: Unique identifier.
        allocation_id: Allocation this consumes from.
        epsilon: Epsilon consumed.
        delta: Delta consumed.
        operation: Type of operation.
        timestamp: When consumed.
        metadata: Additional details.
    """

    consumption_id: str
    allocation_id: str
    epsilon: float
    delta: float = 0.0
    operation: str = "query"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "consumption_id": self.consumption_id,
            "allocation_id": self.allocation_id,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "operation": self.operation,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


@dataclass
class BudgetAlert:
    """Alert about budget status.

    Attributes:
        alert_id: Unique identifier.
        allocation_id: Related allocation.
        severity: Alert severity.
        message: Alert message.
        triggered_at: When triggered.
        acknowledged: Whether acknowledged.
    """

    alert_id: str
    allocation_id: str
    severity: AlertSeverity
    message: str
    triggered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "allocation_id": self.allocation_id,
            "severity": self.severity.value,
            "message": self.message,
            "triggered_at": self.triggered_at,
            "acknowledged": self.acknowledged,
        }


class BudgetAuditLog:
    """Immutable audit log for privacy budget operations."""

    def __init__(self, log_path: Optional[str] = None) -> None:
        self.log_path = Path(log_path) if log_path else None
        self._entries: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        action: str,
        allocation_id: str,
        details: Dict[str, Any],
        user: Optional[str] = None,
    ) -> str:
        """Log an audit entry.

        Args:
            action: Action performed.
            allocation_id: Related allocation.
            details: Action details.
            user: User who performed action.

        Returns:
            Entry ID.
        """
        entry_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Create entry with hash chain
        previous_hash = self._entries[-1]["hash"] if self._entries else "genesis"
        entry_data = json.dumps({
            "entry_id": entry_id,
            "action": action,
            "allocation_id": allocation_id,
            "details": details,
            "user": user,
            "timestamp": timestamp,
            "previous_hash": previous_hash,
        }, sort_keys=True)

        entry_hash = hashlib.sha256(entry_data.encode()).hexdigest()

        entry = {
            "entry_id": entry_id,
            "action": action,
            "allocation_id": allocation_id,
            "details": details,
            "user": user,
            "timestamp": timestamp,
            "previous_hash": previous_hash,
            "hash": entry_hash,
        }

        with self._lock:
            self._entries.append(entry)
            if self.log_path:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")

        return entry_id

    def get_entries(
        self,
        allocation_id: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit entries with filters."""
        entries = self._entries

        if allocation_id:
            entries = [e for e in entries if e["allocation_id"] == allocation_id]
        if action:
            entries = [e for e in entries if e["action"] == action]
        if since:
            entries = [
                e for e in entries
                if datetime.fromisoformat(e["timestamp"]) >= since
            ]

        return entries

    def verify_integrity(self) -> Tuple[bool, Optional[str]]:
        """Verify audit log integrity.

        Returns:
            Tuple of (is_valid, error_message).
        """
        if not self._entries:
            return True, None

        previous_hash = "genesis"
        for i, entry in enumerate(self._entries):
            # Check previous hash
            if entry["previous_hash"] != previous_hash:
                return False, f"Hash chain broken at entry {i}"

            # Recompute hash
            entry_copy = dict(entry)
            stored_hash = entry_copy.pop("hash")
            entry_data = json.dumps(entry_copy, sort_keys=True)
            computed_hash = hashlib.sha256(entry_data.encode()).hexdigest()

            if computed_hash != stored_hash:
                return False, f"Hash mismatch at entry {i}"

            previous_hash = stored_hash

        return True, None


class CompositionCalculator:
    """Calculates privacy composition."""

    @staticmethod
    def sequential(epsilons: List[float]) -> float:
        """Sequential composition: sum of epsilons."""
        return sum(epsilons)

    @staticmethod
    def parallel(epsilons: List[float]) -> float:
        """Parallel composition: maximum epsilon."""
        return max(epsilons) if epsilons else 0.0

    @staticmethod
    def advanced(
        epsilons: List[float],
        delta: float = 1e-5,
    ) -> float:
        """Advanced composition theorem.

        Uses sqrt(2k * ln(1/delta)) * epsilon + k * epsilon^2
        where k is number of compositions.
        """
        if not epsilons:
            return 0.0

        k = len(epsilons)
        avg_eps = np.mean(epsilons)

        # Advanced composition bound
        term1 = np.sqrt(2 * k * np.log(1 / delta)) * avg_eps
        term2 = k * (np.exp(avg_eps) - 1) * avg_eps

        return min(term1 + term2, sum(epsilons))  # Never worse than sequential

    @staticmethod
    def zcdp_to_dp(rho: float, delta: float = 1e-5) -> float:
        """Convert zCDP parameter to (epsilon, delta)-DP."""
        return rho + 2 * np.sqrt(rho * np.log(1 / delta))

    @classmethod
    def compute(
        cls,
        epsilons: List[float],
        theorem: CompositionTheorem,
        delta: float = 1e-5,
    ) -> float:
        """Compute composition using specified theorem."""
        if theorem == CompositionTheorem.SEQUENTIAL:
            return cls.sequential(epsilons)
        elif theorem == CompositionTheorem.PARALLEL:
            return cls.parallel(epsilons)
        elif theorem == CompositionTheorem.ADVANCED:
            return cls.advanced(epsilons, delta)
        elif theorem == CompositionTheorem.ZCDP:
            # Convert epsilons to rho, compose, convert back
            rhos = [(e ** 2) / 2 for e in epsilons]
            total_rho = sum(rhos)
            return cls.zcdp_to_dp(total_rho, delta)
        else:
            return cls.sequential(epsilons)


class PrivacyBudgetOrchestrator:
    """Main privacy budget orchestrator.

    Manages allocation, consumption, and tracking of privacy budgets
    across multiple datasets and operations.
    """

    def __init__(
        self,
        config: Optional[BudgetConfig] = None,
        audit_path: Optional[str] = None,
        alert_callback: Optional[Callable[[BudgetAlert], None]] = None,
    ) -> None:
        """Initialize orchestrator.

        Args:
            config: Budget configuration.
            audit_path: Path for audit log.
            alert_callback: Callback for alerts.
        """
        self.config = config or BudgetConfig()
        self.audit_log = BudgetAuditLog(audit_path)
        self.alert_callback = alert_callback

        self._allocations: Dict[str, BudgetAllocation] = {}
        self._consumptions: List[BudgetConsumption] = []
        self._alerts: List[BudgetAlert] = []
        self._lock = threading.Lock()

        # Track global budget
        self._global_consumed_epsilon = 0.0
        self._global_consumed_delta = 0.0

    @property
    def global_remaining_epsilon(self) -> float:
        """Get globally remaining epsilon."""
        return max(0, self.config.total_epsilon - self._global_consumed_epsilon)

    @property
    def global_usage_fraction(self) -> float:
        """Get global usage fraction."""
        return self._global_consumed_epsilon / self.config.total_epsilon

    def allocate(
        self,
        dataset_id: str,
        epsilon: float,
        delta: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        expires_in: Optional[int] = None,
    ) -> BudgetAllocation:
        """Allocate budget to a dataset.

        Args:
            dataset_id: Dataset identifier.
            epsilon: Epsilon to allocate.
            delta: Delta to allocate.
            metadata: Additional metadata.
            expires_in: Expiration in seconds.

        Returns:
            BudgetAllocation object.

        Raises:
            ValueError: If insufficient budget.
        """
        with self._lock:
            # Check global budget
            if self.config.hard_limit and epsilon > self.global_remaining_epsilon:
                raise ValueError(
                    f"Insufficient global budget: requested {epsilon}, "
                    f"available {self.global_remaining_epsilon}"
                )

            allocation_id = str(uuid.uuid4())
            delta = delta or self.config.total_delta / 100

            expires_at = None
            if expires_in:
                expires_at = (datetime.utcnow() + timedelta(seconds=expires_in)).isoformat()

            allocation = BudgetAllocation(
                allocation_id=allocation_id,
                dataset_id=dataset_id,
                epsilon=epsilon,
                delta=delta,
                metadata=metadata or {},
                expires_at=expires_at,
            )

            self._allocations[allocation_id] = allocation

            # Log audit
            if self.config.audit_enabled:
                self.audit_log.log(
                    action="allocate",
                    allocation_id=allocation_id,
                    details={"epsilon": epsilon, "delta": delta, "dataset_id": dataset_id},
                )

            logger.info(f"Allocated ε={epsilon} to {dataset_id} ({allocation_id})")
            return allocation

    def consume(
        self,
        dataset_id: str,
        epsilon: float,
        delta: float = 0.0,
        operation: str = "query",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BudgetConsumption:
        """Consume budget from a dataset's allocation.

        Args:
            dataset_id: Dataset identifier.
            epsilon: Epsilon to consume.
            delta: Delta to consume.
            operation: Type of operation.
            metadata: Additional metadata.

        Returns:
            BudgetConsumption record.

        Raises:
            ValueError: If insufficient budget or allocation not found.
        """
        with self._lock:
            # Find allocation for dataset
            allocation = self._find_allocation(dataset_id)
            if allocation is None:
                raise ValueError(f"No allocation found for dataset {dataset_id}")

            if allocation.is_depleted:
                raise ValueError(f"Budget depleted for {dataset_id}")

            if self.config.hard_limit and epsilon > allocation.remaining_epsilon:
                raise ValueError(
                    f"Insufficient budget for {dataset_id}: "
                    f"requested {epsilon}, available {allocation.remaining_epsilon}"
                )

            # Record consumption
            consumption_id = str(uuid.uuid4())
            consumption = BudgetConsumption(
                consumption_id=consumption_id,
                allocation_id=allocation.allocation_id,
                epsilon=epsilon,
                delta=delta,
                operation=operation,
                metadata=metadata or {},
            )

            allocation.consumed_epsilon += epsilon
            allocation.consumed_delta += delta
            self._global_consumed_epsilon += epsilon
            self._global_consumed_delta += delta
            self._consumptions.append(consumption)

            # Check for depletion
            if allocation.remaining_epsilon <= 0:
                allocation.status = BudgetStatus.DEPLETED

            # Check for alerts
            self._check_alerts(allocation)

            # Log audit
            if self.config.audit_enabled:
                self.audit_log.log(
                    action="consume",
                    allocation_id=allocation.allocation_id,
                    details={
                        "epsilon": epsilon,
                        "delta": delta,
                        "operation": operation,
                        "remaining": allocation.remaining_epsilon,
                    },
                )

            return consumption

    def remaining(self, dataset_id: str) -> float:
        """Get remaining epsilon for a dataset."""
        allocation = self._find_allocation(dataset_id)
        return allocation.remaining_epsilon if allocation else 0.0

    def get_allocation(self, dataset_id: str) -> Optional[BudgetAllocation]:
        """Get allocation for a dataset."""
        return self._find_allocation(dataset_id)

    def get_all_allocations(self) -> List[BudgetAllocation]:
        """Get all allocations."""
        return list(self._allocations.values())

    def get_consumption_history(
        self,
        dataset_id: Optional[str] = None,
        since: Optional[datetime] = None,
    ) -> List[BudgetConsumption]:
        """Get consumption history."""
        consumptions = self._consumptions

        if dataset_id:
            allocation = self._find_allocation(dataset_id)
            if allocation:
                consumptions = [
                    c for c in consumptions
                    if c.allocation_id == allocation.allocation_id
                ]

        if since:
            consumptions = [
                c for c in consumptions
                if datetime.fromisoformat(c.timestamp) >= since
            ]

        return consumptions

    def compute_composition(
        self,
        epsilons: List[float],
        theorem: Optional[CompositionTheorem] = None,
    ) -> float:
        """Compute privacy composition.

        Args:
            epsilons: List of epsilon values.
            theorem: Composition theorem to use.

        Returns:
            Composed epsilon value.
        """
        theorem = theorem or self.config.composition_theorem
        return CompositionCalculator.compute(
            epsilons, theorem, self.config.total_delta
        )

    def forecast_depletion(
        self,
        dataset_id: str,
        window_days: int = 30,
    ) -> Optional[datetime]:
        """Forecast when budget will be depleted.

        Args:
            dataset_id: Dataset to forecast.
            window_days: Historical window for calculation.

        Returns:
            Estimated depletion datetime or None if stable.
        """
        allocation = self._find_allocation(dataset_id)
        if allocation is None or allocation.is_depleted:
            return None

        # Get consumption in window
        since = datetime.utcnow() - timedelta(days=window_days)
        history = self.get_consumption_history(dataset_id, since)

        if len(history) < 2:
            return None

        # Calculate consumption rate
        total_consumed = sum(c.epsilon for c in history)
        days_elapsed = window_days

        if total_consumed == 0:
            return None

        daily_rate = total_consumed / days_elapsed
        remaining = allocation.remaining_epsilon

        days_to_depletion = remaining / daily_rate
        return datetime.utcnow() + timedelta(days=days_to_depletion)

    def get_summary(self) -> Dict[str, Any]:
        """Get budget summary."""
        return {
            "global": {
                "total_epsilon": self.config.total_epsilon,
                "consumed_epsilon": self._global_consumed_epsilon,
                "remaining_epsilon": self.global_remaining_epsilon,
                "usage_fraction": self.global_usage_fraction,
            },
            "allocations": {
                a.dataset_id: a.to_dict() for a in self._allocations.values()
            },
            "total_allocations": len(self._allocations),
            "total_consumptions": len(self._consumptions),
            "active_alerts": len([a for a in self._alerts if not a.acknowledged]),
        }

    def _find_allocation(self, dataset_id: str) -> Optional[BudgetAllocation]:
        """Find allocation for a dataset."""
        for allocation in self._allocations.values():
            if allocation.dataset_id == dataset_id:
                return allocation
        return None

    def _check_alerts(self, allocation: BudgetAllocation) -> None:
        """Check and trigger alerts for an allocation."""
        # Threshold alert
        if allocation.usage_fraction >= self.config.alert_threshold:
            if allocation.usage_fraction >= 1.0:
                severity = AlertSeverity.CRITICAL
                message = f"Budget depleted for {allocation.dataset_id}"
            elif allocation.usage_fraction >= 0.9:
                severity = AlertSeverity.WARNING
                message = f"Budget >90% used for {allocation.dataset_id}"
            else:
                severity = AlertSeverity.INFO
                message = f"Budget >{self.config.alert_threshold*100:.0f}% used for {allocation.dataset_id}"

            alert = BudgetAlert(
                alert_id=str(uuid.uuid4()),
                allocation_id=allocation.allocation_id,
                severity=severity,
                message=message,
            )
            self._alerts.append(alert)

            if self.alert_callback:
                self.alert_callback(alert)

            logger.warning(f"Budget alert: {message}")


# Convenience decorator
def requires_budget(epsilon: float, delta: float = 0.0):
    """Decorator to require budget for a function.

    Args:
        epsilon: Required epsilon.
        delta: Required delta.

    Example:
        >>> orchestrator = PrivacyBudgetOrchestrator()
        >>> orchestrator.allocate("my_data", epsilon=5.0)
        >>>
        >>> @requires_budget(epsilon=0.5)
        ... def generate_report(data, orchestrator, dataset_id):
        ...     return data.describe()
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            orchestrator = kwargs.get("orchestrator")
            dataset_id = kwargs.get("dataset_id")

            if orchestrator and dataset_id:
                orchestrator.consume(
                    dataset_id=dataset_id,
                    epsilon=epsilon,
                    delta=delta,
                    operation=func.__name__,
                )

            return func(*args, **kwargs)
        return wrapper
    return decorator
