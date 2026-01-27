"""Differential privacy utilities."""

from typing import Any, Dict, List, Optional

import numpy as np


class DPAccountant:
    """Privacy budget accountant for differential privacy.

    Tracks the cumulative privacy budget spent during training.
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
    ) -> None:
        """Initialize the accountant.

        Args:
            epsilon: Total privacy budget
            delta: Probability of privacy breach
        """
        self.total_epsilon = epsilon
        self.delta = delta

        self._spent_epsilon = 0.0
        self._queries: List[Dict[str, float]] = []

    @property
    def remaining_budget(self) -> float:
        """Get remaining privacy budget."""
        return max(0, self.total_epsilon - self._spent_epsilon)

    @property
    def spent_budget(self) -> float:
        """Get spent privacy budget."""
        return self._spent_epsilon

    def spend(self, epsilon: float, description: str = "") -> bool:
        """Spend privacy budget.

        Args:
            epsilon: Amount of budget to spend
            description: Description of the query

        Returns:
            True if budget was available and spent, False otherwise
        """
        if self._spent_epsilon + epsilon > self.total_epsilon:
            return False

        self._spent_epsilon += epsilon
        self._queries.append(
            {
                "epsilon": epsilon,
                "description": description,
                "cumulative": self._spent_epsilon,
            }
        )

        return True

    def get_noise_scale(
        self,
        sensitivity: float,
        epsilon: float,
    ) -> float:
        """Calculate noise scale for Laplace mechanism.

        Args:
            sensitivity: Query sensitivity
            epsilon: Privacy budget for this query

        Returns:
            Scale parameter for Laplace noise
        """
        return sensitivity / epsilon

    def get_gaussian_noise_scale(
        self,
        sensitivity: float,
        epsilon: float,
        delta: Optional[float] = None,
    ) -> float:
        """Calculate noise scale for Gaussian mechanism.

        Args:
            sensitivity: Query sensitivity
            epsilon: Privacy budget for this query
            delta: Delta parameter (uses instance delta if None)

        Returns:
            Standard deviation for Gaussian noise
        """
        delta = delta or self.delta

        # Using the Gaussian mechanism formula
        # sigma = sensitivity * sqrt(2 * ln(1.25/delta)) / epsilon
        return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    def summary(self) -> Dict[str, Any]:
        """Get summary of privacy budget usage."""
        return {
            "total_epsilon": self.total_epsilon,
            "spent_epsilon": self._spent_epsilon,
            "remaining_epsilon": self.remaining_budget,
            "delta": self.delta,
            "n_queries": len(self._queries),
            "queries": self._queries,
        }


def add_laplace_noise(
    data: np.ndarray,
    sensitivity: float,
    epsilon: float,
) -> np.ndarray:
    """Add Laplace noise for differential privacy.

    Args:
        data: Input data
        sensitivity: Query sensitivity
        epsilon: Privacy budget

    Returns:
        Noisy data
    """
    scale = sensitivity / epsilon
    noise = np.random.laplace(0, scale, data.shape)
    return data + noise


def add_gaussian_noise(
    data: np.ndarray,
    sensitivity: float,
    epsilon: float,
    delta: float = 1e-5,
) -> np.ndarray:
    """Add Gaussian noise for differential privacy.

    Args:
        data: Input data
        sensitivity: Query sensitivity
        epsilon: Privacy budget
        delta: Probability parameter

    Returns:
        Noisy data
    """
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise


def clip_gradients(
    gradients: np.ndarray,
    max_norm: float,
) -> np.ndarray:
    """Clip gradients by norm (per-sample).

    Args:
        gradients: Gradient array
        max_norm: Maximum gradient norm

    Returns:
        Clipped gradients
    """
    norms = np.linalg.norm(gradients, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)  # Avoid division by zero
    clip_factor = np.minimum(1.0, max_norm / norms)
    return gradients * clip_factor


class DPOptimizer:
    """Differentially private optimizer wrapper.

    Wraps an optimizer to add gradient clipping and noise
    for differential privacy.
    """

    def __init__(
        self,
        base_optimizer: Any,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        accountant: Optional[DPAccountant] = None,
    ) -> None:
        """Initialize DP optimizer.

        Args:
            base_optimizer: Base optimizer to wrap
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Multiplier for gradient noise
            accountant: Privacy accountant
        """
        self.base_optimizer = base_optimizer
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.accountant = accountant or DPAccountant()

    def compute_private_gradients(
        self,
        gradients: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """Compute privatized gradients.

        Args:
            gradients: Raw gradients
            batch_size: Batch size

        Returns:
            Private gradients
        """
        # Clip per-sample gradients
        clipped = clip_gradients(gradients, self.max_grad_norm)

        # Average
        avg_gradients = np.mean(clipped, axis=0)

        # Add noise
        noise_scale = self.max_grad_norm * self.noise_multiplier / batch_size
        noise = np.random.normal(0, noise_scale, avg_gradients.shape)

        return avg_gradients + noise


def compute_dp_epsilon(
    n_steps: int,
    batch_size: int,
    n_samples: int,
    noise_multiplier: float,
    delta: float = 1e-5,
) -> float:
    """Compute epsilon for DP-SGD using simple composition.

    Args:
        n_steps: Number of training steps
        batch_size: Batch size
        n_samples: Total number of samples
        noise_multiplier: Noise multiplier
        delta: Delta parameter

    Returns:
        Computed epsilon
    """
    # Sampling probability
    q = batch_size / n_samples

    # Using simple composition (Abadi et al. 2016 gives tighter bounds)
    # This is a rough approximation
    sigma = noise_multiplier

    if sigma == 0:
        return float("inf")

    # Per-step epsilon (rough approximation)
    eps_step = q * np.sqrt(2 * np.log(1.25 / delta)) / sigma

    # Composition over steps (simple composition)
    total_epsilon = n_steps * eps_step

    return total_epsilon
