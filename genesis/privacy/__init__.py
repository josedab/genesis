"""Privacy module for Genesis."""

from genesis.core.config import PrivacyConfig
from genesis.privacy.anonymity import (
    check_k_anonymity,
    check_l_diversity,
    enforce_k_anonymity,
    enforce_l_diversity,
    suppress_rare_categories,
)
from genesis.privacy.differential import (
    DPAccountant,
    DPOptimizer,
    add_gaussian_noise,
    add_laplace_noise,
    clip_gradients,
    compute_dp_epsilon,
)
from genesis.privacy.metrics import (
    attribute_disclosure_risk,
    compute_privacy_metrics,
    distance_to_closest_record,
    membership_inference_risk,
    reidentification_risk,
)

__all__ = [
    # Config
    "PrivacyConfig",
    # Differential privacy
    "DPAccountant",
    "DPOptimizer",
    "add_laplace_noise",
    "add_gaussian_noise",
    "clip_gradients",
    "compute_dp_epsilon",
    # Anonymity
    "check_k_anonymity",
    "enforce_k_anonymity",
    "check_l_diversity",
    "enforce_l_diversity",
    "suppress_rare_categories",
    # Metrics
    "compute_privacy_metrics",
    "distance_to_closest_record",
    "reidentification_risk",
    "attribute_disclosure_risk",
    "membership_inference_risk",
]
