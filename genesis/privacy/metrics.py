"""Privacy metrics computation."""

# Re-export from evaluation module for convenience
from genesis.evaluation.privacy import (
    attribute_disclosure_risk,
    compute_privacy_metrics,
    distance_to_closest_record,
    membership_inference_risk,
    reidentification_risk,
)

__all__ = [
    "compute_privacy_metrics",
    "distance_to_closest_record",
    "reidentification_risk",
    "attribute_disclosure_risk",
    "membership_inference_risk",
]
