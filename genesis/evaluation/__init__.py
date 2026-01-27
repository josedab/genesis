"""Evaluation module for Genesis."""

from genesis.evaluation.evaluator import QualityEvaluator, evaluate_synthetic_data
from genesis.evaluation.ml_utility import (
    compare_feature_importance,
    compute_ml_utility,
    train_real_test_synthetic,
    train_synthetic_test_real,
)
from genesis.evaluation.privacy import (
    attribute_disclosure_risk,
    compute_privacy_metrics,
    distance_to_closest_record,
    membership_inference_risk,
    reidentification_risk,
)
from genesis.evaluation.report import QualityReport
from genesis.evaluation.statistical import (
    chi_squared_test,
    compute_statistical_fidelity,
    correlation_comparison,
    distribution_shape_metrics,
    jensen_shannon_divergence,
    kolmogorov_smirnov_test,
    total_variation_distance,
    wasserstein_distance,
)

__all__ = [
    # Evaluator
    "QualityEvaluator",
    "evaluate_synthetic_data",
    # Report
    "QualityReport",
    # Statistical
    "compute_statistical_fidelity",
    "kolmogorov_smirnov_test",
    "chi_squared_test",
    "wasserstein_distance",
    "jensen_shannon_divergence",
    "total_variation_distance",
    "correlation_comparison",
    "distribution_shape_metrics",
    # ML Utility
    "compute_ml_utility",
    "train_synthetic_test_real",
    "train_real_test_synthetic",
    "compare_feature_importance",
    # Privacy
    "compute_privacy_metrics",
    "distance_to_closest_record",
    "reidentification_risk",
    "attribute_disclosure_risk",
    "membership_inference_risk",
]
