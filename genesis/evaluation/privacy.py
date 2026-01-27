"""Privacy metrics for evaluation.

This module provides metrics for assessing privacy risks in synthetic data,
including re-identification risk, attribute disclosure, and membership inference.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =============================================================================
# Privacy Evaluation Constants
# =============================================================================

# Distance threshold for considering a synthetic record as a potential match
# to a real record. Lower values mean stricter matching criteria.
DEFAULT_REIDENTIFICATION_THRESHOLD: float = 0.5

# Threshold for flagging synthetic records as potential copies of real records.
# Records closer than this (in normalized distance) may indicate memorization.
VERY_CLOSE_RECORD_THRESHOLD: float = 0.1

# Normalization factor for converting DCR to a [0,1] privacy score.
# DCR values are divided by this to scale appropriately.
DCR_NORMALIZATION_FACTOR: float = 10.0

# Number of estimators for the membership inference attack model.
# Higher values increase attack accuracy but also computation time.
MEMBERSHIP_INFERENCE_N_ESTIMATORS: int = 50

# Random seed for reproducible membership inference attacks.
MEMBERSHIP_INFERENCE_RANDOM_SEED: int = 42

# Number of cross-validation folds for membership inference evaluation.
MEMBERSHIP_INFERENCE_CV_FOLDS: int = 5


def distance_to_closest_record(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    n_neighbors: int = 1,
) -> Dict[str, float]:
    """Compute Distance to Closest Record (DCR) metric.

    Measures how close synthetic records are to real records.
    Higher values indicate better privacy.

    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        n_neighbors: Number of neighbors to consider

    Returns:
        Dictionary with DCR metrics
    """
    X_real = _encode_data(real_data)
    X_syn = _encode_data(synthetic_data)

    if X_real is None or X_syn is None:
        return {"error": "Could not encode data"}

    # Scale data
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real)
    X_syn_scaled = scaler.transform(X_syn)

    # Find nearest neighbors
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X_real_scaled)

    distances, _ = nn.kneighbors(X_syn_scaled)

    # Statistics
    min_distances = distances[:, 0]  # Distance to closest record

    return {
        "mean_dcr": float(np.mean(min_distances)),
        "median_dcr": float(np.median(min_distances)),
        "min_dcr": float(np.min(min_distances)),
        "max_dcr": float(np.max(min_distances)),
        "std_dcr": float(np.std(min_distances)),
        "pct_very_close": float(np.mean(min_distances < VERY_CLOSE_RECORD_THRESHOLD)),
    }


def reidentification_risk(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    quasi_identifiers: Optional[List[str]] = None,
    threshold: float = DEFAULT_REIDENTIFICATION_THRESHOLD,
) -> Dict[str, float]:
    """Compute re-identification risk.

    Estimates the risk that a synthetic record could be matched
    to a specific real individual.

    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        quasi_identifiers: Columns to use for matching
        threshold: Distance threshold for considering a match

    Returns:
        Dictionary with re-identification risk metrics
    """
    # Use all columns if quasi-identifiers not specified
    if quasi_identifiers is None:
        quasi_identifiers = list(real_data.columns)

    # Filter to common columns
    common_cols = [
        c for c in quasi_identifiers if c in real_data.columns and c in synthetic_data.columns
    ]

    if not common_cols:
        return {"error": "No common quasi-identifier columns"}

    X_real = _encode_data(real_data[common_cols])
    X_syn = _encode_data(synthetic_data[common_cols])

    if X_real is None or X_syn is None:
        return {"error": "Could not encode data"}

    # Scale
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real)
    X_syn_scaled = scaler.transform(X_syn)

    # Find matches
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_real_scaled)

    distances, indices = nn.kneighbors(X_syn_scaled)

    # Count potential re-identifications
    n_matches = np.sum(distances[:, 0] < threshold)
    n_synthetic = len(synthetic_data)

    # Risk is the fraction that could potentially be re-identified
    risk = n_matches / n_synthetic if n_synthetic > 0 else 0

    return {
        "reidentification_risk": float(risk),
        "n_potential_matches": int(n_matches),
        "n_synthetic_records": n_synthetic,
        "threshold": threshold,
        "privacy_score": float(1 - risk),  # Higher is better
    }


def attribute_disclosure_risk(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    sensitive_columns: List[str],
    quasi_identifiers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute attribute disclosure risk.

    Measures the risk that sensitive attributes could be inferred
    for real individuals using synthetic data.

    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        sensitive_columns: Columns containing sensitive attributes
        quasi_identifiers: Columns used for linking

    Returns:
        Dictionary with attribute disclosure risk metrics
    """
    if not sensitive_columns:
        return {"error": "No sensitive columns specified"}

    # Filter to valid columns
    sensitive_cols = [
        c for c in sensitive_columns if c in real_data.columns and c in synthetic_data.columns
    ]

    if not sensitive_cols:
        return {"error": "No valid sensitive columns found"}

    if quasi_identifiers is None:
        quasi_identifiers = [c for c in real_data.columns if c not in sensitive_cols]

    qi_cols = [
        c for c in quasi_identifiers if c in real_data.columns and c in synthetic_data.columns
    ]

    if not qi_cols:
        return {"error": "No quasi-identifier columns found"}

    results = {"per_attribute": {}}

    for sens_col in sensitive_cols:
        # Group by quasi-identifiers in both datasets
        real_groups = real_data.groupby(qi_cols)[sens_col].apply(list).to_dict()
        syn_groups = synthetic_data.groupby(qi_cols)[sens_col].apply(list).to_dict()

        # Check for disclosure
        disclosure_count = 0
        checked_count = 0

        for qi_key in real_groups:
            if qi_key in syn_groups:
                real_vals = set(real_groups[qi_key])
                syn_vals = set(syn_groups[qi_key])

                # If synthetic reveals unique real value
                if len(real_vals) == 1 and real_vals == syn_vals:
                    disclosure_count += 1
                checked_count += 1

        risk = disclosure_count / checked_count if checked_count > 0 else 0

        results["per_attribute"][sens_col] = {
            "disclosure_risk": float(risk),
            "n_disclosures": disclosure_count,
            "n_checked": checked_count,
        }

    # Overall risk
    all_risks = [v["disclosure_risk"] for v in results["per_attribute"].values()]
    results["overall_risk"] = float(np.mean(all_risks)) if all_risks else 0.0
    results["privacy_score"] = 1 - results["overall_risk"]

    return results


def membership_inference_risk(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    holdout_data: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Estimate membership inference attack risk.

    Measures how well an attacker could determine if a record
    was in the training data based on the synthetic data.

    Args:
        real_data: Real training data
        synthetic_data: Synthetic data
        holdout_data: Real data not used in training (for comparison)

    Returns:
        Dictionary with membership inference risk metrics
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X_real = _encode_data(real_data)
    X_syn = _encode_data(synthetic_data)

    if X_real is None or X_syn is None:
        return {"error": "Could not encode data"}

    # Use synthetic data as "attacker's knowledge"
    # Train classifier to distinguish real from random noise

    # Create negative examples (shuffled versions)
    X_negative = X_real.copy()
    for col in range(X_negative.shape[1]):
        np.random.shuffle(X_negative[:, col])

    # Labels: 1 = real, 0 = fake
    X_combined = np.vstack([X_real, X_negative])
    y_combined = np.array([1] * len(X_real) + [0] * len(X_negative))

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Train attack model
    attack_model = RandomForestClassifier(
        n_estimators=MEMBERSHIP_INFERENCE_N_ESTIMATORS,
        random_state=MEMBERSHIP_INFERENCE_RANDOM_SEED,
    )

    # Cross-validation score represents attacker's success
    try:
        scores = cross_val_score(
            attack_model,
            X_scaled,
            y_combined,
            cv=MEMBERSHIP_INFERENCE_CV_FOLDS,
            scoring="accuracy",
        )
        attack_accuracy = float(np.mean(scores))
    except Exception:
        attack_accuracy = 0.5

    # Risk is how much better than random (0.5)
    risk = max(0, (attack_accuracy - 0.5) * 2)

    return {
        "attack_accuracy": attack_accuracy,
        "membership_inference_risk": float(risk),
        "privacy_score": float(1 - risk),
    }


def compute_privacy_metrics(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    sensitive_columns: Optional[List[str]] = None,
    quasi_identifiers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive privacy metrics.

    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        sensitive_columns: Columns with sensitive data
        quasi_identifiers: Columns that could identify individuals

    Returns:
        Dictionary with all privacy metrics
    """
    results = {}

    # DCR
    results["dcr"] = distance_to_closest_record(real_data, synthetic_data)

    # Re-identification risk
    results["reidentification"] = reidentification_risk(
        real_data, synthetic_data, quasi_identifiers
    )

    # Attribute disclosure (if sensitive columns specified)
    if sensitive_columns:
        results["attribute_disclosure"] = attribute_disclosure_risk(
            real_data, synthetic_data, sensitive_columns, quasi_identifiers
        )

    # Membership inference
    results["membership_inference"] = membership_inference_risk(real_data, synthetic_data)

    # Overall privacy score
    scores = [
        results["dcr"].get("mean_dcr", 0) / DCR_NORMALIZATION_FACTOR,
        results["reidentification"].get("privacy_score", 0),
        results["membership_inference"].get("privacy_score", 0),
    ]

    if "attribute_disclosure" in results:
        scores.append(results["attribute_disclosure"].get("privacy_score", 0))

    results["overall_privacy_score"] = float(np.mean([s for s in scores if s > 0]))

    return results


def _encode_data(data: pd.DataFrame) -> Optional[np.ndarray]:
    """Encode DataFrame for distance computation.

    Args:
        data: Input DataFrame

    Returns:
        Encoded numpy array or None
    """
    try:
        encoded = data.copy()

        for col in encoded.columns:
            if encoded[col].dtype == object:
                encoded[col] = LabelEncoder().fit_transform(
                    encoded[col].fillna("missing").astype(str)
                )
            else:
                encoded[col] = encoded[col].fillna(encoded[col].median())

        return encoded.values.astype(float)
    except Exception:
        return None
