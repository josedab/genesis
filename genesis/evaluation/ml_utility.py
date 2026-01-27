"""ML utility metrics for evaluation."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler


def train_synthetic_test_real(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: str,
    model_type: str = "auto",
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Train on synthetic, test on real (TSTR).

    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        target_column: Target column name
        model_type: Model type ('classification', 'regression', 'auto')
        test_size: Test set size
        random_state: Random seed

    Returns:
        Dictionary with TSTR metrics
    """
    # Prepare data
    X_syn, y_syn = _prepare_features(synthetic_data, target_column)
    X_real, y_real = _prepare_features(real_data, target_column)

    if X_syn is None or X_real is None:
        return {"error": "Could not prepare features"}

    # Auto-detect task type
    if model_type == "auto":
        if y_real.dtype == object or len(np.unique(y_real)) < 10:
            model_type = "classification"
        else:
            model_type = "regression"

    # Encode labels for classification
    if model_type == "classification":
        le = LabelEncoder()
        y_syn_encoded = le.fit_transform(y_syn.astype(str))
        y_real_encoded = le.transform(y_real.astype(str))
    else:
        y_syn_encoded = y_syn.astype(float)
        y_real_encoded = y_real.astype(float)

    # Scale features
    scaler = StandardScaler()
    X_syn_scaled = scaler.fit_transform(X_syn)
    X_real_scaled = scaler.transform(X_real)

    # Train on synthetic
    if model_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    model.fit(X_syn_scaled, y_syn_encoded)

    # Test on real
    y_pred = model.predict(X_real_scaled)

    # Compute metrics
    if model_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_real_encoded, y_pred)),
            "f1_macro": float(f1_score(y_real_encoded, y_pred, average="macro")),
        }
        if len(np.unique(y_real_encoded)) == 2:
            try:
                y_prob = model.predict_proba(X_real_scaled)[:, 1]
                metrics["auc"] = float(roc_auc_score(y_real_encoded, y_prob))
            except Exception:
                pass
    else:
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_real_encoded, y_pred))),
            "r2": float(r2_score(y_real_encoded, y_pred)),
        }

    return {
        "model_type": model_type,
        "metrics": metrics,
        "n_train": len(X_syn),
        "n_test": len(X_real),
    }


def train_real_test_synthetic(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: str,
    model_type: str = "auto",
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Train on real, test on synthetic (TRTS).

    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        target_column: Target column name
        model_type: Model type
        test_size: Test set size
        random_state: Random seed

    Returns:
        Dictionary with TRTS metrics
    """
    # Prepare data
    X_real, y_real = _prepare_features(real_data, target_column)
    X_syn, y_syn = _prepare_features(synthetic_data, target_column)

    if X_real is None or X_syn is None:
        return {"error": "Could not prepare features"}

    # Auto-detect task type
    if model_type == "auto":
        if y_real.dtype == object or len(np.unique(y_real)) < 10:
            model_type = "classification"
        else:
            model_type = "regression"

    # Encode labels
    if model_type == "classification":
        le = LabelEncoder()
        y_real_encoded = le.fit_transform(y_real.astype(str))
        try:
            y_syn_encoded = le.transform(y_syn.astype(str))
        except ValueError:
            # Unknown labels in synthetic
            mask = y_syn.astype(str).isin(le.classes_)
            X_syn = X_syn[mask]
            y_syn_encoded = le.transform(y_syn[mask].astype(str))
    else:
        y_real_encoded = y_real.astype(float)
        y_syn_encoded = y_syn.astype(float)

    # Scale features
    scaler = StandardScaler()
    X_real_scaled = scaler.fit_transform(X_real)
    X_syn_scaled = scaler.transform(X_syn)

    # Train on real
    if model_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)

    model.fit(X_real_scaled, y_real_encoded)

    # Test on synthetic
    y_pred = model.predict(X_syn_scaled)

    # Compute metrics
    if model_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_syn_encoded, y_pred)),
            "f1_macro": float(f1_score(y_syn_encoded, y_pred, average="macro")),
        }
    else:
        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_syn_encoded, y_pred))),
            "r2": float(r2_score(y_syn_encoded, y_pred)),
        }

    return {
        "model_type": model_type,
        "metrics": metrics,
        "n_train": len(X_real),
        "n_test": len(X_syn),
    }


def compare_feature_importance(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: str,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Compare feature importance between models trained on real vs synthetic.

    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        target_column: Target column name
        random_state: Random seed

    Returns:
        Dictionary with feature importance comparison
    """
    X_real, y_real = _prepare_features(real_data, target_column)
    X_syn, y_syn = _prepare_features(synthetic_data, target_column)

    if X_real is None or X_syn is None:
        return {"error": "Could not prepare features"}

    feature_names = X_real.columns.tolist()

    # Determine task type
    if y_real.dtype == object or len(np.unique(y_real)) < 10:
        model_class = RandomForestClassifier
        y_real_enc = LabelEncoder().fit_transform(y_real.astype(str))
        y_syn_enc = LabelEncoder().fit_transform(y_syn.astype(str))
    else:
        model_class = RandomForestRegressor
        y_real_enc = y_real.astype(float)
        y_syn_enc = y_syn.astype(float)

    # Train models
    model_real = model_class(n_estimators=100, random_state=random_state)
    model_syn = model_class(n_estimators=100, random_state=random_state)

    model_real.fit(X_real, y_real_enc)
    model_syn.fit(X_syn, y_syn_enc)

    # Get feature importances
    imp_real = model_real.feature_importances_
    imp_syn = model_syn.feature_importances_

    # Compare
    importance_diff = np.abs(imp_real - imp_syn)

    # Rank correlation
    rank_real = np.argsort(np.argsort(-imp_real))
    rank_syn = np.argsort(np.argsort(-imp_syn))

    from scipy.stats import spearmanr

    rank_corr, _ = spearmanr(rank_real, rank_syn)

    return {
        "features": feature_names,
        "importance_real": imp_real.tolist(),
        "importance_synthetic": imp_syn.tolist(),
        "importance_diff": importance_diff.tolist(),
        "mean_diff": float(np.mean(importance_diff)),
        "rank_correlation": float(rank_corr) if not np.isnan(rank_corr) else 0.0,
        "score": float(rank_corr) if not np.isnan(rank_corr) else 0.0,
    }


def compute_ml_utility(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    target_column: Optional[str] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute comprehensive ML utility metrics.

    Args:
        real_data: Real DataFrame
        synthetic_data: Synthetic DataFrame
        target_column: Target column (auto-detected if None)
        random_state: Random seed

    Returns:
        Dictionary with all ML utility metrics
    """
    # Auto-detect target column if not specified
    if target_column is None:
        # Use last column as target (common convention)
        target_column = real_data.columns[-1]

    if target_column not in real_data.columns:
        return {"error": f"Target column '{target_column}' not found"}

    results = {}

    # TSTR
    results["tstr"] = train_synthetic_test_real(
        real_data, synthetic_data, target_column, random_state=random_state
    )

    # TRTS
    results["trts"] = train_real_test_synthetic(
        real_data, synthetic_data, target_column, random_state=random_state
    )

    # Feature importance comparison
    results["feature_importance"] = compare_feature_importance(
        real_data, synthetic_data, target_column, random_state=random_state
    )

    # Overall utility score
    tstr_score = results["tstr"]["metrics"].get("accuracy", results["tstr"]["metrics"].get("r2", 0))
    trts_score = results["trts"]["metrics"].get("accuracy", results["trts"]["metrics"].get("r2", 0))
    fi_score = results["feature_importance"].get("score", 0)

    # Utility score: how well does synthetic data preserve ML utility
    results["utility_score"] = 0.4 * tstr_score + 0.4 * trts_score + 0.2 * fi_score

    return results


def _prepare_features(
    data: pd.DataFrame,
    target_column: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
    """Prepare features and target for ML.

    Args:
        data: Input DataFrame
        target_column: Target column name

    Returns:
        Tuple of (features DataFrame, target Series) or (None, None)
    """
    if target_column not in data.columns:
        return None, None

    y = data[target_column]
    X = data.drop(columns=[target_column])

    # Encode categorical features
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = LabelEncoder().fit_transform(X[col].fillna("missing").astype(str))
        else:
            X[col] = X[col].fillna(X[col].median())

    return X, y
