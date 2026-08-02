"""Feature engineering: encoding, scaling, creation, and selection.

Transforms raw data into features that expose patterns to ML algorithms.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# =========================================================================
# 1. ENCODING CATEGORICAL VARIABLES
# =========================================================================

CARDINALITY_THRESHOLD = 15  # above this → target/high-cardinality encoding


def auto_encode(
    df: pd.DataFrame,
    categorical_cols: list[str],
    target_col: str | None = None,
    cardinality_threshold: int = CARDINALITY_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Automatically choose encoding strategy per column based on cardinality.

    - Low cardinality (<= threshold): One-Hot Encoding
    - High cardinality (> threshold): Target Encoding (if target provided)
                                       or Ordinal Encoding (fallback)
    """
    df = df.copy()
    report: dict[str, dict] = {}
    onehot_cols: list[str] = []
    target_encode_cols: list[str] = []
    ordinal_encode_cols: list[str] = []

    for col in categorical_cols:
        if col not in df.columns:
            continue
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= 1:
            continue
        if n_unique <= cardinality_threshold:
            onehot_cols.append(col)
        elif target_col and target_col in df.columns:
            target_encode_cols.append(col)
        else:
            ordinal_encode_cols.append(col)

    if onehot_cols:
        df, oh_report = one_hot_encode(df, onehot_cols)
        report.update(oh_report)

    if target_encode_cols:
        df, te_report = target_encode(df, target_encode_cols, target_col)
        report.update(te_report)

    if ordinal_encode_cols:
        df, ord_report = ordinal_encode(df, ordinal_encode_cols)
        report.update(ord_report)

    return df, report


def one_hot_encode(
    df: pd.DataFrame, categorical_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """One-Hot Encode low-cardinality categoricals.

    Drops the first category to avoid multicollinearity (dummy trap).
    Returns encoded DataFrame and mapping info.
    """
    df = df.copy()
    report: dict[str, dict] = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue
        categories = df[col].dropna().unique()
        if len(categories) <= 1:
            continue

        sorted_cats = sorted(categories, key=str)
        reference_cat = str(sorted_cats[0])

        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
        new_cols = list(dummies.columns)
        report[col] = {
            "method": "one_hot",
            "encoded": new_cols,
            "reference": reference_cat,
            "categories": [str(c) for c in sorted_cats],
        }
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    return df, report


def target_encode(
    df: pd.DataFrame,
    categorical_cols: list[str],
    target_col: str,
    smoothing: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Target Encoding for high-cardinality categoricals.

    Replaces each category with the smoothed mean of the target variable.
    Smoothing prevents overfitting on rare categories by blending with the
    global mean.

    Formula: smoothed_mean = (n * cat_mean + smoothing * global_mean) / (n + smoothing)
    """
    df = df.copy()
    report: dict[str, dict] = {}

    if target_col not in df.columns:
        return df, report

    global_mean = df[target_col].dropna().mean()

    for col in categorical_cols:
        if col not in df.columns:
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue

        # Compute smoothed means per category
        agg = df.groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
        mapping = smoothed.to_dict()

        # Apply mapping
        new_col = f"{col}_target_encoded"
        df[new_col] = df[col].map(mapping).fillna(global_mean)
        df = df.drop(columns=[col])

        report[col] = {
            "method": "target",
            "new_column": new_col,
            "global_mean": round(float(global_mean), 4),
            "smoothing": smoothing,
            "mapping": {str(k): round(float(v), 4) for k, v in mapping.items()},
        }

    return df, report


def ordinal_encode(
    df: pd.DataFrame,
    categorical_cols: list[str],
    ordinal_maps: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Ordinal Encoding: map categories to integers 0, 1, 2, ...

    If ordinal_maps is provided, uses the specified ordering.
    Otherwise, sorts alphabetically (arbitrary but deterministic).
    """
    df = df.copy()
    report: dict[str, dict] = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) <= 1:
            continue

        if ordinal_maps and col in ordinal_maps:
            order = ordinal_maps[col]
        else:
            order = sorted(unique_vals, key=str)

        mapping = {v: i for i, v in enumerate(order)}
        df[col] = df[col].map(mapping)

        report[col] = {
            "method": "ordinal",
            "mapping": {str(k): v for k, v in mapping.items()},
            "order": [str(v) for v in order],
        }

    return df, report


# =========================================================================
# 2. FEATURE SCALING
# =========================================================================

def auto_scale(
    df: pd.DataFrame,
    numeric_cols: list[str],
    method: str = "standard",
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Scale numeric features.

    Args:
        method: "standard" (Z-score) or "minmax" (0-1 scaling)
        exclude_cols: columns to skip (e.g. dependent variable)
    """
    if method == "minmax":
        return minmax_scale(df, numeric_cols, exclude_cols)
    return standard_scale(df, numeric_cols, exclude_cols)


def standard_scale(
    df: pd.DataFrame,
    numeric_cols: list[str],
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """StandardScaler: Z-score normalization (mean=0, std=1).

    Best for: Linear Regression, Logistic Regression, SVM, Neural Networks,
    PCA, and any algorithm assuming normally distributed features.
    """
    df = df.copy()
    params: dict[str, dict] = {}
    exclude = set(exclude_cols or [])

    for col in numeric_cols:
        if col in exclude or col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        vals = df[col].dropna()
        if len(vals) < 2:
            continue
        mean = float(vals.mean())
        std = float(vals.std(ddof=0))
        if std == 0:
            continue
        df[col] = (df[col] - mean) / std
        params[col] = {"method": "standard", "mean": round(mean, 6), "std": round(std, 6)}

    return df, params


def minmax_scale(
    df: pd.DataFrame,
    numeric_cols: list[str],
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """MinMaxScaler: scale features to [0, 1] range.

    Best for: KNN, Neural Networks, image processing, algorithms sensitive
    to feature magnitudes. Preserves zero entries in sparse data.
    """
    df = df.copy()
    params: dict[str, dict] = {}
    exclude = set(exclude_cols or [])

    for col in numeric_cols:
        if col in exclude or col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        vals = df[col].dropna()
        if len(vals) < 2:
            continue
        col_min = float(vals.min())
        col_max = float(vals.max())
        range_val = col_max - col_min
        if range_val == 0:
            continue
        df[col] = (df[col] - col_min) / range_val
        params[col] = {"method": "minmax", "min": round(col_min, 6), "max": round(col_max, 6)}

    return df, params


def inverse_scale(
    df: pd.DataFrame,
    scaled_cols: list[str],
    params: dict[str, dict],
) -> pd.DataFrame:
    """Reverse scaling to restore original feature values."""
    df = df.copy()
    for col in scaled_cols:
        if col not in params or col not in df.columns:
            continue
        p = params[col]
        if p["method"] == "standard":
            df[col] = df[col] * p["std"] + p["mean"]
        elif p["method"] == "minmax":
            df[col] = df[col] * (p["max"] - p["min"]) + p["min"]
    return df


# =========================================================================
# 3. FEATURE CREATION
# =========================================================================

def extract_datetime_features(
    df: pd.DataFrame,
    datetime_cols: list[str],
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Extract datetime components as numeric features.

    Default features: year, month, day, day_of_week, hour (if present),
    is_weekend, quarter, week_of_year.
    """
    if features is None:
        features = ["year", "month", "day", "day_of_week", "hour", "is_weekend", "quarter"]

    df = df.copy()
    created: dict[str, list[str]] = {}

    for col in datetime_cols:
        if col not in df.columns:
            continue

        dt = pd.to_datetime(df[col], errors="coerce")
        if dt.isna().sum() == len(dt):
            continue

        new_cols: list[str] = []

        if "year" in features:
            df[f"{col}_year"] = dt.dt.year
            new_cols.append(f"{col}_year")

        if "month" in features:
            df[f"{col}_month"] = dt.dt.month
            new_cols.append(f"{col}_month")

        if "day" in features:
            df[f"{col}_day"] = dt.dt.day
            new_cols.append(f"{col}_day")

        if "day_of_week" in features:
            df[f"{col}_day_of_week"] = dt.dt.dayofweek
            new_cols.append(f"{col}_day_of_week")

        if "hour" in features:
            has_time = dt.dt.hour.nunique() > 1 or dt.dt.hour.max() > 0
            if has_time:
                df[f"{col}_hour"] = dt.dt.hour
                new_cols.append(f"{col}_hour")

        if "is_weekend" in features:
            df[f"{col}_is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
            new_cols.append(f"{col}_is_weekend")

        if "quarter" in features:
            df[f"{col}_quarter"] = dt.dt.quarter
            new_cols.append(f"{col}_quarter")

        if "week_of_year" in features:
            df[f"{col}_week_of_year"] = dt.dt.isocalendar().week.astype(int)
            new_cols.append(f"{col}_week_of_year")

        created[col] = new_cols

    return df, created


def create_ratio_features(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> tuple[pd.DataFrame, list[str]]:
    """Create ratio features from pairs of numeric columns.

    For each pair (a, b), creates column a/b (avoiding division by zero).
    """
    df = df.copy()
    created: list[str] = []

    for col_a, col_b in pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        if not np.issubdtype(df[col_a].dtype, np.number):
            continue
        if not np.issubdtype(df[col_b].dtype, np.number):
            continue

        new_col = f"{col_a}_div_{col_b}"
        denominator = df[col_b].replace(0, np.nan)
        df[new_col] = df[col_a] / denominator
        created.append(new_col)

    return df, created


def create_aggregation_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    group_col: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Create aggregation features: row-level stats across numeric columns.

    If group_col is provided, also computes group-level aggregations.
    """
    df = df.copy()
    created: list[str] = []

    valid_nums = [c for c in numeric_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    if len(valid_nums) < 2:
        return df, created

    # Row-level aggregations
    df["_row_mean"] = df[valid_nums].mean(axis=1)
    df["_row_std"] = df[valid_nums].std(axis=1)
    df["_row_max"] = df[valid_nums].max(axis=1)
    df["_row_min"] = df[valid_nums].min(axis=1)
    df["_row_range"] = df["_row_max"] - df["_row_min"]
    created.extend(["_row_mean", "_row_std", "_row_max", "_row_min", "_row_range"])

    # Group-level aggregations (mean per group for each numeric column)
    if group_col and group_col in df.columns:
        for col in valid_nums:
            group_mean = df.groupby(group_col)[col].transform("mean")
            new_col = f"{col}_group_mean"
            df[new_col] = group_mean
            created.append(new_col)

    return df, created


def create_interaction_features(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> tuple[pd.DataFrame, list[str]]:
    """Create interaction features: product and difference of numeric pairs."""
    df = df.copy()
    created: list[str] = []

    for col_a, col_b in pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        if not np.issubdtype(df[col_a].dtype, np.number):
            continue
        if not np.issubdtype(df[col_b].dtype, np.number):
            continue

        prod_col = f"{col_a}_x_{col_b}"
        diff_col = f"{col_a}_minus_{col_b}"
        df[prod_col] = df[col_a] * df[col_b]
        df[diff_col] = df[col_a] - df[col_b]
        created.extend([prod_col, diff_col])

    return df, created


# =========================================================================
# 4. DIMENSIONALITY REDUCTION & FEATURE SELECTION
# =========================================================================

def filter_correlated_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    threshold: float = 0.95,
) -> tuple[list[str], dict[str, Any]]:
    """Remove features with correlation > threshold (keep the first encountered).

    Returns list of columns to keep and the removal report.
    """
    valid = [c for c in numeric_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    if len(valid) < 2:
        return valid, {"removed": [], "correlations": {}}

    corr_matrix = df[valid].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop: list[str] = []
    high_corr_pairs: dict[str, str] = {}

    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        for c in correlated:
            if c not in to_drop and col not in to_drop:
                to_drop.append(c)
                high_corr_pairs[c] = col

    keep = [c for c in valid if c not in to_drop]
    return keep, {
        "removed": to_drop,
        "kept": keep,
        "high_correlations": high_corr_pairs,
        "threshold": threshold,
    }


def select_by_correlation_with_target(
    df: pd.DataFrame,
    predictor_cols: list[str],
    target_col: str,
    threshold: float = 0.05,
) -> tuple[list[str], dict[str, Any]]:
    """Select features that have at least `threshold` absolute correlation with target."""
    valid = [c for c in predictor_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    if target_col not in df.columns or not valid:
        return valid, {"selected": valid, "removed": [], "correlations": {}}

    target_vals = df[target_col]
    if not np.issubdtype(target_vals.dtype, np.number):
        return valid, {"selected": valid, "removed": [], "correlations": {}}

    correlations: dict[str, float] = {}
    for col in valid:
        r = df[col].corr(target_vals)
        if not np.isnan(r):
            correlations[col] = round(float(r), 4)

    selected = [col for col, r in correlations.items() if abs(r) >= threshold]
    removed = [col for col in valid if col not in selected]

    return selected, {
        "selected": selected,
        "removed": removed,
        "correlations": correlations,
        "threshold": threshold,
    }


def apply_pca(
    df: pd.DataFrame,
    numeric_cols: list[str],
    n_components: int | None = None,
    variance_threshold: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply PCA for dimensionality reduction.

    If n_components is None, uses variance_threshold to auto-select
    the number of components that explain the threshold of total variance.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    valid = [c for c in numeric_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    if len(valid) < 2:
        return df, {"components": 0, "variance_explained": [], "columns_used": valid}

    X = df[valid].dropna()
    if len(X) < 3:
        return df, {"components": 0, "variance_explained": [], "columns_used": valid}

    # Standardize before PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if n_components is None:
        # Auto-select: find minimum components that explain variance_threshold
        pca_full = PCA()
        pca_full.fit(X_scaled)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
        n_components = min(n_components, len(valid))

    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(X_scaled)

    # Build new column names
    new_cols = [f"PC{i+1}" for i in range(n_components)]

    # Replace original columns with PCs
    result_df = df.drop(columns=[c for c in valid if c in df.columns]).copy()
    for i, name in enumerate(new_cols):
        result_df[name] = principal_components[:, i]

    return result_df, {
        "components": n_components,
        "variance_explained": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "cumulative_variance": [round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)],
        "columns_used": valid,
        "new_columns": new_cols,
    }


def select_by_lasso(
    df: pd.DataFrame,
    predictor_cols: list[str],
    target_col: str,
    alpha: float = 0.01,
    max_iter: int = 1000,
) -> tuple[list[str], dict[str, Any]]:
    """Use Lasso (L1) regression to select features with non-zero coefficients.

    Features with zero coefficients are dropped as irrelevant.
    """
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    valid = [c for c in predictor_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    if target_col not in df.columns or len(valid) < 2:
        return valid, {"selected": valid, "removed": [], "coefficients": {}}

    clean = df[valid + [target_col]].dropna()
    if len(clean) < 10:
        return valid, {"selected": valid, "removed": [], "coefficients": {}}

    X = clean[valid].values
    y = clean[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
    lasso.fit(X_scaled, y)

    coefficients = {col: round(float(c), 4) for col, c in zip(valid, lasso.coef_)}
    selected = [col for col, c in zip(valid, lasso.coef_) if abs(c) > 1e-6]
    removed = [col for col in valid if col not in selected]

    return selected, {
        "selected": selected,
        "removed": removed,
        "coefficients": coefficients,
        "alpha": alpha,
    }


def select_by_feature_importance(
    df: pd.DataFrame,
    predictor_cols: list[str],
    target_col: str,
    top_k: int | None = None,
    threshold: float = 0.01,
) -> tuple[list[str], dict[str, Any]]:
    """Use Random Forest feature importance to select top features."""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    valid = [c for c in predictor_cols if c in df.columns and np.issubdtype(df[c].dtype, np.number)]
    if target_col not in df.columns or len(valid) < 2:
        return valid, {"selected": valid, "removed": [], "importances": {}}

    clean = df[valid + [target_col]].dropna()
    if len(clean) < 10:
        return valid, {"selected": valid, "removed": [], "importances": {}}

    X = clean[valid].values
    y = clean[target_col].values

    # Choose classifier vs regressor based on target
    n_unique = len(np.unique(y))
    if n_unique <= 10 and all(v == int(v) for v in y[:100]):
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    rf.fit(X, y)
    importances = {col: round(float(imp), 4) for col, imp in zip(valid, rf.feature_importances_)}

    # Sort by importance
    sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)

    if top_k is not None:
        selected = [col for col, _ in sorted_features[:top_k]]
    else:
        selected = [col for col, imp in sorted_features if imp >= threshold]

    removed = [col for col in valid if col not in selected]

    return selected, {
        "selected": selected,
        "removed": removed,
        "importances": dict(sorted_features),
        "top_k": top_k,
        "threshold": threshold,
    }


def apply_vif_filter(
    df: pd.DataFrame,
    predictor_cols: list[str],
    vif_threshold: float = 10.0,
) -> tuple[list[str], dict[str, Any]]:
    """Iteratively remove features with VIF > threshold (multicollinearity)."""
    from .preprocessing import compute_vif

    cols = list(predictor_cols)
    removed: list[str] = []
    vif_history: list[dict] = []

    while len(cols) > 1:
        vif_results = compute_vif(df, cols)
        vif_map = {r["predictor"]: r["value"] for r in vif_results}
        max_vif_col = max(vif_map, key=vif_map.get)
        max_vif_val = vif_map[max_vif_col]

        vif_history.append({"column": max_vif_col, "vif": round(max_vif_val, 4), "remaining": len(cols)})

        if max_vif_val <= vif_threshold:
            break

        cols.remove(max_vif_col)
        removed.append(max_vif_col)

    return cols, {
        "selected": cols,
        "removed": removed,
        "vif_threshold": vif_threshold,
        "history": vif_history,
    }
