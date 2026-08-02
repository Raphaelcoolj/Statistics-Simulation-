"""Model explainability: feature importance, SHAP/LIME, partial dependence.

Provides model-agnostic and model-specific explanations with graceful
fallback when SHAP/LIME are not installed.
"""

import warnings
from typing import Optional

import numpy as np
from sklearn.inspection import permutation_importance as sk_permutation_importance
from sklearn.inspection import partial_dependence as sk_partial_dependence

warnings.filterwarnings("ignore")

try:
    import shap as _shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime as _lime
    import lime.lime_tabular as _lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


# =========================================================================
# 1. PERMUTATION IMPORTANCE (model-agnostic, always available)
# =========================================================================

def permutation_importance_analysis(
    model,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    is_classification: bool = True,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict:
    """Compute permutation importance (works for any sklearn-compatible model)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    scoring = "accuracy" if is_classification else "r2"

    try:
        result = sk_permutation_importance(
            model, X, y, n_repeats=n_repeats,
            random_state=random_state, scoring=scoring,
        )
        importance = [
            {
                "feature": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                "importance": round(float(result.importances_mean[i]), 4),
                "std": round(float(result.importances_std[i]), 4),
                "rank": 0,
            }
            for i in range(len(feature_names))
        ]
        importance.sort(key=lambda x: x["importance"], reverse=True)
        for rank, item in enumerate(importance):
            item["rank"] = rank + 1
        return {
            "method": "permutation_importance",
            "features": importance,
            "topFeatures": [f["feature"] for f in importance[:5]],
        }
    except Exception as e:
        return {"method": "permutation_importance", "error": str(e), "features": []}


# =========================================================================
# 2. SHAP VALUES (optional, best-in-class explainability)
# =========================================================================

def shap_analysis(
    model,
    X: np.ndarray,
    feature_names: list[str],
    max_samples: int = 200,
) -> dict:
    """Compute SHAP values for global + local explanations.

    Falls back to an empty report if SHAP is not installed.
    """
    if not SHAP_AVAILABLE:
        return {"method": "shap", "available": False, "note": "Install shap: pip install shap"}

    X = np.asarray(X, dtype=float)
    if len(X) > max_samples:
        idx = np.random.default_rng(42).choice(len(X), max_samples, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    try:
        explainer = _shap.TreeExplainer(model) if hasattr(model, "estimators_") else _shap.KernelExplainer(model.predict, _shap.sample(X_sample, min(50, len(X_sample))))
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        mean_abs = np.mean(np.abs(shap_values), axis=0)
        total = mean_abs.sum()
        importance = []
        for i in range(min(len(feature_names), len(mean_abs))):
            imp = float(mean_abs[i])
            importance.append({
                "feature": feature_names[i],
                "meanAbsShap": round(imp, 4),
                "percentage": round(imp / total * 100, 2) if total > 0 else 0,
                "rank": 0,
            })
        importance.sort(key=lambda x: x["meanAbsShap"], reverse=True)
        for rank, item in enumerate(importance):
            item["rank"] = rank + 1

        top5 = importance[:5]

        return {
            "method": "shap",
            "available": True,
            "features": importance,
            "topFeatures": [f["feature"] for f in top5],
            "shapValues": {
                "values": shap_values.tolist() if len(shap_values) <= 500 else shap_values[:500].tolist(),
                "featureNames": feature_names[:len(mean_abs)],
            },
        }
    except Exception as e:
        return {"method": "shap", "available": True, "error": str(e), "features": []}


# =========================================================================
# 3. LIME EXPLANATIONS (optional, per-instance explanations)
# =========================================================================

def lime_explanation(
    model,
    X_train: np.ndarray,
    feature_names: list[str],
    instance: np.ndarray,
    is_classification: bool = True,
    class_names: list[str] | None = None,
    num_features: int = 10,
) -> dict:
    """LIME explanation for a single instance.

    Falls back to coefficient-based explanation if LIME is not installed.
    """
    if not LIME_AVAILABLE:
        return {"method": "lime", "available": False, "note": "Install lime: pip install lime"}

    X_train = np.asarray(X_train, dtype=float)
    instance = np.asarray(instance, dtype=float).reshape(1, -1)

    try:
        explainer = _lime_tabular.LimeTabularExplainer(
            X_train, feature_names=feature_names,
            class_names=class_names,
            mode="classification" if is_classification else "regression",
            random_state=42,
        )

        predict_fn = model.predict_proba if is_classification and hasattr(model, "predict_proba") else model.predict
        explanation = explainer.explain_instance(
            instance.flatten(), predict_fn, num_features=num_features,
        )

        feature_contributions = []
        for feat, weight in explanation.as_list():
            feature_contributions.append({
                "feature": feat,
                "contribution": round(float(weight), 4),
            })

        return {
            "method": "lime",
            "available": True,
            "features": feature_contributions,
            "predictedClass": int(explanation.predict_proba.argmax()) if is_classification else None,
        }
    except Exception as e:
        return {"method": "lime", "available": True, "error": str(e), "features": []}


# =========================================================================
# 4. PARTIAL DEPENDENCE (model-agnostic, always available)
# =========================================================================

def partial_dependence_analysis(
    model,
    X: np.ndarray,
    feature_names: list[str],
    features: list[int] | None = None,
    grid_resolution: int = 50,
) -> dict:
    """Compute partial dependence for selected features."""
    X = np.asarray(X, dtype=float)
    if features is None:
        features = list(range(min(5, X.shape[1])))
    features = [f for f in features if f < X.shape[1]]

    if not features:
        return {"method": "partial_dependence", "features": []}

    try:
        pd_result = sk_partial_dependence(
            model, X, features=features, grid_resolution=grid_resolution,
        )
        result_features = []
        for i, feat_idx in enumerate(features):
            feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"feature_{feat_idx}"
            result_features.append({
                "feature": feat_name,
                "featureIndex": feat_idx,
                "values": pd_result["grid_values"][i].tolist() if hasattr(pd_result["grid_values"][i], "tolist") else list(pd_result["grid_values"][i]),
                "average": pd_result["average"][i].tolist() if hasattr(pd_result["average"][i], "tolist") else list(pd_result["average"][i]),
            })
        return {"method": "partial_dependence", "features": result_features}
    except Exception as e:
        return {"method": "partial_dependence", "error": str(e), "features": []}


# =========================================================================
# 5. COEFFICIENT ANALYSIS (linear/logistic models)
# =========================================================================

def coefficient_analysis(
    model,
    feature_names: list[str],
    is_classification: bool = True,
) -> dict:
    """Extract and interpret model coefficients."""
    if not hasattr(model, "coef_"):
        return {"method": "coefficients", "available": False, "note": "Model has no coef_ attribute"}

    coefs = model.coef_.flatten() if model.coef_.ndim > 1 else model.coef_

    features = []
    total_abs = np.sum(np.abs(coefs))
    for i in range(min(len(feature_names), len(coefs))):
        coef = float(coefs[i])
        abs_coef = abs(coef)
        pct = (abs_coef / total_abs * 100) if total_abs > 0 else 0

        feature_info = {
            "feature": feature_names[i],
            "coefficient": round(coef, 4),
            "absCoefficient": round(abs_coef, 4),
            "percentage": round(pct, 2),
            "direction": "positive" if coef > 0 else "negative",
            "rank": 0,
        }

        if is_classification:
            try:
                odds_ratio = round(float(np.exp(coef)), 4)
                feature_info["oddsRatio"] = odds_ratio
                if odds_ratio > 1.01:
                    feature_info["interpretation"] = f"+1 unit increases odds by {round((odds_ratio - 1) * 100, 1)}%"
                elif odds_ratio < 0.99:
                    feature_info["interpretation"] = f"+1 unit decreases odds by {round((1 - odds_ratio) * 100, 1)}%"
                else:
                    feature_info["interpretation"] = "Negligible effect"
            except Exception:
                pass
        else:
            if abs_coef >= 0.01:
                feature_info["interpretation"] = f"+1 unit in {feature_names[i]} → {coef:+.2f} change in target"
            else:
                feature_info["interpretation"] = "Negligible effect"

        features.append(feature_info)

    features.sort(key=lambda x: x["absCoefficient"], reverse=True)
    for rank, item in enumerate(features):
        item["rank"] = rank + 1

    has_intercept = hasattr(model, "intercept_")
    intercept = float(model.intercept_.flatten()[0]) if has_intercept and hasattr(model.intercept_, "flatten") else (float(model.intercept_) if has_intercept else None)

    return {
        "method": "coefficients",
        "available": True,
        "intercept": round(intercept, 4) if intercept is not None else None,
        "features": features,
        "topFeatures": [f["feature"] for f in features[:5]],
    }


# =========================================================================
# 6. BUILT-IN FEATURE IMPORTANCE (tree-based models)
# =========================================================================

def builtin_feature_importance(
    model,
    feature_names: list[str],
) -> dict:
    """Extract sklearn's native feature_importances_ (tree-based models only)."""
    if not hasattr(model, "feature_importances_"):
        return {"method": "builtin_importance", "available": False}

    imp = model.feature_importances_
    total = imp.sum()
    features = []
    for i in range(min(len(feature_names), len(imp))):
        val = float(imp[i])
        features.append({
            "feature": feature_names[i],
            "importance": round(val, 4),
            "percentage": round(val / total * 100, 2) if total > 0 else 0,
            "rank": 0,
        })
    features.sort(key=lambda x: x["importance"], reverse=True)
    for rank, item in enumerate(features):
        item["rank"] = rank + 1

    return {
        "method": "builtin_importance",
        "available": True,
        "features": features,
        "topFeatures": [f["feature"] for f in features[:5]],
    }


# =========================================================================
# 7. COMPREHENSIVE EXPLAINABILITY REPORT
# =========================================================================

def build_explainability_report(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    is_classification: bool = True,
    model_name: str = "model",
) -> dict:
    """Build a full explainability report combining all available methods."""
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test)

    report: dict = {"methods": {}, "topFeatures": {}, "summary": ""}

    # 1. Built-in importance (fast, always first)
    builtin = builtin_feature_importance(model, feature_names)
    if builtin.get("available"):
        report["methods"]["builtin"] = builtin
        report["topFeatures"]["builtin"] = builtin.get("topFeatures", [])

    # 2. Permutation importance (always works)
    perm = permutation_importance_analysis(model, X_test, y_test, feature_names, is_classification)
    if perm.get("features"):
        report["methods"]["permutation"] = perm
        report["topFeatures"]["permutation"] = perm.get("topFeatures", [])

    # 3. SHAP (if available)
    shap = shap_analysis(model, X_train, feature_names)
    if shap.get("available"):
        report["methods"]["shap"] = shap
        report["topFeatures"]["shap"] = shap.get("topFeatures", [])

    # 4. LIME (if available, for a sample test instance)
    if LIME_AVAILABLE and len(X_test) > 0:
        lime = lime_explanation(model, X_train, feature_names, X_test[0], is_classification)
        if lime.get("available"):
            report["methods"]["lime"] = lime

    # 5. Coefficients (linear/logistic)
    coeff = coefficient_analysis(model, feature_names, is_classification)
    if coeff.get("available"):
        report["methods"]["coefficients"] = coeff
        report["topFeatures"]["coefficients"] = coeff.get("topFeatures", [])

    # 6. Partial dependence for top features
    top_feature_indices = []
    for method in ["shap", "permutation", "builtin", "coefficients"]:
        if method in report["topFeatures"]:
            for fname in report["topFeatures"][method][:3]:
                if fname in feature_names:
                    idx = feature_names.index(fname)
                    if idx not in top_feature_indices:
                        top_feature_indices.append(idx)
            break

    if top_feature_indices:
        pd_result = partial_dependence_analysis(model, X_train, feature_names, top_feature_indices[:4])
        if pd_result.get("features"):
            report["methods"]["partialDependence"] = pd_result

    # Build consensus ranking
    report["consensusRanking"] = _build_consensus_ranking(report["methods"], feature_names)
    report["topFeatures"]["consensus"] = [f["feature"] for f in report["consensusRanking"][:5]]

    # Generate human-readable summary
    report["summary"] = _generate_explainability_summary(report, is_classification, model_name)

    return report


# =========================================================================
# INTERNALS
# =========================================================================

def _build_consensus_ranking(methods: dict, feature_names: list[str]) -> list:
    """Average rankings across all available methods to produce a consensus."""
    rankings: dict[str, list[int]] = {}
    for method_name, method_data in methods.items():
        features = method_data.get("features", [])
        for item in features:
            fname = item.get("feature", "")
            if fname not in rankings:
                rankings[fname] = []
            rank = item.get("rank", len(features))
            rankings[fname].append(rank)

    consensus = []
    for fname, ranks in rankings.items():
        avg_rank = sum(ranks) / len(ranks)
        consensus.append({
            "feature": fname,
            "averageRank": round(avg_rank, 2),
            "nMethods": len(ranks),
        })
    consensus.sort(key=lambda x: x["averageRank"])
    for i, item in enumerate(consensus):
        item["consensusRank"] = i + 1
    return consensus


def _generate_explainability_summary(report: dict, is_classification: bool, model_name: str) -> str:
    """Generate a concise summary of model explainability."""
    parts = [f"Model: {model_name}."]

    consensus = report.get("consensusRanking", [])
    if consensus:
        top = consensus[:3]
        parts.append(
            f"Most important features: {', '.join(f['feature'] for f in top)}."
        )

    methods_used = [k for k in report.get("methods", {}).keys() if k not in ("partialDependence",)]
    if methods_used:
        parts.append(f"Explainability methods used: {', '.join(methods_used)}.")

    shap_data = report.get("methods", {}).get("shap", {})
    if shap_data.get("available") and shap_data.get("features"):
        top = shap_data["features"][0]
        parts.append(
            f"Top feature '{top['feature']}' explains {top.get('percentage', 0):.1f}% of model decisions (SHAP)."
        )

    coeff = report.get("methods", {}).get("coefficients", {})
    if coeff.get("available") and coeff.get("features"):
        top = coeff["features"][0]
        direction = "increases" if top["direction"] == "positive" else "decreases"
        if is_classification and "oddsRatio" in top:
            parts.append(
                f"'{top['feature']}' {direction} odds by {round(abs(top['oddsRatio'] - 1) * 100, 1)}% per unit (OR={top['oddsRatio']})."
            )
        elif "interpretation" in top:
            parts.append(f"'{top['feature']}': {top['interpretation']}.")

    return " ".join(parts)
