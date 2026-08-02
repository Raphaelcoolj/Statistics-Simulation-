"""Business translation: convert model metrics into actionable business insights.

Translates statistical metrics (R², RMSE, accuracy, F1, etc.) into
plain-English business outcomes, cost-benefit estimates, and recommendations.
"""

from typing import Optional


def translate_regression_metrics(
    metrics: dict,
    dependent: str = "target",
    feature_names: list[str] | None = None,
    coefficients: list[float] | None = None,
    avg_target: float | None = None,
    target_std: float | None = None,
) -> dict:
    """Translate regression metrics into business-understandable outcomes."""
    result: dict = {"insights": [], "confidence": "low", "impact": {}, "summary": ""}

    r2 = metrics.get("r2")
    rmse = metrics.get("rmse")
    mae = metrics.get("mae")

    # --- R² interpretation ---
    if r2 is not None:
        r2_pct = round(r2 * 100, 1)
        if r2 > 0.8:
            quality = "excellent"
            quality_desc = f"The model explains {r2_pct}% of variation in '{dependent}' — very reliable for prediction."
        elif r2 > 0.6:
            quality = "good"
            quality_desc = f"The model captures {r2_pct}% of variation in '{dependent}' — useful for most business decisions."
        elif r2 > 0.4:
            quality = "moderate"
            quality_desc = f"The model explains {r2_pct}% of variation in '{dependent}' — provides directional guidance but has notable uncertainty."
        elif r2 > 0.2:
            quality = "weak"
            quality_desc = f"Only {r2_pct}% of variation in '{dependent}' is explained — predictions should be used with caution."
        else:
            quality = "poor"
            quality_desc = f"The model explains just {r2_pct}% of variation — predictions are unreliable."
        result["insights"].append({"type": "modelQuality", "text": quality_desc, "r2": r2})
        result["impact"]["quality"] = quality

    # --- Error interpretation ---
    if rmse is not None:
        if avg_target is not None and avg_target != 0:
            rmse_pct = abs(rmse / avg_target) * 100
            if rmse_pct < 5:
                err_desc = f"Typical prediction error is {rmse:.2f}, just {rmse_pct:.1f}% of the average '{dependent}' — highly accurate."
            elif rmse_pct < 15:
                err_desc = f"Typical prediction error is {rmse:.2f} ({rmse_pct:.1f}% of average) — acceptable for most use cases."
            elif rmse_pct < 30:
                err_desc = f"Prediction error averages {rmse:.2f} ({rmse_pct:.1f}% of average) — significant room for improvement."
            else:
                err_desc = f"Prediction error is {rmse:.2f} ({rmse_pct:.1f}% of average) — model needs refinement."
        else:
            err_desc = f"Root Mean Squared Error is {rmse:.2f}."
        result["insights"].append({"type": "predictionError", "text": err_desc, "rmse": rmse})
        result["impact"]["errorMagnitude"] = "low" if rmse_pct < 10 else "medium"

    if mae is not None and avg_target is not None and avg_target != 0:
        mae_pct = abs(mae / avg_target) * 100
        result["insights"].append({
            "type": "averageDeviation",
            "text": f"On average, predictions deviate by {mae:.2f} from the actual '{dependent}' ({mae_pct:.1f}% of average).",
            "mae": mae,
        })

    # --- Coefficient interpretations ---
    if coefficients and feature_names:
        top_effects = []
        for i, (fname, coef) in enumerate(zip(feature_names, coefficients)):
            if abs(coef) >= 0.01:
                direction = "increases" if coef > 0 else "decreases"
                top_effects.append({
                    "feature": fname,
                    "effect": f"+1 in {fname} → {coef:+.2f} change in {dependent}",
                    "direction": "positive" if coef > 0 else "negative",
                    "magnitude": "strong" if abs(coef) > 1 else "moderate" if abs(coef) > 0.1 else "small",
                })
        if top_effects:
            top_effects.sort(key=lambda x: abs(float(x["effect"].split("→")[1].split(" ")[0])), reverse=True)
            result["insights"].append({
                "type": "featureEffects",
                "effects": top_effects[:5],
                "text": f"Key drivers: {', '.join(e['feature'] for e in top_effects[:3])}.",
            })

    # --- Baseline comparison ---
    if target_std is not None and rmse is not None and target_std > 0:
        ratio = rmse / target_std
        if ratio < 0.5:
            result["insights"].append({
                "type": "baselineComparison",
                "text": f"Model error is {ratio:.1f}× the natural variation — significantly better than guessing the average.",
            })
        elif ratio < 1.0:
            result["insights"].append({
                "type": "baselineComparison",
                "text": f"Model error is {ratio:.1f}× the natural variation — better than guessing, but modest improvement.",
            })
        else:
            result["insights"].append({
                "type": "baselineComparison",
                "text": f"Model error is {ratio:.1f}× the natural variation — not much better than guessing the average.",
            })

    # --- Confidence level ---
    if r2 is not None:
        if r2 > 0.7:
            result["confidence"] = "high"
        elif r2 > 0.4:
            result["confidence"] = "moderate"
        else:
            result["confidence"] = "low"

    # --- Summary ---
    result["summary"] = _build_regression_summary(result, dependent)
    return result


def translate_classification_metrics(
    metrics: dict,
    dependent: str = "target",
    class_labels: list[str] | None = None,
    prevalence: float | None = None,
) -> dict:
    """Translate classification metrics into business-understandable outcomes."""
    result: dict = {"insights": [], "confidence": "low", "impact": {}, "summary": ""}

    accuracy = metrics.get("accuracy")
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    f1 = metrics.get("f1")
    auc_roc = metrics.get("auc_roc")

    if class_labels is None:
        class_labels = ["class 0", "class 1"]

    # --- Accuracy ---
    if accuracy is not None:
        acc_pct = round(accuracy * 100, 1)
        if accuracy > 0.9:
            acc_desc = f"Correctly identifies {acc_pct}% of cases — highly reliable."
        elif accuracy > 0.8:
            acc_desc = f"Correctly identifies {acc_pct}% of cases — good for most applications."
        elif accuracy > 0.7:
            acc_desc = f"Correctly identifies {acc_pct}% of cases — acceptable for screening or triage."
        elif accuracy > 0.6:
            acc_desc = f"Correctly identifies {acc_pct}% of cases — only marginally better than random."
        else:
            acc_desc = f"Correctly identifies {acc_pct}% of cases — not reliable."
        result["insights"].append({"type": "accuracy", "text": acc_desc, "accuracy": accuracy})
        result["impact"]["accuracy"] = accuracy

    # --- Precision ---
    if precision is not None:
        prec_pct = round(precision * 100, 1)
        if precision > 0.85:
            prec_desc = f"When the model predicts '{class_labels[-1] if len(class_labels) > 1 else 'positive'}', it's correct {prec_pct}% of the time — very trustworthy predictions."
        elif precision > 0.7:
            prec_desc = f"When the model predicts positive, it's correct {prec_pct}% of the time — few false alarms."
        elif precision > 0.5:
            prec_desc = f"Positive predictions are correct only {prec_pct}% of the time — moderate false alarm rate."
        else:
            prec_desc = f"Only {prec_pct}% of positive predictions are correct — high false alarm rate."
        result["insights"].append({"type": "precision", "text": prec_desc, "precision": precision})

    # --- Recall ---
    if recall is not None:
        rec_pct = round(recall * 100, 1)
        if recall > 0.85:
            rec_desc = f"Catches {rec_pct}% of all actual positive cases — very few missed."
        elif recall > 0.7:
            rec_desc = f"Catches {rec_pct}% of positives — some cases slip through."
        elif recall > 0.5:
            rec_desc = f"Catches only {rec_pct}% of positives — many cases are missed."
        else:
            rec_desc = f"Catches just {rec_pct}% of positives — most cases are missed."
        result["insights"].append({"type": "recall", "text": rec_desc, "recall": recall})

    # --- F1 Score ---
    if f1 is not None:
        f1_pct = round(f1 * 100, 1)
        if f1 > 0.85:
            f1_desc = f"Overall model quality (F1) is {f1_pct}% — strong balance of precision and recall."
        elif f1 > 0.7:
            f1_desc = f"Overall quality (F1) is {f1_pct}% — reasonable balance, room to improve."
        elif f1 > 0.5:
            f1_desc = f"Overall quality (F1) is {f1_pct}% — imbalanced performance."
        else:
            f1_desc = f"Overall quality (F1) is just {f1_pct}% — poor model."
        result["insights"].append({"type": "f1", "text": f1_desc, "f1": f1})

    # --- AUC-ROC ---
    if auc_roc is not None:
        if auc_roc > 0.9:
            auc_desc = f"AUC-ROC of {auc_roc:.3f} — excellent discrimination between classes."
        elif auc_roc > 0.8:
            auc_desc = f"AUC-ROC of {auc_roc:.3f} — good discrimination."
        elif auc_roc > 0.7:
            auc_desc = f"AUC-ROC of {auc_roc:.3f} — moderate discrimination."
        elif auc_roc > 0.6:
            auc_desc = f"AUC-ROC of {auc_roc:.3f} — weak discrimination."
        else:
            auc_desc = f"AUC-ROC of {auc_roc:.3f} — no better than random."
        result["insights"].append({"type": "aucRoc", "text": auc_desc, "aucRoc": auc_roc})
        result["impact"]["discrimination"] = auc_roc

    # --- Class balance ---
    if prevalence is not None:
        prev_pct = round(prevalence * 100, 1)
        result["insights"].append({
            "type": "classBalance",
            "text": f"Positive class prevalence is {prev_pct}% — {'balanced' if 30 < prevalence < 70 else 'imbalanced'} dataset.",
            "prevalence": prevalence,
        })

    # --- Confidence ---
    if f1 is not None and auc_roc is not None:
        score = (f1 + auc_roc) / 2
        if score > 0.85:
            result["confidence"] = "high"
        elif score > 0.7:
            result["confidence"] = "moderate"
        else:
            result["confidence"] = "low"
    elif accuracy is not None:
        result["confidence"] = "high" if accuracy > 0.85 else "moderate" if accuracy > 0.7 else "low"

    result["summary"] = _build_classification_summary(result, dependent, class_labels)
    return result


def translate_feature_importance(
    importance: list[dict],
    dependent: str = "target",
    domain: str | None = None,
) -> dict:
    """Translate feature importance into business-relevant insights."""
    result: dict = {"insights": [], "recommendations": [], "topFeatures": []}

    if not importance:
        return result

    total_imp = sum(item.get("importance", item.get("meanAbsShap", 0)) for item in importance)

    for item in importance[:5]:
        fname = item.get("feature", "")
        imp = item.get("importance", item.get("meanAbsShap", 0))
        pct = item.get("percentage", (imp / total_imp * 100) if total_imp > 0 else 0)

        if pct > 30:
            level = "dominant"
            desc = f"'{fname}' is the dominant driver ({pct:.1f}% of model decisions)."
        elif pct > 15:
            level = "major"
            desc = f"'{fname}' is a major factor ({pct:.1f}% of decisions)."
        elif pct > 5:
            level = "moderate"
            desc = f"'{fname}' has moderate influence ({pct:.1f}%)."
        else:
            level = "minor"
            desc = f"'{fname}' has minor influence ({pct:.1f}%)."

        result["topFeatures"].append({
            "feature": fname,
            "percentage": round(pct, 1),
            "level": level,
            "description": desc,
        })

    if domain:
        result["domain"] = domain
        for feat in result["topFeatures"][:3]:
            if feat["level"] in ("dominant", "major"):
                result["recommendations"].append(
                    f"Prioritize monitoring/optimizing '{feat['feature']}' — it {('dominantly' if feat['level'] == 'dominant' else 'significantly')} drives {dependent}."
                )

    result["summary"] = (
        f"Feature importance analysis: {', '.join(f['feature'] for f in result['topFeatures'][:3])} "
        f"are the top drivers of '{dependent}'."
    )
    return result


def generate_recommendations(
    explainability: dict,
    business_translation: dict,
    dependent: str = "target",
) -> list[dict]:
    """Generate actionable recommendations based on all analysis."""
    recs: list[dict] = []
    confidence = business_translation.get("confidence", "low")

    # Model quality recommendations
    quality = business_translation.get("impact", {}).get("quality")
    if quality in ("poor", "weak"):
        recs.append({
            "category": "modelImprovement",
            "priority": "high",
            "action": f"Improve model quality: current {quality} fit ({quality}% of variance explained). Consider adding more features, collecting more data, or trying non-linear models.",
            "rationale": "Low model quality means unreliable predictions.",
        })
    elif quality == "moderate":
        recs.append({
            "category": "modelImprovement",
            "priority": "medium",
            "action": f"Model is moderately predictive. Consider feature engineering or ensemble methods to improve from {quality} to strong fit.",
            "rationale": "Moderate fit can often be improved with targeted engineering.",
        })

    # Feature-based recommendations
    top_features = explainability.get("topFeatures", {}).get("consensus", [])
    if top_features:
        top3 = top_features[:3]
        recs.append({
            "category": "focusArea",
            "priority": "high",
            "action": f"Focus data collection and monitoring on: {', '.join(top3)} — these drive predictions most.",
            "rationale": "Investing in the most impactful features yields highest ROI.",
        })

    # Feature removal recommendation
    features_ranking = explainability.get("consensusRanking", [])
    low_features = [f["feature"] for f in features_ranking if f.get("averageRank", 99) > len(features_ranking) * 0.8]
    if low_features:
        recs.append({
            "category": "simplification",
            "priority": "low",
            "action": f"Consider removing low-impact features ({', '.join(low_features[:3])}) to simplify the model.",
            "rationale": "Simpler models are easier to maintain and deploy.",
        })

    # Precision/recall trade-off
    insights = business_translation.get("insights", [])
    for insight in insights:
        if insight["type"] == "precision" and insight.get("precision", 1) < 0.7:
            recs.append({
                "category": "threshold",
                "priority": "high",
                "action": "Lower the classification threshold to improve recall if missing positives is costly.",
                "rationale": "Current precision is low; consider adjusting decision threshold.",
            })
        if insight["type"] == "recall" and insight.get("recall", 1) < 0.7:
            recs.append({
                "category": "threshold",
                "priority": "high",
                "action": "Raise the classification threshold if false positives are costly.",
                "rationale": "Current recall is low; many positives are being missed.",
            })

    return recs


# =========================================================================
# INTERNALS
# =========================================================================

def _build_regression_summary(result: dict, dependent: str) -> str:
    parts = []
    for insight in result.get("insights", []):
        if "text" in insight:
            parts.append(insight["text"])
    if not parts:
        return f"Regression analysis of '{dependent}' completed."
    return " ".join(parts[:4])


def _build_classification_summary(result: dict, dependent: str, class_labels: list[str]) -> str:
    parts = []
    for insight in result.get("insights", []):
        if "text" in insight:
            parts.append(insight["text"])
    if not parts:
        return f"Classification analysis of '{dependent}' completed."
    return " ".join(parts[:4])
