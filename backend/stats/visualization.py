"""Model insight visualization: chart data generation for dashboards.

Generates chart-ready data structures for feature importance, partial
dependence, model comparison, and business impact visualizations.
"""


def feature_importance_chart(explainability: dict, top_n: int = 10) -> dict:
    """Build a horizontal bar chart data structure for feature importance.

    Returns chart data compatible with the frontend Recharts library.
    """
    consensus = explainability.get("consensusRanking", [])
    if not consensus:
        return {"chartType": "bar", "title": "Feature Importance", "data": [], "note": "No importance data available"}

    features = consensus[:top_n]
    data = [
        {
            "name": f["feature"],
            "importance": round(1.0 / f["averageRank"], 4) if f["averageRank"] > 0 else 0,
            "rank": f.get("consensusRank", i + 1),
            "methods": f.get("nMethods", 1),
        }
        for i, f in enumerate(features)
    ]

    max_imp = max(d["importance"] for d in data) if data else 1
    for d in data:
        d["normalizedImportance"] = round(d["importance"] / max_imp, 4) if max_imp > 0 else 0

    return {
        "chartType": "bar",
        "title": "Feature Importance (Consensus across methods)",
        "axis": {"x": "Feature", "y": "Relative Importance"},
        "data": data,
    }


def shap_summary_chart(explainability: dict) -> dict:
    """Build a SHAP summary (beeswarm-like) chart data structure."""
    shap_data = explainability.get("methods", {}).get("shap", {})
    if not shap_data.get("available"):
        return {"chartType": "scatter", "title": "SHAP Summary", "data": [], "note": "SHAP not available"}

    features = shap_data.get("features", [])
    raw_values = shap_data.get("shapValues", {})

    if not features:
        return {"chartType": "scatter", "title": "SHAP Summary", "data": []}

    # Build feature importance bar chart (since actual beeswarm needs per-sample values)
    data = [
        {
            "name": f["feature"],
            "meanAbsShap": f.get("meanAbsShap", 0),
            "percentage": f.get("percentage", 0),
            "rank": f.get("rank", 0),
        }
        for f in features[:15]
    ]

    return {
        "chartType": "bar",
        "title": "SHAP Feature Importance",
        "subtitle": "Mean |SHAP value| — impact on model output",
        "axis": {"x": "Feature", "y": "Mean |SHAP value|"},
        "data": data,
    }


def partial_dependence_chart(explainability: dict) -> dict:
    """Build partial dependence plot data for top features."""
    pd_data = explainability.get("methods", {}).get("partialDependence", {})
    if not pd_data.get("features"):
        return {"chartType": "line", "title": "Partial Dependence", "data": [], "note": "No partial dependence data"}

    charts = []
    for feat in pd_data["features"][:4]:
        values = feat.get("values", [])
        average = feat.get("average", [])
        if values and average:
            chart_data = [
                {"x": round(float(v), 4), "y": round(float(a), 4)}
                for v, a in zip(values, average)
            ]
            charts.append({
                "chartType": "line",
                "title": f"Partial Dependence: {feat['feature']}",
                "subtitle": f"How {feat['feature']} affects the predicted outcome",
                "axis": {"x": feat["feature"], "y": "Predicted Outcome"},
                "data": chart_data,
            })

    return {"charts": charts}


def model_comparison_chart(models_report: dict, is_classification: bool = True) -> dict:
    """Build a model comparison bar chart."""
    if not models_report:
        return {"chartType": "bar", "title": "Model Comparison", "data": []}

    data = []
    for name, entry in models_report.items():
        if not isinstance(entry, dict) or "testMetrics" not in entry:
            continue
        test_m = entry["testMetrics"]
        train_m = entry.get("trainMetrics", {})
        item = {"name": name}

        if is_classification:
            item["testAccuracy"] = test_m.get("accuracy", 0)
            item["testF1"] = test_m.get("f1", 0)
            item["testAucRoc"] = test_m.get("auc_roc", 0)
            item["trainAccuracy"] = train_m.get("accuracy", 0)
        else:
            item["testR2"] = test_m.get("r2", 0)
            item["testRmse"] = test_m.get("rmse", 0)
            item["trainR2"] = train_m.get("r2", 0)

        tuning = entry.get("tuning", {})
        item["tuningScore"] = tuning.get("bestScore") or 0
        item["tuningTime"] = tuning.get("duration_sec", 0)
        data.append(item)

    if is_classification:
        return {
            "chartType": "grouped_bar",
            "title": "Model Comparison (Test Set)",
            "axis": {"x": "Model", "y": "Score"},
            "series": [
                {"key": "testAccuracy", "name": "Test Accuracy", "color": "#3b82f6"},
                {"key": "testF1", "name": "Test F1", "color": "#10b981"},
                {"key": "testAucRoc", "name": "Test AUC-ROC", "color": "#f59e0b"},
            ],
            "data": data,
        }
    else:
        return {
            "chartType": "grouped_bar",
            "title": "Model Comparison (Test Set)",
            "axis": {"x": "Model", "y": "Score"},
            "series": [
                {"key": "testR2", "name": "Test R²", "color": "#3b82f6"},
                {"key": "testRmse", "name": "Test RMSE", "color": "#ef4444"},
            ],
            "data": data,
        }


def business_impact_dashboard(business_translation: dict) -> dict:
    """Build a business impact dashboard with KPI cards."""
    cards = []
    insights = business_translation.get("insights", [])
    impact = business_translation.get("impact", {})
    confidence = business_translation.get("confidence", "low")

    # Confidence card
    cards.append({
        "type": "confidence",
        "title": "Model Confidence",
        "value": confidence.upper(),
        "color": "green" if confidence == "high" else "amber" if confidence == "moderate" else "red",
        "detail": f"Confidence level: {confidence}.",
    })

    for insight in insights:
        itype = insight.get("type", "")

        if itype == "modelQuality":
            cards.append({
                "type": "metric",
                "title": "Model Quality",
                "value": f"{insight.get('r2', 0) * 100:.1f}%",
                "color": "green" if insight.get("r2", 0) > 0.6 else "amber" if insight.get("r2", 0) > 0.3 else "red",
                "detail": insight.get("text", ""),
            })

        if itype == "predictionError":
            cards.append({
                "type": "metric",
                "title": "Prediction Error (RMSE)",
                "value": f"{insight.get('rmse', 0):.2f}",
                "color": "green" if impact.get("errorMagnitude") == "low" else "amber",
                "detail": insight.get("text", ""),
            })

        if itype == "accuracy":
            cards.append({
                "type": "metric",
                "title": "Accuracy",
                "value": f"{insight.get('accuracy', 0) * 100:.1f}%",
                "color": "green" if insight.get("accuracy", 0) > 0.8 else "amber" if insight.get("accuracy", 0) > 0.6 else "red",
                "detail": insight.get("text", ""),
            })

        if itype == "aucRoc":
            cards.append({
                "type": "metric",
                "title": "AUC-ROC",
                "value": f"{insight.get('aucRoc', 0):.3f}",
                "color": "green" if insight.get("aucRoc", 0) > 0.8 else "amber" if insight.get("aucRoc", 0) > 0.6 else "red",
                "detail": insight.get("text", ""),
            })

        if itype == "f1":
            cards.append({
                "type": "metric",
                "title": "F1 Score",
                "value": f"{insight.get('f1', 0) * 100:.1f}%",
                "color": "green" if insight.get("f1", 0) > 0.8 else "amber" if insight.get("f1", 0) > 0.6 else "red",
                "detail": insight.get("text", ""),
            })

    return {
        "chartType": "dashboard",
        "title": "Business Impact",
        "cards": cards,
        "summary": business_translation.get("summary", ""),
    }


def residuals_chart(predictions, actuals) -> dict:
    """Build a residuals distribution histogram."""
    if not predictions or not actuals:
        return {"chartType": "histogram", "title": "Residuals Distribution", "data": []}

    predictions = [float(p) for p in predictions]
    actuals = [float(a) for a in actuals]
    residuals = [a - p for a, p in zip(actuals, predictions)]
    n = len(residuals)
    if n < 2:
        return {"chartType": "histogram", "title": "Residuals Distribution", "data": []}

    min_r = min(residuals)
    max_r = max(residuals)
    bin_count = min(max(int(n ** 0.5), 5), 20)
    bin_width = (max_r - min_r) / bin_count if bin_count > 0 else 1

    bins = []
    for i in range(bin_count):
        lo = min_r + i * bin_width
        hi = lo + bin_width
        count = sum(1 for r in residuals if lo <= r < hi or (i == bin_count - 1 and r == hi))
        bins.append({
            "name": f"{lo:.2f}–{hi:.2f}",
            "midpoint": round((lo + hi) / 2, 4),
            "count": count,
            "density": round(count / (n * bin_width), 4) if bin_width > 0 else 0,
        })

    return {
        "chartType": "histogram",
        "title": "Residuals Distribution",
        "subtitle": "Predicted errors should be normally distributed around zero",
        "axis": {"x": "Residual", "y": "Count"},
        "data": bins,
        "stats": {
            "mean": round(sum(residuals) / n, 4),
            "std": round(float((sum((r - sum(residuals) / n) ** 2 for r in residuals) / (n - 1)) ** 0.5), 4) if n > 1 else 0,
            "min": round(min_r, 4),
            "max": round(max_r, 4),
        },
    }


def predicted_vs_actual_chart(predictions, actuals) -> dict:
    """Build a predicted vs actual scatter chart."""
    if not predictions or not actuals:
        return {"chartType": "scatter", "title": "Predicted vs Actual", "data": []}

    predictions = [float(p) for p in predictions]
    actuals = [float(a) for a in actuals]

    data = [
        {"actual": round(float(a), 4), "predicted": round(float(p), 4)}
        for a, p in zip(actuals, predictions)
    ]

    all_vals = [d["actual"] for d in data] + [d["predicted"] for d in data]
    min_val = min(all_vals)
    max_val = max(all_vals)

    return {
        "chartType": "scatter",
        "title": "Predicted vs Actual",
        "subtitle": "Points on the diagonal line indicate perfect predictions",
        "axis": {"x": "Actual", "y": "Predicted"},
        "diagonalLine": {"x1": min_val, "y1": min_val, "x2": max_val, "y2": max_val},
        "data": data,
    }


def confidence_interval_chart(forecast: list[dict]) -> dict:
    """Build a confidence interval chart for time series forecasts."""
    if not forecast:
        return {"chartType": "line", "title": "Forecast with Confidence Intervals", "data": []}

    data = [
        {
            "name": f.get("label", f"Period {i + 1}"),
            "predicted": f.get("predicted", 0),
            "lower": f.get("lower", 0),
            "upper": f.get("upper", 0),
        }
        for i, f in enumerate(forecast)
    ]

    return {
        "chartType": "area_line",
        "title": "Forecast with 95% Confidence Intervals",
        "axis": {"x": "Period", "y": "Value"},
        "series": [
            {"key": "upper", "name": "Upper Bound", "color": "#dbeafe"},
            {"key": "predicted", "name": "Forecast", "color": "#3b82f6"},
            {"key": "lower", "name": "Lower Bound", "color": "#dbeafe"},
        ],
        "data": data,
    }


def build_all_model_charts(
    explainability: dict,
    business_translation: dict,
    models_report: dict,
    predictions: list[float] | None = None,
    actuals: list[float] | None = None,
    forecast: list[dict] | None = None,
    is_classification: bool = True,
) -> dict:
    """Build all model insight charts in one call."""
    charts: dict = {
        "featureImportance": feature_importance_chart(explainability),
        "shapSummary": shap_summary_chart(explainability),
        "partialDependence": partial_dependence_chart(explainability),
        "modelComparison": model_comparison_chart(models_report, is_classification),
        "businessDashboard": business_impact_dashboard(business_translation),
    }

    if predictions and actuals:
        charts["residuals"] = residuals_chart(predictions, actuals)
        charts["predictedVsActual"] = predicted_vs_actual_chart(predictions, actuals)

    if forecast:
        charts["forecast"] = confidence_interval_chart(forecast)

    return charts
