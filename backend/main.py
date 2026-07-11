"""
FastAPI backend for StatLab analysis.

Run with:
    uvicorn backend.main:app --reload --port 8000

Or from project root:
    python -m uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import json
import io

from .models import (
    AnalysisRequest, AnalyseResponse, AnalysisResult,
    ChartSuggestion, MissingValueReport, DatasetSchema,
    ColumnType, ModelType,
)
from .stats.parser import parse_csv, apply_missing_strategy
from .stats.descriptive import compute_descriptive
from .stats.inferential import (
    compute_correlations, compute_hypothesis_tests, compute_regression,
)
from .stats.predictive import run_predictive

app = FastAPI(title="StatLab Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _generate_chart_suggestions(
    columns: list,
    analyses: dict,
    desc_results: list | None,
    inf_result,
    pred_result,
) -> list[ChartSuggestion]:
    """Mirrors the TS generateChartSuggestions function."""
    suggestions: list[ChartSuggestion] = []
    mode = analyses.get("mode", "manual")

    if mode == "manual":
        desc_config = analyses.get("descriptive")
        if desc_config:
            cols = desc_config.get("columns", [])
            for col_name in cols:
                col = next((c for c in columns if c.name == col_name), None)
                if col and col.type == ColumnType.continuous:
                    suggestions.append(ChartSuggestion(
                        chartType="histogram",
                        title=f"Distribution of {col_name}",
                        reason="Descriptive analysis of continuous variable",
                        column=col_name,
                    ))

        inf_config = analyses.get("inferential", {})
        corr_pairs = inf_config.get("correlationPairs", [])
        for a, b in corr_pairs:
            suggestions.append(ChartSuggestion(
                chartType="scatter",
                title=f"{a} vs {b}",
                reason="Correlation analysis",
                x=a, y=b,
            ))

        regression = inf_config.get("regression")
        if regression:
            dep = regression["dependent"]
            for p in regression.get("predictors", []):
                suggestions.append(ChartSuggestion(
                    chartType="scatter",
                    title=f"Regression: {dep} vs {p}",
                    reason="Regression analysis with trendline",
                    x=p, y=dep,
                ))

        pred_config = analyses.get("predictive")
        if pred_config:
            mt = pred_config.get("modelType")
            if mt == "logistic" or (not mt and pred_result and pred_result.modelType == "logistic"):
                suggestions.append(ChartSuggestion(
                    chartType="confusion_matrix",
                    title="Confusion Matrix",
                    reason="Logistic regression classification results",
                ))
            if mt == "timeseries":
                suggestions.append(ChartSuggestion(
                    chartType="line",
                    title="Time Series Forecast",
                    reason="Time series regression with forecast",
                    x="date",
                    y=pred_config["dependent"],
                ))

    return suggestions


@app.post("/analyse")
async def analyse(
    file: UploadFile = File(...),
    analyses: str = Form(...),
    strategies: str = Form(None),
) -> AnalyseResponse:
    try:
        analyses_data: dict = json.loads(analyses)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid analyses JSON")

    strategies_data: dict | None = None
    if strategies:
        try:
            strategies_data = json.loads(strategies)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid strategies JSON")

    # Parse CSV
    contents = await file.read()
    schema, df, missing_report = parse_csv(contents, file.filename or "uploaded.csv")

    # Apply missing value strategies
    if strategies_data:
        df = apply_missing_strategy(df, strategies_data)
    else:
        auto_strategies = {
            col: info.suggestedStrategy
            for col, info in missing_report.byColumn.items()
        }
        df = apply_missing_strategy(df, auto_strategies)

    result = AnalysisResult(chartSuggestions=[])

    # Descriptive
    desc_config = analyses_data.get("descriptive")
    if desc_config:
        desc_cols = desc_config.get("columns", [])
        desc_results = compute_descriptive(df, desc_cols)
        result.descriptive = desc_results

    # Inferential
    inf_config = analyses_data.get("inferential")
    if inf_config:
        from .models import InferentialResult
        inf_result = InferentialResult()

        corr_pairs = inf_config.get("correlationPairs")
        if corr_pairs:
            pairs = [(p[0], p[1]) for p in corr_pairs]
            inf_result.correlations = compute_correlations(df, pairs)

        hyp_tests = inf_config.get("hypothesisTests")
        if hyp_tests:
            inf_result.hypothesisTests = compute_hypothesis_tests(df, hyp_tests)

        regression = inf_config.get("regression")
        if regression:
            dep = regression["dependent"]
            preds = regression.get("predictors", [])
            inf_result.regression = compute_regression(df, dep, preds)

        result.inferential = inf_result

    # Predictive
    pred_config = analyses_data.get("predictive")
    pred_result = None
    if pred_config:
        dep = pred_config["dependent"]
        preds = pred_config.get("predictors", [])
        mt_override = pred_config.get("modelType")

        col_types = {c.name: c.type for c in schema.columns}
        model_type = ModelType(mt_override) if mt_override else None

        pred_result = run_predictive(df, dep, preds, col_types, model_type)
        result.predictive = pred_result

    # Chart suggestions
    result.chartSuggestions = _generate_chart_suggestions(
        schema.columns, analyses_data,
        result.descriptive, result.inferential, pred_result,
    )

    return AnalyseResponse(
        success=True,
        result=result,
        missingValueReport=missing_report,
        schema=schema,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
