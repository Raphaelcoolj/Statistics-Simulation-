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
import os

from .models import (
    AnalysisRequest, AnalyseResponse, AnalysisResult,
    ChartSuggestion, MissingValueReport, DatasetSchema,
    ColumnType, ModelType,
    PreprocessingConfig, CleaningReport,
    FeatureEngineeringConfig, FeatureEngineeringReport,
    ModelTrainingConfig, ModelTrainingReport,
)
from .stats.parser import parse_file, apply_missing_strategy, apply_codebook
from .stats.parser import parse_file, apply_missing_strategy, apply_codebook
from .stats.descriptive import compute_descriptive
from .stats.inferential import (
    compute_correlations, compute_hypothesis_tests, compute_regression,
)
from .stats.predictive import run_predictive
from .stats.model_training import run_model_training
from .stats.cleaning import (
    handle_outliers, standardize_categoricals, standardize_states,
    parse_dates, fix_typos, remove_exact_duplicates, remove_fuzzy_duplicates,
)

app = FastAPI(title="StatLab Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        os.environ.get("FRONTEND_URL", ""),
    ],
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
    codebook: str = Form(None),
    preprocessing: str = Form(None),
    feature_engineering: str = Form(None),
    model_training: str = Form(None),
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

    codebook_data: dict | None = None
    if codebook:
        try:
            codebook_data = json.loads(codebook)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid codebook JSON")

    preproc_config: PreprocessingConfig | None = None
    if preprocessing:
        try:
            preproc_config = PreprocessingConfig(**json.loads(preprocessing))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid preprocessing JSON")

    fe_config: FeatureEngineeringConfig | None = None
    if feature_engineering:
        try:
            fe_config = FeatureEngineeringConfig(**json.loads(feature_engineering))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid feature engineering JSON")

    mt_config: ModelTrainingConfig | None = None
    if model_training:
        try:
            mt_config = ModelTrainingConfig(**json.loads(model_training))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid model training JSON")

    # Parse CSV / Excel
    contents = await file.read()
    schema, df, missing_report = parse_file(contents, file.filename or "uploaded.csv")

    # Attach human-readable labels to coded columns (optional but recommended).
    if codebook_data:
        schema = apply_codebook(schema, codebook_data)

    rows_before = len(df)
    cleaning_report = CleaningReport(rowsBefore=rows_before)

    # --- Apply preprocessing pipeline ---
    if preproc_config:
        cfg = preproc_config

        # 1. Deduplication (do first to reduce dataset size)
        if cfg.removeExactDupes:
            df, dup_report = remove_exact_duplicates(df)
            cleaning_report.duplicatesRemoved = dup_report

        if cfg.removeFuzzyDupes:
            df, fuzzy_report = remove_fuzzy_duplicates(
                df, cfg.fuzzyColumns, cfg.fuzzyThreshold or 0.9,
            )
            cleaning_report.fuzzyDuplicatesRemoved = fuzzy_report

        # 2. Fix inconsistencies
        if cfg.standardizeCase:
            df, cat_report = standardize_categoricals(
                df, cfg.standardizeCase, cfg.standardizeCase,
            )
            cleaning_report.categoricalsStandardized = cat_report

        if cfg.stateColumns:
            df, state_report = standardize_states(df, cfg.stateColumns, cfg.stateFormat or "full")
            cleaning_report.statesStandardized = state_report

        if cfg.parseDates:
            df, date_report = parse_dates(df, cfg.dateColumns)
            cleaning_report.datesParsed = date_report

        if cfg.typoCorrections:
            df, typo_report = fix_typos(df, cfg.typoColumns, cfg.typoCorrections)
            cleaning_report.typosFixed = typo_report

        # 3. Handle outliers
        if cfg.outlierAction and cfg.outlierAction not in ("none", "detect"):
            df, outlier_report = handle_outliers(
                df, cfg.outlierColumns, cfg.outlierMethod or "iqr",
                cfg.outlierAction, cfg.outlierFactor or 1.5,
            )
            cleaning_report.outliersHandled = outlier_report

    # Apply missing value strategies (after dedup/cleaning so strategies apply to clean data)
    if strategies_data:
        df = apply_missing_strategy(df, strategies_data)
    else:
        auto_strategies = {
            col: info.suggestedStrategy
            for col, info in missing_report.byColumn.items()
        }
        df = apply_missing_strategy(df, auto_strategies)

    # Advanced imputation (KNN/Iterative) — applied to all numeric columns with missing values
    if preproc_config and preproc_config.advancedImputation:
        from .stats.parser import apply_knn_imputation, apply_iterative_imputation
        if preproc_config.advancedImputation == "knn":
            df = apply_knn_imputation(df)
        elif preproc_config.advancedImputation == "iterative":
            df = apply_iterative_imputation(df)

    cleaning_report.rowsAfter = len(df)

    # Rebuild schema after cleaning (column types may have changed)
    from .stats.parser import _build_schema
    schema, missing_report = _build_schema(df, schema.fileName)

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
            col_types = {c.name: c.type.value for c in schema.columns}
            inf_result.correlations = compute_correlations(df, pairs, col_types)

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
    fe_report = None
    if pred_config:
        dep = pred_config["dependent"]
        preds = pred_config.get("predictors", [])
        mt_override = pred_config.get("modelType")

        col_types = {c.name: c.type for c in schema.columns}
        model_type = ModelType(mt_override) if mt_override else None

        pred_result, fe_report = run_predictive(df, dep, preds, col_types, model_type, fe_config)
        result.predictive = pred_result

    # Model training / tuning / evaluation (optional)
    model_training_report = None
    if mt_config and mt_config.enabled:
        pred_cfg = analyses_data.get("predictive")
        if not pred_cfg:
            raise HTTPException(
                status_code=400,
                detail="Model training requires a 'predictive' configuration to define target/predictors.",
            )
        mt_dep = pred_cfg["dependent"]
        mt_preds = pred_cfg.get("predictors", [])

        col_types = {c.name: c.type.value for c in schema.columns}
        raw = run_model_training(
            df, mt_dep, mt_preds, col_types,
            config=mt_config.model_dump(),
        )
        model_training_report = ModelTrainingReport(**raw)

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
        cleaningReport=cleaning_report,
        featureEngineeringReport=fe_report,
        modelTrainingReport=model_training_report,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
