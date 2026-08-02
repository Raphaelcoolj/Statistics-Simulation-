import pandas as pd
import numpy as np
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression as SkLogistic,
    Ridge,
)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split as sk_train_test_split
from models import (
    ModelType, RegressionResult, TestMetrics,
    PredictiveResult, ColumnType, FeatureEngineeringConfig, FeatureEngineeringReport,
)
from stats.preprocessing import preprocess_for_model, one_hot_encode, compute_vif
from stats.feature_engineering import (
    auto_encode, auto_scale,
    extract_datetime_features, create_ratio_features,
    create_aggregation_features, create_interaction_features,
    filter_correlated_features, select_by_correlation_with_target,
    apply_pca, select_by_lasso, select_by_feature_importance,
    apply_vif_filter,
)


def select_model(
    dependent_type: ColumnType,
    predictor_types: list[ColumnType],
    n_rows: int,
    data_df: pd.DataFrame,
    dependent: str,
    predictor_names: list[str],
) -> ModelType:
    if dependent_type == ColumnType.binary:
        return ModelType.logistic

    if len(predictor_types) > 1:
        return ModelType.multiple

    has_datetime = any(p == ColumnType.datetime for p in predictor_types)
    if has_datetime:
        return ModelType.timeseries

    if len(predictor_types) == 1 and len(data_df) >= 10:
        y = data_df[dependent].dropna().astype(float)
        x = data_df[predictor_names[0]].dropna().astype(float)
        if len(y) >= 10 and len(x) >= 10:
            common = data_df[[dependent, predictor_names[0]]].dropna()
            if len(common) >= 10:
                xv = common[predictor_names[0]].astype(float).values.reshape(-1, 1)
                yv = common[dependent].astype(float).values
                reg = LinearRegression().fit(xv, yv)
                r_squared = r2_score(yv, reg.predict(xv))
                if r_squared < 0.6:
                    poly = PolynomialFeatures(degree=2)
                    x_poly = poly.fit_transform(xv)
                    poly_reg = LinearRegression().fit(x_poly, yv)
                    poly_r2 = r2_score(yv, poly_reg.predict(x_poly))
                    if poly_r2 > r_squared + 0.1:
                        return ModelType.polynomial
                    return ModelType.linear
                return ModelType.linear
        return ModelType.linear

    if dependent_type == ColumnType.continuous and len(predictor_types) == 1:
        return ModelType.linear

    if n_rows > 500 or len(predictor_types) > 5:
        return ModelType.randomforest

    return ModelType.linear


def _extract_model_data(
    df: pd.DataFrame, dependent: str, predictors: list[str]
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Extract numeric X, y from df for the given predictors/dependent."""
    cols = [dependent] + predictors
    clean = df[cols].dropna()
    if len(clean) < 3:
        return np.array([]), np.array([]), clean
    y = clean[dependent].astype(float).values
    X = clean[predictors].astype(float).values
    return X, y, clean


def run_linear_regression(
    df: pd.DataFrame, dependent: str, predictor: str
) -> RegressionResult:
    X, y, clean = _extract_model_data(df, dependent, [predictor])
    if len(y) < 10:
        return RegressionResult(
            modelType=ModelType.linear, dependent=dependent, predictors=[predictor],
            coefficients=[], intercept=0.0, predictions=[],
            note="Insufficient data (minimum 10 rows)",
        )

    lr = LinearRegression().fit(X, y)
    intercept = float(lr.intercept_)
    coef = float(lr.coef_[0])
    predictions = lr.predict(X).tolist()
    r_squared = float(r2_score(y, predictions))
    residuals = (y - lr.predict(X)).tolist()
    rmse = float(np.sqrt(mean_squared_error(y, predictions)))

    result = RegressionResult(
        modelType=ModelType.linear,
        dependent=dependent,
        predictors=[predictor],
        coefficients=[coef],
        intercept=intercept,
        rSquared=round(r_squared, 4),
        rmse=round(rmse, 4),
        predictions=predictions,
        residuals=residuals,
    )

    if len(clean) >= 30:
        X_train, X_test, y_train, y_test = sk_train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        lr_test = LinearRegression().fit(X_train, y_train)
        test_preds = lr_test.predict(X_test)
        result.testPredictions = test_preds.tolist()
        result.testMetrics = TestMetrics(
            rSquared=round(r2_score(y_test, test_preds), 4),
            rmse=round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 4),
            sampleSize=len(y_test),
        )

    return result


def run_multiple_regression(
    df: pd.DataFrame, dependent: str, predictors: list[str]
) -> RegressionResult:
    X, y, clean = _extract_model_data(df, dependent, predictors)
    if len(y) < 10:
        return RegressionResult(
            modelType=ModelType.multiple, dependent=dependent, predictors=predictors,
            coefficients=[], intercept=0.0, predictions=[], note="Insufficient data",
        )

    try:
        lr = LinearRegression().fit(X, y)
        note = None
    except np.linalg.LinAlgError:
        lam = 1.0 if len(predictors) > 5 else 0.001
        ridge = Ridge(alpha=lam)
        ridge.fit(X, y)
        intercept = float(ridge.intercept_)
        coefficients = ridge.coef_.tolist()
        predictions = ridge.predict(X).tolist()
        r_squared = float(r2_score(y, predictions))
        rmse = float(np.sqrt(mean_squared_error(y, predictions)))
        note = f"Ridge regression (L2 penalty λ={lam}) used due to singular matrix"
        n = len(y)
        p = len(predictors)
        adjusted_r2 = float(1 - (1 - r_squared) * (n - 1) / (n - p - 1)) if n > p + 1 else r_squared
        result = RegressionResult(
            modelType=ModelType.multiple,
            dependent=dependent, predictors=predictors,
            coefficients=coefficients, intercept=intercept,
            rSquared=round(r_squared, 4), adjustedRSquared=round(adjusted_r2, 4),
            rmse=round(rmse, 4), predictions=predictions,
            note=note,
        )
        result.vif = compute_vif(clean, predictors)
        return result

    intercept = float(lr.intercept_)
    coefficients = lr.coef_.tolist()
    predictions = lr.predict(X).tolist()
    r_squared = float(r2_score(y, predictions))
    rmse = float(np.sqrt(mean_squared_error(y, predictions)))

    n = len(y)
    p = len(predictors)
    adjusted_r2 = float(1 - (1 - r_squared) * (n - 1) / (n - p - 1)) if n > p + 1 else r_squared

    result = RegressionResult(
        modelType=ModelType.multiple,
        dependent=dependent, predictors=predictors,
        coefficients=coefficients, intercept=intercept,
        rSquared=round(r_squared, 4), adjustedRSquared=round(adjusted_r2, 4),
        rmse=round(rmse, 4), predictions=predictions,
        note=note,
    )

    if len(clean) >= 30:
        X_train, X_test, y_train, y_test = sk_train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        lr_test = LinearRegression().fit(X_train, y_train)
        test_preds = lr_test.predict(X_test)
        result.testPredictions = test_preds.tolist()
        result.testMetrics = TestMetrics(
            rSquared=round(r2_score(y_test, test_preds), 4),
            rmse=round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 4),
            sampleSize=len(y_test),
        )

    result.vif = compute_vif(clean, predictors)

    return result


def run_logistic_regression(
    df: pd.DataFrame, dependent: str, predictors: list[str]
) -> RegressionResult:
    X, y, clean = _extract_model_data(df, dependent, predictors)
    if len(y) < 10:
        return RegressionResult(
            modelType=ModelType.logistic, dependent=dependent, predictors=predictors,
            coefficients=[], intercept=0.0, predictions=[], note="Insufficient data",
        )

    # Encode y as 0/1 if not already
    unique_y = np.unique(y)
    if np.array_equal(unique_y, [0, 1]) or np.array_equal(unique_y, [0]) or np.array_equal(unique_y, [1]):
        pass  # Already 0/1 encoded
    elif len(unique_y) == 2:
        # Binary: map min→0, max→1 explicitly
        sorted_vals = sorted(unique_y)
        y = np.where(y == sorted_vals[1], 1, 0)
    elif len(unique_y) > 2:
        # Multi-class: keep only the two most frequent classes
        from collections import Counter
        counts = Counter(y)
        top_two = [val for val, _ in counts.most_common(2)]
        mask = np.isin(y, top_two)
        y = np.where(y == top_two[0], 0, 1)
        # Filter to only the two classes
        X = X[mask]
        y = y[mask]

    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = SkLogistic(max_iter=5000, C=100.0, random_state=42)
    model.fit(X_train, y_train)

    if model.n_iter_[0] >= 5000:
        model = SkLogistic(max_iter=10000, C=10.0, random_state=42)
        model.fit(X_train, y_train)

    intercept = float(model.intercept_[0])
    coefficients = model.coef_[0].tolist()

    train_preds = model.predict(X_train)
    train_probs = model.predict_proba(X_train)[:, 1]
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, test_preds))
    precision = float(precision_score(y_test, test_preds, zero_division=0))
    recall_val = float(recall_score(y_test, test_preds, zero_division=0))
    f1 = float(f1_score(y_test, test_preds, zero_division=0))
    auc = float(roc_auc_score(y_test, test_probs)) if len(np.unique(y_test)) > 1 else 0.5

    train_accuracy = float(accuracy_score(y_train, train_preds))

    # McFadden pseudo-R² on training set
    ll_full = np.sum(y_train * np.log(train_probs + 1e-15) +
                     (1 - y_train) * np.log(1 - train_probs + 1e-15))
    y_mean = np.mean(y_train)
    ll_null = np.sum(y_train * np.log(y_mean + 1e-15) +
                     (1 - y_train) * np.log(1 - y_mean + 1e-15))
    r_squared = float(1 - ll_full / ll_null) if ll_null != 0 else 0.0

    return RegressionResult(
        modelType=ModelType.logistic,
        dependent=dependent, predictors=predictors,
        coefficients=coefficients, intercept=intercept,
        rSquared=round(r_squared, 4),
        accuracy=round(train_accuracy, 4),
        predictions=train_preds.tolist(),
        testPredictions=test_probs.tolist(),
        testMetrics=TestMetrics(
            rSquared=round(r_squared, 4),
            rmse=round(float(np.sqrt(mean_squared_error(y_test, test_probs))), 4),
            accuracy=round(accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall_val, 4),
            f1=round(f1, 4),
            aucRoc=round(auc, 4),
            sampleSize=len(y_test),
        ),
        note=f"Training accuracy: {(train_accuracy * 100):.1f}%",
    )


def run_polynomial_regression(
    df: pd.DataFrame, dependent: str, predictor: str, degree: int = 2
) -> RegressionResult:
    X, y, clean = _extract_model_data(df, dependent, [predictor])
    if len(y) < 10:
        return run_linear_regression(df, dependent, predictor)

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    lr = LinearRegression().fit(X_poly, y)
    predictions = lr.predict(X_poly).tolist()
    r_squared = float(r2_score(y, predictions))
    rmse = float(np.sqrt(mean_squared_error(y, predictions)))
    intercept = float(lr.intercept_)
    coefficients = lr.coef_.tolist()

    result = RegressionResult(
        modelType=ModelType.polynomial,
        dependent=dependent, predictors=[predictor],
        coefficients=coefficients, intercept=intercept,
        rSquared=round(r_squared, 4), rmse=round(rmse, 4),
        predictions=predictions,
        note=f"Polynomial degree: {degree}",
    )

    if len(clean) >= 30:
        X_train, X_test, y_train, y_test = sk_train_test_split(
            X_poly, y, test_size=0.2, random_state=42
        )
        lr_test = LinearRegression().fit(X_train, y_train)
        test_preds = lr_test.predict(X_test)
        result.testPredictions = test_preds.tolist()
        result.testMetrics = TestMetrics(
            rSquared=round(r2_score(y_test, test_preds), 4),
            rmse=round(float(np.sqrt(mean_squared_error(y_test, test_preds))), 4),
            sampleSize=len(y_test),
        )

    return result


def run_timeseries(df: pd.DataFrame, dependent: str, predictor: str) -> PredictiveResult:
    clean = df[[dependent]].dropna().reset_index(drop=True)
    if len(clean) < 5:
        return PredictiveResult(
            modelType=ModelType.timeseries,
            regressionResult=RegressionResult(
                modelType=ModelType.timeseries, dependent=dependent, predictors=[predictor],
                coefficients=[], intercept=0.0, predictions=[], note="Insufficient data",
            ),
        )

    y = clean[dependent].astype(float).values
    X = np.arange(len(y)).reshape(-1, 1)
    lr = LinearRegression().fit(X, y)
    predictions = lr.predict(X).tolist()
    r_squared = float(r2_score(y, predictions))
    rmse = float(np.sqrt(mean_squared_error(y, predictions)))
    intercept = float(lr.intercept_)
    coef = float(lr.coef_[0])

    future_X = np.arange(len(y), len(y) + 5).reshape(-1, 1)
    future_preds = lr.predict(future_X)
    residuals_std = float(np.std(y - lr.predict(X), ddof=1)) if len(y) > 1 else 0.0
    ci = 1.96 * residuals_std

    forecast = [
        {
            "label": f"Period {len(y) + i + 1}",
            "predicted": round(float(fp), 4),
            "lower": round(float(fp - ci), 4),
            "upper": round(float(fp + ci), 4),
        }
        for i, fp in enumerate(future_preds)
    ]

    reg_result = RegressionResult(
        modelType=ModelType.timeseries,
        dependent=dependent, predictors=[predictor],
        coefficients=[coef], intercept=intercept,
        rSquared=round(r_squared, 4), rmse=round(rmse, 4),
        predictions=predictions,
    )

    return PredictiveResult(
        modelType=ModelType.timeseries,
        regressionResult=reg_result,
        forecast=forecast,
    )


def run_random_forest(df: pd.DataFrame, dependent: str, predictors: list[str]) -> RegressionResult:
    X, y, clean = _extract_model_data(df, dependent, predictors)
    if len(y) < 30:
        return RegressionResult(
            modelType=ModelType.randomforest, dependent=dependent, predictors=predictors,
            coefficients=[], intercept=0.0, predictions=[], note="Insufficient data (min 30 rows)",
        )

    is_classification = clean[dependent].nunique() <= 2

    X_train, X_test, y_train, y_test = sk_train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if is_classification:
        rf = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        train_preds = rf.predict(X_train)
        test_preds = rf.predict(X_test)
        accuracy = float(accuracy_score(y_test, test_preds))
        precision = float(precision_score(y_test, test_preds, zero_division=0))
        recall_val = float(recall_score(y_test, test_preds, zero_division=0))
        f1 = float(f1_score(y_test, test_preds, zero_division=0))
        r2 = float(r2_score(y_test, test_preds))
        rmse = None
        test_metrics = TestMetrics(
            rSquared=round(r2, 4), accuracy=round(accuracy, 4),
            precision=round(precision, 4), recall=round(recall_val, 4),
            f1=round(f1, 4), sampleSize=len(y_test),
        )
    else:
        rf = RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        )
        rf.fit(X_train, y_train)
        train_preds = rf.predict(X_train)
        test_preds = rf.predict(X_test)
        r2 = float(r2_score(y_test, test_preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
        test_metrics = TestMetrics(
            rSquared=round(r2, 4), rmse=round(rmse, 4), sampleSize=len(y_test),
        )

    imp = rf.feature_importances_
    total = imp.sum()
    if total > 0:
        imp = imp / total
    feature_importance = [
        {"feature": predictors[i], "importance": round(float(imp[i]), 4)}
        for i in range(len(predictors))
    ]
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)

    return RegressionResult(
        modelType=ModelType.randomforest,
        dependent=dependent, predictors=predictors,
        coefficients=[], intercept=0.0,
        rSquared=round(r2, 4), rmse=round(rmse, 4) if rmse is not None else None,
        predictions=train_preds.tolist(),
        testPredictions=test_preds.tolist(),
        testMetrics=test_metrics,
        featureImportance=feature_importance,
    )


def run_predictive(
    df: pd.DataFrame,
    dependent: str,
    predictors: list[str],
    column_types: dict[str, ColumnType],
    model_type_override: ModelType | None = None,
    fe_config: FeatureEngineeringConfig | None = None,
) -> tuple[PredictiveResult, FeatureEngineeringReport]:
    report = FeatureEngineeringReport()

    if dependent not in df.columns:
        return PredictiveResult(
            modelType=ModelType.linear,
            regressionResult=RegressionResult(
                modelType=ModelType.linear, dependent=dependent, predictors=predictors,
                coefficients=[], intercept=0.0, predictions=[], note="Dependent column not found",
            ),
        ), report

    valid_preds = [p for p in predictors if p in df.columns]
    if not valid_preds:
        return PredictiveResult(
            modelType=ModelType.linear,
            regressionResult=RegressionResult(
                modelType=ModelType.linear, dependent=dependent, predictors=predictors,
                coefficients=[], intercept=0.0, predictions=[], note="No valid predictors",
            ),
        ), report

    dependent_type = column_types.get(dependent, ColumnType.continuous)
    predictor_types = [column_types.get(p, ColumnType.continuous) for p in valid_preds]

    # Select model type
    if model_type_override:
        model_type = model_type_override
    else:
        model_type = select_model(
            dependent_type, predictor_types, len(df), df, dependent, valid_preds
        )

    # Preprocess: impute missing values
    proc_df = preprocess_for_model(df, dependent, valid_preds)

    # --- Feature Engineering Pipeline ---
    cols_before = len(proc_df.columns)

    # 1. Feature creation (datetime, ratios, interactions, aggregations)
    if fe_config:
        if fe_config.datetimeColumns:
            proc_df, dt_created = extract_datetime_features(
                proc_df, fe_config.datetimeColumns, fe_config.datetimeFeatures,
            )
            report.datetimeFeatures = dt_created
            # Add extracted datetime cols to predictors
            for orig, new_cols in dt_created.items():
                valid_preds.extend(c for c in new_cols if c in proc_df.columns)

        if fe_config.ratioPairs:
            proc_df, ratio_cols = create_ratio_features(proc_df, fe_config.ratioPairs)
            report.ratioFeatures = ratio_cols
            valid_preds.extend(c for c in ratio_cols if c in proc_df.columns)

        if fe_config.interactionPairs:
            proc_df, inter_cols = create_interaction_features(proc_df, fe_config.interactionPairs)
            report.interactionFeatures = inter_cols
            valid_preds.extend(c for c in inter_cols if c in proc_df.columns)

        if fe_config.aggregationColumns:
            proc_df, agg_cols = create_aggregation_features(
                proc_df, fe_config.aggregationColumns, fe_config.aggregationGroupBy,
            )
            report.aggregationFeatures = agg_cols
            valid_preds.extend(c for c in agg_cols if c in proc_df.columns)

    # 2. Identify categorical and numeric predictors
    cat_preds = [
        p for p in valid_preds
        if p in proc_df.columns and column_types.get(p, ColumnType.continuous) in (ColumnType.categorical, ColumnType.binary)
    ]
    num_preds = [p for p in valid_preds if p not in cat_preds and p in proc_df.columns]

    # 3. Encoding
    encoded_mapping = {}
    if fe_config and fe_config.encodingStrategy:
        strategy = fe_config.encodingStrategy
        enc_cols = fe_config.encodingColumns or cat_preds
        if strategy == "auto":
            proc_df, enc_report = auto_encode(proc_df, enc_cols, dependent)
        elif strategy == "target" and dependent in proc_df.columns:
            from stats.feature_engineering import target_encode
            proc_df, enc_report = target_encode(proc_df, enc_cols, dependent)
        elif strategy == "ordinal":
            from stats.feature_engineering import ordinal_encode
            proc_df, enc_report = ordinal_encode(proc_df, enc_cols, fe_config.ordinalMaps)
        else:
            # Default: one-hot
            proc_df, enc_report = one_hot_encode(proc_df, enc_cols)
        report.encoding = enc_report
    elif cat_preds:
        # Auto-encode categoricals (one-hot for low cardinality, target for high)
        proc_df, enc_report = auto_encode(proc_df, cat_preds, dependent)
        report.encoding = enc_report

    # Build expanded predictor list after encoding
    expanded_predictors = []
    for p in valid_preds:
        if p in proc_df.columns:
            expanded_predictors.append(p)
    # Add any new columns from encoding
    for col in proc_df.columns:
        if col not in expanded_predictors and col != dependent and col in proc_df.columns:
            # Check if this is an encoded column derived from a predictor
            if any(col.startswith(cp + "_") or col.startswith(cp + ".") for cp in cat_preds):
                expanded_predictors.append(col)

    # Remove duplicates while preserving order
    seen = set()
    unique_predictors = []
    for p in expanded_predictors:
        if p not in seen and p in proc_df.columns:
            seen.add(p)
            unique_predictors.append(p)
    expanded_predictors = unique_predictors

    # 4. Feature selection (before scaling)
    if fe_config and expanded_predictors:
        if fe_config.removeCorrelated:
            keep_cols, corr_report = filter_correlated_features(
                proc_df, expanded_predictors, fe_config.correlationThreshold or 0.95,
            )
            expanded_predictors = [p for p in keep_cols if p in proc_df.columns]
            report.correlatedFilter = corr_report

        if fe_config.targetCorrelationFilter and dependent in proc_df.columns:
            selected, tcorr_report = select_by_correlation_with_target(
                proc_df, expanded_predictors, dependent,
                fe_config.targetCorrelationThreshold or 0.05,
            )
            expanded_predictors = [p for p in selected if p in proc_df.columns]
            report.targetCorrelationFilter = tcorr_report

        if fe_config.applyVifFilter and len(expanded_predictors) > 2:
            selected, vif_report = apply_vif_filter(
                proc_df, expanded_predictors, fe_config.vifThreshold or 10.0,
            )
            expanded_predictors = [p for p in selected if p in proc_df.columns]
            report.vifFilter = vif_report

        if fe_config.applyLassoSelection and dependent in proc_df.columns:
            selected, lasso_report = select_by_lasso(
                proc_df, expanded_predictors, dependent, fe_config.lassoAlpha or 0.01,
            )
            expanded_predictors = [p for p in selected if p in proc_df.columns]
            report.lassoSelection = lasso_report

        if fe_config.applyPca:
            proc_df, pca_report = apply_pca(
                proc_df, expanded_predictors, variance_threshold=fe_config.pcaVarianceThreshold or 0.95,
            )
            expanded_predictors = [c for c in pca_report.get("new_columns", []) if c in proc_df.columns]
            report.pcaResult = pca_report

        if fe_config.applyFeatureImportance and dependent in proc_df.columns:
            selected, fi_report = select_by_feature_importance(
                proc_df, expanded_predictors, dependent,
                top_k=fe_config.featureImportanceTopK,
            )
            expanded_predictors = [p for p in selected if p in proc_df.columns]
            report.featureImportanceSelection = fi_report

    # 5. Scaling (after selection, before model training)
    if fe_config and fe_config.scalingMethod:
        exclude = [dependent] + (fe_config.scalingExclude or [])
        proc_df, scale_params = auto_scale(
            proc_df, expanded_predictors, fe_config.scalingMethod, exclude,
        )
        report.scaling = scale_params

    report.columnsBefore = cols_before
    report.columnsAfter = len(proc_df.columns)

    # For timeseries, use original data (need temporal order)
    if model_type == ModelType.timeseries and len(valid_preds) >= 1:
        time_col = valid_preds[0]
        ts_result = run_timeseries(proc_df, dependent, time_col)
        ts_result.encodedColumns = report.encoding if report.encoding else None
        return ts_result, report

    # Run the appropriate model
    if model_type == ModelType.linear and len(expanded_predictors) == 1:
        reg_result = run_linear_regression(proc_df, dependent, expanded_predictors[0])
    elif model_type == ModelType.polynomial and len(expanded_predictors) == 1:
        reg_result = run_polynomial_regression(proc_df, dependent, expanded_predictors[0])
    elif model_type == ModelType.logistic:
        reg_result = run_logistic_regression(proc_df, dependent, expanded_predictors)
    elif model_type == ModelType.randomforest:
        reg_result = run_random_forest(proc_df, dependent, expanded_predictors)
    else:
        reg_result = run_multiple_regression(proc_df, dependent, expanded_predictors)

    # Build preprocessing notes
    notes = []
    if report.encoding:
        encoded_cols = []
        for col, info in report.encoding.items():
            if info.get("method") == "one_hot":
                encoded_cols.append(f'{col} (one-hot, ref: {info.get("reference", "?")})')
            elif info.get("method") == "target":
                encoded_cols.append(f'{col} (target encoded)')
            elif info.get("method") == "ordinal":
                encoded_cols.append(f'{col} (ordinal)')
        if encoded_cols:
            notes.append(f"Encoded: {', '.join(encoded_cols)}")

    if report.scaling:
        notes.append(f"Scaled: {list(report.scaling.keys())}")

    if report.pcaResult and report.pcaResult.get("components", 0) > 0:
        notes.append(f"PCA: {report.pcaResult['components']} components ({sum(report.pcaResult.get('variance_explained', [])):.1%} variance)")

    if reg_result.note:
        notes.insert(0, reg_result.note)

    if notes:
        reg_result.note = " | ".join(notes)

    return PredictiveResult(
        modelType=model_type,
        regressionResult=reg_result,
        encodedColumns=report.encoding if report.encoding else None,
    ), report
