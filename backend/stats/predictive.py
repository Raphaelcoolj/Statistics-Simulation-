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
from ..models import (
    ModelType, RegressionResult, TestMetrics,
    PredictiveResult, ColumnType,
)
from .preprocessing import preprocess_for_model, one_hot_encode, compute_vif


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
    if not np.array_equal(unique_y, [0, 1]) and not np.array_equal(unique_y, [0]) and not np.array_equal(unique_y, [1]):
        y = np.where(y == np.max(unique_y), 1, 0)

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
) -> PredictiveResult:
    if dependent not in df.columns:
        return PredictiveResult(
            modelType=ModelType.linear,
            regressionResult=RegressionResult(
                modelType=ModelType.linear, dependent=dependent, predictors=predictors,
                coefficients=[], intercept=0.0, predictions=[], note="Dependent column not found",
            ),
        )

    valid_preds = [p for p in predictors if p in df.columns]
    if not valid_preds:
        return PredictiveResult(
            modelType=ModelType.linear,
            regressionResult=RegressionResult(
                modelType=ModelType.linear, dependent=dependent, predictors=predictors,
                coefficients=[], intercept=0.0, predictions=[], note="No valid predictors",
            ),
        )

    dependent_type = column_types.get(dependent, ColumnType.continuous)
    predictor_types = [column_types.get(p, ColumnType.continuous) for p in valid_preds]

    # Select model type
    if model_type_override:
        model_type = model_type_override
    else:
        model_type = select_model(
            dependent_type, predictor_types, len(df), df, dependent, valid_preds
        )

    # Preprocess: impute missing values (using all rows first)
    proc_df = preprocess_for_model(df, dependent, valid_preds)

    # One-hot encode categorical/binary predictors
    cat_preds = [
        p for p in valid_preds
        if column_types.get(p, ColumnType.continuous) in (ColumnType.categorical, ColumnType.binary)
    ]
    encoded_mapping = {}
    if cat_preds:
        proc_df, encoded_mapping = one_hot_encode(proc_df, cat_preds)

    # Build expanded predictor list (categoricals → their encoded columns)
    expanded_predictors = []
    for p in valid_preds:
        if p in encoded_mapping:
            expanded_predictors.extend(encoded_mapping[p]["encoded"])
        else:
            expanded_predictors.append(p)

    # For timeseries, use original data (need temporal order, no train/test split)
    if model_type == ModelType.timeseries and len(valid_preds) >= 1:
        time_col = valid_preds[0]
        ts_result = run_timeseries(proc_df, dependent, time_col)
        ts_result.encodedColumns = encoded_mapping if encoded_mapping else None
        return ts_result

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
    if cat_preds:
        total_encoded = sum(len(v["encoded"]) for v in encoded_mapping.values())
        refs = [f'{col} (reference: {info["reference"]})' for col, info in encoded_mapping.items()]
        notes.append(f"One-hot encoded: {', '.join(refs)}")

    if reg_result.note:
        notes.insert(0, reg_result.note)

    if notes:
        reg_result.note = " | ".join(notes)

    return PredictiveResult(
        modelType=model_type,
        regressionResult=reg_result,
        encodedColumns=encoded_mapping if encoded_mapping else None,
    )
