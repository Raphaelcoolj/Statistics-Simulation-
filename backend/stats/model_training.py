"""Model training, tuning, and evaluation.

Trains and benchmarks ML algorithms with proper train/validation/test
splitting, cross-validation, hyperparameter tuning, and business-relevant
evaluation metrics.
"""

import time
import warnings
from typing import Optional

import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, mean_absolute_error, matthews_corrcoef,
)
from sklearn.model_selection import (
    train_test_split as sk_train_test_split,
    cross_val_score, StratifiedKFold, KFold, GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.neural_network import MLPClassifier, MLPRegressor

warnings.filterwarnings("ignore")

# Optional libraries (guarded imports, gracefully absent if not installed)
try:
    import xgboost as _xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as _lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


# =========================================================================
# 1. DATA SPLITTING (no data leakage)
# =========================================================================

def split_data(
    X,
    y,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
) -> dict:
    """Split into Train / Validation / Test.

    Test set is carved out first so that all transforms/tuning only touch
    train/validation, avoiding leakage. Validation is a subset of the rest.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    is_cls = _is_classification(y)

    # First: train+val  vs  test
    strat1 = y if is_cls else None
    X_rest, X_test, y_rest, y_test = sk_train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat1,
    )

    if val_size > 0:
        val_frac = val_size / (1 - test_size)
        if len(y_rest) > 1:
            strat2 = y_rest if is_cls else None
            X_train, X_val, y_train, y_val = sk_train_test_split(
                X_rest, y_rest, test_size=val_frac,
                random_state=random_state, stratify=strat2,
            )
        else:
            X_train, X_val, y_train, y_val = X_rest, np.array([]), y_rest, np.array([])
    else:
        X_train, X_val, y_train, y_val = X_rest, np.array([]), y_rest, np.array([])

    return {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }


def _is_classification(y) -> bool:
    y = np.asarray(y)
    if len(y) == 0:
        return True
    uniq = np.asarray(np.unique(y)).astype(float)
    # A low-cardinality target that only holds small integers / 0-1 values is
    # treated as a classification problem.
    if len(uniq) <= 10:
        finite = uniq[np.isfinite(uniq)]
        if len(finite) > 0 and np.all(finite == np.floor(finite)):
            # only treat as classification when 0/1 or small integer codes
            if finite.max() - finite.min() <= 10:
                return True
    return False


def build_cv(y, n_folds: int = 5, is_classification: bool = True, random_state: int = 42):
    """Return a Stratified (if classification) or plain K-Fold splitter."""
    y = np.asarray(y)
    if is_classification and len(np.unique(y)) >= 2 and len(y) >= n_folds:
        return StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    return KFold(n_splits=n_folds, shuffle=True, random_state=random_state)


def cross_validate(model, X, y, is_classification: bool, n_folds: int = 5, scoring=None) -> dict:
    """Cross-validate a model, returning per-fold scores and summary."""
    X = np.asarray(X)
    y = np.asarray(y)
    is_cls = is_classification if is_classification else _is_classification(y)
    if scoring is None:
        scoring = (
            "roc_auc" if is_cls and len(np.unique(y)) == 2 else
            "accuracy" if is_cls else "r2"
        )
    cv = build_cv(y, n_folds, is_cls)
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    return {
        "scores": [round(float(s), 4) for s in scores],
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
    }


# =========================================================================
# 2. EVALUATION METRICS
# =========================================================================

def evaluate_predictions(y_true, y_pred, is_classification: bool, y_prob=None) -> dict:
    """Compute a metric set for actual vs predicted values.

    is_classification True -> accuracy/precision/recall/f1/(AUC/MCC)
    else                       -> R2/RMSE/MAE/MSE
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics: dict = {}

    if is_classification:
        n_cls = len(np.unique(y_true))
        metrics["accuracy"] = round(float(accuracy_score(y_true, y_pred)), 4)
        avg = "binary" if n_cls == 2 else "macro"
        metrics["precision"] = round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4)
        metrics["recall"] = round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4)
        metrics["f1"] = round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4)
        try:
            metrics["matthews_corrcoef"] = round(float(matthews_corrcoef(y_true, y_pred)), 4)
        except Exception:
            pass
        if y_prob is not None and n_cls == 2:
            try:
                metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_prob[:, 1])), 4)
            except Exception:
                pass
    else:
        metrics["r2"] = round(float(r2_score(y_true, y_pred)), 4)
        metrics["mse"] = round(float(mean_squared_error(y_true, y_pred)), 4)
        metrics["rmse"] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
        metrics["mae"] = round(float(mean_absolute_error(y_true, y_pred)), 4)

    metrics["n"] = len(y_true)
    return metrics


# =========================================================================
# 3. BASELINE MODELS
# =========================================================================

def _fit_baseline(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def train_baselines(X_train, y_train, X_test, y_test, is_classification: bool) -> dict:
    """Train a trivial dummy baseline and a linear/logistic benchmark."""
    baselines: dict = {}

    # Dummy baseline (predicts most-frequent class / mean)
    try:
        dummy = DummyClassifier(strategy="most_frequent") if is_classification else DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        preds = dummy.predict(X_test)
        prob = dummy.predict_proba(X_test) if is_classification and hasattr(dummy, "predict_proba") else None
        baselines["dummy"] = evaluate_predictions(y_test, preds, is_classification, prob)
    except Exception as e:  # pragma: no cover
        baselines["dummy"] = {"error": str(e)}

    # Linear / Logistic regression benchmark
    try:
        if is_classification:
            lin = LogisticRegression(max_iter=2000, random_state=42)
            lin.fit(X_train, y_train)
            preds = lin.predict(X_test)
            prob = lin.predict_proba(X_test) if len(np.unique(y_test)) == 2 else None
        else:
            lin = LinearRegression()
            lin.fit(X_train, y_train)
            preds = lin.predict(X_test)
            prob = None
        baselines["linear_logistic"] = evaluate_predictions(y_test, preds, is_classification, prob)
    except Exception as e:  # pragma: no cover
        baselines["linear_logistic"] = {"error": str(e)}

    return baselines


# =========================================================================
# 4. MODEL DEFINITIONS
# =========================================================================

def available_models(is_classification: bool) -> dict:
    """Return name -> default-parameter model (with optional libs guarded)."""
    if is_classification:
        models = {
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "logistic": LogisticRegression(max_iter=2000, random_state=42),
            "neural_network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=42, early_stopping=True),
        }
        if XGBOOST_AVAILABLE:
            models["xgboost"] = _xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
        if LIGHTGBM_AVAILABLE:
            models["lightgbm"] = _lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    else:
        models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "ridge": Ridge(alpha=1.0),
            "neural_network": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42, early_stopping=True),
        }
        if XGBOOST_AVAILABLE:
            models["xgboost"] = _xgb.XGBRegressor(n_estimators=100, random_state=42)
        if LIGHTGBM_AVAILABLE:
            models["lightgbm"] = _lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

    return models


def hyperparameter_grid(name: str, is_classification: bool) -> dict:
    """Return a small search space for a given model name."""
    grids = {
        "random_forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "max_features": ["sqrt", "log2"],
        },
        "logistic": {"C": [0.01, 0.1, 1.0, 10.0], "max_iter": [1000, 2000]},
        "ridge": {"alpha": [0.01, 0.1, 1.0, 10.0]},
        "neural_network": {
            "hidden_layer_sizes": [(32,), (64,), (32, 16)],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "adaptive"],
        },
    }
    if XGBOOST_AVAILABLE:
        grids["xgboost"] = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1],
        }
    if LIGHTGBM_AVAILABLE:
        grids["lightgbm"] = {
            "n_estimators": [50, 100, 200],
            "num_leaves": [15, 31, 63],
            "learning_rate": [0.01, 0.05, 0.1],
        }
    return grids.get(name, {})


def _build_model(name: str, params: dict, is_classification: bool):
    """Instantiate a fresh estimator for `name` with `params` applied."""
    factory = {
        "random_forest": (RandomForestClassifier if is_classification else RandomForestRegressor),
        "logistic": LogisticRegression,
        "ridge": Ridge,
        "neural_network": (MLPClassifier if is_classification else MLPRegressor),
    }
    if name in ("xgboost",) and XGBOOST_AVAILABLE:
        return _xgb.XGBClassifier(**params) if is_classification else _xgb.XGBRegressor(**params)
    if name in ("lightgbm",) and LIGHTGBM_AVAILABLE:
        return _lgb.LGBMClassifier(**params) if is_classification else _lgb.LGBMRegressor(**params)

    cls = factory.get(name)
    if cls is None:
        raise ValueError(f"Unknown model: {name}")
    return cls(**params)


# =========================================================================
# 5. HYPERPARAMETER TUNING
# =========================================================================

def tune_model(
    name: str,
    is_classification: bool,
    X_train, y_train,
    method: str = "grid",
    n_iter: int = 15,
    n_folds: int = 3,
    scoring=None,
    random_state: int = 42,
) -> dict:
    """Tune a model via grid / random / optuna.

    Returns {'method', 'best_model', 'best_params', 'best_score',
             'cv', 'duration_sec', 'note'?}
    """
    y_train = np.asarray(y_train)
    if scoring is None:
        scoring = (
            "roc_auc" if is_classification and len(np.unique(y_train)) == 2 else
            "accuracy" if is_classification else "r2"
        )

    grid = hyperparameter_grid(name, is_classification)
    cv = build_cv(y_train, n_folds, is_classification, random_state)
    start = time.time()

    if method != "none" and grid:
        base = _build_model(name, {}, is_classification)
        if method == "grid":
            search = GridSearchCV(base, grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
        else:
            search = RandomizedSearchCV(
                base, grid, n_iter=n_iter, cv=cv, scoring=scoring,
                n_jobs=-1, random_state=random_state, refit=True,
            )
        search.fit(X_train, y_train)
        return {
            "method": method,
            "best_model": search.best_estimator_,
            "best_params": search.best_params_,
            "best_score": round(float(search.best_score_), 4),
            "cv_folds": n_folds,
            "duration_sec": round(time.time() - start, 2),
        }

    # No grid / no tuning: train defaults directly
    if method == "bayesian" and not OPTUNA_AVAILABLE:
        note = "Optuna not installed; used default params."
    else:
        note = None
    model = _build_model(name, {}, is_classification)
    model.fit(X_train, y_train)
    result = {
        "method": method if grid else "none",
        "best_model": model,
        "best_params": {},
        "best_score": None,
        "cv_folds": n_folds,
        "duration_sec": round(time.time() - start, 2),
    }
    if note:
        result["note"] = note
    return result


# =========================================================================
# 6. FULL EXPERIMENT PIPELINE
# =========================================================================

def run_model_experiment(
    X,
    y,
    is_classification: Optional[bool] = None,
    enabled_models: Optional[list] = None,
    tuning_method: str = "random",
    tuning_iterations: int = 15,
    cv_folds: int = 5,
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
    feature_names: Optional[list] = None,
) -> dict:
    """Split, train baselines, tune+eval a set of models, pick the best."""
    X = np.asarray(X)
    y = np.asarray(y)
    if is_classification is None:
        is_classification = _is_classification(y)

    split = split_data(X, y, test_size, val_size, random_state)
    X_train, X_val, X_test = split["X_train"], split["X_val"], split["X_test"]
    y_train, y_val, y_test = split["y_train"], split["y_val"], split["y_test"]

    report = {
        "problemType": "classification" if is_classification else "regression",
        "split": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
            "testSize": test_size,
            "valSize": val_size,
        },
        "baselines": train_baselines(X_train, y_train, X_test, y_test, is_classification),
        "models": {},
        "bestModel": None,
    }

    candidates = available_models(is_classification)
    if enabled_models:
        candidates = {k: v for k, v in candidates.items() if k in enabled_models}
    if not candidates:
        report["error"] = "No candidate models available (check installed libs)."
        return report

    for name in candidates:
        entry = {}
        try:
            tuned = tune_model(
                name, is_classification, X_train, y_train,
                method=tuning_method, n_iter=tuning_iterations,
                n_folds=cv_folds, random_state=random_state,
            )
            model = tuned["best_model"]
            entry["tuning"] = {k: v for k, v in tuned.items() if k != "best_model"}
            entry["trainedModel"] = model  # keep for explainability (not serialized)

            entry["trainMetrics"] = evaluate_predictions(
                y_train, model.predict(X_train), is_classification,
                _prob_of(model, X_train) if is_classification else None,
            )
            entry["testMetrics"] = evaluate_predictions(
                y_test, model.predict(X_test), is_classification,
                _prob_of(model, X_test) if is_classification else None,
            )

            entry["valMetrics"] = None
            if len(X_val) > 0:
                entry["valMetrics"] = evaluate_predictions(
                    y_val, model.predict(X_val), is_classification,
                    _prob_of(model, X_val) if is_classification else None,
                )

            entry["crossValidation"] = cross_validate(
                model, X_train, y_train, is_classification,
                n_folds=min(cv_folds, max(2, len(y_train))),
            )

            entry["testMape"] = _tracking(y_test, model.predict(X_test)) if not is_classification else None
            report["models"][name] = entry
        except Exception as e:  # pragma: no cover
            report["models"][name] = {"error": str(e)}

    report["bestModel"] = _select_best_model(report["models"], is_classification)

    # --- Explainability + Business Translation + Visualization ---
    best = report["bestModel"]
    if best and best.get("model"):
        best_name = best["model"]
        best_entry = report["models"].get(best_name, {})
        best_model_obj = best_entry.get("trainedModel")

        if best_model_obj is not None:
            if feature_names is None or len(feature_names) != X_train.shape[1]:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

            try:
                from .explainability import build_explainability_report
                report["explainability"] = build_explainability_report(
                    best_model_obj, X_train, y_train, X_test, y_test,
                    feature_names, is_classification, best_name,
                )
            except Exception as e:
                report["explainability"] = {"error": str(e)}

            try:
                from .business_translation import (
                    translate_regression_metrics, translate_classification_metrics,
                    translate_feature_importance, generate_recommendations,
                )
                test_metrics = best_entry.get("testMetrics", {})
                avg_target = float(np.mean(y_train))
                target_std = float(np.std(y_train))
                prevalence = float(np.mean(y_train == 1)) if is_classification else None

                if is_classification:
                    report["businessTranslation"] = translate_classification_metrics(
                        test_metrics, "target", prevalence=prevalence,
                    )
                else:
                    report["businessTranslation"] = translate_regression_metrics(
                        test_metrics, "target", avg_target=avg_target, target_std=target_std,
                    )

                explain = report.get("explainability", {})
                consensus = explain.get("consensusRanking", [])
                if consensus:
                    imp_for_biz = [
                        {"feature": f["feature"], "importance": 1.0 / f["averageRank"]}
                        for f in consensus
                    ]
                    report["featureInsights"] = translate_feature_importance(imp_for_biz, "target")

                report["recommendations"] = generate_recommendations(
                    explain, report.get("businessTranslation", {}), "target",
                )
            except Exception as e:
                report["businessTranslation"] = {"error": str(e)}

            try:
                from .visualization import build_all_model_charts
                preds = best_model_obj.predict(X_test).tolist() if hasattr(best_model_obj, "predict") else []
                actuals = y_test.tolist() if hasattr(y_test, "tolist") else list(y_test)
                report["charts"] = build_all_model_charts(
                    report.get("explainability", {}),
                    report.get("businessTranslation", {}),
                    report["models"],
                    preds, actuals,
                    is_classification=is_classification,
                )
            except Exception as e:
                report["charts"] = {"error": str(e)}

    # Strip non-serializable sklearn model objects before returning
    for name, entry in report.get("models", {}).items():
        if isinstance(entry, dict):
            entry.pop("trainedModel", None)

    return report


def _prob_of(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        if p.shape[1] == 2:
            return p
    return None


def _tracking(y_true, y_pred) -> float:
    """Mean absolute percentage error (MAPE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0 or len(y_true) == 0:
        return None
    return round(float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100), 4)


def _select_best_model(models_report: dict, is_classification: bool):
    """Pick best model by test F1 (classification) or R2 (regression)."""
    best_name = None
    best_key = "f1" if is_classification else "r2"
    best_score = -float("inf")
    for name, entry in models_report.items():
        if not isinstance(entry, dict) or "testMetrics" not in entry:
            continue
        score = entry["testMetrics"].get(best_key)
        if score is None:
            score = entry["testMetrics"].get("accuracy")
        if score is not None and score > best_score:
            best_score = score
            best_name = name
    if best_name is None:
        return None
    return {"model": best_name, "score": round(float(best_score), 4)}


def prepare_model_matrix(
    df: pd.DataFrame,
    dependent: str,
    predictors: list[str],
    column_types: Optional[dict] = None,
) -> tuple[np.ndarray, np.ndarray, list, bool, Optional[str]]:
    """Build a clean numeric (X, y, feature_names, is_classification) matrix.

    Lightweight: one-hot encodes categorical predictors and drops incomplete
    rows so the result is usable by the training pipeline. If impossible,
    returns (empty, empty, [], False, error_message).
    """
    import pandas as pd

    if dependent not in df.columns:
        return np.array([]), np.array([]), [], False, f"Dependent column {dependent!r} not found."

    valid_preds = [p for p in predictors if p in df.columns]
    if not valid_preds:
        return np.array([]), np.array([]), [], False, "No valid predictor columns."

    column_types = column_types or {}
    Cat = "categorical"

    cat_preds = [p for p in valid_preds if column_types.get(p, "continuous") == Cat]
    num_preds = [p for p in valid_preds if p not in cat_preds]

    X_df = df[num_preds].copy() if num_preds else pd.DataFrame(index=df.index)
    for c in cat_preds:
        if c in df.columns:
            dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
            X_df = X_df.join(dummies)

    y = df[dependent]
    combined = X_df.copy()
    combined["__y__"] = y

    clean = combined.dropna()
    if len(clean) < 10:
        return np.array([]), np.array([]), [], False, "Insufficient rows after dropping missing values (minimum 10)."

    feature_names = list(X_df.columns)
    if not feature_names:
        return np.array([]), np.array([]), [], False, "No usable features after preprocessing."

    X = clean[feature_names].astype(float).values
    y_vals = clean["__y__"].astype(float).values

    is_cls = _is_classification(y_vals)
    return X, y_vals, feature_names, is_cls, None


def run_model_training(
    df: pd.DataFrame,
    dependent: str,
    predictors: Optional[list] = None,
    column_types: Optional[dict] = None,
    config: Optional[dict] = None,
) -> dict:
    """High-level entry point used by the API route.

    config may carry: models, tuningMethod, tuningIterations, cvFolds,
    testSize, valSize, randomSeed, problemType override.
    """
    config = config or {}
    predictors = predictors or []
    if isinstance(predictors, str):
        predictors = [predictors]

    X, y, feature_names, is_classification, error = prepare_model_matrix(
        df, dependent, predictors, column_types,
    )
    if error:
        return {"problemType": None, "split": None, "baselines": {},
                "models": {}, "bestModel": None, "error": error}

    problem = config.get("problemType")
    if problem:
        is_classification = problem == "classification"

    return run_model_experiment(
        X, y,
        is_classification=is_classification,
        enabled_models=config.get("models"),
        tuning_method=config.get("tuningMethod", "random"),
        tuning_iterations=config.get("tuningIterations", 15),
        cv_folds=config.get("cvFolds", 5),
        test_size=config.get("testSize", 0.2),
        val_size=config.get("valSize", 0.15),
        random_state=config.get("randomSeed", 42),
        feature_names=feature_names,
    )