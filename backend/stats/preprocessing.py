import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.model_selection import train_test_split as sk_train_test_split


def preprocess_for_model(
    df: pd.DataFrame,
    dependent: str,
    predictors: list[str],
) -> pd.DataFrame:
    df = df.copy()

    for col in predictors:
        if col not in df.columns:
            continue
        if df[col].isna().sum() == 0:
            continue
        if np.issubdtype(df[col].dtype, np.number):
            # Use median for skewed distributions, mean for symmetric
            vals = df[col].dropna()
            if len(vals) >= 8:
                skewness = abs(scipy_stats.skew(vals))
                if skewness > 1.0:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mean())
        else:
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
            else:
                df[col] = df[col].fillna("Unknown")

    df = df.dropna(subset=[dependent])

    return df


def one_hot_encode(
    df: pd.DataFrame, categorical_cols: list[str]
) -> tuple[pd.DataFrame, dict[str, dict]]:
    df = df.copy()
    added_columns: dict[str, dict] = {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        categories = df[col].dropna().unique()
        if len(categories) <= 1:
            continue

        sorted_cats = sorted(categories)
        reference_cat = str(sorted_cats[0])  # first alphabetically is dropped

        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        new_cols = list(dummies.columns)
        added_columns[col] = {
            "encoded": new_cols,
            "reference": reference_cat,
        }
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df, added_columns


def standardize(df: pd.DataFrame, continuous_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    params = {}
    for col in continuous_cols:
        if col not in df.columns:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue
        vals = df[col].dropna()
        if len(vals) < 2:
            continue
        mean = vals.mean()
        std = vals.std(ddof=0)
        if std == 0:
            continue
        df[col] = (df[col] - mean) / std
        params[col] = {"mean": float(mean), "std": float(std)}
    return df, params


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return sk_train_test_split(X, y, test_size=test_size, random_state=random_state)


def compute_vif(df: pd.DataFrame, predictors: list[str]) -> list[dict]:
    results = []
    for i, col in enumerate(predictors):
        y = df[col].astype(float).values
        other_cols = [c for j, c in enumerate(predictors) if j != i]
        if len(other_cols) == 0:
            results.append({"predictor": col, "value": 1.0})
            continue
        X = df[other_cols].astype(float).values
        X = np.column_stack([np.ones(len(X)), X])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            preds = X @ beta
            ss_res = np.sum((y - preds) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            vif = 1 / (1 - r2) if r2 < 1 else 10_000
        except np.linalg.LinAlgError:
            vif = 10_000
        results.append({"predictor": col, "value": round(float(vif), 4)})
    return results
