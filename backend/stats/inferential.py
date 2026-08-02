import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from models import CorrelationResult, HypothesisResult, RegressionResult, ModelType
from stats.utils import pyval


def compute_correlations(
    df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    column_types: dict[str, str] | None = None,
) -> list[CorrelationResult]:
    results: list[CorrelationResult] = []

    for col_a, col_b in pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        clean = df[[col_a, col_b]].dropna()
        if len(clean) < 3:
            continue

        vals_a = clean[col_a].astype(float)
        vals_b = clean[col_b].astype(float)

        # Skip constant columns (zero variance)
        if vals_a.nunique() <= 1 or vals_b.nunique() <= 1:
            continue

        # Determine method: consult column types first, then fall back to cardinality
        use_spearman = False
        if column_types:
            type_a = column_types.get(col_a, "continuous")
            type_b = column_types.get(col_b, "continuous")
            if type_a in ("ordinal", "binary") or type_b in ("ordinal", "binary"):
                use_spearman = True
        if not use_spearman:
            n_unique_a = vals_a.nunique()
            n_unique_b = vals_b.nunique()
            use_spearman = n_unique_a <= 15 or n_unique_b <= 15

        if use_spearman:
            r, p_value = scipy_stats.spearmanr(vals_a, vals_b)
            method = "spearman"
        else:
            r, p_value = scipy_stats.pearsonr(vals_a, vals_b)
            method = "pearson"

        r = float(r)
        abs_r = abs(r)
        if abs_r >= 0.7:
            direction = "positive" if r > 0 else "negative"
            interpretation = f"strong {direction}"
        elif abs_r >= 0.4:
            direction = "positive" if r > 0 else "negative"
            interpretation = f"moderate {direction}"
        else:
            interpretation = "weak"

        # Fisher z-transformation for 95% confidence interval
        ci_lower = ci_upper = None
        if abs_r < 1.0 and len(clean) > 3:
            z = 0.5 * np.log((1 + r) / (1 - r))
            se = 1.0 / np.sqrt(len(clean) - 3)
            ci_lower = round(float(np.tanh(z - 1.96 * se)), 4)
            ci_upper = round(float(np.tanh(z + 1.96 * se)), 4)

        results.append(CorrelationResult(
            columnA=col_a,
            columnB=col_b,
            r=round(r, 4),
            method=method,
            interpretation=interpretation,
            pValue=round(float(p_value), 4) if p_value is not None else None,
            confidenceIntervalLower=ci_lower,
            confidenceIntervalUpper=ci_upper,
        ))

    return results


def compute_hypothesis_tests(df: pd.DataFrame, tests: list[dict]) -> list[HypothesisResult]:
    results: list[HypothesisResult] = []

    for test in tests:
        test_type = test["type"]
        columns = test["columns"]

        if test_type == "t-test":
            if len(columns) != 2:
                continue
            val_col, group_col = columns[0], columns[1]
            clean = df[[val_col, group_col]].dropna()
            if len(clean) < 3:
                continue
            groups = clean.groupby(group_col)[val_col].apply(lambda x: x.astype(float).values)
            if len(groups) != 2:
                continue
            g1, g2 = groups.iloc[0], groups.iloc[1]
            if len(g1) < 2 or len(g2) < 2:
                continue
            statistic, p_value = scipy_stats.ttest_ind(g1, g2, equal_var=False)
            # Welch-Satterthwaite degrees of freedom
            n1, n2 = len(g1), len(g2)
            s1, s2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
            df_welch = ((s1/n1 + s2/n2)**2 /
                        ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))) if (s1/n1 + s2/n2) > 0 else 0
            results.append(HypothesisResult(
                testType="t-test",
                statistic=round(float(statistic), 4),
                pValue=round(float(p_value), 4),
                significant=bool(p_value < 0.05),
                confidenceLevel=0.95,
                columns=columns,
                degreesOfFreedom=round(df_welch, 2),
            ))

        elif test_type == "chi-square":
            if len(columns) < 2:
                continue
            clean = df[columns].dropna()
            if len(clean) < 2:
                continue
            for c in columns:
                clean[c] = clean[c].astype(str)
            contingency = pd.crosstab(clean[columns[0]], clean[columns[1]])
            if contingency.size < 2:
                continue
            statistic, p_value, dof, expected = scipy_stats.chi2_contingency(contingency)
            results.append(HypothesisResult(
                testType="chi-square",
                statistic=round(float(statistic), 4),
                pValue=round(float(p_value), 4),
                significant=bool(p_value < 0.05),
                confidenceLevel=0.95,
                columns=columns,
                degreesOfFreedom=dof,
            ))

        elif test_type == "anova":
            if len(columns) != 2:
                continue
            val_col, group_col = columns[0], columns[1]
            clean = df[[val_col, group_col]].dropna()
            if len(clean) < 3:
                continue
            groups = [
                g.astype(float).values
                for _, g in clean.groupby(group_col)[val_col]
                if len(g) >= 2
            ]
            if len(groups) < 2:
                continue
            statistic, p_value = scipy_stats.f_oneway(*groups)
            # ANOVA degrees of freedom: (k-1, N-k)
            k = len(groups)
            n_total = sum(len(g) for g in groups)
            df_between = k - 1
            df_within = n_total - k
            results.append(HypothesisResult(
                testType="anova",
                statistic=round(float(statistic), 4),
                pValue=round(float(p_value), 4),
                significant=bool(p_value < 0.05),
                confidenceLevel=0.95,
                columns=columns,
                degreesOfFreedom=(df_between, df_within),
            ))

    return results


def compute_regression(df: pd.DataFrame, dependent: str, predictors: list[str]) -> RegressionResult | None:
    if dependent not in df.columns:
        return None
    cols = [dependent] + [p for p in predictors if p in df.columns]
    if len(cols) < 2:
        return None
    clean = df[cols].dropna()
    if len(clean) < 3:
        return None

    # Filter out constant predictors (zero variance)
    valid_predictors = []
    for p in predictors:
        if p not in clean.columns:
            continue
        if clean[p].astype(float).nunique() > 1:
            valid_predictors.append(p)
    if not valid_predictors:
        return None
    predictors = valid_predictors

    X = clean[predictors].astype(float).values
    y = clean[dependent].astype(float).values

    X = np.column_stack([np.ones(len(X)), X])

    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    intercept = float(beta[0])
    coefficients = [float(b) for b in beta[1:]]
    predictions = (X @ beta).tolist()
    residuals = (y - X @ beta).tolist()

    y_mean = np.mean(y)
    ss_res = np.sum((y - X @ beta) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    n = len(y)
    p = len(predictors)
    adjusted_r2 = float(1 - (1 - r_squared) * (n - 1) / (n - p - 1)) if n > p + 1 else r_squared

    mse = float(np.mean((y - X @ beta) ** 2))
    rmse = float(np.sqrt(mse))

    return RegressionResult(
        modelType=ModelType.linear,
        dependent=dependent,
        predictors=predictors,
        coefficients=coefficients,
        intercept=intercept,
        rSquared=round(r_squared, 4),
        adjustedRSquared=round(adjusted_r2, 4),
        mse=round(mse, 4),
        rmse=round(rmse, 4),
        predictions=predictions,
        residuals=residuals,
    )
