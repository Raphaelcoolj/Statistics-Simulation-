import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from models import DescriptiveResult, ColumnType
from stats.utils import pyval


def _is_numeric(series: pd.Series) -> bool:
    try:
        if np.issubdtype(series.dtype, np.number):
            return True
    except TypeError:
        pass
    try:
        import pandas.api.types as pd_types
        return bool(pd_types.is_numeric_dtype(series))
    except Exception:
        return False


def compute_descriptive(df: pd.DataFrame, columns: list[str]) -> list[DescriptiveResult]:
    results: list[DescriptiveResult] = []

    for col_name in columns:
        if col_name not in df.columns:
            continue
        series = df[col_name]
        series_clean = series.dropna()
        n_null = int(series.isna().sum())
        count = len(series_clean)

        result = DescriptiveResult(column=col_name, count=count, nullCount=n_null)

        if _is_numeric(series) and len(series_clean) > 0:
            vals = series_clean.astype(float)
            result.mean = pyval(float(np.mean(vals)))
            result.median = pyval(float(np.median(vals)))
            result.stdDev = pyval(float(np.std(vals, ddof=1))) if len(vals) > 1 else 0.0
            result.variance = pyval(float(np.var(vals, ddof=1))) if len(vals) > 1 else 0.0
            result.min = pyval(float(np.min(vals)))
            result.max = pyval(float(np.max(vals)))
            result.range = pyval(float(np.max(vals) - np.min(vals)))
            q1, q3 = np.percentile(vals, [25, 75])
            result.iqr = pyval(float(q3 - q1))
            if len(vals) >= 3:
                result.skewness = pyval(float(scipy_stats.skew(vals)))
                result.kurtosis = pyval(float(scipy_stats.kurtosis(vals, fisher=True)))
            else:
                result.skewness = 0.0
                result.kurtosis = 0.0

            lower = q1 - 1.5 * (q3 - q1)
            upper = q3 + 1.5 * (q3 - q1)
            result.outlierCount = pyval(int(np.sum((vals < lower) | (vals > upper))))

            mode_res = scipy_stats.mode(vals, keepdims=True)
            result.mode = pyval(float(mode_res.mode[0])) if mode_res.count[0] > 0 else None
        else:
            freq = series_clean.value_counts().to_dict()
            result.frequencyTable = pyval({str(k): int(v) for k, v in freq.items()})
            result.mode = str(series_clean.mode().iloc[0]) if not series_clean.mode().empty else None
            result.note = "Non-numeric column; limited descriptive measures available"

        results.append(result)

    return results
