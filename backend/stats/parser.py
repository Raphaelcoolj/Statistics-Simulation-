import pandas as pd
import numpy as np
from io import BytesIO
from ..models import DatasetSchema, Column, ColumnType, MissingValueReport, MissingValueInfo
from .utils import pyval


BINARY_TRUE_VALUES = {"true", "yes", "1", 1, 1.0, True}
BINARY_FALSE_VALUES = {"false", "no", "0", 0, 0.0, False}


def _is_numeric_dtype(series: pd.Series) -> bool:
    try:
        return bool(np.issubdtype(series.dtype, np.number))
    except TypeError:
        pass
    try:
        import pandas.api.types as pd_types
        return bool(pd_types.is_numeric_dtype(series))
    except Exception:
        return False


def _detect_column_type(series: pd.Series) -> ColumnType:
    n_unique = series.nunique(dropna=False)

    try:
        if np.issubdtype(series.dtype, np.datetime64):
            return ColumnType.datetime
    except TypeError:
        pass

    is_numeric = _is_numeric_dtype(series)

    if not is_numeric:
        unique_vals = {str(v).lower().strip() for v in series.dropna().unique() if v is not None and v != ""}
        if len(unique_vals) <= 2 and len(unique_vals) > 0:
            if unique_vals.issubset(BINARY_TRUE_VALUES | BINARY_FALSE_VALUES | {""}):
                return ColumnType.binary
        if n_unique <= 15:
            return ColumnType.categorical
        return ColumnType.categorical
    else:
        unique_vals = set(series.dropna().unique())
        if n_unique == 2 and unique_vals.issubset({0, 1, 0.0, 1.0}):
            return ColumnType.binary
        if n_unique <= 15:
            return ColumnType.ordinal
        return ColumnType.continuous


def _infer_missing_strategy(col_type: ColumnType, null_pct: float) -> str:
    if null_pct > 50:
        return "drop_column"
    if col_type == ColumnType.continuous:
        return "mean"
    if col_type == ColumnType.ordinal:
        return "median"
    return "mode"


def parse_csv(buffer: bytes, file_name: str = "uploaded.csv") -> tuple[DatasetSchema, pd.DataFrame, MissingValueReport]:
    df = pd.read_csv(BytesIO(buffer), skipinitialspace=True)
    df = df.where(pd.notna(df), None)

    columns_list: list[Column] = []
    total_missing = 0
    by_column = {}

    for col_name in df.columns:
        series = df[col_name]
        series_clean = series.dropna()
        n_null = int(series.isna().sum())
        total_missing += n_null
        null_pct = round(n_null / max(len(series), 1) * 100, 2)

        col_type = _detect_column_type(series)

        col_info = Column(
            name=col_name,
            type=col_type,
            nullCount=n_null,
        )

        if col_type in (ColumnType.continuous, ColumnType.ordinal) and len(series_clean) > 0:
            vals = series_clean.astype(float)
            col_info.mean = pyval(vals.mean())
            col_info.median = pyval(vals.median())
            col_info.min = pyval(vals.min())
            col_info.max = pyval(vals.max())

        if col_type == ColumnType.categorical:
            col_info.uniqueValues = pyval(list(series_clean.unique()))
        elif col_type == ColumnType.ordinal:
            col_info.uniqueValues = pyval(sorted(series_clean.unique()))
        elif col_type == ColumnType.binary:
            col_info.uniqueValues = pyval(sorted(series_clean.unique()))

        col_info.sampleValues = pyval(series.head(5).to_list())

        by_column[col_name] = MissingValueInfo(
            count=n_null,
            percentage=null_pct,
            suggestedStrategy=_infer_missing_strategy(col_type, null_pct),
        )
        columns_list.append(col_info)

    schema = DatasetSchema(
        fileName=file_name,
        rowCount=len(df),
        columnCount=len(df.columns),
        columns=columns_list,
        sampleRows=pyval(df.head(10).to_dict(orient="records")),
    )

    report = MissingValueReport(
        totalMissing=total_missing,
        byColumn=by_column,
        requiresAttention=total_missing > 0,
    )

    return schema, df, report


def apply_missing_strategy(df: pd.DataFrame, strategies: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    for col, strategy in strategies.items():
        if col not in df.columns:
            continue
        if strategy == "drop_rows":
            df = df.dropna(subset=[col])
        elif strategy == "drop_column":
            df = df.drop(columns=[col])
        elif strategy == "mean":
            if _is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0]) if not df[col].mode().empty else df[col]
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val.iloc[0]) if not mode_val.empty else df[col]
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "forward_fill":
            df[col] = df[col].ffill()
        elif strategy == "backward_fill":
            df[col] = df[col].bfill()
    return df
