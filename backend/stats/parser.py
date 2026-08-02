import os
import re
import pandas as pd
import numpy as np
from io import BytesIO

from ..models import (
    DatasetSchema, Column, ColumnType,
    MissingValueReport, MissingValueInfo,
)
from .utils import pyval

BINARY_TRUE_VALUES = {"true", "yes", "1", 1, 1.0, True}
BINARY_FALSE_VALUES = {"false", "no", "0", 0, 0.0, False}

CSV_EXTS = {".csv", ".tsv", ".txt"}

# ---------------------------------------------------------------------------
# Column-name heuristics
#
# The whole point: a column like `sex` holding values {1, 2} is NOT a numeric
# measurement — 1 and 2 are *codes* for categories. We consult the column name
# (plus its values) so the profiler classifies coded columns correctly instead
# of calling every short numeric column `ordinal`.
#
# Tokens are matched as whole words against the lowercased, non-alphanumeric-
# stripped column name (e.g. "Sex/Category" -> {"sex", "category"}).
# ---------------------------------------------------------------------------

BINARY_NAME_TOKENS = {
    "sex", "gender", "sexe", "gen", "married", "marital", "alive", "survival",
    "survived", "pregnant", "smoker", "smoking", "obese", "diabetic",
    "hypertensive", "infected", "positive", "dead", "deceased", "pass", "fail",
}

CATEGORICAL_NAME_TOKENS = {
    "category", "categor", "cat", "class", "clas", "group", "grp", "grpcd",
    "type", "typ", "kind", "status", "education", "school", "occupation", "job",
    "profession", "region", "state", "country", "province", "sector", "sect",
    "company", "species", "breed", "race", "ethnic", "religion", "ward",
    "location", "department", "diagnosis", "symptom", "disease", "condition",
    "treatment", "drug", "outcome", "reason", "mode", "color", "colour", "tier",
}

ORDINAL_NAME_TOKENS = {
    "level", "stage", "rank", "band", "grade", "order", "quartile", "decile",
    "percentile", "agecat", "age_group", "agegrp", "agegroup", "age_grp",
    "grp", "group", "class", "clas",
}

# Binaries are categorical subsets; these override to binary when only 2 values.
def _name_tokens(name: str):
    cleaned = str(name).lower().replace("_", " ").replace("-", " ")
    words = set(cleaned.split())
    return {w for w in words if w.isalpha()}


def _name_hint(name: str):
    """Return (binary, categorical, ordinal) booleans from the column name."""
    toks = _name_tokens(name)
    if not toks:
        return False, False, False
    binary = bool(toks & BINARY_NAME_TOKENS)
    ordinal = bool(toks & ORDINAL_NAME_TOKENS)
    categorical = bool(toks & CATEGORICAL_NAME_TOKENS)
    return binary, ordinal, categorical


def _is_numeric_dtype(series: pd.Series) -> bool:
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


def _integer_values(series: pd.Series) -> set:
    """Return the set of values cast to canonical form, marking if all integers."""
    vals = []
    all_int = True
    for v in series.dropna().unique():
        try:
            f = float(v)
            if f == int(f):
                vals.append(int(f))
            else:
                vals.append(f)
                all_int = False
        except (TypeError, ValueError):
            all_int = False
            vals.append(v)
    return set(vals), all_int


def _looks_like_codes(
    name: str,
    unique_vals: set,
    all_int: bool,
    n_unique: int,
    col_type: ColumnType,
):
    """Decide if a numeric column holds category/scale *codes* rather than raw
    measurements, using both the column name and the value profile.

    Returns (bool) whether the column is a coded variable plus a short note.
    """
    binary_hint, ordinal_hint, cat_hint = _name_hint(name)
    has_hint = binary_hint or ordinal_hint or cat_hint

    # Heuristic decisions are only made when the name (or a clean 0/1 binary)
    # gives a firm signal. Opaque-name columns fall through to `_is_code_uncertain`.
    if binary_hint and n_unique == 2:
        return True
    if has_hint and floats_low(unique_vals):
        return True
    return False


def _is_code_uncertain(name: str, unique_vals: set, all_int: bool, n_unique: int) -> bool:
    """True when the value profile suggests codes but the heuristic can't decide:
    low-cardinality, small-integer numeric column with no name hint."""
    binary_hint, ordinal_hint, cat_hint = _name_hint(name)
    has_hint = binary_hint or ordinal_hint or cat_hint

    # Genuine 0/1 binary is unambiguous.
    if set(unique_vals) == {0, 1}:
        return False
    # A name that implies coding/ordinal is decided by heuristics.
    if has_hint:
        return False
    small_integers = all_int and all(0 <= v <= 20 for v in unique_vals if isinstance(v, (int, np.integer)))
    # Low cardinality, small integers, opaque name -> unsure (AI callback).
    return small_integers and 2 <= n_unique <= 15 and floats_low(unique_vals)


# Maximum value considered "small" for coded-variable detection.
# Values above this threshold are treated as raw measurements, not category codes.
# For example, a column with values {0, 1, 2, 30} would NOT be treated as codes
# because 30 > CODE_MAX_VALUE.
CODE_MAX_VALUE = 30


def floats_low(unique_vals) -> bool:
    """Check if all numeric values stay at or below CODE_MAX_VALUE.

    Columns with small bounded values (e.g. 0-5, 1-10) are typical of coded
    variables ( Likert scales, category codes, binary indicators). Values
    exceeding CODE_MAX_VALUE suggest genuine measurements rather than codes.
    """
    nums = [v for v in unique_vals if isinstance(v, (int, float, np.integer, np.floating))]
    if not nums:
        return False
    for v in nums:
        try:
            if float(v) > CODE_MAX_VALUE:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _detect_column_type(
    series: pd.Series,
    name: str,
    precomputed_int_vals: tuple[set, bool] | None = None,
) -> ColumnType:
    """Detect the analysis type for a column, honoring the column name.

    Priority: name-driven classification beats the raw value-based default, so
    coded columns (e.g. sex={1,2}) are not mislabeled as continuous numbers.

    Args:
        precomputed_int_vals: Optional (unique_vals, all_int) from _integer_values
            to avoid recomputing when the caller already has it.
    """
    try:
        if np.issubdtype(series.dtype, np.datetime64):
            return ColumnType.datetime
    except TypeError:
        pass

    is_numeric = _is_numeric_dtype(series)
    n_unique = series.nunique(dropna=False)
    binary_hint, ordinal_hint, cat_hint = _name_hint(name)

    if is_numeric:
        unique_vals, all_int = (
            precomputed_int_vals if precomputed_int_vals is not None
            else _integer_values(series)
        )
        non_null = series.dropna()

        # Name says it's a binary category and we see exactly 2 distinct values.
        if binary_hint and n_unique == 2:
            return ColumnType.binary
        # Name-driven categorical (e.g. "category", "sex" with 3+ codes).
        if cat_hint and not ordinal_hint:
            return ColumnType.categorical
        # Name-driven ordinal (e.g. age_group, education_level).
        if ordinal_hint:
            return ColumnType.ordinal
        # Value-level fallback (no name signal).
        if n_unique == 2 and unique_vals.issubset({0, 1, 0.0, 1.0}):
            return ColumnType.binary
        if n_unique <= 15:
            return ColumnType.ordinal
        return ColumnType.continuous

    # ---- non-numeric ----
    unique_vals = {str(v).lower().strip() for v in series.dropna().unique() if v is not None and v != ""}
    if len(unique_vals) <= 2 and len(unique_vals) > 0:
        if unique_vals.issubset(BINARY_TRUE_VALUES | BINARY_FALSE_VALUES | {""}):
            return ColumnType.binary
    if n_unique <= 15:
        return ColumnType.categorical
    return ColumnType.categorical


def _infer_missing_strategy(col_type: ColumnType, null_pct: float) -> str:
    if null_pct > 50:
        return "drop_column"
    if null_pct > 20 and col_type == ColumnType.continuous:
        return "knn"
    if col_type == ColumnType.continuous:
        return "mean"
    if col_type == ColumnType.ordinal:
        return "median"
    return "mode"


def _build_schema(df: pd.DataFrame, file_name: str) -> tuple[DatasetSchema, MissingValueReport]:
    columns_list: list[Column] = []
    total_missing = 0
    by_column = {}

    for col_name in df.columns:
        series = df[col_name]
        series_clean = series.dropna()
        n_null = int(series.isna().sum())
        total_missing += n_null
        null_pct = round(n_null / max(len(series), 1) * 100, 2)

        # Compute integer values once (used by both type detection and coded detection)
        int_vals = _integer_values(series) if _is_numeric_dtype(series) else None
        col_type = _detect_column_type(series, col_name, precomputed_int_vals=int_vals)

        # Detect whether this column is a coded variable (numeric codes standing
        # in for category/scale labels), even when its type resolves to numeric.
        coded = False
        code_note = None
        code_uncertain = False
        if col_type not in (ColumnType.datetime, ColumnType.categorical):
            unique_vals, all_int = int_vals if int_vals is not None else _integer_values(series)
            if _looks_like_codes(col_name, unique_vals, all_int, series.nunique(dropna=False), col_type):
                coded = True
                code_note = "Values look like discrete codes, not raw measurements."
            elif _is_code_uncertain(col_name, unique_vals, all_int, series.nunique(dropna=False)):
                code_uncertain = True
                coded = None

        if col_type in (ColumnType.continuous, ColumnType.ordinal) and len(series_clean) > 0:
            vals = series_clean.astype(float)
            col_mean = pyval(vals.mean())
            col_median = pyval(vals.median())
            col_min = pyval(vals.min())
            col_max = pyval(vals.max())
            # Detect constant column (zero variance)
            if vals.nunique() <= 1:
                code_note = "Constant column (zero variance) — will be skipped in correlation and regression analyses."
                coded = True
        else:
            col_mean = col_median = col_min = col_max = None

        if col_type == ColumnType.categorical:
            uniq = pyval(list(series_clean.unique()))
        elif col_type in (ColumnType.ordinal, ColumnType.binary):
            uniq = pyval(sorted(series_clean.unique()))
        else:
            uniq = None

        col_info = Column(
            name=col_name,
            type=col_type,
            coded=coded,
            codeNote=code_note,
            codeUncertain=code_uncertain,
            uniqueValues=uniq,
            min=col_min,
            max=col_max,
            mean=col_mean,
            median=col_median,
            sampleValues=pyval(series.head(5).to_list()),
            nullCount=n_null,
        )
        by_column[col_name] = MissingValueInfo(
            count=n_null,
            percentage=null_pct,
            suggestedStrategy=_infer_missing_strategy(col_type, null_pct),
        )
        columns_list.append(col_info)

    duplicate_count = int(df.duplicated().sum())

    # Full data for charting — all rows, all columns (used by frontend scatter plots)
    full_data = pyval(df.to_dict(orient="records"))

    schema = DatasetSchema(
        fileName=file_name,
        rowCount=len(df),
        columnCount=len(df.columns),
        columns=columns_list,
        sampleRows=pyval(df.to_dict(orient="records")),
        fullData=full_data,
        duplicateRowCount=duplicate_count,
    )

    report = MissingValueReport(
        totalMissing=total_missing,
        byColumn=by_column,
        requiresAttention=total_missing > 0,
    )

    # Generate warnings for columns with high missingness that weren't dropped
    warnings = []
    for col_name, info in by_column.items():
        if info.percentage > 30 and info.suggestedStrategy != "drop_column":
            warnings.append(
                f"Column '{col_name}' has {info.percentage}% missing values "
                f"({info.count} rows). Consider dropping or using advanced imputation."
            )
    if warnings:
        report.warnings = warnings

    return schema, report


def _read_frame(buffer: bytes, file_name: str) -> pd.DataFrame:
    """Read a CSV or Excel byte buffer into a DataFrame, by file extension."""
    ext = os.path.splitext(file_name)[1].lower()
    if ext == ".xlsx" or ext == ".xls":
        engine = None
        if ext == ".xlsx":
            engine = "openpyxl"
        else:
            engine = "xlrd"
        return pd.read_excel(BytesIO(buffer), engine=engine)
    if ext in CSV_EXTS:
        return pd.read_csv(BytesIO(buffer), skipinitialspace=True)
    # Unknown extension: sniff; default to CSV.
    return pd.read_csv(BytesIO(buffer), skipinitialspace=True)


def parse_file(buffer: bytes, file_name: str = "uploaded.csv") -> tuple[DatasetSchema, pd.DataFrame, MissingValueReport]:
    """Parse a CSV/Excel file into a schema + cleaned DataFrame + missing report."""
    df = _read_frame(buffer, file_name)
    df = df.where(pd.notna(df), None)
    schema, report = _build_schema(df, file_name)
    return schema, df, report


def parse_csv(buffer: bytes, file_name: str = "uploaded.csv") -> tuple[DatasetSchema, pd.DataFrame, MissingValueReport]:
    """Backwards-compatible CSV-only entrypoint."""
    return parse_file(buffer, file_name)


def apply_codebook(
    schema: DatasetSchema,
    codebook: dict[str, dict],
) -> DatasetSchema:
    """Attach human-readable labels to coded columns, e.g.
    {sex: {"1": "Male", "2": "Female"}}.

    Values in the codebook are normalized to strings so they match the
    frequency-table keys rendered by the frontend. Existing integer codes such
    as 1/2 are turned into "1"/"2" automatically.
    """
    if not codebook:
        return schema

    for col in schema.columns:
        mapping = codebook.get(col.name)
        if not mapping:
            continue
        labels = {}
        for code, label in mapping.items():
            labels[str(code)] = str(label)
        col.labels = labels
        col.coded = True
        col.codeNote = "Values mapped to human-readable labels via codebook."
        col.codeUncertain = False
        # Codebook clarifies a numeric column is a category code, not a measure.
        if col.type == ColumnType.continuous:
            label_count = len(set(str(l) for l in labels.values()))
            col.type = ColumnType.categorical if label_count > 2 else ColumnType.binary
        elif col.type == ColumnType.ordinal:
            col.type = ColumnType.ordinal
    return schema


def apply_missing_strategy(df: pd.DataFrame, strategies: dict[str, str]) -> pd.DataFrame:
    # Collect columns needing knn/iterative so we can batch them
    numeric_impute_cols: dict[str, list[str]] = {}  # strategy -> [cols]

    df = df.copy()
    for col, strategy in strategies.items():
        if col not in df.columns:
            continue
        if strategy == "drop_rows":
            df = df.dropna(subset=[col])
        elif strategy == "drop_column":
            df = df.drop(columns=[col])
        elif strategy in ("knn", "iterative"):
            # Defer to batch processing below
            numeric_impute_cols.setdefault(strategy, []).append(col)
        elif strategy == "mean":
            if _is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0]) if not df[col].mode().empty else df[col]
        elif strategy == "median":
            if _is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0]) if not df[col].mode().empty else df[col]
        elif strategy == "mode":
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val.iloc[0]) if not mode_val.empty else df[col]
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "forward_fill":
            df[col] = df[col].ffill()
        elif strategy == "backward_fill":
            df[col] = df[col].bfill()

    # Batch imputation for advanced methods (must operate on all numeric cols together)
    if "knn" in numeric_impute_cols:
        df = apply_knn_imputation(df, numeric_impute_cols["knn"])
    if "iterative" in numeric_impute_cols:
        df = apply_iterative_imputation(df, numeric_impute_cols["iterative"])

    return df


def apply_knn_imputation(df: pd.DataFrame, columns: list[str] | None = None, n_neighbors: int = 5) -> pd.DataFrame:
    """Impute missing values using KNN (k-nearest neighbors) imputer.

    Uses the k nearest samples (by other numeric features) to estimate missing
    values. Best for numeric columns with moderate missingness and inter-feature
    correlations.
    """
    from sklearn.impute import KNNImputer

    numeric_cols = columns or [c for c in df.columns if _is_numeric_dtype(df[c])]
    if not numeric_cols:
        return df

    imputer = KNNImputer(n_neighbors=min(n_neighbors, max(1, len(df) - 1)))
    df = df.copy()
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df


def apply_iterative_imputation(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    max_iter: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Impute missing values using IterativeImputer (multivariate imputation).

    Models each feature with missing values as a function of other features
    in a round-robin fashion. Handles complex relationships between features.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    numeric_cols = columns or [c for c in df.columns if _is_numeric_dtype(df[c])]
    if not numeric_cols:
        return df

    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    df = df.copy()
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    return df