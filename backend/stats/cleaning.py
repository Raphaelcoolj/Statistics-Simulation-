"""Data cleaning utilities: outlier handling, inconsistency fixes, deduplication."""

import re
import unicodedata
from difflib import get_close_matches

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# 1. OUTLIER HANDLING
# ---------------------------------------------------------------------------

def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
) -> dict[str, dict]:
    """Detect outliers using the IQR method.

    Returns a dict mapping column name to:
        {count, indices, lower_bound, upper_bound}
    """
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    result: dict[str, dict] = {}

    for col in cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 4:
            continue
        q1 = float(vals.quantile(0.25))
        q3 = float(vals.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        outlier_indices = df.index[mask].tolist()
        result[col] = {
            "count": len(outlier_indices),
            "indices": outlier_indices[:100],  # cap for payload size
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
        }
    return result


def detect_outliers_zscore(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 3.0,
) -> dict[str, dict]:
    """Detect outliers using the Z-score method.

    Points with |z| > threshold are flagged.
    Returns same shape as detect_outliers_iqr.
    """
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    result: dict[str, dict] = {}

    for col in cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 4:
            continue
        mean = vals.mean()
        std = vals.std(ddof=1)
        if std == 0:
            continue
        z = (df[col] - mean).abs() / std
        mask = z > threshold
        outlier_indices = df.index[mask].tolist()
        result[col] = {
            "count": len(outlier_indices),
            "indices": outlier_indices[:100],
            "mean": round(float(mean), 4),
            "std": round(float(std), 4),
            "threshold": threshold,
        }
    return result


def handle_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "iqr",
    action: str = "clip",
    factor: float = 1.5,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Handle outliers in numeric columns.

    Args:
        method: "iqr" or "zscore" for detection
        action: "remove" - drop outlier rows
                "clip"   - winsorize (clip to bounds)
                "log"    - apply log1p transform (positive values only)
                "sqrt"   - apply sqrt transform (non-negative values only)
                "none"   - detect only, don't modify

    Returns:
        (cleaned_df, outlier_report)
    """
    cols = columns or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df = df.copy()

    if method == "zscore":
        report = detect_outliers_zscore(df, cols, factor)
    else:
        report = detect_outliers_iqr(df, cols, factor)

    if action == "none" or action == "detect":
        return df, report

    for col in cols:
        if col not in report:
            continue
        info = report[col]
        if info["count"] == 0:
            continue

        if method == "zscore":
            vals = df[col].dropna()
            if len(vals) < 4:
                continue
            mean = vals.mean()
            std = vals.std(ddof=1)
            if std == 0:
                continue
            lower = mean - factor * std
            upper = mean + factor * std
        else:
            lower = info["lower_bound"]
            upper = info["upper_bound"]

        if action == "clip":
            df[col] = df[col].clip(lower=lower, upper=upper)
        elif action == "remove":
            mask = (df[col] < lower) | (df[col] > upper)
            df = df[~mask]
        elif action == "log":
            # Only apply to positive values; clip negatives to small positive
            if (df[col].dropna() > 0).all():
                df[col] = np.log1p(df[col])
            else:
                min_pos = df[col][df[col] > 0].min() if (df[col] > 0).any() else 1.0
                df[col] = df[col].clip(lower=min_pos)
                df[col] = np.log1p(df[col])
        elif action == "sqrt":
            if (df[col].dropna() >= 0).all():
                df[col] = np.sqrt(df[col])
            else:
                df[col] = df[col].clip(lower=0)
                df[col] = np.sqrt(df[col])

    return df, report


# ---------------------------------------------------------------------------
# 2. FIX INCONSISTENCIES
# ---------------------------------------------------------------------------

# Common US state abbreviations and their full names
_STATE_ABBREV: dict[str, str] = {
    "al": "Alabama", "ak": "Alaska", "az": "Arizona", "ar": "Arkansas",
    "ca": "California", "co": "Colorado", "ct": "Connecticut", "de": "Delaware",
    "fl": "Florida", "ga": "Georgia", "hi": "Hawaii", "id": "Idaho",
    "il": "Illinois", "in": "Indiana", "ia": "Iowa", "ks": "Kansas",
    "ky": "Kentucky", "la": "Louisiana", "me": "Maine", "md": "Maryland",
    "ma": "Massachusetts", "mi": "Michigan", "mn": "Minnesota", "ms": "Mississippi",
    "mo": "Missouri", "mt": "Montana", "ne": "Nebraska", "nv": "Nevada",
    "nh": "New Hampshire", "nj": "New Jersey", "nm": "New Mexico", "ny": "New York",
    "nc": "North Carolina", "nd": "North Dakota", "oh": "Ohio", "ok": "Oklahoma",
    "or": "Oregon", "pa": "Pennsylvania", "ri": "Rhode Island", "sc": "South Carolina",
    "sd": "South Dakota", "tn": "Tennessee", "tx": "Texas", "ut": "Utah",
    "vt": "Vermont", "va": "Virginia", "wa": "Washington", "wv": "West Virginia",
    "wi": "Wisconsin", "wy": "Wyoming",
}


def standardize_categoricals(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    case: str = "title",
    strip_whitespace: bool = True,
    collapse_whitespace: bool = True,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Standardize categorical string values.

    Args:
        case: "lower", "upper", "title", "capitalize", or "none"
        strip_whitespace: remove leading/trailing whitespace
        collapse_whitespace: collapse multiple spaces to single

    Returns:
        (standardized_df, changes_report)
    """
    cols = columns or [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) and not pd.api.types.is_numeric_dtype(df[c])
    ]
    df = df.copy()
    report: dict[str, dict] = {}

    for col in cols:
        if col not in df.columns:
            continue
        original = df[col].copy()
        series = df[col].dropna()

        if strip_whitespace:
            series = series.str.strip()
        if collapse_whitespace:
            series = series.str.replace(r"\s+", " ", regex=True)

        if case == "lower":
            series = series.str.lower()
        elif case == "upper":
            series = series.str.upper()
        elif case == "title":
            series = series.str.title()
        elif case == "capitalize":
            series = series.str.capitalize()
        # case == "none" -> no case change

        df.loc[series.index, col] = series

        # Track changes
        changed = (original.fillna("__NA__") != df[col].fillna("__NA__")).sum()
        if changed > 0:
            old_vals = set(original.dropna().unique())
            new_vals = set(df[col].dropna().unique())
            report[col] = {
                "values_changed": int(changed),
                "old_unique_count": len(old_vals),
                "new_unique_count": len(new_vals),
            }

    return df, report


def standardize_states(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    output_format: str = "full",
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Standardize US state names/abbreviations.

    Args:
        output_format: "full" (e.g. "New York") or "abbrev" (e.g. "NY")
    """
    cols = columns or [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c])
    ]
    df = df.copy()
    report: dict[str, dict] = {}

    abbrev_to_full = {k.upper(): v for k, v in _STATE_ABBREV.items()}
    full_to_abbrev = {v.lower(): k.upper() for k, v in _STATE_ABBREV.items()}
    # Also handle case-insensitive full names
    full_lower_map = {v.lower(): (k.upper(), v) for k, v in _STATE_ABBREV.items()}

    for col in cols:
        if col not in df.columns:
            continue
        original = df[col].copy()
        series = df[col].dropna()
        changed_count = 0

        new_vals = []
        for val in series:
            s = str(val).strip()
            s_lower = s.lower()

            # Already a 2-letter abbreviation
            if s.upper() in abbrev_to_full:
                if output_format == "full":
                    new_vals.append(abbrev_to_full[s.upper()])
                else:
                    new_vals.append(s.upper())
                changed_count += 1
                continue

            # Full name match
            if s_lower in full_lower_map:
                abbrev, full = full_lower_map[s_lower]
                if output_format == "full":
                    new_vals.append(full)
                else:
                    new_vals.append(abbrev)
                changed_count += 1
                continue

            # Fuzzy match if no exact match
            matches = get_close_matches(s_lower, full_lower_map.keys(), n=1, cutoff=0.8)
            if matches:
                abbrev, full = full_lower_map[matches[0]]
                if output_format == "full":
                    new_vals.append(full)
                else:
                    new_vals.append(abbrev)
                changed_count += 1
                continue

            new_vals.append(s)

        if changed_count > 0:
            df.loc[series.index, col] = new_vals
            report[col] = {
                "values_changed": changed_count,
                "total_rows": len(series),
            }

    return df, report


def parse_dates(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    infer_format: bool = True,
    dayfirst: bool = False,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Convert date/time strings to proper datetime objects.

    Tries common formats automatically. Reports columns that couldn't be parsed.
    """
    cols = columns or [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c])
    ]
    df = df.copy()
    report: dict[str, dict] = {}

    for col in cols:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Quick check: do values look like dates?
        sample = series.head(20).astype(str)
        has_date_patterns = sample.str.contains(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}", regex=True).any()
        if not has_date_patterns:
            continue

        try:
            parsed = pd.to_datetime(
                df[col],
                dayfirst=dayfirst,
                errors="coerce",
            )
            n_parsed = parsed.notna().sum()
            n_original = series.notna().sum()
            n_failed = n_original - n_parsed

            if n_parsed > n_original * 0.3:  # At least 30% parsed successfully
                df[col] = parsed
                report[col] = {
                    "parsed": int(n_parsed),
                    "failed": int(n_failed),
                    "dtype": "datetime64",
                }
            else:
                report[col] = {
                    "parsed": int(n_parsed),
                    "failed": int(n_failed),
                    "dtype": "unchanged",
                    "note": "Too many values failed to parse; column left as-is.",
                }
        except Exception as e:
            report[col] = {"error": str(e), "dtype": "unchanged"}

    return df, report


def fix_typos(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    correction_map: dict[str, dict[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Fix common typos and map known incorrect values to correct ones.

    Args:
        correction_map: per-column mapping of {wrong: correct}, e.g.
            {"city": {"NYC": "New York", "Nwe York": "New York"}}

    If no correction_map is provided, uses fuzzy matching to suggest corrections
    for values that appear very infently (likely typos of more common values).
    """
    cols = columns or [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c])
    ]
    df = df.copy()
    report: dict[str, dict] = {}

    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue

        changed = 0

        # Apply explicit corrections
        if correction_map and col in correction_map:
            mapping = correction_map[col]
            for wrong, correct in mapping.items():
                mask = df[col].astype(str).str.strip().str.lower() == wrong.strip().lower()
                n = mask.sum()
                if n > 0:
                    df.loc[mask, col] = correct
                    changed += n

        else:
            # Auto-detect: find values that appear only once and are very similar
            # to more common values (likely typos)
            value_counts = series.value_counts()
            if len(value_counts) < 3:
                continue

            all_values = list(value_counts.index.astype(str))
            corrections: dict[str, str] = {}

            for val in all_values:
                if value_counts[val] > 1:
                    continue  # Only flag singleton values as potential typos
                # Find close matches among more common values
                common_vals = [v for v in all_values if v != val and value_counts.get(v, 0) > 1]
                if not common_vals:
                    continue
                matches = get_close_matches(val, common_vals, n=1, cutoff=0.85)
                if matches:
                    corrections[val] = matches[0]

            if corrections:
                for wrong, correct in corrections.items():
                    mask = df[col].astype(str).str.strip() == wrong.strip()
                    n = mask.sum()
                    if n > 0:
                        df.loc[mask, col] = correct
                        changed += n

        if changed > 0:
            report[col] = {"values_corrected": changed}

    return df, report


# ---------------------------------------------------------------------------
# 3. DEDUPLICATION
# ---------------------------------------------------------------------------

def detect_exact_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> dict:
    """Detect exact duplicate rows.

    Returns report with count and duplicate row indices.
    """
    mask = df.duplicated(subset=subset, keep="first")
    dup_count = int(mask.sum())
    dup_indices = df.index[mask].tolist()

    return {
        "exact_duplicate_count": dup_count,
        "total_rows": len(df),
        "duplicate_indices": dup_indices[:100],  # cap for payload
        "columns_checked": subset or list(df.columns),
    }


def remove_exact_duplicates(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: str = "first",
) -> tuple[pd.DataFrame, dict]:
    """Remove exact duplicate rows.

    Args:
        subset: column names to check; None = all columns
        keep: "first", "last", or False (drop all dupes)
    """
    report = detect_exact_duplicates(df, subset)
    df_clean = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    report["rows_removed"] = len(df) - len(df_clean)
    report["rows_remaining"] = len(df_clean)
    return df_clean, report


def detect_fuzzy_duplicates(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.9,
) -> dict:
    """Detect near-duplicate rows using fuzzy string matching.

    Compares string/categorical columns pairwise and flags rows where all
    compared columns have similarity >= threshold.

    Returns list of duplicate groups (each group is a list of row indices).
    """
    cols = columns or [
        c for c in df.columns
        if pd.api.types.is_object_dtype(df[c]) and not pd.api.types.is_numeric_dtype(df[c])
    ]
    if not cols:
        return {"fuzzy_groups": [], "total_groups": 0}

    # Normalize values for comparison
    norm_df = df[cols].copy()
    for col in cols:
        norm_df[col] = norm_df[col].fillna("").astype(str).str.strip().str.lower()

    groups: list[list[int]] = []
    visited: set[int] = set()

    for i in range(len(norm_df)):
        if i in visited:
            continue
        group = [i]
        row_i = norm_df.iloc[i]

        for j in range(i + 1, len(norm_df)):
            if j in visited:
                continue
            row_j = norm_df.iloc[j]

            # Check if all columns match above threshold
            all_match = True
            for col in cols:
                vi = row_i[col]
                vj = row_j[col]
                if vi == vj:
                    continue
                if vi == "" or vj == "":
                    continue  # Skip empty comparisons
                # Simple similarity: ratio of matching chars
                from difflib import SequenceMatcher
                sim = SequenceMatcher(None, vi, vj).ratio()
                if sim < threshold:
                    all_match = False
                    break

            if all_match:
                group.append(j)
                visited.add(j)

        if len(group) > 1:
            groups.append(group)
            visited.update(group)

    return {
        "fuzzy_groups": [g for g in groups[:50]],  # cap for payload
        "total_groups": len(groups),
        "columns_compared": cols,
        "threshold": threshold,
    }


def remove_fuzzy_duplicates(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.9,
    keep: str = "first",
) -> tuple[pd.DataFrame, dict]:
    """Remove fuzzy duplicate rows, keeping one representative per group.

    Args:
        keep: "first" keeps the first row in each group, "last" keeps the last
    """
    report = detect_fuzzy_duplicates(df, columns, threshold)
    df_clean = df.copy()

    rows_to_drop: list[int] = []
    for group in report["fuzzy_groups"]:
        if keep == "first":
            rows_to_drop.extend(group[1:])
        elif keep == "last":
            rows_to_drop.extend(group[:-1])
        else:
            rows_to_drop.extend(group)

    if rows_to_drop:
        df_clean = df_clean.drop(rows_to_drop).reset_index(drop=True)

    report["rows_removed"] = len(df) - len(df_clean)
    report["rows_remaining"] = len(df_clean)
    return df_clean, report
