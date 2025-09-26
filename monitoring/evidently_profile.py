import pandas as pd
from typing import List, Tuple, Optional

# Evidently 0.5.1 APIs
from evidently.report import Report
from evidently.metric_preset import ClassificationPreset
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset


# ---------- helpers ----------
def _align_columns(ref: pd.DataFrame, cur: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Align schemas in-memory (do NOT modify files)."""
    all_cols = sorted(set(ref.columns) | set(cur.columns))
    for c in all_cols:
        if c not in ref.columns:
            ref[c] = pd.NA
        if c not in cur.columns:
            cur[c] = pd.NA
    ref = ref[all_cols]
    cur = cur[all_cols]
    return ref, cur


def _infer_feature_lists(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    target_col: str,
    pred_col: Optional[str],
) -> Tuple[List[str], List[str]]:
    """Union numeric/categorical across datasets; exclude target/pred from features."""
    num = set(ref.select_dtypes(include="number").columns) | set(cur.select_dtypes(include="number").columns)
    cat = set(ref.select_dtypes(exclude="number").columns) | set(cur.select_dtypes(exclude="number").columns)
    for col in [target_col, pred_col]:
        if col:
            num.discard(col)
            cat.discard(col)
    return sorted(num), sorted(cat)


def _load_and_prepare(
    reference_path: str,
    current_path: str,
    target_col: str,
    pred_col: Optional[str],
):
    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_path)

    # align schemas (in-memory only)
    ref_df, cur_df = _align_columns(ref_df, cur_df)

    # cast label-like cols to string (robust for 0/1 or text)
    for df in (ref_df, cur_df):
        if target_col in df.columns:
            df[target_col] = df[target_col].astype("string")
        if pred_col and pred_col in df.columns:
            df[pred_col] = df[pred_col].astype("string")

    # features
    numeric_features, categorical_features = _infer_feature_lists(ref_df, cur_df, target_col, pred_col)

    # column mapping (only map prediction if present in BOTH to avoid partially present errors)
    cm = ColumnMapping()
    cm.target = target_col if (target_col in ref_df.columns or target_col in cur_df.columns) else None
    cm.prediction = pred_col if (pred_col and pred_col in ref_df.columns and pred_col in cur_df.columns) else None
    cm.numerical_features = numeric_features
    cm.categorical_features = categorical_features

    return ref_df, cur_df, cm


# --------- report (for /model-drift, /model-drift.html) --------------
def build_classification_report(
    reference_path: str,
    current_path: str,
    target_col: str = "loan_status",
    pred_col: str = "prediction",
) -> Report:
    # Reuse the same loader/mapping
    ref_df, cur_df, cm = _load_and_prepare(reference_path, current_path, target_col, pred_col)

    # --- ClassificationPreset needs consistent label dtypes across both datasets ---
    # Convert to numeric ints (0/1), to avoid mixed str/int issues during label sorting.
    for name, df in (("reference", ref_df), ("current", cur_df)):
        if target_col not in df.columns:
            raise ValueError(f"ClassificationPreset requires {name} to contain '{target_col}'")
        if pred_col not in df.columns:
            raise ValueError(f"ClassificationPreset requires {name} to contain '{pred_col}'")

        # If CSVs already have ints, this is a no-op. If they were strings, this coerces cleanly.
        df[target_col] = pd.to_numeric(df[target_col], errors="raise").astype(int)
        df[pred_col] = pd.to_numeric(df[pred_col], errors="raise").astype(int)

    # Build and run the classification performance report
    report = Report(metrics=[ClassificationPreset()])
    # v0.5.1 ordering: reference first, then current
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=cm)
    return report


# ---------- report (for /drift, /drift.html) ----------

def build_report(
    reference_path: str,
    current_path: str,
    target_col: str = "loan_status",
    pred_col: str = "prediction",
) -> Report:
    ref_df, cur_df, cm = _load_and_prepare(reference_path, current_path, target_col, pred_col)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=cm)
    return report


# ---------- tests (for /tests, /tests.html) ----------

def build_drift_suite(
    reference_path: str,
    current_path: str,
    target_col: str = "loan_status",
    pred_col: str = "prediction",
) -> TestSuite:
    ref_df, cur_df, cm = _load_and_prepare(reference_path, current_path, target_col, pred_col)
    suite = TestSuite(tests=[DataDriftTestPreset()])
    suite.run(reference_data=ref_df, current_data=cur_df, column_mapping=cm)
    return suite


def suite_json(
    reference_path: str,
    current_path: str,
    target_col: str = "loan_status",
    pred_col: str = "prediction",
) -> str:
    """Return Evidently TestSuite JSON string (safer than jsonify for numpy types)."""
    return build_drift_suite(reference_path, current_path, target_col, pred_col).json()


def save_suite_html(
    out_path: str,
    reference_path: str,
    current_path: str,
    target_col: str = "loan_status",
    pred_col: str = "prediction",
) -> str:
    """Save TestSuite HTML and return the path."""
    suite = build_drift_suite(reference_path, current_path, target_col, pred_col)
    suite.save_html(out_path)
    return out_path
