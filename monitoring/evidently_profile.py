import pandas as pd
from typing import Tuple, Optional

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping

from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset


# ---- configure the feature set for data drift (no target/pred) ----
FEATURE_NUMERIC = [
    "person_age",
    "person_income",
    "loan_amnt",
    "loan_percent_income",
    "loan_int_rate",
    "credit_score",
]
FEATURE_CATEGORICAL = [
    "previous_loan_defaults_on_file",
]
FEATURES_ALL = FEATURE_NUMERIC + FEATURE_CATEGORICAL


def _align_and_filter_features(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, ColumnMapping]:
    """Keep only the intersection of the configured feature set present in both frames.
       Coerce numeric features safely; keep categorical as string.
    """
    # Keep only columns present in both datasets
    present = sorted(set(ref.columns) & set(cur.columns) & set(FEATURES_ALL))
    if not present:
        raise ValueError("No common feature columns found between reference and current data.")

    # Select only those features
    ref = ref[present].copy()
    cur = cur[present].copy()

    # Split what’s present into numeric/categorical
    numeric_features = [c for c in FEATURE_NUMERIC if c in present]
    categorical_features = [c for c in FEATURE_CATEGORICAL if c in present]

    # Coerce numerics, keep categorical as string
    for col in numeric_features:
        if col in ref.columns:
            ref[col] = pd.to_numeric(ref[col], errors="coerce")
        if col in cur.columns:
            cur[col] = pd.to_numeric(cur[col], errors="coerce")

    for col in categorical_features:
        if col in ref.columns:
            ref[col] = ref[col].astype("string")
        if col in cur.columns:
            cur[col] = cur[col].astype("string")

    # Drop rows ONLY if a required numeric feature is NaN
    if numeric_features:
        ref = ref.dropna(subset=numeric_features).reset_index(drop=True)
        cur = cur.dropna(subset=numeric_features).reset_index(drop=True)

    if len(ref) == 0:
        raise ValueError("Reference DataFrame empty after cleaning required numeric features.")
    if len(cur) == 0:
        raise ValueError("Current DataFrame empty after cleaning required numeric features.")

    # Build ColumnMapping for feature-only drift (no target/pred)
    cm = ColumnMapping()
    cm.target = None
    cm.prediction = None
    cm.numerical_features = numeric_features
    cm.categorical_features = categorical_features

    return ref, cur, cm


def _load_and_prepare(
    reference_path: str,
    current_path: str,
    target_col: str,          # kept for signature compatibility, unused in drift-only
    pred_col: Optional[str],  # kept for signature compatibility, unused in drift-only
):
    # Read files (already materialized to CSV in app.py for current=BQ)
    ref_df = pd.read_csv(reference_path)
    cur_df = pd.read_csv(current_path)

    # Only use feature columns, coerce/clean minimally
    ref_df, cur_df, cm = _align_and_filter_features(ref_df, cur_df)
    return ref_df, cur_df, cm


def build_report(
    reference_path: str,
    current_path: str,
    target_col: str = "loan_status",   # ignored in drift-only build
    pred_col: str = "prediction",      # ignored in drift-only build
) -> Report:
    ref_df, cur_df, cm = _load_and_prepare(reference_path, current_path, target_col, pred_col)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df, column_mapping=cm)
    return report


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
    return build_drift_suite(reference_path, current_path, target_col, pred_col).json()


def save_suite_html(
    out_path: str,
    reference_path: str,
    current_path: str,
    target_col: str = "loan_status",
    pred_col: str = "prediction",
) -> str:
    suite = build_drift_suite(reference_path, current_path, target_col, pred_col)
    suite.save_html(out_path)
    return out_path
