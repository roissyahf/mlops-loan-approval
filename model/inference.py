import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import time

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

# Load model (can be a pipeline)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# The exact features the model was trained on (post-transform)
REQUIRED_COLUMNS = [
    "person_age",
    "person_income",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "credit_score",
    "previous_loan_defaults_on_file",
]

def _extract_features_dict(input_payload: dict) -> dict:
    """
    Accepts common payload shapes:
      - flat: {"person_age": ..., ...}
      - wrapped: {"features": {...}} / {"input": {...}} / {"data": {...}} / {"payload": {...}}
    Returns a flat dict containing ONLY model feature keys (others are ignored).
    Missing keys are filled with None (the model/pipeline may still error if truly required).
    """
    if not isinstance(input_payload, dict):
        raise ValueError("Input payload must be a JSON object (dict).")

    # Unwrap common containers
    candidate = input_payload
    for k in ("features", "input", "data", "payload"):
        if k in input_payload and isinstance(input_payload[k], dict):
            candidate = input_payload[k]
            break

    out = {}
    for col in REQUIRED_COLUMNS:
        out[col] = candidate.get(col, None)
    return out

def predict(input_json):
    t0 = time.perf_counter()

    # Normalize/align input to the model's columns
    features_dict = _extract_features_dict(input_json)
    X = pd.DataFrame([features_dict], columns=REQUIRED_COLUMNS)

    # Optional: lightweight numeric coercion for numeric fields (keeps strings like "Yes"/"No" as-is)
    numeric_cols = [
        "person_age",
        "person_income",
        "loan_amnt",
        "loan_int_rate",
        "loan_percent_income",
        "credit_score",
    ]
    for c in numeric_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Predict
    y_pred = model.predict(X)[0]
    pred_proba = None
    probs_list = []

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        probs_list = probs.tolist()
        # binary classification; positive class proba usually at index 1
        if len(probs) >= 2:
            try:
                pred_proba = float(probs[1])
            except Exception:
                pred_proba = None

    latency_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "prediction": int(y_pred) if y_pred is not None else None,
        "prediction_proba": pred_proba,     # single scalar for API logger
        "probabilities": probs_list,        # keep full vector
        "latency_ms": latency_ms,
    }
