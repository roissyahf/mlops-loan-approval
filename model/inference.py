import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np
from pathlib import Path
import mlflow
import pandas as pd
import time
import joblib


# Detect if running inside container (/app/) or locally
DEFAULT_PATH = Path(__file__).parent / "model.pkl"
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_PATH))


def ensure_model_present():
    """Fetch model from MLflow Registry if not present (or FORCE_REFRESH_MODEL=1)."""
    need_fetch = os.getenv("FORCE_REFRESH_MODEL", "0") == "1" or not os.path.exists(MODEL_PATH)

    if not need_fetch:
        return

    uri   = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    user  = os.getenv("MLFLOW_TRACKING_USERNAME", "").strip()
    pwd   = os.getenv("MLFLOW_TRACKING_PASSWORD", "").strip()
    name  = os.getenv("MLFLOW_MODEL_NAME", "XGB-best-model-manual").strip()
    stage = os.getenv("MLFLOW_MODEL_STAGE", "Production").strip()

    if not uri or not user or not pwd:
        raise RuntimeError("Missing MLflow creds/URI: MLFLOW_TRACKING_URI/USERNAME/PASSWORD")

    os.environ["MLFLOW_TRACKING_USERNAME"] = user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd
    mlflow.set_tracking_uri(uri)

    model_uri = f"models:/{name}/{stage}"
    print(f"[model-bootstrap] Loading {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    joblib.dump(model, MODEL_PATH)
    print("[model-bootstrap] Saved", MODEL_PATH)

# call before creating routes
ensure_model_present()
model = joblib.load(MODEL_PATH)

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
