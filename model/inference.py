import os
import time
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import joblib
from pathlib import Path

# Detect if running inside container (/app/) or locally
DEFAULT_PATH = Path(__file__).parent / "model.pkl"
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_PATH))

# Global model variable
model = None

def _load_from_registry_or_latest_artifact():
    """Try registry first; fallback to latest run's model artifact."""
    uri   = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    user  = os.getenv("MLFLOW_TRACKING_USERNAME", "").strip()
    pwd   = os.getenv("MLFLOW_TRACKING_PASSWORD", "").strip()
    name  = os.getenv("MLFLOW_MODEL_NAME", "XGB-retraining").strip()
    stage = os.getenv("MLFLOW_MODEL_STAGE", "Production").strip()
    exp   = os.getenv("MLFLOW_EXPERIMENT_NAME", "").strip()

    if not uri or not user or not pwd:
        raise RuntimeError("Missing MLflow creds/URI: MLFLOW_TRACKING_URI/USERNAME/PASSWORD")

    os.environ["MLFLOW_TRACKING_USERNAME"] = user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = pwd
    mlflow.set_tracking_uri(uri)

    # 1) Try Registry stage
    try:
        model_uri = f"models:/{name}/{stage}"
        print(f"[model-bootstrap] Trying registry: {model_uri}", flush=True)
        m = mlflow.sklearn.load_model(model_uri)
        print(f"[model-bootstrap] Successfully loaded model from registry", flush=True)
        return m
    except Exception as e:
        print(f"[model-bootstrap] Registry load failed: {e}", flush=True)

    # 2) Fallback: latest run's model artifact (requires experiment name)
    if exp:
        try:
            exp_obj = mlflow.get_experiment_by_name(exp)
            if exp_obj is None:
                raise RuntimeError(f"Experiment '{exp}' not found")
            df = mlflow.search_runs(
                experiment_ids=[exp_obj.experiment_id],
                order_by=["attributes.start_time DESC"],
                max_results=1,
                filter_string="attributes.status = 'FINISHED'"
            )
            if df is not None and not df.empty:
                run_id = df.iloc[0]["run_id"]
                run_uri = f"runs:/{run_id}/model"
                print(f"[model-bootstrap] Falling back to latest run: {run_uri}", flush=True)
                m = mlflow.sklearn.load_model(run_uri)
                return m
            else:
                raise RuntimeError("No finished runs found to fallback to.")
        except Exception as e:
            print(f"[model-bootstrap] Fallback to latest run failed: {e}", flush=True)

    # 3) Try to find any available model version
    try:
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{name}'")
        if versions:
            latest_version = max(versions, key=lambda x: int(x.version))
            model_uri = f"models:/{name}/{latest_version.version}"
            print(f"[model-bootstrap] Trying latest version: {model_uri}", flush=True)
            m = mlflow.sklearn.load_model(model_uri)
            return m
    except Exception as e:
        print(f"[model-bootstrap] Latest version load failed: {e}", flush=True)

    return None

def ensure_model_present():
    """Fetch model from MLflow if not present"""
    need_fetch = os.getenv("FORCE_REFRESH_MODEL", "0") == "1" or not os.path.exists(MODEL_PATH)
    if not need_fetch:
        return True

    try:
        print("[model-bootstrap] Downloading model from MLflow...", flush=True)
        m = _load_from_registry_or_latest_artifact()
        if m is None:
            print("[model-bootstrap] No model could be loaded; service will start but predictions will fail", flush=True)
            return False

        joblib.dump(m, MODEL_PATH)
        print(f"[model-bootstrap] Saved model to {MODEL_PATH}", flush=True)
        return True
    except Exception as e:
        print(f"[model-bootstrap] Model download failed: {e}", flush=True)
        return False

def load_model():
    """Load model with retry logic"""
    global model
    
    if model is not None:
        return model
    
    # Try to load from file first
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            print(f"[model-bootstrap] Loaded model from {MODEL_PATH}", flush=True)
            return model
        except Exception as e:
            print(f"[model-bootstrap] Failed to load from file: {e}", flush=True)
    
    # Try to download from MLflow
    if ensure_model_present():
        try:
            model = joblib.load(MODEL_PATH)
            return model
        except Exception as e:
            print(f"[model-bootstrap] Failed to load downloaded model: {e}", flush=True)
    
    return None

# Initialize model at startup (but don't fail if it doesn't work)
try:
    print("[model-bootstrap] Starting model initialization...", flush=True)
    model = load_model()
    if model:
        print("[model-bootstrap] Model loaded successfully", flush=True)
    else:
        print("[model-bootstrap] Model not loaded - will retry on first prediction", flush=True)
except Exception as e:
    print(f"[model-bootstrap] Model initialization failed: {e}", flush=True)
    model = None

REQUIRED_COLUMNS = [
    "person_age",
    "person_income", 
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "credit_score",
    "previous_loan_defaults_on_file",
]

def predict(input_json):
    global model
    
    # Lazy load model if not available
    if model is None:
        print("[model-bootstrap] Model not loaded, attempting to load...", flush=True)
        model = load_model()
        if model is None:
            raise RuntimeError("Model is not available. Please check MLflow connection and model registration.")
    
    t0 = time.perf_counter()
    features_dict = _extract_features_dict(input_json)
    X = pd.DataFrame([features_dict], columns=REQUIRED_COLUMNS)
    
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

    y_pred = model.predict(X)[0]
    pred_proba = None
    probs_list = []

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        probs_list = probs.tolist()
        if len(probs) >= 2:
            try:
                pred_proba = float(probs[1])
            except Exception:
                pred_proba = None

    latency_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "prediction": int(y_pred) if y_pred is not None else None,
        "prediction_proba": pred_proba,
        "probabilities": probs_list,
        "latency_ms": latency_ms,
    }

def _extract_features_dict(input_payload: dict) -> dict:
    """Extract features from input payload"""
    if not isinstance(input_payload, dict):
        raise ValueError("Input payload must be a JSON object (dict).")

    candidate = input_payload
    for k in ("features", "input", "data", "payload"):
        if k in input_payload and isinstance(input_payload[k], dict):
            candidate = input_payload[k]
            break

    out = {}
    for col in REQUIRED_COLUMNS:
        out[col] = candidate.get(col, None)
    return out