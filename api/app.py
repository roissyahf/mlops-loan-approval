from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import requests
import json
import uuid
import time
from datetime import datetime, timezone
from urllib.parse import urljoin
import google.cloud.logging
from google.cloud.logging_v2.handlers import StructuredLogHandler
from google.oauth2 import id_token
from google.auth.transport.requests import Request

app = Flask(__name__)
CORS(app)

# ---- Config ----
MODEL_SERVICE_URL = os.environ.get("MODEL_URL", "").rstrip("/")  # set by Cloud Run deploy
#MODEL_SERVICE_URL = os.environ.get("MODEL_URL", "http://model:5000/predict") # for local deployment
if not MODEL_SERVICE_URL:
    raise RuntimeError("MODEL URL is not set!")
print(json.dumps({"message": "Using MODEL_URL", "url": MODEL_SERVICE_URL}))
PREDICT_URL = urljoin(MODEL_SERVICE_URL + "/", "predict") 

LOG_PATH = os.environ.get("LOG_PATH", "/tmp/current.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
LOG_PATH = os.path.abspath(LOG_PATH)

MODEL_NAME = (os.environ.get("MODEL_NAME", "loan-xgb-tuning") or "").strip()
MODEL_VERSION = (os.environ.get("MODEL_VERSION", "prod-1") or "").strip()
FEATURE_PIPELINE_VERSION = (os.environ.get("FEATURE_PIPELINE_VERSION", "local-dev") or "").strip()
SCHEMA_VERSION = "1.0"


print(json.dumps({"message": "Using MODEL_URL", "url": MODEL_SERVICE_URL}))
print(json.dumps({"message": "Logging to path", "path": LOG_PATH}))
print(json.dumps({
    "message": "Model meta",
    "model_name": MODEL_NAME,
    "model_version": MODEL_VERSION,
    "feature_pipeline_version": FEATURE_PIPELINE_VERSION
}))


def get_id_token():
    """
    Fetches a Google-signed ID token for the given audience (usually the target service URL).
    """
    try:
        token = id_token.fetch_id_token(Request(), MODEL_SERVICE_URL)
        return token
    except Exception as e:
        print(json.dumps({"message": "Failed to fetch ID token", "error": str(e)}))
        raise


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_model_response(resp_json):
    """
    Extraction of prediction and probability from the model response.
    Accepts common key variants. Returns (prediction, prediction_proba or None).
    """
    if resp_json is None:
        return None, None

    candidates = []
    if isinstance(resp_json, dict):
        candidates.append(resp_json)
        if isinstance(resp_json.get("prediction"), dict):
            candidates.append(resp_json["prediction"])

    pred_keys = ["prediction", "pred", "label", "output", "y_hat"]
    proba_keys = ["prediction_proba", "proba", "probability", "score"]

    pred = None
    proba = None
    for d in candidates:
        for k in pred_keys:
            if k in d:
                pred = d[k]
                break
        for k in proba_keys:
            if k in d:
                proba = d[k]
                break
        if pred is not None:
            break

    return pred, proba


def _write_jsonl(path: str, obj: dict):
    # for locally write to data/simulation/current.jsonl
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        print(json.dumps({"message": "Failed to append inference event to JSONL", "path": path, "error": str(e)}))
    

def _log_and_persist_event(event: dict):
    # persist locally (for retraining JSONL)
    try:
        _write_jsonl(LOG_PATH, event)
    except Exception as e:
        print(json.dumps({"message": "Failed to append inference event to JSONL", "path": LOG_PATH, "error": str(e)}))

    # emit to stdout for Cloud Logging
    try:
        # Direct print of JSON object for Cloud Logging
        print(json.dumps({
            "message": "prediction_event",
            "event": event
        }))
    except Exception as e:
        print(json.dumps({"message": "Failed to log event to stdout", "error": str(e)}))


def log_current_row(features_dict: dict, prediction, prediction_proba=None):
    """
    Use this anywhere (batch or online) to keep logs consistent with /predict.
    Writes to the same JSONL with the same schema, so we can call it from anywhere and stay consistent.
    """
    event = {
        "schema_version": SCHEMA_VERSION,
        "request_id": str(uuid.uuid4()),
        "timestamp_utc": _utc_now_iso(),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "request_source": "api",
        "endpoint": "/predict",
        "latency_ms": None,        
        **(features_dict or {}),
        "prediction": prediction
    }
    if prediction_proba is not None:
        event["prediction_proba"] = prediction_proba

    _write_jsonl(LOG_PATH, event)

    # Direct print of JSON object for Cloud Logging
    print(json.dumps({
        "message": "prediction_event",
        "event": event
    }))


@app.route("/", methods=["GET"])
def index():
    return "API Service is Running", 200


@app.route("/health", methods=["GET"])
def health():
    # Direct print of JSON object for Cloud Logging
    print(json.dumps({
        "service": "api",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "custom_metric": "service_health"
    }))
    return jsonify(status="ok", model_url=MODEL_SERVICE_URL), 200


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict_endpoint():
    """
    Make prediction, also Appends one JSON object per request to LOG_PATH (default ./data/simulation/current.jsonl)
    """
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    aud = MODEL_SERVICE_URL
    tok = id_token.fetch_id_token(Request(), aud)
    print(json.dumps({"message": "CALLING MODEL", "url": PREDICT_URL, "audience": aud, "token_len": len(tok) if tok else 0}))

    request_id = str(uuid.uuid4())
    started = time.perf_counter()

    try:
        response = requests.post(PREDICT_URL, json=payload,
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type":"application/json"}, timeout=10)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        print(json.dumps({"message": "Model response", "status_code": response.status_code, "text": response.text}))

        resp_json = None
        try:
            resp_json = response.json()
        except Exception:
            pass

        prediction, prediction_proba = _parse_model_response(resp_json)

        event = {
            "schema_version": SCHEMA_VERSION,
            "request_id": request_id,
            "timestamp_utc": _utc_now_iso(),
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
            "latency_ms": elapsed_ms,
            "request_source": "api",
            "endpoint": "/predict",
        }

        if isinstance(payload, dict):
            event.update(payload)

        event["prediction"] = prediction
        if prediction_proba is not None:
            event["prediction_proba"] = prediction_proba

        _log_and_persist_event(event)

        if resp_json is not None:
            return jsonify(resp_json), response.status_code
        else:
            return response.text, response.status_code

    except Exception as e:
        print(json.dumps({"message": "Model service error", "error": str(e)}))
        return jsonify({"error": "Model service failed",
                         "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080)) # port 8000 for API local development
    app.run(host="0.0.0.0", port=port)
