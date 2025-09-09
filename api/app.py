from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import requests
import json
import uuid
import time
from datetime import datetime, timezone

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- Config ----
MODEL_SERVICE_URL = os.environ.get("MODEL_URL", "").rstrip("/")  # set by Cloud Run deploy
#MODEL_SERVICE_URL = os.environ.get("MODEL_URL", "http://model:5000/predict") # for local deployment
if not MODEL_SERVICE_URL:
    raise RuntimeError("MODEL URL is not set!")
logger.info(f"Using MODEL_URL={MODEL_SERVICE_URL!r}")


LOG_PATH = os.environ.get("LOG_PATH", "./data/simulation/current.jsonl")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
LOG_PATH = os.path.abspath(LOG_PATH)


MODEL_NAME = (os.environ.get("MODEL_NAME", "loan-xgb-tuning") or "").strip()
MODEL_VERSION = (os.environ.get("MODEL_VERSION", "prod-1") or "").strip()
FEATURE_PIPELINE_VERSION = (os.environ.get("FEATURE_PIPELINE_VERSION", "local-dev") or "").strip()
SCHEMA_VERSION = "1.0"


logger.info(f"Using MODEL_URL={MODEL_SERVICE_URL!r}")
logger.info(f"Logging to {LOG_PATH!r}")
logger.info(f"Model meta: name={MODEL_NAME!r}, version={MODEL_VERSION!r}, pipeline={FEATURE_PIPELINE_VERSION!r}")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_model_response(resp_json):
    """
    Best-effort extraction of prediction and probability from the model response.
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
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.error(f"Failed to append inference event to {path}: {e}")


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
        "request_source": "api",   # adjust if calling from batch
        "endpoint": "/predict",
        "latency_ms": None,        
        **(features_dict or {}),
        "prediction": prediction
    }
    if prediction_proba is not None:
        event["prediction_proba"] = prediction_proba

    _write_jsonl(LOG_PATH, event)


@app.route("/", methods=["GET"])
def index():
    return "API Service is Running", 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", model_url=MODEL_SERVICE_URL), 200


@app.route("/predict", methods=["POST", "OPTIONS"])
def predict_endpoint():
    """
    Make prediction, also Appends one JSON object per request to LOG_PATH (default ./data/simulation/current.jsonl)
    """
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json()
    logger.info(f"Received input: {payload}")

    request_id = str(uuid.uuid4())
    started = time.perf_counter()

    try:
        response = requests.post(MODEL_SERVICE_URL, json=payload, timeout=10)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        logger.info(f"Model response: {response.status_code} - {response.text}")

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

        _write_jsonl(LOG_PATH, event)

        if resp_json is not None:
            return jsonify(resp_json), response.status_code
        else:
            return response.text, response.status_code

    except Exception as e:
        logger.error(f"Model service error: {e}")
        return jsonify({"error": "Model service failed",
                        "details": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080)) # port 8000 for API local development
    app.run(host="0.0.0.0", port=port)
