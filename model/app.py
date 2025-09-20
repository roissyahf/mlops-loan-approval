from flask import Flask, request, jsonify
import logging
from inference import predict, load_model
import os
import logging
import json
from datetime import datetime
import time

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/", methods=["GET"])
def index():
    return "Model Service is Running", 200


@app.route("/health", methods=["GET"])
def health():
    # Check if model is loaded
    try:
        model = load_model()
        model_status = "loaded" if model is not None else "not_loaded"
    except Exception:
        model_status = "error"
    
    logger.info(json.dumps({
        "service": "model",
        "status": "healthy",
        "model_status": model_status,
        "timestamp": datetime.now().isoformat(),
        "custom_metric": "service_health"
    }))
    return jsonify(status="ok", model_status=model_status)


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    try:
        data = request.get_json()
        result = predict(data)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080")) # port 5000 for local deployment
    time.sleep(2)
    app.run(host="0.0.0.0", port=port)