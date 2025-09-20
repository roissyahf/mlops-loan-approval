from flask import Flask, request, jsonify
import logging
from inference import predict
import os
import logging
import json
from datetime import datetime
#import mlflow
#import joblib
#from pathlib import Path

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/", methods=["GET"])
def index():
    return "Model Service is Running", 200

@app.route("/health", methods=["GET"])
def health():
    # Log structured metrics for Cloud Monitoring
    logger.info(json.dumps({
        "service": "model",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "custom_metric": "service_health"
    }))
    return jsonify(status="ok")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    result = predict(data)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080")) # port 5000 for model local development
    app.run(host="0.0.0.0", port=port)
