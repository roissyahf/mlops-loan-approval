from flask import Flask, request, app, jsonify
from flask_cors import CORS 
import os
import logging
import requests

MODEL_SERVICE_URL = os.environ.get("MODEL_URL", "http://model:5000/predict")  # internal Docker DNS
#MODEL_SERVICE_URL = os.getenv("MODEL_URL", "http://localhost:5000/predict") # for local test

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    logger.info(f"Received input: {data}")

    try:
        response = requests.post(MODEL_SERVICE_URL, json=data)
        logger.info(f"Model response: {response.status_code} - {response.text}")
        return jsonify(response.json())
    except Exception as e:
        logger.error(f"Model service error: {e}")
        return jsonify({"error": "Model service failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
