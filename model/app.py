from flask import Flask, request, jsonify
import logging
from inference import predict
import os

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/", methods=["GET"])
def index():
    return "Model Service is Running", 200

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    result = predict(data)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080")) # port 5000 for model local development
    app.run(host="0.0.0.0", port=port)
