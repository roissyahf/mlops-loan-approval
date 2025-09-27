from flask import Flask, request, jsonify
import logging
from inference_dev import predict

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok")

@app.route("/", methods=["GET"])
def index():
    return "Model Service is Running", 200

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    data = request.get_json()
    result = predict(data)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)