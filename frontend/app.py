from flask import Flask, render_template, jsonify
from flask_cors import CORS
import os
import logging
import json
from datetime import datetime

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_URL = os.environ.get("API_URL", "")  # set by Cloud Run deploy

@app.route("/health", methods=["GET"])
def health():
    # Log structured metrics for Cloud Monitoring
    logger.info(json.dumps({
        "service": "frontend",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "custom_metric": "service_health"
    }))
    return jsonify(status="ok", model_url=API_URL), 200

@app.route("/")
def home():
    return render_template("index.html", api_url=API_URL) # api_url will be used in index.html (case sensitive)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080")) # port 3001 for frontend local development
    app.run(host="0.0.0.0", port=port)
