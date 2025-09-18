import os, subprocess, sys
from flask import Flask, jsonify, send_file, abort, Response
from flask_cors import CORS
from evidently_profile import build_report, build_classification_report, suite_json, save_suite_html
import pandas as pd
import logging

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flags for data availability
DATA_READY = False
DVC_ATTEMPTED = False

def dvc_pull_async():
    """
    Try DVC pull in background, don't block startup
    """
    global DATA_READY, DVC_ATTEMPTED
    
    if DVC_ATTEMPTED:
        return
    
    DVC_ATTEMPTED = True
    user = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    if not user or not token:
        logger.warning("[dvc] DAGSHUB_USERNAME / DAGSHUB_TOKEN not set; skipping pull")
        return

    try:
        logger.info("[dvc] Starting background pull...")
        
        # Configure DVC remote auth (scoped to container)
        subprocess.run(
            ["dvc", "remote", "modify", "dagshub", "auth", "basic"],
            check=True, capture_output=True, timeout=30
        )
        subprocess.run(
            ["dvc", "remote", "modify", "dagshub", "user", user],
            check=True, capture_output=True, timeout=30
        )
        subprocess.run(
            ["dvc", "remote", "modify", "dagshub", "--local", "password", token],
            check=True, capture_output=True, timeout=30
        )

        # Pull only what this service needs with timeout
        subprocess.run(["dvc","pull",
                        "data/simulation/reference_data.csv",
                        "data/simulation/current_data.csv"], 
                      check=True, timeout=120)

        logger.info("[dvc] pull completed successfully")
        DATA_READY = True
        
    except subprocess.TimeoutExpired:
        logger.error("[dvc] pull timeout - continuing without data")
    except subprocess.CalledProcessError as e:
        logger.error(f"[dvc] failed: {e}\nSTDERR: {e.stderr.decode() if e.stderr else ''}")
    except Exception as e:
        logger.error(f"[dvc] unexpected error: {e}")

def resolve_path(p: str) -> str:
    """
    Make env-provided paths robust across:
    - Windows / Linux
    - running from project root vs monitoring/
    - absolute vs relative
    """
    if p is None:
        return ""
    p = p.strip()

    # Leave GCS URIs or other schemes alone
    if p.startswith("gs://"):
        return p

    # Absolute path: Normalize and return.
    if os.path.isabs(p):
        return os.path.normpath(p)

    # Try relative to current working directory
    cwd_path = os.path.normpath(os.path.join(os.getcwd(), p))
    if os.path.exists(cwd_path):
        return cwd_path

    # Fallback: relative to this file's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path = os.path.normpath(os.path.join(script_dir, p))
    return alt_path

# Run from project root: python monitoring/app.py
REF_DATA_PATH = resolve_path(os.getenv("REF_DATA_PATH", "data/simulation/reference_data.csv"))
CURR_DATA_PATH = resolve_path(os.getenv("CURR_DATA_PATH", "data/simulation/current_data.csv"))

@app.route("/", methods=["GET"])
def index():
    return "Monitoring Service is Running", 200

@app.route("/health", methods=["GET"])
def health():
    """
    Always return healthy for Cloud Run startup - data issues are not fatal
    """
    ref_exists = os.path.exists(REF_DATA_PATH)
    curr_exists = os.path.exists(CURR_DATA_PATH)
    
    # Try DVC pull if not attempted and data missing
    if not DATA_READY and not DVC_ATTEMPTED and (not ref_exists or not curr_exists):
        dvc_pull_async()
        # Re-check after attempt
        ref_exists = os.path.exists(REF_DATA_PATH)
        curr_exists = os.path.exists(CURR_DATA_PATH)
    
    status = "ok"
    return jsonify({
        "status": status,
        "ref": REF_DATA_PATH,
        "ref_exists": ref_exists,
        "curr": CURR_DATA_PATH,
        "curr_exists": curr_exists,
        "cwd": os.getcwd(),
        "data_ready": DATA_READY,
        "dvc_attempted": DVC_ATTEMPTED
    })

def check_data_files():
    """Helper to ensure data is available before processing"""
    if not DATA_READY and not DVC_ATTEMPTED:
        dvc_pull_async()
    
    if not (os.path.exists(REF_DATA_PATH) and os.path.exists(CURR_DATA_PATH)):
        return False
    return True

# --------- (for /model-drift, /model-drift.html) --------------
@app.route("/model-drift", methods=["GET"])
def model_drift_json():
    if not check_data_files():
        return jsonify({"error": "reference or current file not found", 
                       "data_ready": DATA_READY}), 404
    
    try:
        # Build classification performance report
        report = build_classification_report(REF_DATA_PATH, CURR_DATA_PATH)
        # Return Evidently's JSON (safer than jsonify for numpy types)
        return Response(report.json(), mimetype="application/json")
    except Exception as e:
        logger.error(f"Error generating model drift report: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/model-drift.html", methods=["GET"])
def model_drift_html():
    if not check_data_files():
        return jsonify({"error": "reference or current file not found"}), 404
    
    try:
        report = build_classification_report(REF_DATA_PATH, CURR_DATA_PATH)
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_drift_report.html")
        report.save_html(out_path)
        return send_file(out_path, mimetype="text/html")
    except Exception as e:
        logger.error(f"Error generating model drift HTML: {e}")
        return jsonify({"error": str(e)}), 500

# --------- (for /tests, /tests.html) --------------
@app.route("/tests", methods=["GET"])
def tests_json():
    if not check_data_files():
        return jsonify({"error": "reference or current file not found"}), 404
    
    try:
        return Response(suite_json(REF_DATA_PATH, CURR_DATA_PATH), mimetype="application/json")
    except Exception as e:
        logger.error(f"Error generating tests: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/tests.html", methods=["GET"])
def tests_html():
    if not check_data_files():
        return jsonify({"error": "reference or current file not found"}), 404
    
    try:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_tests.html")
        save_suite_html(out_path, REF_DATA_PATH, CURR_DATA_PATH)
        return send_file(out_path, mimetype="text/html")
    except Exception as e:
        logger.error(f"Error generating tests HTML: {e}")
        return jsonify({"error": str(e)}), 500

# --------- (for /drift, /drift.html) --------------
@app.route("/drift", methods=["GET"])
def drift_json():
    if not check_data_files():
        return jsonify({"error": "reference or current file not found"}), 404
    
    try:
        report = build_report(REF_DATA_PATH, CURR_DATA_PATH)
        # Use Evidently's serializer to avoid numpy/typing issues with jsonify
        return Response(report.json(), mimetype="application/json")
    except Exception as e:
        logger.error(f"Error generating drift report: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/drift.html", methods=["GET"])
def drift_html():
    if not check_data_files():
        return jsonify({"error": "reference or current file not found"}), 404
    
    try:
        report = build_report(REF_DATA_PATH, CURR_DATA_PATH)
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_report.html")
        report.save_html(out_path)
        return send_file(out_path, mimetype="text/html")
    except Exception as e:
        logger.error(f"Error generating drift HTML: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))  # Use PORT env var from Cloud Run
    logger.info(f"Starting monitoring service on port {port}")
    logger.info(f"REF_DATA_PATH: {REF_DATA_PATH}")
    logger.info(f"CURR_DATA_PATH: {CURR_DATA_PATH}")
    
    app.run(host="0.0.0.0", port=port)