import os, subprocess, sys
from flask import Flask, jsonify, send_file, abort, Response
from flask_cors import CORS
from evidently_profile import build_report, build_classification_report, suite_json, save_suite_html
import pandas as pd
import logging
import json
from datetime import datetime
import time
import threading

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global flags for data availability
DATA_READY = False
DVC_ATTEMPTED = False
DVC_IN_PROGRESS = False

def dvc_pull_background():
    """Run DVC pull in background thread to avoid blocking startup"""
    global DATA_READY, DVC_ATTEMPTED, DVC_IN_PROGRESS
    
    if DVC_ATTEMPTED or DVC_IN_PROGRESS:
        return
    
    DVC_IN_PROGRESS = True
    DVC_ATTEMPTED = True
    
    try:
        user = os.getenv("DAGSHUB_USERNAME")
        token = os.getenv("DAGSHUB_TOKEN")

        logger.info(f"[dvc] Starting background pull with user: {user[:5] if user else 'None'}...")
        
        if not user or not token:
            logger.error("[dvc] DAGSHUB_USERNAME / DAGSHUB_TOKEN not set")
            return

        # Initialize git repo if it doesn't exist (required for DVC)
        if not os.path.exists('.git'):
            logger.info("[dvc] Initializing git repository...")
            subprocess.run(["git", "init", "."], check=True, capture_output=True, timeout=30)
            subprocess.run(["git", "config", "user.email", "monitoring@example.com"], check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Monitoring Service"], check=True, capture_output=True)

        # Check if .dvc directory exists
        if not os.path.exists('.dvc'):
            logger.error("[dvc] .dvc directory missing!")
            return

        logger.info("[dvc] Configuring DVC remote...")

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
        logger.info("[dvc] Starting data pull...")
        subprocess.run(["dvc","pull",
                        "data/simulation/reference_data.csv",
                        "data/simulation/current_data.csv"], 
                      check=True, timeout=180)

        logger.info("[dvc] Data pull completed successfully")
        DATA_READY = True
        
    except subprocess.TimeoutExpired:
        logger.error("[dvc] Data pull timeout - service will continue without data")
    except subprocess.CalledProcessError as e:
        logger.error(f"[dvc] Data pull failed: {e}")
        if e.stderr:
            logger.error(f"[dvc] STDERR: {e.stderr.decode()}")
    except Exception as e:
        logger.error(f"[dvc] Unexpected error during data pull: {e}")
    finally:
        DVC_IN_PROGRESS = False

def resolve_path(p: str) -> str:
    """Make env-provided paths robust across different environments"""
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

# Path configuration
REF_DATA_PATH = resolve_path(os.getenv("REF_DATA_PATH", "data/simulation/reference_data.csv"))
CURR_DATA_PATH = resolve_path(os.getenv("CURR_DATA_PATH", "data/simulation/current_data.csv"))

@app.route("/", methods=["GET"])
def index():
    return "Monitoring Service is Running", 200

@app.route("/health", methods=["GET"])
def health():
    """Always return healthy - data issues are not fatal for startup"""
    ref_exists = os.path.exists(REF_DATA_PATH)
    curr_exists = os.path.exists(CURR_DATA_PATH)
    
    # Start background DVC pull if data is missing and not attempted yet
    if not DATA_READY and not DVC_ATTEMPTED and not DVC_IN_PROGRESS:
        if not ref_exists or not curr_exists:
            # Start background thread for DVC pull
            thread = threading.Thread(target=dvc_pull_background, daemon=True)
            thread.start()
    
    logger.info(json.dumps({
        "service": "monitoring",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "custom_metric": "service_health"
    }))

    return jsonify({
        "status": "ok",
        "ref_exists": ref_exists,
        "curr_exists": curr_exists,
        "data_ready": DATA_READY,
        "dvc_attempted": DVC_ATTEMPTED,
        "dvc_in_progress": DVC_IN_PROGRESS
    })

def check_data_files():
    """Helper to ensure data is available before processing"""
    ref_exists = os.path.exists(REF_DATA_PATH)
    curr_exists = os.path.exists(CURR_DATA_PATH)
    
    # Try to trigger background pull if not done yet
    if not DATA_READY and not DVC_ATTEMPTED and not DVC_IN_PROGRESS:
        if not ref_exists or not curr_exists:
            thread = threading.Thread(target=dvc_pull_background, daemon=True)
            thread.start()
    
    return ref_exists and curr_exists

@app.route("/model-drift", methods=["GET"])
def model_drift_json():
    if not check_data_files():
        return jsonify({
            "error": "reference or current file not found", 
            "data_ready": DATA_READY,
            "dvc_in_progress": DVC_IN_PROGRESS,
            "message": "Data may still be downloading in background"
        }), 404
    
    try:
        report = build_classification_report(REF_DATA_PATH, CURR_DATA_PATH)
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

@app.route("/drift", methods=["GET"])
def drift_json():
    if not check_data_files():
        return jsonify({"error": "reference or current file not found"}), 404
    
    try:
        report = build_report(REF_DATA_PATH, CURR_DATA_PATH)
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
    port = int(os.getenv("PORT", "8080"))
    logger.info(f"Starting monitoring service on port {port}")
    logger.info(f"REF_DATA_PATH: {REF_DATA_PATH}")
    logger.info(f"CURR_DATA_PATH: {CURR_DATA_PATH}")
    
    # Start the Flask app immediately
    app.run(host="0.0.0.0", port=port)