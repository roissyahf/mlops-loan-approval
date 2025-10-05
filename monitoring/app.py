import os, subprocess
from flask import Flask, jsonify, send_file, Response
from flask_cors import CORS
from evidently_profile import build_report, suite_json, save_suite_html 
import pandas as pd
import logging
from google.cloud import bigquery
from dotenv import load_dotenv, find_dotenv
import tempfile

# load environment variables from .env file
load_dotenv(find_dotenv())

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --------- (reference data config) --------------
# Global flags for data availability
DATA_READY = False
DVC_ATTEMPTED = False

def dvc_pull_async():
    global DATA_READY, DVC_ATTEMPTED
    
    if DVC_ATTEMPTED:
        return
    
    DVC_ATTEMPTED = True
    user = os.getenv("DAGSHUB_USERNAME")
    token = os.getenv("DAGSHUB_TOKEN")

    logger.info(f"[dvc] Starting pull with user: {user[:5] if user else 'None'}...")
    logger.info(f"[dvc] Token length: {len(token) if token else 0}")
    
    if not user or not token:
        logger.error("[dvc] DAGSHUB_USERNAME / DAGSHUB_TOKEN not set")
        return

    try:
        # Initialize git repo if it doesn't exist (required for DVC)
        if not os.path.exists('.git'):
            logger.info("[dvc] Initializing git repository...")
            subprocess.run(["git", "init", "."], check=True, capture_output=True, timeout=30)
            subprocess.run(["git", "config", "user.email", "monitoring@example.com"], check=True, capture_output=True)
            subprocess.run(["git", "config", "user.name", "Monitoring Service"], check=True, capture_output=True)

        # Show current working directory and files
        logger.info(f"[dvc] CWD: {os.getcwd()}")
        logger.info(f"[dvc] Files: {os.listdir('.')}")
        
        # Check if .dvc directory exists
        if os.path.exists('.dvc'):
            logger.info(f"[dvc] .dvc contents: {os.listdir('.dvc')}")
        else:
            logger.error("[dvc] .dvc directory missing!")
            return

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
                         "data/simulation/reference_data.csv"], 
                       check=True, timeout=120)

        logger.info("[dvc] pull completed successfully")
        DATA_READY = True
        
    except subprocess.TimeoutExpired:
        logger.error("[dvc] pull timeout - continuing without data")
    except subprocess.CalledProcessError as e:
        logger.error(f"[dvc] failed: {e}\nSTDERR: {e.stderr.decode() if e.stderr else ''}")
    except Exception as e:
        logger.error(f"[dvc] unexpected error: {e}")

def resolve_ref_path(p: str) -> str:
    """
    Make env-provided paths robust.
    """
    if p is None:
        return ""
    p = p.strip()
    if p.startswith("gs://"):
        return p
    if os.path.isabs(p):
        return os.path.normpath(p)

    cwd_path = os.path.normpath(os.path.join(os.getcwd(), p))
    if os.path.exists(cwd_path):
        return cwd_path

    script_dir = os.path.dirname(os.path.abspath(__file__))
    alt_path = os.path.normpath(os.path.join(script_dir, p))
    return alt_path

REF_DATA_PATH = resolve_ref_path(os.getenv("REF_DATA_PATH", "data/simulation/reference_data.csv"))

# --------- (current data config) --------------
# Current data comes from BigQuery
CURR_BQ_PROJECT = os.getenv("GCP_PROJECT_ID")
CURR_BQ_DATASET = "loan_prediction_dataset"
CURR_BQ_TABLE = "prediction_events_clean" # table changed, the schema has enforced

# Global path for current data - initialized to None (fetched on demand)
CURR_DATA_PATH = None

def fetch_current_from_bq_to_csv() -> str:
    """Pull current rows from BigQuery with enhanced debugging."""
    if not (CURR_BQ_PROJECT and CURR_BQ_DATASET and CURR_BQ_TABLE):
        raise RuntimeError("Missing envs: GCP_PROJECT_ID / BIGQUERY_DATASET / BIGQUERY_TABLE")

    bq_id = f"{CURR_BQ_PROJECT}.{CURR_BQ_DATASET}.{CURR_BQ_TABLE}"
    query = f"""
        SELECT
            person_age, 
            person_income,
            loan_amnt,
            loan_percent_income,
            loan_int_rate,
            credit_score,
            previous_loan_defaults_on_file,
            prediction
        FROM `{bq_id}`
    """
    
    logger.info(f"[bq] Fetching current from {bq_id}")
    logger.info(f"[bq] Query: {query[:100]}...")  # Log first 100 chars
    
    try:
        client = bigquery.Client(project=CURR_BQ_PROJECT)
        logger.info("[bq] Client created successfully")
        
        # Run query
        query_job = client.query(query)
        logger.info(f"[bq] Query job started: {query_job.job_id}")
        
        # Wait for results with timeout
        df = query_job.result(timeout=60).to_dataframe()  # Add 60s timeout
        logger.info(f"[bq] Query completed, fetched {len(df)} rows")
        logger.info(f"[bq-DEBUG] First 5 rows:\n{df.head()}")
        logger.info(f"[bq-DEBUG] Non-null counts:\n{df.count()}")
        
    except Exception as e:
        logger.error(f"[bq] FETCH FAILED - Error type: {type(e).__name__}")
        logger.error(f"[bq] Error message: {str(e)}")
        import traceback
        logger.error(f"[bq] Full traceback:\n{traceback.format_exc()}")
        raise

    if df.empty:
        raise ValueError(f"[bq] No rows returned from {bq_id}")

    # Use a persistent temp path or create the directory
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, "current_bq_data.csv")
    
    df.to_csv(tmp_path, index=False)
    logger.info(f"[bq] Saved {len(df)} rows to {tmp_path}")
    
    # Verify the file was created
    if not os.path.exists(tmp_path):
        raise RuntimeError(f"[bq] Failed to create CSV file at {tmp_path}")
    
    logger.info(f"[bq] File verified at {tmp_path}, size: {os.path.getsize(tmp_path)} bytes")
    return tmp_path


# --------- (the app) --------------
@app.route("/", methods=["GET"])
def index():
    return "Monitoring Service is Running", 200

@app.route("/health", methods=["GET"])
def health():
    ref_exists = os.path.exists(REF_DATA_PATH)

    # DVC for reference only
    if not DATA_READY and not DVC_ATTEMPTED and not ref_exists:
        dvc_pull_async()
        ref_exists = os.path.exists(REF_DATA_PATH)

    bq_ok = False
    bq_error = None
    if CURR_BQ_PROJECT and CURR_BQ_DATASET and CURR_BQ_TABLE:
        try:
            client = bigquery.Client(project=CURR_BQ_PROJECT)
            client.query(
                f"SELECT 1 FROM `{CURR_BQ_PROJECT}.{CURR_BQ_DATASET}.{CURR_BQ_TABLE}` LIMIT 1"
            ).result()
            bq_ok = True
        except Exception as e:
            bq_error = str(e).split('\n')[0] 
            logger.error(f"[bq] health check failed: {e}")

    return jsonify({
        "status": "ok",
        "ref": REF_DATA_PATH,
        "ref_exists": ref_exists,
        "bq_mode": True,
        "bq_ok": bq_ok,
        "bq_error": bq_error,
        "cwd": os.getcwd(),
        "data_ready": DATA_READY,
        "dvc_attempted": DVC_ATTEMPTED
    })

def check_data_files():
    """Helper to ensure data is available before processing"""
    global CURR_DATA_PATH 

    # Check/Pull DVC data
    if not DATA_READY and not DVC_ATTEMPTED:
        dvc_pull_async()

    # Check/Fetch BQ data (Current Data)
    if CURR_DATA_PATH is None:
        try:
            logger.info("[data] Attempting to fetch current data from BigQuery...")
            CURR_DATA_PATH = fetch_current_from_bq_to_csv()
            logger.info(f"[data] Current data path set to: {CURR_DATA_PATH}")
        except Exception as e:
            # Log the actual error instead of silently passing
            logger.error(f"[data] Failed to fetch from BigQuery: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # Final check for both files
    ref_file_exists = os.path.exists(REF_DATA_PATH)
    curr_file_exists = CURR_DATA_PATH and os.path.exists(CURR_DATA_PATH)
    
    if not (ref_file_exists and curr_file_exists):
        if not ref_file_exists:
             logger.error(f"[data] Missing reference file: {REF_DATA_PATH}")
        if not curr_file_exists:
             logger.error(f"[data] Missing current file: {CURR_DATA_PATH}")
        return False
        
    return True

# --------- (for /tests, /tests.html) --------------
@app.route("/tests", methods=["GET"])
def tests_json():
    if not check_data_files():
        return jsonify({"error": "Reference or current file not found (DVC/BQ failure). Check console logs for details."}), 404
    
    try:
        return Response(suite_json(REF_DATA_PATH, CURR_DATA_PATH), mimetype="application/json")
    except Exception as e:
        logger.error(f"Error generating tests: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/tests.html", methods=["GET"])
def tests_html():
    if not check_data_files():
        return jsonify({"error": "Reference or current file not found (DVC/BQ failure). Check console logs for details."}), 404
    
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
        return jsonify({"error": "Reference or current file not found (DVC/BQ failure). Check console logs for details."}), 404
    
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
        return jsonify({"error": "Reference or current file not found (DVC/BQ failure). Check console logs for details."}), 404
    
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
    logger.info(f"CURR from BQ? {'yes' if (CURR_BQ_PROJECT and CURR_BQ_DATASET and CURR_BQ_TABLE) else 'no'}")

    app.run(host="0.0.0.0", port=port)