import os
from flask import Flask, jsonify, send_file, abort, Response
from flask_cors import CORS
from evidently_profile import build_report, suite_json, save_suite_html
from convert_dev_logs import convert_jsonl_to_csv


app = Flask(__name__)
CORS(app)


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


REF_DATA_PATH = resolve_path(os.getenv("REF_DATA_PATH", "data/simulation/reference_data.csv"))
curr_proceed = convert_jsonl_to_csv("data/simulation/current.jsonl", "data/simulation/prediction_dev_log.csv")
CURR_DATA_PATH = resolve_path(os.getenv("CURR_DATA_PATH", curr_proceed))


@app.route("/", methods=["GET"])
def index():
    return "Monitoring Service is Running", 200

@app.route("/health", methods=["GET"])
def health():
    ref_exists = os.path.exists(REF_DATA_PATH)
    curr_exists = os.path.exists(CURR_DATA_PATH)
    status = "ok" if (ref_exists and curr_exists) else ("missing_reference" if not ref_exists else "missing_current")
    return jsonify({
        "status": status,
        "ref": REF_DATA_PATH,
        "ref_exists": ref_exists,
        "curr": CURR_DATA_PATH,
        "curr_exists": curr_exists,
        "cwd": os.getcwd()
    })

# --------- (for /tests, /tests.html) --------------
@app.route("/tests", methods=["GET"])
def tests_json():
    if not (os.path.exists(REF_DATA_PATH) and os.path.exists(CURR_DATA_PATH)):
        return abort(404, description="reference or current file not found")
    return Response(suite_json(REF_DATA_PATH, CURR_DATA_PATH), mimetype="application/json")

@app.route("/tests.html", methods=["GET"])
def tests_html():
    if not (os.path.exists(REF_DATA_PATH) and os.path.exists(CURR_DATA_PATH)):
        return abort(404, description="reference or current file not found")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_tests.html")
    save_suite_html(out_path, REF_DATA_PATH, CURR_DATA_PATH)
    return send_file(out_path, mimetype="text/html")


# --------- (for /drift, /drift.html) --------------
@app.route("/drift", methods=["GET"])
def drift_json():
    if not (os.path.exists(REF_DATA_PATH) and os.path.exists(CURR_DATA_PATH)):
        return abort(404, description="reference or current file not found")
    report = build_report(REF_DATA_PATH, CURR_DATA_PATH)
    # Use Evidently’s serializer to avoid numpy/typing issues with jsonify
    return Response(report.json(), mimetype="application/json")

@app.route("/drift.html", methods=["GET"])
def drift_html():
    if not (os.path.exists(REF_DATA_PATH) and os.path.exists(CURR_DATA_PATH)):
        return abort(404, description="reference or current file not found")
    report = build_report(REF_DATA_PATH, CURR_DATA_PATH)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_report.html")
    report.save_html(out_path)
    return send_file(out_path, mimetype="text/html")



if __name__ == "__main__":
    port = int(os.getenv("PORT", "5003"))
    app.run(host="0.0.0.0", port=port)
