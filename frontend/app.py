from flask import Flask, render_template
from flask_cors import CORS
import os

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

API_URL = os.environ.get("API_URL", "")  # set by Cloud Run deploy

@app.route("/")
def home():
    return render_template("index.html", api_url=API_URL) # api_url will be used in index.html (case sensitive)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080")) # port 3001 for frontend local development
    app.run(host="0.0.0.0", port=port)
