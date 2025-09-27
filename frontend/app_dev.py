from flask import Flask, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

@app.route("/")
def home():
    return render_template("index_dev.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3001, debug=True)