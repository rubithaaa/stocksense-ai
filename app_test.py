from flask import Flask, send_from_directory, jsonify
import os

app = Flask(__name__, static_folder="frontend")

@app.route("/")
def home():
    return send_from_directory("frontend", "index.html")

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "dataset_loaded": True,
        "gemini_configured": bool(os.getenv("GEMINI_API_KEY"))
    })

if __name__ == "__main__":
    print("STOCKSENSE FRONTEND SERVER")
    print("Open: http://localhost:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)
