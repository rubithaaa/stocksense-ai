from __future__ import annotations

import os
import traceback

from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

from src.data_engine import data_engine
from src import analytics
from src import ai_engine

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


@app.after_request
def cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def error(message, status=400):
    return jsonify({"error": message}), status


def require_data():
    if not data_engine.is_loaded:
        return error("No dataset loaded.", 409)
    return None


# ---------- FRONTEND ----------
@app.route("/")
def home():
    return send_from_directory(FRONTEND_DIR, "index.html")


# ---------- HEALTH ----------
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "dataset_loaded": data_engine.is_loaded,
        "gemini_configured": bool(os.environ.get("GEMINI_API_KEY"))
    })


# ---------- UPLOAD ----------
@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return error("No CSV file uploaded.")

    file = request.files["file"]

    if not file.filename.lower().endswith(".csv"):
        return error("Only CSV files are supported.")

    result = data_engine.load_from_bytes(
        file.read(),
        filename=file.filename
    )

    if not result.is_valid:
        return jsonify({
            "status": "rejected",
            "validation": result.to_dict()
        }), 422

    return jsonify({
        "status": "loaded",
        "summary": data_engine.summary(),
        "validation": result.to_dict()
    })


# ---------- DATA ----------
@app.route("/api/dataset/summary")
def dataset_summary():
    e = require_data()
    if e:
        return e

    return jsonify({
        "summary": data_engine.summary(),
        "validation": data_engine.get_last_validation()
    })


@app.route("/api/products")
def products():
    e = require_data()
    if e:
        return e
    return jsonify({"products": data_engine.list_products()})


@app.route("/api/stores")
def stores():
    e = require_data()
    if e:
        return e
    return jsonify({"stores": data_engine.list_stores()})


@app.route("/api/categories")
def categories():
    e = require_data()
    if e:
        return e
    return jsonify({"categories": data_engine.list_categories()})


# ---------- ANALYTICS ----------
@app.route("/api/analytics/summary")
def summary():
    e = require_data()
    if e:
        return e

    return jsonify(
        analytics.overall_summary(data_engine.get_dataframe())
    )


@app.route("/api/analytics/stock-risk")
def stock_risk():
    e = require_data()
    if e:
        return e

    return jsonify({
        "stock_risk": analytics.detect_stock_risks(
            data_engine.get_dataframe()
        )
    })


@app.route("/api/analytics/non-moving")
def non_moving():
    e = require_data()
    if e:
        return e

    return jsonify({
        "non_moving": analytics.detect_non_moving_products(
            data_engine.get_dataframe()
        )
    })


@app.route("/api/analytics/top-sellers")
def top_sellers():
    e = require_data()
    if e:
        return e

    return jsonify({
        "top_sellers": analytics.rank_products(
            data_engine.get_dataframe(),
            by="revenue",
            top_n=10
        )
    })


@app.route("/api/analytics/trends")
def trends():
    e = require_data()
    if e:
        return e

    return jsonify(
        analytics.detect_sales_trends(
            data_engine.get_dataframe()
        )
    )


# ---------- AI ----------
@app.route("/api/query", methods=["POST"])
def query():
    e = require_data()
    if e:
        return e

    body = request.get_json(silent=True) or {}
    question = str(body.get("question", "")).strip()

    if not question:
        return error("Question is required.")

    try:
        df = data_engine.get_dataframe()

        detected = analytics.detect_intent(
            question,
            data_engine
        )

        evidence = analytics.build_evidence(
            df,
            detected["intent"],
            detected["entities"]
        )

        response = ai_engine.ask_gemini(
            question,
            evidence,
            [
                f"Dataset contains {len(df)} rows.",
                f"Dataset spans {data_engine.days_of_history()} days."
            ]
        )

        return jsonify({
            "question": question,
            "intent": detected["intent"],
            "evidence": evidence,
            "gemini": response
        })

    except Exception as exc:
        app.logger.error(
            "Query error: %s\n%s",
            exc,
            traceback.format_exc()
        )

        return jsonify({
            "status": "HUMAN_REVIEW",
            "answer": "AI reasoning is currently unavailable. Review the deterministic analytics before making a decision."
        })


# ---------- ERROR ----------
@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404


if __name__ == "__main__":
    # Automatically load our demo dataset on startup
    dataset_path = os.path.join(BASE_DIR, "data", "retail_data.csv")

    if os.path.exists(dataset_path) and not data_engine.is_loaded:
        try:
            data_engine.load_from_path(dataset_path)
            print("✓ Demo retail dataset loaded")
        except Exception as exc:
            print("Dataset load warning:", exc)

    print("✓ StockSense AI running on http://localhost:8000")

    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False
    )