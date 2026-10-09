from flask import Flask, render_template, request, jsonify
import sqlite3
import datetime
from utils.tfidf_model import recommend_tfidf
from utils.neural_model import recommend_neural
from config import DB_PATH

app = Flask(__name__)


# -----------------------------
# DATABASE INITIALIZATION
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # History Table
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            model TEXT,
            timestamp TEXT
        )
    """)

    # Saved Recommendations Table
    # url column added to support clickable course links on dashboard
    c.execute("""
        CREATE TABLE IF NOT EXISTS saved (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT,
            model TEXT,
            course_title TEXT,
            subject TEXT,
            score REAL,
            timestamp TEXT,
            url TEXT
        )
    """)

    # Safe migration: add url column to existing databases that don't have it
    # This ensures backward compatibility — no data loss on existing installs
    try:
        c.execute("ALTER TABLE saved ADD COLUMN url TEXT")
    except sqlite3.OperationalError:
        # Column already exists — safe to ignore
        pass

    conn.commit()
    conn.close()


# -----------------------------
# PAGE ROUTES
# -----------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/recommend")
def recommend_page():
    return render_template("recommend.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/about")
def about():
    return render_template("about.html")


# -----------------------------
# API ROUTES
# -----------------------------
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data       = request.json
    query      = data.get("query")
    model_type = data.get("model")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    if model_type not in ["tfidf", "neural", "both"]:
        return jsonify({"error": "Invalid model type. Use 'tfidf', 'neural', or 'both'"}), 400

    # -------------------------------------------
    # BOTH models: side-by-side comparison mode
    # -------------------------------------------
    if model_type == "both":
        tfidf_results  = recommend_tfidf(query,  top_k=10)
        neural_results = recommend_neural(query, top_k=10)
        _save_history(query, "both")
        return jsonify({
            "tfidf_results":  tfidf_results,
            "neural_results": neural_results,
        })

    # -------------------------------------------
    # SINGLE model: backward-compatible responses
    # -------------------------------------------
    if model_type == "tfidf":
        results = recommend_tfidf(query, top_k=10)
        _save_history(query, "tfidf")
        return jsonify({"tfidf_results": results})

    if model_type == "neural":
        results = recommend_neural(query, top_k=10)
        _save_history(query, "neural")
        return jsonify({"neural_results": results})


# -----------------------------
# HISTORY HELPER
# -----------------------------
def _save_history(query, model_type):
    """
    Internal helper to log a query + model type to history table.
    """
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute(
        "INSERT INTO history (query, model, timestamp) VALUES (?, ?, ?)",
        (query, model_type, str(datetime.datetime.now()))
    )
    conn.commit()
    conn.close()


# -----------------------------
# GET HISTORY
# -----------------------------
@app.route("/api/history", methods=["GET"])
def api_history():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT * FROM history ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify(rows)


# -----------------------------
# SAVE RECOMMENDATION
# -----------------------------
@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.json

    # Extract url field — defaults to empty string if not provided
    # This ensures backward compatibility with any older save calls
    url = data.get("url", "")

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        INSERT INTO saved (query, model, course_title, subject, score, timestamp, url)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        data["query"],
        data["model"],
        data["course_title"],
        data["subject"],
        data["score"],
        str(datetime.datetime.now()),
        url
    ))

    conn.commit()
    conn.close()

    return jsonify({"message": "Saved successfully!"})


# -----------------------------
# GET SAVED RECOMMENDATIONS
# -----------------------------
@app.route("/api/saved", methods=["GET"])
def api_saved():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("SELECT * FROM saved ORDER BY timestamp DESC")
    rows = c.fetchall()
    conn.close()
    return jsonify(rows)


# -----------------------------
# DELETE SAVED (UNSAVE)
# -----------------------------
@app.route("/api/delete_saved", methods=["POST"])
def api_delete_saved():
    data         = request.json
    query        = data.get("query")
    course_title = data.get("course_title")

    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    c.execute("""
        DELETE FROM saved
        WHERE query = ? AND course_title = ?
    """, (query, course_title))

    conn.commit()
    conn.close()

    return jsonify({"message": "Recommendation deleted successfully"})


# -----------------------------
# CLEAR HISTORY
# -----------------------------
@app.route("/api/clear_history", methods=["POST"])
def api_clear_history():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()
    c.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return jsonify({"message": "History cleared successfully"})


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    init_db()
    app.run(debug=True)

