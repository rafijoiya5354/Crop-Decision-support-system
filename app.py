"""
AI Agricultural Decision Intelligence Platform
Production-ready Flask backend
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, send_file
from functools import wraps
import sqlite3
import bcrypt
import json
import os
from datetime import datetime

from utils.predict import run_prediction
from utils.explain import get_shap_explanation
from utils.gpt_explain import generate_gpt_explanation
from utils.chat import chat_with_context
from utils.recommend import get_recommendations
from utils.report import generate_pdf_report
from utils.db import init_db, get_db

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "agri-ai-secret-key-change-in-prod")

# ─────────────────────────────────────────────
# DB INIT
# ─────────────────────────────────────────────
with app.app_context():
    init_db()


# ─────────────────────────────────────────────
# AUTH DECORATORS
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            return redirect(url_for("login"))
        if session.get("role") != "admin":
            return jsonify({"error": "Admin access required"}), 403
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────────
# AUTH ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    data = request.get_json(silent=True) or {}  # FIX 8: silent=True prevents crash on missing Content-Type
    username = data.get("username", "").strip()
    password = data.get("password", "")
    email = data.get("email", "").strip()

    if not username or not password or not email:
        return jsonify({"error": "All fields required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    db = get_db()
    try:
        db.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (username, email, hashed, "user"),
        )
        db.commit()
        return jsonify({"success": True, "message": "Account created. Please login."})
    except sqlite3.IntegrityError:
        return jsonify({"error": "Username or email already exists"}), 409
    finally:
        db.close()


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    data = request.get_json(silent=True) or {}  # FIX 7: silent=True prevents 400 on missing Content-Type
    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "Username and password required"}), 400

    db = get_db()
    try:
        user = db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    finally:
        db.close()  # FIX 7: ensure connection is always closed

    if not user or not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
        return jsonify({"error": "Invalid credentials"}), 401

    session["username"] = user["username"]
    session["role"] = user["role"]
    return jsonify({"success": True, "role": user["role"]})


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ─────────────────────────────────────────────
# MAIN PAGES
# ─────────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template(
        "dashboard.html",
        username=session["username"],
        role=session["role"],
    )


@app.route("/analytics")
@login_required
def analytics():
    return render_template("analytics.html", username=session["username"])


@app.route("/admin")
@admin_required
def admin():
    db = get_db()
    users = db.execute("SELECT id, username, email, role, created_at FROM users").fetchall()
    db.close()
    return render_template("admin.html", users=[dict(u) for u in users])


# ─────────────────────────────────────────────
# PREDICTION API
# ─────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
@login_required
def predict():
    data = request.get_json(silent=True) or {}  # FIX 9: silent prevents 400 on wrong Content-Type

    from utils.predict import NUMERIC_FEATURES, CATEGORICAL_FEATURES, FEATURE_DEFAULTS

    # Build features dict — accept any/all of the 21 model columns;
    # missing numeric values are coerced to float with dataset-mean defaults,
    # missing categoricals fall back to FEATURE_DEFAULTS inside predict.py.
    features = {}
    for col in NUMERIC_FEATURES:
        raw = data.get(col, FEATURE_DEFAULTS[col])
        try:
            features[col] = float(raw)
        except (ValueError, TypeError):
            features[col] = float(FEATURE_DEFAULTS[col])

    for col in CATEGORICAL_FEATURES:
        features[col] = data.get(col, FEATURE_DEFAULTS[col])

    # ── Run pipeline — all steps wrapped so any crash returns JSON, not HTML ───
    try:
        predictions = run_prediction(features)
    except FileNotFoundError as e:
        return jsonify({
            "error": "Model files not found. Place the .pkl files in the Models/ directory.",
            "detail": str(e),
        }), 500
    except ModuleNotFoundError as e:
        # Catches "No module named 'numpy._core'" and similar pickle/numpy version mismatches
        return jsonify({
            "error": "Dependency version mismatch loading model.",
            "detail": str(e),
            "fix": "Run: pip install --upgrade numpy scikit-learn joblib scipy",
        }), 500
    except Exception as e:
        detail = str(e)
        # Provide a helpful hint for common numpy/pickle incompatibilities
        if "numpy" in detail.lower() or "_core" in detail or "pickle" in detail.lower():
            return jsonify({
                "error": "NumPy/pickle version mismatch — model was saved with a different NumPy version.",
                "detail": detail,
                "fix": "Run: pip install --upgrade numpy scikit-learn joblib scipy",
            }), 500
        return jsonify({"error": "Prediction failed", "detail": detail}), 500

    try:
        shap_result = get_shap_explanation(features)
    except Exception:
        shap_result = {"top_features": [], "method": "unavailable", "base_value": 0.0}

    try:
        recommendations = get_recommendations(features, predictions)
    except Exception:
        recommendations = []

    try:
        gpt_explanation = generate_gpt_explanation(features, predictions, shap_result)
    except Exception:
        gpt_explanation = {"summary": "Explanation unavailable.", "source": "error"}

    # Store in history (non-fatal)
    db = get_db()
    try:
        db.execute(
            "INSERT INTO history (username, input_data, prediction, shap_data, timestamp) VALUES (?, ?, ?, ?, ?)",
            (
                session["username"],
                json.dumps(features),
                json.dumps(predictions),
                json.dumps(shap_result),
                datetime.utcnow().isoformat(),
            ),
        )
        db.commit()
    except Exception:
        pass  # History save failure should not block returning results
    finally:
        db.close()

    return jsonify(
        {
            "predictions": predictions,
            "shap": shap_result,
            "recommendations": recommendations,
            "gpt_explanation": gpt_explanation,
            "input": features,
        }
    )


# ─────────────────────────────────────────────
# CHAT API
# ─────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True) or {}  # FIX 10
    user_message = data.get("message", "").strip()
    history = data.get("history", [])
    context = data.get("context", {})

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    response = chat_with_context(user_message, history[-8:], context)
    return jsonify({"response": response})


# ─────────────────────────────────────────────
# ANALYTICS DATA API
# ─────────────────────────────────────────────
@app.route("/api/analytics")
@login_required
def analytics_data():
    db = get_db()
    rows = db.execute(
        "SELECT prediction, timestamp FROM history WHERE username = ? ORDER BY timestamp DESC LIMIT 50",
        (session["username"],),
    ).fetchall()
    db.close()

    failure_trend = []
    stress_dist = {"Low": 0, "Medium": 0, "High": 0}
    suitability_vals = []

    for row in rows:
        pred = json.loads(row["prediction"])
        ts = row["timestamp"]

        failure_trend.append(
            {
                "timestamp": ts,
                "failure_risk": pred.get("failure_probability", 0),
            }
        )

        stress = pred.get("stress_level", "Medium")
        if stress in stress_dist:
            stress_dist[stress] += 1
        else:
            stress_dist["Medium"] += 1

        suitability_vals.append(pred.get("suitability_score", 0))

    avg_suitability = (
        round(sum(suitability_vals) / len(suitability_vals), 2) if suitability_vals else 0
    )

    return jsonify(
        {
            "failure_trend": failure_trend[::-1],
            "stress_distribution": stress_dist,
            "avg_suitability": avg_suitability,
            "total_predictions": len(rows),
        }
    )


# ─────────────────────────────────────────────
# HISTORY API
# ─────────────────────────────────────────────
@app.route("/api/history")
@login_required
def history():
    db = get_db()
    rows = db.execute(
        "SELECT id, input_data, prediction, timestamp FROM history WHERE username = ? ORDER BY timestamp DESC LIMIT 20",
        (session["username"],),
    ).fetchall()
    db.close()

    result = []
    for row in rows:
        result.append(
            {
                "id": row["id"],
                "input": json.loads(row["input_data"]),
                "prediction": json.loads(row["prediction"]),
                "timestamp": row["timestamp"],
            }
        )

    return jsonify(result)


# ─────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────
@app.route("/api/report/<int:history_id>")
@login_required
def download_report(history_id):
    db = get_db()
    row = db.execute(
        "SELECT * FROM history WHERE id = ? AND username = ?",
        (history_id, session["username"]),
    ).fetchone()
    db.close()

    if not row:
        return jsonify({"error": "Record not found"}), 404

    pdf_path = generate_pdf_report(
        username=session["username"],
        input_data=json.loads(row["input_data"]),
        predictions=json.loads(row["prediction"]),
        timestamp=row["timestamp"],
    )

    return send_file(pdf_path, as_attachment=True, download_name="agri_report.pdf")


@app.route("/api/report/latest")
@login_required
def download_latest_report():
    db = get_db()
    row = db.execute(
        "SELECT * FROM history WHERE username = ? ORDER BY timestamp DESC LIMIT 1",
        (session["username"],),
    ).fetchone()
    db.close()

    if not row:
        return jsonify({"error": "No predictions yet"}), 404

    pdf_path = generate_pdf_report(
        username=session["username"],
        input_data=json.loads(row["input_data"]),
        predictions=json.loads(row["prediction"]),
        timestamp=row["timestamp"],
    )

    return send_file(pdf_path, as_attachment=True, download_name="agri_report.pdf")


# ─────────────────────────────────────────────
# ADMIN APIS
# ─────────────────────────────────────────────
@app.route("/api/admin/users")
@admin_required
def admin_users():
    db = get_db()
    users = db.execute(
        "SELECT id, username, email, role, created_at FROM users"
    ).fetchall()
    total_preds = db.execute("SELECT COUNT(*) as cnt FROM history").fetchone()["cnt"]
    db.close()
    return jsonify(
        {
            "users": [dict(u) for u in users],
            "total_predictions": total_preds,
        }
    )


@app.route("/api/admin/promote", methods=["POST"])
@admin_required
def promote_user():
    data = request.get_json(silent=True) or {}  # FIX 11: silent=True prevents crash
    username = data.get("username")
    role = data.get("role", "user")
    if not username:
        return jsonify({"error": "Username required"}), 400  # FIX 11: guard missing username
    if role not in ("user", "admin"):
        return jsonify({"error": "Invalid role"}), 400
    db = get_db()
    try:
        db.execute("UPDATE users SET role = ? WHERE username = ?", (role, username))
        db.commit()
    finally:
        db.close()  # FIX 11: ensure connection always closed
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)


# ─────────────────────────────────────────────
# GLOBAL ERROR HANDLERS — always return JSON, never HTML
# ─────────────────────────────────────────────
@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": "Bad request", "detail": str(e)}), 400

@app.errorhandler(401)
def unauthorized(e):
    return jsonify({"error": "Unauthorized"}), 401

@app.errorhandler(403)
def forbidden(e):
    return jsonify({"error": "Forbidden"}), 403

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "Internal server error", "detail": str(e)}), 500


# ─────────────────────────────────────────────
# DIAGNOSTIC ROUTE — helps debug model loading
# ─────────────────────────────────────────────
@app.route("/api/status")
@login_required
def status():
    from utils.predict import _FAILURE_PATH, _STRESS_PATH, _SUITABILITY_PATH
    return jsonify({
        "status": "ok",
        "models": {
            "failure":     os.path.exists(_FAILURE_PATH),
            "stress":      os.path.exists(_STRESS_PATH),
            "suitability": os.path.exists(_SUITABILITY_PATH),
        },
        "model_dir": os.path.dirname(_FAILURE_PATH),
    })
