import json
import hmac
import math
import os
import re
import secrets
import sqlite3
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, abort, g, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from health_agent import generate_health_report


app = Flask(__name__)
app.secret_key = os.environ.get("HEALTHPULSE_SECRET_KEY") or secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("HEALTHPULSE_COOKIE_SECURE") == "1",
)

# Load the trained model once when the application starts.
PROJECT_FOLDER = Path(__file__).resolve().parent
DATABASE_PATH = Path(
    os.environ.get(
        "HEALTHPULSE_DATABASE_PATH",
        str(PROJECT_FOLDER / "data" / "healthpulse.db"),
    )
)
MODEL_PATH = PROJECT_FOLDER / "models" / "health_model.pkl"
FEATURE_IMPORTANCE_PATH = PROJECT_FOLDER / "models" / "feature_importances.json"
model = joblib.load(MODEL_PATH)
feature_importances = []
if FEATURE_IMPORTANCE_PATH.exists():
    with FEATURE_IMPORTANCE_PATH.open(encoding="utf-8") as importance_file:
        feature_importances = sorted(
            json.load(importance_file)["features"],
            key=lambda item: item["importance"],
            reverse=True,
        )
feature_importance_max = max(
    (item["importance"] for item in feature_importances),
    default=1,
)

# Keep this order aligned with the features used to train the model.
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

INPUT_VALIDATION = {
    "Pregnancies": (0, 20, True),
    "Glucose": (0, 300, False),
    "BloodPressure": (0, 200, False),
    "SkinThickness": (0, 150, False),
    "Insulin": (0, 1000, False),
    "BMI": (0, 80, False),
    "DiabetesPedigreeFunction": (0, 3, False),
    "Age": (1, 120, True),
}

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def connect_db():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                pregnancies REAL NOT NULL,
                glucose REAL NOT NULL,
                blood_pressure REAL NOT NULL,
                skin_thickness REAL NOT NULL,
                insulin REAL NOT NULL,
                bmi REAL NOT NULL,
                diabetes_pedigree_function REAL NOT NULL,
                age REAL NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE INDEX IF NOT EXISTS idx_assessments_user_created
                ON assessments (user_id, created_at DESC);
            """
        )


def csrf_token():
    token = session.get("_csrf_token")
    if token is None:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def csrf_is_valid():
    expected = session.get("_csrf_token", "")
    submitted = request.form.get("csrf_token", "")
    return bool(expected and submitted and hmac.compare_digest(expected, submitted))


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        user_id = session.get("user_id")
        if user_id is None:
            return redirect(url_for("login"))

        connection = connect_db()
        try:
            user = connection.execute(
                "SELECT id, name, email FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        finally:
            connection.close()

        if user is None:
            session.clear()
            return redirect(url_for("login"))

        g.current_user = user
        return view(*args, **kwargs)

    return wrapped_view


@app.context_processor
def inject_csrf_token():
    return {"csrf_token": csrf_token}


@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id") is not None:
        return redirect(url_for("home"))

    error = None
    if request.method == "POST":
        if not csrf_is_valid():
            error = "Your session expired. Please refresh the page and try again."
        else:
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm_password = request.form.get("confirm_password", "")

            if not name:
                error = "Please enter your full name."
            elif not email:
                error = "Please enter your email address."
            elif not EMAIL_PATTERN.fullmatch(email):
                error = "Please enter a valid email address."
            elif len(password) < 8:
                error = "Your password must be at least 8 characters long."
            elif password != confirm_password:
                error = "The passwords do not match."
            else:
                connection = connect_db()
                try:
                    connection.execute(
                        """
                        INSERT INTO users (name, email, password_hash)
                        VALUES (?, ?, ?)
                        """,
                        (name, email, generate_password_hash(password)),
                    )
                    user_id = connection.execute(
                        "SELECT id FROM users WHERE email = ?",
                        (email,),
                    ).fetchone()["id"]
                    connection.commit()
                except sqlite3.IntegrityError:
                    connection.rollback()
                    error = "An account with that email already exists."
                except sqlite3.Error:
                    connection.rollback()
                    app.logger.exception("Could not create a HealthPulse account.")
                    error = "We couldn't create your account right now. Please try again."
                finally:
                    connection.close()

                if error is None:
                    session.clear()
                    session["user_id"] = user_id
                    session["user_name"] = name
                    return redirect(url_for("home"))

    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id") is not None:
        return redirect(url_for("home"))

    error = None
    if request.method == "POST":
        if not csrf_is_valid():
            error = "Your session expired. Please refresh the page and try again."
        else:
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            connection = connect_db()
            try:
                user = connection.execute(
                    "SELECT id, name, email, password_hash FROM users WHERE email = ?",
                    (email,),
                ).fetchone()
            except sqlite3.Error:
                app.logger.exception("Could not look up a HealthPulse account.")
                error = "We couldn't sign you in right now. Please try again."
                user = None
            finally:
                connection.close()

            if error is None:
                if user is None or not check_password_hash(user["password_hash"], password):
                    error = "Invalid email or password."
                else:
                    session.clear()
                    session["user_id"] = user["id"]
                    session["user_name"] = user["name"]
                    return redirect(url_for("home"))

    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    if not csrf_is_valid():
        abort(400)
    session.clear()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    result = None
    probability = None
    report = None
    error = None

    if request.method == "POST":
        if not csrf_is_valid():
            error = "Your session expired. Please refresh the page and try again."
        else:
            values = []
            for name in FEATURE_NAMES:
                raw_value = request.form.get(name, "").strip()
                minimum, maximum, must_be_integer = INPUT_VALIDATION[name]

                if not raw_value:
                    error = f"Please enter a value for {name}."
                    break

                try:
                    value = float(raw_value)
                except ValueError:
                    error = f"{name} must be a numeric value."
                    break

                if not math.isfinite(value):
                    error = f"{name} must be a finite number between {minimum} and {maximum}."
                    break

                if must_be_integer and not value.is_integer():
                    error = f"{name} must be a whole number between {minimum} and {maximum}."
                    break

                if value < minimum or value > maximum:
                    error = f"{name} must be between {minimum} and {maximum}."
                    break

                values.append(value)

            if error is None:
                input_data = pd.DataFrame([values], columns=FEATURE_NAMES)

                prediction = model.predict(input_data)[0]
                result = (
                    "Higher Risk Indicator"
                    if prediction == 1
                    else "Lower Risk Indicator"
                )

                # Show the model's probability for class 1 when probability estimates exist.
                if hasattr(model, "predict_proba"):
                    class_one_index = list(model.classes_).index(1)
                    probability = model.predict_proba(input_data)[0][class_one_index]

                # Generate the plain-language report after the model prediction.
                report = generate_health_report(
                    dict(zip(FEATURE_NAMES, values)),
                    prediction,
                    probability,
                )

                if probability is None:
                    error = "The assessment was generated, but its probability could not be saved."
                else:
                    connection = connect_db()
                    try:
                        connection.execute(
                            """
                            INSERT INTO assessments (
                                user_id, pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi,
                                diabetes_pedigree_function, age, prediction, probability
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                g.current_user["id"],
                                *values,
                                int(prediction),
                                float(probability),
                            ),
                        )
                        connection.commit()
                    except sqlite3.Error:
                        connection.rollback()
                        app.logger.exception(
                            "Could not save assessment for user_id=%s",
                            g.current_user["id"],
                        )
                        error = "Your assessment was generated but could not be saved to history."
                    finally:
                        connection.close()

    return render_template(
        "index.html",
        feature_names=FEATURE_NAMES,
        result=result,
        probability=probability,
        report=report,
        feature_importances=feature_importances,
        feature_importance_max=feature_importance_max,
        error=error,
        form_values=request.form,
        user_name=g.current_user["name"],
        current_user=g.current_user,
    )


@app.route("/history")
@login_required
def history():
    connection = connect_db()
    try:
        assessments = connection.execute(
            """
            SELECT id, created_at, prediction, probability, glucose, bmi, age
            FROM assessments
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (g.current_user["id"],),
        ).fetchall()
    finally:
        connection.close()

    return render_template(
        "history.html",
        assessments=assessments,
        user_name=g.current_user["name"],
        current_user=g.current_user,
    )


@app.route("/history/<int:assessment_id>")
@login_required
def assessment_detail(assessment_id):
    connection = connect_db()
    try:
        assessment = connection.execute(
            """
            SELECT *
            FROM assessments
            WHERE id = ? AND user_id = ?
            """,
            (assessment_id, g.current_user["id"]),
        ).fetchone()
    finally:
        connection.close()

    if assessment is None:
        abort(404)

    values = [
        assessment["pregnancies"],
        assessment["glucose"],
        assessment["blood_pressure"],
        assessment["skin_thickness"],
        assessment["insulin"],
        assessment["bmi"],
        assessment["diabetes_pedigree_function"],
        assessment["age"],
    ]
    measurements = [
        {"name": name, "value": value}
        for name, value in zip(FEATURE_NAMES, values)
    ]
    report = generate_health_report(
        dict(zip(FEATURE_NAMES, values)),
        assessment["prediction"],
        assessment["probability"],
    )

    return render_template(
        "assessment_detail.html",
        assessment=assessment,
        report=report,
        measurements=measurements,
        feature_names=FEATURE_NAMES,
        feature_importances=feature_importances,
        feature_importance_max=feature_importance_max,
        user_name=g.current_user["name"],
        current_user=g.current_user,
    )


init_db()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)