from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import os
import pickle
import numpy as np

app = Flask(__name__)
app.secret_key = "your_secret_key"

# Paths
DB_PATH = "database/users.db"
MODEL_PATH = "models/model.pkl"

# Ensure database exists
os.makedirs("database", exist_ok=True)
if not os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Load model and preprocessor
with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)
model = data["model"]
preprocessor = data["preprocessor"]

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash("Signup successful! Please login.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "error")
        finally:
            conn.close()

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session["user"] = username
            return redirect(url_for("predict"))
        else:
            flash("Invalid credentials!", "error")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    if "user" not in session:
        flash("Please login first!", "error")
        return redirect(url_for("login"))

    if request.method == "POST":
        try:
            bedrooms = float(request.form["bedrooms"])
            bathrooms = float(request.form["bathrooms"])
            sqft = float(request.form["sqft"])
            location = request.form["location"]

            # Prepare input for model
            input_df = {"bedrooms": [bedrooms], "bathrooms": [bathrooms], "sqft": [sqft], "location": [location]}
            import pandas as pd
            X_input = pd.DataFrame(input_df)

            # Preprocess input
            X_processed = preprocessor.transform(X_input)

            # Predict
            pred = model.predict(X_processed)[0]
            prediction = round(pred, 2)
        except Exception as e:
            flash(f"Error in prediction: {e}", "error")

    return render_template("predict.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
