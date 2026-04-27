from flask import Flask, render_template, request, jsonify
import pickle
import os
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

app = Flask(__name__, template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECT_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
STATS_PATH = os.path.join(BASE_DIR, "stats.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECT_PATH, "rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def load_stats():
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH, "rb") as f:
            stats = pickle.load(f)
    else:
        stats = {"Genuine": 0, "Fake": 0, "TotalReviews": 0}
    return stats

def save_stats(stats):
    with open(STATS_PATH, "wb") as f:
        pickle.dump(stats, f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/analyze", methods=["POST"])
def analyze():

    review = request.form.get("review", "")

    cleaned = clean_text(review)
    X = vectorizer.transform([cleaned])

    pred = model.predict(X)[0]

    # REAL CONFIDENCE FIX
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0]
        confidence = round(max(prob) * 100, 2)
    else:
        confidence = 80

    stats = load_stats()

    if pred == 1:
        label = "Genuine"
        stats["Genuine"] += 1
    else:
        label = "Fake"
        stats["Fake"] += 1

    stats["TotalReviews"] += 1
    save_stats(stats)

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "text": review,
        "stats": stats
    })

@app.route("/dashboard_data")
def dashboard_data():
    return jsonify(load_stats())

if __name__ == "__main__":
    app.run(debug=True)