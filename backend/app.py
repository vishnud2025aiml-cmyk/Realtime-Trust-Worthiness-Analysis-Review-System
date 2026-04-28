from flask import Flask, render_template, request, jsonify
import pickle
import os
import re
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

app = Flask(__name__, template_folder="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model = pickle.load(open(os.path.join(BASE_DIR,"model.pkl"),"rb"))
vectorizer = pickle.load(open(os.path.join(BASE_DIR,"vectorizer.pkl"),"rb"))

STATS_PATH = os.path.join(BASE_DIR, "stats.pkl")

stop_words = set(stopwords.words('english'))

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ---------------- LOAD STATS ----------------
def load_stats():
    if os.path.exists(STATS_PATH):
        stats = pickle.load(open(STATS_PATH,"rb"))
    else:
        stats = {}

    stats.setdefault("Genuine", 0)
    stats.setdefault("Fake", 0)
    stats.setdefault("TotalReviews", 0)
    stats.setdefault("history", [])

    return stats

# ---------------- SAVE STATS ----------------
def save_stats(stats):
    pickle.dump(stats, open(STATS_PATH,"wb"))

# ---------------- RULE BOOST ----------------
def rule_based_fake(review):
    review = review.lower()
    if "!!!" in review:
        return True
    if "buy now" in review:
        return True
    if "100%" in review:
        return True
    return False

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# ---------------- ANALYZE ----------------
@app.route("/analyze", methods=["POST"])
def analyze():

    review = request.form.get("review","")

    cleaned = clean_text(review)
    X = vectorizer.transform([cleaned])

    prob = model.predict_proba(X)[0]
    fake_prob = prob[0]
    genuine_prob = prob[1]

    # prediction
    if rule_based_fake(review):
        prediction = "Fake"
        confidence = 90
    else:
        if fake_prob > genuine_prob:
            prediction = "Fake"
            confidence = round(fake_prob * 100, 2)
        else:
            prediction = "Genuine"
            confidence = round(genuine_prob * 100, 2)

    stats = load_stats()

    # ✅ FIX: SIMPLE COUNT (NO DECIMAL ADD)
    if prediction == "Fake":
        stats["Fake"] += 1
    else:
        stats["Genuine"] += 1

    stats["TotalReviews"] += 1

    # ✅ FIX: STORE EACH REVIEW SEPARATELY
    stats["history"].append({
        "type": prediction,     # Fake or Genuine
        "value": confidence     # 0–100
    })

    # keep only last 20
    if len(stats["history"]) > 20:
        stats["history"] = stats["history"][-20:]

    save_stats(stats)

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "stats": stats
    })

# ---------------- DASHBOARD DATA ----------------
@app.route("/dashboard_data")
def dashboard_data():
    return jsonify(load_stats())

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)