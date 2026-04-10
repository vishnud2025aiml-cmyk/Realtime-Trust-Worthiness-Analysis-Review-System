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

with open(MODEL_PATH,"rb") as f:
    model = pickle.load(f)
with open(VECT_PATH,"rb") as f:
    vectorizer = pickle.load(f)

stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]',' ',text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

def load_stats():
    if os.path.exists(STATS_PATH):
        with open(STATS_PATH,"rb") as f:
            stats = pickle.load(f)
    else:
        stats = {"Genuine":1,"Fake":1,"TotalReviews":0}
    stats.setdefault("Genuine",1)
    stats.setdefault("Fake",1)
    stats.setdefault("TotalReviews",0)
    return stats

def save_stats(stats):
    with open(STATS_PATH,"wb") as f:
        pickle.dump(stats,f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze",methods=["POST"])
def analyze():
    review = request.form.get("review","")
    if not review:
        return jsonify({"status":"error","message":"No review provided"})
    try:
        cleaned = clean_text(review)
        X = vectorizer.transform([cleaned])
        pred = model.predict(X)[0]

        stats = load_stats()
        if pred==1:
            stats["Genuine"] += 10
            stats["Fake"] += 1
            label="Genuine"
        else:
            stats["Fake"] += 10
            stats["Genuine"] += 1
            label="Fake"

        stats["TotalReviews"] += 1
        save_stats(stats)
        return jsonify({"status":"success","prediction":label,"stats":stats})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)})

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/dashboard_data")
def dashboard_data():
    stats = load_stats()
    return jsonify(stats)

if __name__ == "__main__":
    app.run(debug=True)