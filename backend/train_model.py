import pandas as pd
import re
import pickle
import os
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

nltk.download('stopwords')

dataset_path = "../dataset"
all_data = []

# ---------------- PROCESS DATASET ----------------
def process_dataset(df, filename):
    df.columns = [col.lower() for col in df.columns]

    # Amazon dataset
    if 'review_body' in df.columns and 'star_rating' in df.columns:
        df = df[['review_body', 'star_rating']].dropna()
        df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')
        df = df.dropna()
        df['label'] = df['star_rating'].apply(lambda x: 1 if x >= 3 else 0)
        df = df.rename(columns={'review_body': 'text'})
        return df[['text', 'label']]

    # text_ dataset
    if 'text_' in df.columns and 'label' in df.columns:
        df = df[['text_', 'label']].dropna()
        df = df.rename(columns={'text_': 'text'})
        return df[['text', 'label']]

    # general dataset
    if 'text' in df.columns and 'label' in df.columns:
        return df[['text', 'label']].dropna()

    return None


# ---------------- LOAD DATA ----------------
for file in os.listdir(dataset_path):
    if file.endswith(".csv"):
        try:
            df = pd.read_csv(os.path.join(dataset_path, file))
            processed = process_dataset(df, file)
            if processed is not None and len(processed) > 0:
                all_data.append(processed)
        except Exception as e:
            print("Skipping:", file, e)

if len(all_data) == 0:
    raise Exception("❌ No valid dataset found!")

df = pd.concat(all_data, ignore_index=True)

# ---------------- CLEAN LABELS ----------------
df['label'] = df['label'].astype(str).str.lower().str.strip()

df['label'] = df['label'].replace({
    'real': 1,
    'genuine': 1,
    'positive': 1,
    'fake': 0,
    'spam': 0,
    'negative': 0,
    '1': 1,
    '0': 0,
    'true': 1,
    'false': 0
})

df['label'] = pd.to_numeric(df['label'], errors='coerce')
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# ---------------- CHECK DATA ----------------
print("\nLabel distribution:")
print(df['label'].value_counts())

if df['label'].nunique() < 2:
    raise Exception("❌ Dataset must contain BOTH Fake and Genuine reviews!")

# ---------------- FIX: SAFE BALANCING ----------------
fake_count = len(df[df['label'] == 0])
real_count = len(df[df['label'] == 1])

min_count = int(min(fake_count, real_count))

# avoid crash if dataset is too small
if min_count == 0:
    raise Exception("❌ One class is empty after cleaning!")

df_fake = df[df['label'] == 0].sample(n=min_count, random_state=42)
df_real = df[df['label'] == 1].sample(n=min_count, random_state=42)

df = pd.concat([df_fake, df_real]).sample(frac=1, random_state=42)

# ---------------- CLEAN TEXT ----------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return " ".join([w for w in text.split() if w not in stop_words])

df['cleaned'] = df['text'].apply(clean_text)

# ---------------- TF-IDF ----------------
vectorizer = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ---------------- MODEL ----------------
model = MultinomialNB()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("\n✅ Accuracy:", round(accuracy * 100, 2), "%")

# ---------------- SAVE MODEL ----------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

# ---------------- STATS ----------------
stats = {
    "Genuine": int((df['label'] == 1).sum()),
    "Fake": int((df['label'] == 0).sum()),
    "TotalReviews": len(df),
    "history": []
}

pickle.dump(stats, open("stats.pkl", "wb"))

print("\n✅ Model + Stats saved successfully!")