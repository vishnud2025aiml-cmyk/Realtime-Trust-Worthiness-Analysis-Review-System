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

# ----------------------------- Process Dataset -----------------------------
def process_dataset(df, filename):
    df.columns = [col.lower() for col in df.columns]

    # Amazon dataset
    if 'review_body' in df.columns and 'star_rating' in df.columns:
        df = df[['review_body', 'star_rating']].dropna()
        df['star_rating'] = pd.to_numeric(df['star_rating'], errors='coerce')
        df = df.dropna()
        df['label'] = df['star_rating'].apply(lambda x: 1 if x >= 3 else 0)
        df.columns = ['text', 'rating', 'label']
        return df[['text', 'label']]

    # Fake dataset
    if 'text_' in df.columns and 'label' in df.columns:
        df = df[['text_', 'label']].dropna()
        df.columns = ['text', 'label']
        return df

    # General detection
    text_cols = ['reviewtext', 'review', 'text', 'content', 'summary']
    rating_cols = ['rating', 'overall', 'score', 'stars']
    label_cols = ['label', 'sentiment', 'fake']

    text_col = rating_col = label_col = None
    for col in df.columns:
        if col in text_cols: text_col = col
        if col in rating_cols: rating_col = col
        if col in label_cols: label_col = col

    if text_col and rating_col:
        df = df[[text_col, rating_col]].dropna()
        df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
        df = df.dropna()
        df['label'] = df[rating_col].apply(lambda x: 1 if x >= 3 else 0)
        df = df[[text_col, 'label']]
        df.columns = ['text', 'label']
        return df

    if text_col and label_col:
        df = df[[text_col, label_col]].dropna()
        df.columns = ['text', 'label']
        return df

    return None

# ----------------------------- Load All CSVs -----------------------------
for file in os.listdir(dataset_path):
    if file.endswith(".csv"):
        try:
            df = pd.read_csv(os.path.join(dataset_path, file))
            processed = process_dataset(df, file)
            if processed is not None and len(processed) > 0:
                all_data.append(processed)
        except:
            continue

# ----------------------------- Combine -----------------------------
if len(all_data) == 0:
    raise Exception("No valid datasets found!")
df = pd.concat(all_data, ignore_index=True)

# ----------------------------- Final Label Fix -----------------------------
df['label'] = df['label'].astype(str).str.lower().str.strip()
df['label'] = df['label'].map({
    '1':1,'0':0,'real':1,'fake':0,'genuine':1,'positive':1,'negative':0
})
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

# ----------------------------- Text Cleaning -----------------------------
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

df['cleaned'] = df['text'].apply(clean_text)

# ----------------------------- TF-IDF -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned'])
y = df['label']

# ----------------------------- Train Model -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = MultinomialNB()
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# ----------------------------- Save -----------------------------
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(vectorizer, open("vectorizer.pkl","wb"))

# ----------------------------- Stats -----------------------------
df['label_name'] = df['label'].map({1:"Genuine",0:"Fake"})
stats = df['label_name'].value_counts().to_dict()
stats.setdefault("Genuine",1)
stats.setdefault("Fake",1)
stats["TotalReviews"] = 0
pickle.dump(stats, open("stats.pkl","wb"))
print("Model + stats saved!")