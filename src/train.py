import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from preprocess import clean_text

# Load JSON dataset
with open("data/intent-corpus-basic.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

# Parse dataset structure
for item in data["sentences"]:
    if item.get("training", True):
        text = item.get("text", "")
        intent = item.get("intent", "unknown")
        if isinstance(text, str) and text.strip():
            texts.append(text)
            labels.append(intent)

df = pd.DataFrame({"text": texts, "label": labels})
print(f"Loaded {len(df)} training samples")

# Clean text
df["text"] = df["text"].astype(str).apply(clean_text)

X = df["text"]
y = df["label"]

# Train-test split (no stratify to avoid rare class issues)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ML Pipeline (better NLP features)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2
    )),
    ("clf", LogisticRegression(max_iter=2000))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(pipeline, "chatbot_model.pkl")
print("\n✅ Model trained and saved as chatbot_model.pkl")