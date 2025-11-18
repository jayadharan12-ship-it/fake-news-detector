import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report


# ----------------------------------------------------------
# TEXT CLEANING FUNCTION
# ----------------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------
# IMPORTANT:
# Your CSV must contain:  text , label
# Label should be 'fake' or 'real'
df = pd.read_csv("fake_news_dataset.csv")

# Clean text column
df["text"] = df["text"].astype(str).apply(clean_text)

# Convert labels to numeric
# fake = 1 , real = 0
df["label"] = df["label"].map({"fake": 1, "real": 0})

# Drop rows with missing labels
df = df.dropna(subset=["label"])


# ----------------------------------------------------------
# TRAIN / TEST SPLIT
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)


# ----------------------------------------------------------
# TF-IDF VECTORIZER
# ----------------------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=50000,        # large feature space = better accuracy
    ngram_range=(1, 2),        # unigrams + bigrams
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)


# ----------------------------------------------------------
# TRAIN SVM MODEL
# ----------------------------------------------------------
model = LinearSVC()
model.fit(X_train_vec, y_train)

print("Training completed successfully.")
print("Evaluating model...\n")


# ----------------------------------------------------------
# MODEL EVALUATION
# ----------------------------------------------------------
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

print(classification_report(y_test, y_pred))


# ----------------------------------------------------------
# SAVE MODEL + VECTORIZER
# ----------------------------------------------------------
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel and vectorizer saved successfully!")
print("Files generated:")
print("  - fake_news_model.pkl")
print("  - vectorizer.pkl")
