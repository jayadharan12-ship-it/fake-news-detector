import streamlit as st
import pickle
import re

# Load model and vectorizer
@st.cache_resource
def load_files():
    model = pickle.load(open("fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_files()

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

st.title("📰 Fake News Detector by Jaya")

user_input = st.text_area("Enter news text...", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]

        # -------------------------------
        # FIX: handle both numeric + string outputs
        # -------------------------------
        if isinstance(pred, (int, float)):
            # Example mapping (common)
            # 1 → fake
            # 0 → real
            if pred == 1:
                st.error("❌ Fake News Detected!")
            else:
                st.success("✔️ Real News Detected!")

        else:
            pred = str(pred).lower()
            if "fake" in pred:
                st.error("❌ Fake News Detected!")
            else:
                st.success("✔️ Real News Detected!")



