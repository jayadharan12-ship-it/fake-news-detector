import streamlit as st
import pickle
import re
import numpy as np

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
st.write("Enter the news text below and I will analyze it.")

user_input = st.text_area("News content:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some news text!")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])

        # Model prediction
        pred = model.predict(vec)[0]

        # Try probability-based detection (best + most accurate)
        try:
            proba = model.predict_proba(vec)[0]

            # Assume class 1 is "fake" and class 0 is "real" (most ML models follow this)
            fake_prob = proba[1]
            real_prob = proba[0]

            if fake_prob > real_prob:
                st.error(f"❌ Fake News Detected! (Confidence: {fake_prob:.2f})")
            else:
                st.success(f"✔️ Real News Detected! (Confidence: {real_prob:.2f})")

        except:
            # Fallback if predict_proba is unavailable
            pred_str = str(pred).lower()

            if pred_str in ["1", "fake", "true", "yes"]:
                st.error("❌ Fake News Detected!")
            else:
                st.success("✔️ Real News Detected!")

