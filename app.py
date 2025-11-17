import streamlit as st
import pickle
import re

# Load model and vectorizer safely
@st.cache_resource
def load_files():
    model = pickle.load(open("fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_files()

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------- UI ----------------
st.title("📰 Fake News Detector Web App by Jaya")
st.write("Enter any news text below and I will tell you if it's **REAL** or **FAKE**.")

user_input = st.text_area("Enter news text here...", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please type some news text before predicting.")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]

        # If your model outputs 0/1
        try:
            prediction = prediction.lower()
        except:
            pass  # for int values

        # Handle both int and string model outputs
        if prediction in ["fake", 0, "0"]:
            st.error("❌ Fake News Detected!")
        else:
            st.success("✔️ Real News Detected!")


