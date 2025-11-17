import streamlit as st
import pickle
import re

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

st.title("📰 Fake News Detector Web App by Jaya")
st.write("Enter any news headline or paragraph, and I'll predict whether it's REAL or FAKE.")

user_input = st.text_area("Enter news text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
