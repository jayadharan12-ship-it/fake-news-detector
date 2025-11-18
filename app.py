import streamlit as st
import pickle
import re
import os
import numpy as np

st.set_page_config(page_title="Fake News Detector by Jaya", layout="centered")

@st.cache_resource
def load_files(model_path="fake_news_model.pkl", vec_path="vectorizer.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vec_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vec_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # keep numbers in case helpful
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def predict_with_confidence(model, vectorizer, text: str):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])

    # Get prediction
    pred_raw = model.predict(X)[0]

    # Try to get probability / confidence
    prob = None
    try:
        # many classifiers support predict_proba
        probs = model.predict_proba(X)[0]
        # choose probability of predicted class
        # find predicted class index in model.classes_
        if hasattr(model, "classes_"):
            # find index of pred_raw in classes_ (works for string or numeric)
            # map each element to string for robust matching
            classes = list(model.classes_)
            idx = None
            for i, c in enumerate(classes):
                # compare casted strings (robust to int/str label types)
                if str(c).lower() == str(pred_raw).lower():
                    idx = i
                    break
            if idx is None:
                # fallback: take argmax
                idx = int(np.argmax(probs))
            prob = float(probs[idx])
        else:
            # fallback: take max probability
            prob = float(np.max(probs))
    except Exception:
        # model does not support predict_proba; leave prob as None
        prob = None

    # Normalize label to "fake" or "real"
    label = None
    try:
        # if numeric label
        if isinstance(pred_raw, (int, float)):
            # common convention: 1 -> fake, 0 -> real
            if pred_raw == 1:
                label = "fake"
            else:
                label = "real"
        else:
            s = str(pred_raw).strip().lower()
            if "fake" in s or "false" in s or "1" == s:
                label = "fake"
            elif "real" in s or "true" in s or "0" == s:
                label = "real"
            else:
                # fallback: if model.classes_ available, try to interpret them
                if hasattr(model, "classes_"):
                    classes = [str(c).lower() for c in model.classes_]
                    # try to find which class is more likely if prob exists
                    if prob is not None:
                        idx = int(np.argmax(model.predict_proba(vectorizer.transform([cleaned]))[0]))
                        chosen = classes[idx]
                        if "fake" in chosen or "false" in chosen or chosen == "1":
                            label = "fake"
                        else:
                            label = "real"
                    else:
                        # last fallback: if pred_raw contains '0' or '1'
                        if "1" in s and "0" not in s:
                            label = "fake"
                        else:
                            label = "real"
                else:
                    label = "real"
    except Exception:
        label = "real"

    return label, prob

st.title("📰 Fake News Detector by Jaya")
st.caption("Upload a trained `fake_news_model.pkl` and `vectorizer.pkl` in the app folder if not present.")

# File existence check and friendly message
try:
    model, vectorizer = load_files()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Put your `fake_news_model.pkl` and `vectorizer.pkl` next to this file, or change paths in load_files().")
    st.stop()

st.write("Enter the news text below and click **Predict**. The app will show a label and a confidence (if available).")

user_input = st.text_area("News text", height=220, value="")

col1, col2 = st.columns([3,1])
with col1:
    if st.button("Predict"):
        if user_input.strip() == "":
            st.warning("Please enter some text to predict.")
        else:
            label, prob = predict_with_confidence(model, vectorizer, user_input)

            if prob is not None:
                conf_pct = round(prob * 100, 2)
                conf_text = f"(Confidence: {conf_pct}%)"
            else:
                conf_text = "(Confidence: N/A)"

            if label == "fake":
                st.error(f"❌ Fake News Detected! {conf_text}")
            else:
                st.success(f"✔️ Real News Detected! {conf_text}")

with col2:
    st.markdown("### Quick tests")
    st.write("Click a test to copy into the input box:")
    if st.button("Real - ISRO launch"):
        st.session_state['test_text'] = "The Indian Space Research Organisation announced the successful launch of its latest satellite into orbit today."
        st.experimental_set_query_params()  # noop but ensures session state
        # place into text area by writing directly (workaround)
        st.write("Copied — paste into the input box if needed:")
        st.code(st.session_state['test_text'])
    if st.button("Fake - Cola cures cancer"):
        st.session_state['test_text'] = "Scientists have confirmed that drinking two liters of cola every morning can cure all types of cancer, according to a secret study."
        st.write("Copied — paste into the input box if needed:")
        st.code(st.session_state['test_text'])
    if st.button("Ambiguous / Satire"):
        st.session_state['test_text'] = "A viral claim says that eating only mangoes will instantly make you immune to all diseases, but authorities warn it's misleading."
        st.write("Copied — paste into the input box if needed:")
        st.code(st.session_state['test_text'])

st.markdown("---")
st.markdown("**Notes & tips**")
st.markdown("""
- Ensure your `vectorizer.pkl` was fit the same way your model expects (same preprocessing).
- If your model doesn't support `predict_proba`, the app shows `Confidence: N/A`.
- Common label encodings: `1` = fake, `0` = real. This app tries to infer label meaning automatically.
- If you still get unexpected results, try printing `model.classes_` and sample predictions in a separate script to confirm label mapping.
""")
