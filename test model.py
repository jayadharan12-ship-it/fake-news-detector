import pickle

model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

X = vectorizer.transform(["this is a fake news example"])
print("Prediction:", model.predict(X)[0])
