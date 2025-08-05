import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_text(text):
    x = vectorizer.transform([text])
    prediction = model.predict(x)[0]
    return "SPAM" if prediction == 1 else "HAM"