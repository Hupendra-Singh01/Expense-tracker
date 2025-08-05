from data_loader import load_data, preprocess
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib

def train_and_save_model():
    df = load_data()
    (X_train, X_test, y_train, y_test), vectorizer = preprocess(df)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model trained with accuracy: {acc * 100:.2f}%")

    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

if __name__ == "__main__":
    train_and_save_model()