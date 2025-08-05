import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def load_data():
    df = pd.read_csv('dataset/spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['label_num'] = df.label.map({'ham': 0, 'spam': 1})
    return df

def preprocess(df):
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['text'])
    y = df['label_num']
    return train_test_split(X, y, test_size=0.25, random_state=42), vectorizer