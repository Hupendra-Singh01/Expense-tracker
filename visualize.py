import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_analysis():
    df = pd.read_csv('dataset/spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'text']
    df['length'] = df['text'].apply(len)

    plt.figure(figsize=(10,5))
    sns.histplot(data=df, x='length', hue='label', bins=50, kde=True)
    plt.title("Message Length Distribution by Label")
    plt.xlabel("Message Length")
    plt.ylabel("Frequency")
    plt.show()