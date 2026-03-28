import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load stopwords once
stop_words = set(stopwords.words("english"))

def clean_text(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)

    # 3. Remove special characters & numbers
    text = re.sub(r"[^a-zA-Z\s]", '', text)

    # 4. Tokenization
    tokens = word_tokenize(text)

    # 5. Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # 6. Join back to string
    return " ".join(tokens)

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)

    tokens = word_tokenize(text)

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    return " ".join(tokens)

import pandas as pd

df = pd.read_csv("data/processed/combined_data.csv")

df["cleaned_content"] = df["content"].apply(clean_text)

df.to_csv("data/processed/cleaned_news.csv", index=False)
