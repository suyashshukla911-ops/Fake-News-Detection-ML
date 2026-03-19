import pandas as pd
import numpy as np
import nltk
import re
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

nltk.download('stopwords')
from nltk.corpus import stopwords

# -----------------------------
# Load Dataset
# -----------------------------

fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0
true["label"] = 1

data = pd.concat([fake, true])

data = data[["text", "label"]]

print("Dataset Shape:", data.shape)

# -----------------------------
# Text Cleaning
# -----------------------------

stop_words = set(stopwords.words("english"))

def clean_text(text):

    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

data["text"] = data["text"].apply(clean_text)

# -----------------------------
# Train Test Split
# -----------------------------

X = data["text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------

vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------

model = LogisticRegression()

model.fit(X_train_vec, y_train)

# -----------------------------
# Predictions
# -----------------------------

y_pred = model.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel Saved Successfully!")
def predict_news(news):

    news = clean_text(news)
    news = vectorizer.transform([news])

    prediction = model.predict(news)

    if prediction[0] == 1:
        return "Real News"
    else:
        return "Fake News"


print("\nExample Prediction:")
print(predict_news("Breaking news: government announces new economic policy"))