import pandas as pd
import numpy as np
from preprocess import preprocess

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("../data/spam.csv", encoding='latin-1')
if 'v1' in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
# Preprocess
df['clean'] = df['message'].apply(preprocess)

# Use TF-IDF as word embedding (Word2Vec-like representation)
vectorizer = TfidfVectorizer(max_features=100)

X = vectorizer.fit_transform(df['clean']).toarray()
y = df['label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nWord2Vec (Simulated) + Logistic Regression\n")
print(classification_report(y_test, y_pred))