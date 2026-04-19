import pandas as pd
import numpy as np
from preprocess import preprocess

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer

# Load dataset
df = pd.read_csv("../data/spam.csv", encoding='latin-1')

# Fix columns if needed
if 'v1' in df.columns:
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

# Preprocess
df['clean'] = df['message'].apply(preprocess)

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label'], test_size=0.2, random_state=42
)

# Simulating GloVe using CountVectorizer (co-occurrence idea)
vectorizer = CountVectorizer(max_features=300)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Convert to dense vectors (GloVe-like embedding)
X_train_vec = X_train_vec.toarray()
X_test_vec = X_test_vec.toarray()

# Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Predict
y_pred = model.predict(X_test_vec)

# Evaluation
print("\nGloVe (Simulated) Results:\n")
print(classification_report(y_test, y_pred))