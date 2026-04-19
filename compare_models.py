import pandas as pd
import time
from preprocess import preprocess

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("../data/spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Preprocess
print("Preprocessing...")
df['message'] = df['message'].apply(preprocess)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return (
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, pos_label='spam'),
        recall_score(y_test, y_pred, pos_label='spam'),
        f1_score(y_test, y_pred, pos_label='spam')
    )

results = []

# =========================
# 🔹 BoW + Logistic Regression
# =========================
start = time.time()

bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_bow, y_train)

metrics = evaluate(model, X_test_bow, y_test)
time_taken = time.time() - start

results.append(("BoW + LR", *metrics, time_taken))


# =========================
# 🔹 TF-IDF + Logistic Regression
# =========================
start = time.time()

tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

metrics = evaluate(model, X_test_tfidf, y_test)
time_taken = time.time() - start

results.append(("TF-IDF + LR", *metrics, time_taken))


# =========================
# 🔹 N-Grams (Bigrams) + SVM
# =========================
start = time.time()

ngram = CountVectorizer(ngram_range=(1, 2))
X_train_ng = ngram.fit_transform(X_train)
X_test_ng = ngram.transform(X_test)

model = SVC(kernel='linear')
model.fit(X_train_ng, y_train)

metrics = evaluate(model, X_test_ng, y_test)
time_taken = time.time() - start

results.append(("N-Grams + SVM", *metrics, time_taken))


# =========================
# 📊 Create DataFrame Table
# =========================
columns = ["Method", "Accuracy", "Precision", "Recall", "F1 Score", "Time (s)"]
df_results = pd.DataFrame(results, columns=columns)

print("\n FINAL COMPARISON TABLE:\n")
print(df_results.to_string(index=False))
df_results.to_csv("../results/comparison.csv", index=False)