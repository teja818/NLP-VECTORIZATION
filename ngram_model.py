import pandas as pd
from preprocess import preprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("../data/spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Preprocess
df['message'] = df['message'].apply(preprocess)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Vectorization (N-grams: bigrams)
vectorizer = CountVectorizer(ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model (SVM)
model = SVC(kernel='linear')
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)

print("\nN-Grams (Bigrams) + SVM\n")
print(classification_report(y_test, y_pred))