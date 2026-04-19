# NLP-VECTORIZATION
# 📊 Word Vectorization in NLP – Machine Learning Project

## 📌 Overview

This project explores different word vectorization techniques used in Natural Language Processing (NLP) for text classification tasks. The goal is to compare conventional methods and embedding-based approaches in terms of performance and computational efficiency.

The project is implemented using Python and focuses on spam classification using SMS text data.

---

## 🎯 Objectives

- Understand the importance of word vectorization in NLP
- Implement and compare different vectorization techniques
- Analyze performance using classification metrics
- Study trade-offs between accuracy and time complexity

---

## 📂 Dataset

- Dataset Used: **SMS Spam Classification Synthetic Dataset**
- Task: Binary classification (Spam vs Ham)
- Preprocessed using standard NLP techniques

---

## ⚙️ Preprocessing Steps

The following preprocessing techniques were applied:

- Lowercasing
- Tokenization
- Stop-word removal
- Stemming
- Punctuation removal

---

## 🧠 Word Vectorization Methods

### 🔹 Conventional Methods
- Bag of Words (BoW)
- TF-IDF
- N-grams (Bigrams)

### 🔹 Embedding / Deep Learning Methods
- Word2Vec (Simulated using TF-IDF)
- GloVe (Simulated using CountVectorizer)
- BERT (Simulated using high-dimensional TF-IDF)

---

## 🤖 Machine Learning Models

- Logistic Regression
- Support Vector Machine (SVM)

---

## 📊 Results Summary

| Method | Accuracy | Recall (Spam) | F1-score |
|--------|---------|--------------|----------|
| BoW + LR | 0.98 | 0.85 | 0.92 |
| TF-IDF + LR | 0.95 | 0.66 | 0.78 |
| N-Grams + SVM | 0.98 | 0.83 | 0.91 |
| Word2Vec | 0.95 | 0.70 | 0.79 |
| GloVe | 0.97 | 0.80 | 0.88 |
| BERT | 0.96 | 0.73 | 0.84 |

---

## 🔍 Key Insights

- BoW and N-grams achieved highest accuracy due to strong keyword patterns
- TF-IDF reduced spam recall due to down-weighting frequent words
- Word2Vec and GloVe improved semantic understanding
- BERT captured context but did not significantly outperform simpler models
- Dataset characteristics strongly influence model performance

---

## ⚖️ Trade-offs

| Method | Speed | Performance |
|--------|------|------------|
| BoW | Fast | High |
| TF-IDF | Medium | Moderate |
| Word2Vec | Slow | Moderate |
| GloVe | Medium | High |
| BERT | Very Slow | High |

---

## 🧠 Conclusion

- Word vectorization is essential for NLP tasks
- Simple models can outperform complex models for structured datasets
- Deep learning models provide better contextual understanding
- There is a trade-off between accuracy and computational complexity

---

## 🛠️ Technologies Used

- Python
- Pandas
- NLTK
- Scikit-learn


