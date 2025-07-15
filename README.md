# 🧠 Fake News Detection using Machine Learning 

This project focuses on classifying fake vs real news articles using multiple machine learning algorithms and text feature extraction techniques. After preprocessing the text data and converting it into numerical vectors using TF-IDF, we evaluate different classifiers to determine the best performing model.

---

## 🚀 Tech Stack

### 🗂️ Tools & Libraries
- **Python 3.10+**
- **Jupyter Notebook / VS Code**
- **Pandas** – Data manipulation
- **NumPy** – Numerical operations
- **Matplotlib & Seaborn** – Data visualization
- **NLTK** – Text cleaning and stopword removal
- **Scikit-learn (sklearn)** – ML models, TF-IDF, evaluation
- **XGBoost** – Gradient boosting classifier

---

## 🔄 Workflow

### 1. **Data Preprocessing**
- Convert text to lowercase
- Remove punctuation
- Remove stopwords using `nltk.corpus.stopwords`
- Optional: Tokenization with `word_tokenize`

### 2. **Feature Extraction**
- **TF-IDF (Term Frequency-Inverse Document Frequency)** with top 5000 features
- Converted the cleaned text into numerical vectors using `TfidfVectorizer`

### 3. **Model Training**
Trained and evaluated the following models:
- ✅ **XGBoost Classifier** *(Best Performer)*
- Logistic Regression
- Random Forest
- Multinomial Naive Bayes
- *(SVM skipped due to performance issues with large TF-IDF features)*

### 4. **Evaluation Metrics**
Used `classification_report` and `confusion_matrix` to compare:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**

Also visualized model comparison using bar charts and confusion matrices.

---

## ✅ Conclusion

- **XGBoost** delivered the best results on our TF-IDF vectorized dataset.
- TF-IDF + XGBoost is a powerful combination for textual classification tasks.
- Future improvements can include deep learning (e.g., BERT) and word embeddings (Word2Vec/GloVe).

---

## 📌 Dependencies to be installed

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn xgboost
```

---

