# 📰 Fake News Detection using NLP & Machine Learning

## 📌 Project Overview

This project focuses on building a machine learning model to detect **fake vs real news articles** using Natural Language Processing (NLP) techniques. The goal is to help media organizations and fact-checking platforms automatically classify news content and combat misinformation.

---

## 🎯 Objectives

* Classify news articles as **Fake (0)** or **Real (1)**
* Extract meaningful textual features using **TF-IDF**
* Train a baseline model using **Logistic Regression**
* Evaluate performance using standard ML metrics
* Analyze patterns in fake vs real news content

---

## 📂 Dataset

* Source: Kaggle – *Fake and Real News Dataset*
* Files:

  * `Fake.csv`
  * `True.csv`

### Features:

* `title` – headline of the article
* `text` – main content
* `subject` – category (not used for modeling)
* `date` – publication date (not used)

---

## ⚙️ Project Workflow

### 1️⃣ Data Loading & Merging

* Loaded both datasets (`Fake.csv`, `True.csv`)
* Added labels:

  * Fake → `0`
  * Real → `1`
* Combined into a single dataset
* Shuffled data to avoid ordering bias

---

### 2️⃣ Exploratory Data Analysis (EDA)

* Checked dataset size and structure
* Verified class balance:

  * Fake: 23,481 samples
  * Real: 21,417 samples
* Identified and removed:

  * 209 duplicate rows
* Observed text length differences:

  * Fake: ~2642 characters
  * Real: ~2443 characters

---

### 3️⃣ Data Preprocessing

Performed text cleaning to improve feature quality:

* Lowercasing text
* Removing URLs, punctuation, and special characters
* Tokenization
* Stopword removal (using NLTK)
* Lemmatization

Combined:

```text
content = title + text
```

Created:

```text
cleaned_content
```

Handled missing values:

* Removed rows with NaN text after preprocessing

---

### 4️⃣ Feature Engineering (TF-IDF)

Used **TF-IDF Vectorization** to convert text into numerical features:

* `max_features = 5000`
* `ngram_range = (1,2)` → unigrams + bigrams

Key idea:

* TF → word frequency in document
* IDF → penalizes common words

---

### 5️⃣ Train-Test Split

* Split dataset into:

  * 80% training
  * 20% testing
* Used **stratified sampling** to preserve class balance

---

### 6️⃣ Model Training

Trained a **Logistic Regression** model:

* Fast and interpretable baseline
* Effective for high-dimensional sparse data

Saved artifacts:

* `logistic_model.pkl`
* `tfidf_vectorizer.pkl`

---

### 7️⃣ Model Evaluation

Evaluated on **test dataset only** (to avoid data leakage)

### Metrics:

* Accuracy: **99.22%**
* Precision, Recall, F1-score computed
* Confusion Matrix analyzed

### ⚠️ Important Note:

High accuracy required validation to ensure:

* No data leakage
* Proper train/test separation
* No feature leakage (e.g., subject column)

---

## 📊 Key Insights

* Fake news often contains:

  * Repetitive phrases
  * Sensational wording
* Real news tends to be:

  * More structured
  * Slightly shorter in length
* TF-IDF + Logistic Regression performs **surprisingly well** due to strong keyword patterns

---

## 🧠 Learnings

* Importance of **EDA before modeling**
* Handling **text preprocessing correctly**
* Avoiding **data leakage**
* Evaluating on **unseen data only**
* Simple models can outperform complex ones in structured NLP tasks

---

## 🚀 Future Improvements

* Try deep learning models:

  * LSTM (sequence-based learning)
  * BERT (contextual embeddings)
* Use advanced features:

  * Word embeddings (Word2Vec, GloVe)
* Hyperparameter tuning
* Model deployment using:

  * Flask / FastAPI
* Build a real-time fake news detection API

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* NLTK
* Matplotlib, Seaborn
* Joblib

---

## 📁 Project Structure

```
fake-news-detection/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── src/
│   ├── preprocessing.py
│
├── notebooks/
│   ├── eda.ipynb
│   ├── train.ipynb
│   ├── evaluate.ipynb
│
├── models/
│   ├── logistic_model.pkl
│   ├── tfidf_vectorizer.pkl
│
├── requirements.txt
└── README.md
```

---

## 🧩 How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run preprocessing:

```bash
python src/preprocessing.py
```

3. Train model:

```bash
python src/train.py
```

4. Evaluate:

```bash
python src/evaluate.py
```

---

## 🏁 Conclusion

This project demonstrates a **complete NLP pipeline** from raw data to model evaluation. Despite its simplicity, the model achieves high performance, showing that well-engineered features and clean preprocessing can be highly effective for text classification tasks.

---

## 📌 Author

**K Preetham**
