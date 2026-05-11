# Finance_Sentiment_Analyzer_nlp
# 📈 Financial News Sentiment Analyzer

A machine learning web app that analyzes financial news headlines and predicts whether they carry a **positive** or **negative** market signal.

🔗 **Live Demo** → [https://financesentimentanalyzernlp-fhgnhrzhw4jfeypdwvrdni.streamlit.app/](https://financesentimentanalyzernlp-fhgnhrzhw4jfeypdwvrdni.streamlit.app/?utm_source=chatgpt.com)

---

## 📌 What it does
Paste any financial headline and the app instantly tells you:
- Whether the news is positive or negative for the market
- How confident the model is in its prediction

---

## 🛠️ How it works
1. Headline is cleaned — punctuation, numbers and stopwords removed
2. TF-IDF converts words into numerical features
3. VADER adds a sentiment score based on language tone
4. A Logistic Regression model makes the final prediction

---

## 📊 Model Performance

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | 88% | 0.86 |
| XGBoost | 86% | 0.83 |
| Naive Bayes | 77% | 0.63 |

Logistic Regression outperformed both models — proving that simpler models can beat complex ones on small datasets.

---

## 📁 Dataset
**FinancialPhraseBank** — 4846 financial news sentences labelled as positive, negative or neutral by finance experts. Available on [Kaggle](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news).

---

## 🔑 Key Findings
- Growth words like *rose, grew, increase* strongly signal positive news
- Decline words like *decreased, fell, layoffs* strongly signal negative news
- VADER score was the **second most powerful feature** in the model
- Naive Bayes failed on subtle financial language due to its word independence assumption

---

## 🧰 Tech Stack
- Python
- Scikit-learn — TF-IDF, Logistic Regression
- XGBoost
- NLTK — text preprocessing
- VADER — rule based sentiment scoring
- Streamlit — web app

---

## 🚀 Run locally
```bash
pip install -r requirements.txt
streamlit run NewsApp.py
```
