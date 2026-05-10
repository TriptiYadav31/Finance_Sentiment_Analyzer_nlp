import streamlit as st
import pickle
import string
import nltk
import scipy.sparse as sp
import numpy as np
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

with open("sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words and w.isalpha()]
    return " ".join(words)

st.set_page_config(page_title="Financial Sentiment Analyzer", page_icon="📈", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #ffffff;
    }

    .header {
        text-align: center;
        padding: 2.5rem 0 1rem 0;
    }

    .header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.3rem;
    }

    .header p {
        font-size: 1.1rem;
        color: #64748b;
        margin-top: 0;
    }

    .input-label {
        font-size: 1rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 0.4rem;
    }

    .stTextInput input {
        font-size: 1rem;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        color: #0f172a;
        background-color: #f8fafc;
    }

    .stTextInput input:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.15);
    }

    .stButton button {
        width: 100%;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: 700;
        border-radius: 10px;
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: white;
        border: none;
        cursor: pointer;
        margin-top: 0.5rem;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #4f46e5, #3730a3);
    }

    .result-positive {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 2px solid #22c55e;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-negative {
        background: linear-gradient(135deg, #fff1f2, #ffe4e6);
        border: 2px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }

    .result-emoji {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }

    .result-label {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
    }

    .result-positive .result-label { color: #15803d; }
    .result-negative .result-label { color: #b91c1c; }

    .result-confidence {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1rem;
    }

    .result-meaning {
        font-size: 0.95rem;
        color: #334155;
        background: rgba(255,255,255,0.7);
        border-radius: 8px;
        padding: 0.6rem 1rem;
        display: inline-block;
    }

    .divider {
        border: none;
        border-top: 1px solid #e2e8f0;
        margin: 2rem 0;
    }

    .howit {
        background-color: #f8fafc;
        border-radius: 14px;
        padding: 1.5rem 2rem;
        margin-top: 1rem;
    }

    .howit h3 {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
    }

    .step {
        display: flex;
        align-items: flex-start;
        margin-bottom: 0.8rem;
        font-size: 0.95rem;
        color: #334155;
    }

    .step-num {
        background: #6366f1;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        margin-right: 0.8rem;
        flex-shrink: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="header">
        <h1>📈 Financial Sentiment Analyzer</h1>
        <p>Paste any financial headline and instantly know if it's good or bad news for the market</p>
    </div>
""", unsafe_allow_html=True)

# Input
st.markdown('<p class="input-label">Enter a financial headline</p>', unsafe_allow_html=True)
headline = st.text_input("", placeholder="e.g. Apple reports record profits beating analyst expectations...")

analyze = st.button("Analyze Headline")

if analyze:
    if headline.strip() == "":
        st.warning("Please enter a headline first.")
    else:
        cleaned = clean_text(headline)
        tfidf_features = tfidf.transform([cleaned])
        vader_score = analyzer.polarity_scores(headline)["compound"]
        vader_sparse = sp.csr_matrix(np.array([[vader_score]]))
        X = sp.hstack([tfidf_features, vader_sparse])

        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X)[0]

        if prediction == 1:
            conf_pct = confidence[1] * 100
            st.markdown(f"""
                <div class="result-positive">
                    <div class="result-emoji">📈</div>
                    <div class="result-label">Positive Market Signal</div>
                    <div class="result-confidence">Model confidence: {conf_pct:.1f}%</div>
                    <div class="result-meaning">This headline suggests <b>good news</b> for the company or market — investors are likely to react positively.</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            conf_pct = confidence[0] * 100
            st.markdown(f"""
                <div class="result-negative">
                    <div class="result-emoji">📉</div>
                    <div class="result-label">Negative Market Signal</div>
                    <div class="result-confidence">Model confidence: {conf_pct:.1f}%</div>
                    <div class="result-meaning">This headline suggests <b>bad news</b> for the company or market — investors may react with concern.</div>
                </div>
            """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# How it works
st.markdown("""
    <div class="howit">
        <h3>⚙️ How it works</h3>
        <div class="step"><div class="step-num">1</div>Your headline is cleaned — punctuation, numbers and filler words are removed</div>
        <div class="step"><div class="step-num">2</div>TF-IDF converts the words into numbers the model understands</div>
        <div class="step"><div class="step-num">3</div>VADER adds a sentiment score based on the tone of the language</div>
        <div class="step"><div class="step-num">4</div>A Logistic Regression model trained on 1967 real financial headlines makes the final call</div>
    </div>
""", unsafe_allow_html=True)