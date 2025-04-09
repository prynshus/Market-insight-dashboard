import streamlit as st
import pandas as pd
import datetime
import yfinance as yf
from transformers import pipeline
from sklearn.ensemble import IsolationForest
import numpy as np

# --- CONFIG ---
st.set_page_config(page_title="📊 Market Sentiment Dashboard", layout="wide")
st.title("📈 Real-Time Market Intelligence with Credibility & Anomaly Detection")

# --- CACHING MODELS ---
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# --- CREDIBILITY SCORING ---
def credibility_score(user_metadata):
    score = 0
    if user_metadata.get("verified"): score += 0.5
    if "financial" in user_metadata.get("bio", "").lower(): score += 0.3
    if user_metadata.get("followers_count", 0) > 10000: score += 0.2
    return round(min(score, 1.0), 2)

# --- ANOMALY DETECTION ---
def detect_anomaly_iforest(sentiment_scores):
    model = IsolationForest(contamination=0.1)
    preds = model.fit_predict(np.array(sentiment_scores).reshape(-1, 1))
    return preds[-1] == -1

# --- PRICE FETCH ---
@st.cache_data(ttl=60)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="7d", interval="15m")
    df["MA20"] = df["Close"].rolling(window=5).mean()
    df["Volatility"] = df["Close"].rolling(window=5).std()
    return df.dropna()

# --- MOCK SOCIAL DATA ---
social_data = [
    {"text": "Paytm stock crash again. Too much pain.", "verified": True, "followers_count": 50000, "bio": "Finance writer"},
    {"text": "Some say Paytm is oversold. RSI looks bullish.", "verified": False, "followers_count": 200, "bio": "Trader"},
    {"text": "Big volumes building up. Might reverse soon!", "verified": False, "followers_count": 8000, "bio": "Market observer"},
]

# --- EXECUTION ---
sentiment_analyzer = load_sentiment_model()
ticker = "PAYTM.NS"
df = get_stock_data(ticker)

latest_price = df["Close"].iloc[-1]
ma20 = df["MA20"].iloc[-1]
volatility = df["Volatility"].iloc[-1]

sentiment_scores = []
cred_scores = []
labels = []
for entry in social_data:
    result = sentiment_analyzer(entry["text"])[0]
    sentiment_scores.append(result["score"] if result["label"] == "POSITIVE" else -result["score"])
    cred_scores.append(credibility_score(entry))
    labels.append(result["label"])

avg_sentiment = np.mean(sentiment_scores)

# --- TOP METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"₹{latest_price:.2f}")
col2.metric("Avg Sentiment", "Positive" if avg_sentiment > 0 else "Negative", f"{avg_sentiment:.2f}")
col3.metric("Volatility", f"{volatility:.2f}")

# --- PRICE CHART ---
st.markdown("### 📉 Price & MA20")
st.line_chart(df[["Close", "MA20"]])

# --- SENTIMENT TIMELINE ---
st.markdown("### 🧠 Sentiment Score Timeline")
timeline = pd.DataFrame({
    "Time": pd.date_range(end=datetime.datetime.now(), periods=len(sentiment_scores), freq="min"),
    "Sentiment Score": sentiment_scores,
    "Credibility": cred_scores
})
st.line_chart(timeline.set_index("Time"))

# --- ALERTS ---
st.markdown("### 🚨 Alerts")
if detect_anomaly_iforest(sentiment_scores):
    st.error("Anomaly Detected in Sentiment - Possible Divergence or Contradiction")
if avg_sentiment > 0.5 and latest_price > ma20 and volatility > 10:
    st.success("📈 Bullish Composite Signal: Positive Sentiment + Price Breakout + High Volatility")

# --- DETAIL TABLE ---
st.markdown("### 🔍 Source Credibility Breakdown")
st.dataframe(pd.DataFrame({
    "Text": [d["text"] for d in social_data],
    "Label": labels,
    "Score": [f"{abs(s):.2f}" for s in sentiment_scores],
    "Credibility": cred_scores
}))

from streamlit_autorefresh import st_autorefresh
st.markdown("⏱️ _Auto-refreshing every 60 seconds..._")
st_autorefresh(interval=60000, key="auto_refresh")
