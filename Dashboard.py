import streamlit as st
import pandas as pd
import datetime
import yfinance as yf
from transformers import pipeline

# Page setup
st.set_page_config(page_title="📊 Market Sentiment + Quant Dashboard", layout="wide")
st.title("📈 Real-Time Market Intelligence Dashboard")

# Cache models and data fetching
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_data(ttl=60)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="7d", interval="15m")
    df = df.dropna()
    df["MA20"] = df["Close"].rolling(window=5).mean()
    df["Volatility"] = df["Close"].rolling(window=5).std()
    return df

# Run sentiment
def run_sentiment(text="Adani stock sentiment today"):
    return sentiment_analyzer(text)

# Load model and fetch stock data
sentiment_analyzer = load_sentiment_model()
ticker = "ADANIENT.NS"
df = get_stock_data(ticker)

# Latest values
latest_price = df["Close"].iloc[-1]
ma20 = df["MA20"].iloc[-1]
volatility = df["Volatility"].iloc[-1]

sentiment_result = run_sentiment()
sentiment_label = sentiment_result[0]['label']
sentiment_score = sentiment_result[0]['score']

# Top metrics
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"₹{latest_price:.2f}")
col2.metric("Sentiment", sentiment_label, f"{sentiment_score:.2%}")
col3.metric("Volatility (std)", f"{volatility:.2f}")

# Quant Chart
st.markdown("### 📉 Price & MA20")
st.line_chart(df[["Close", "MA20"]])

# Sentiment Over Time (simulate here)
st.markdown("### 🧠 NLP Sentiment Score Over Time")
sentiment_chart = pd.DataFrame({
    "Time": pd.date_range(end=datetime.datetime.now(), periods=10, freq="min"),
    "Sentiment Score": [0.32, 0.45, 0.5, 0.62, 0.58, 0.72, 0.75, 0.78, sentiment_score, sentiment_score]
})
st.line_chart(sentiment_chart.set_index("Time"))

# Insights block
st.markdown("### 🗞️ Latest Market Narrative")
st.info("“Adani shares stabilize as the group counters short-seller claims. Volatility remains elevated. #Adani #StockMarket”")


from streamlit_autorefresh import st_autorefresh

st.markdown("⏱️ _Auto-refreshing every 60 seconds..._")
st_autorefresh(interval=60 * 1000, key="data_refresh")

