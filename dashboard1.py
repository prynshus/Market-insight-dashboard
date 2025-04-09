import streamlit as st
import pandas as pd
import datetime
import yfinance as yf
from transformers import pipeline
from streamlit_autorefresh import st_autorefresh

# 🚀 Setup
st.set_page_config(page_title="📊 Fintech Sentiment + Quant Dashboard", layout="wide")
st.title("💹 Real-Time Fintech Intelligence: Paytm")

# ⚙️ Load sentiment model once
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",revision="714eb0f")

# 📈 Get 7-day stock data with quant metrics
@st.cache_data(ttl=60)
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="7d", interval="15m")
    df = df.dropna()
    df["MA20"] = df["Close"].rolling(window=5).mean()
    df["Volatility"] = df["Close"].rolling(window=5).std()
    return df

# 🧠 Run sentiment on news headline or default
def run_sentiment(text="Paytm sees rising investor interest amid digital payment growth."):
    return sentiment_analyzer(text)

# 🔧 Choose fintech company
ticker = "PAYTM.NS"
company_name = "Paytm"

# 🔍 Load model and data
sentiment_analyzer = load_sentiment_model()
df = get_stock_data(ticker)

# 🔢 Latest market stats
latest_price = df["Close"].iloc[-1]
ma20 = df["MA20"].iloc[-1]
volatility = df["Volatility"].iloc[-1]

# 📊 Run sentiment analysis
sentiment_result = run_sentiment()
sentiment_label = sentiment_result[0]['label']
sentiment_score = sentiment_result[0]['score']

# 📌 Dashboard Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"₹{latest_price:.2f}")
col2.metric("Sentiment", sentiment_label, f"{sentiment_score:.2%}")
col3.metric("Volatility (σ)", f"{volatility:.2f}")

# 📉 Quant: Price + MA20
st.markdown(f"### 📊 {company_name} Price with Moving Average (MA20)")
st.line_chart(df[["Close", "MA20"]])

# 📈 Sentiment Timeline (simulated)
st.markdown("### 🧠 Sentiment Trend")
sentiment_chart = pd.DataFrame({
    "Time": pd.date_range(end=datetime.datetime.now(), periods=10, freq="min"),
    "Sentiment Score": [0.35, 0.45, 0.52, 0.58, 0.63, 0.69, 0.71, 0.74, sentiment_score, sentiment_score]
})
st.line_chart(sentiment_chart.set_index("Time"))

# 🗞️ News/Narrative
st.markdown("### 🗞️ Market Narrative")
st.info("“Paytm stock rebounds as digital transaction volume soars. Positive sentiment from retail investors continues. #Fintech #India”")

# 🔄 Auto-refresh every 60 sec
st.markdown("⏱️ _Auto-refreshing every 60 seconds..._")
st_autorefresh(interval=60 * 1000, key="data_refresh")

try:
    sentiment_analyzer = load_sentiment_model()
except Exception as e:
    st.error("⚠️ Failed to load sentiment model.")
    st.stop()
