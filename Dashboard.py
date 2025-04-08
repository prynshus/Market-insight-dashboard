import streamlit as st
import pandas as pd
import datetime
from transformers import pipeline
import time
import yfinance as yf

st.set_page_config(page_title="Real-Time Market Sentiment", layout="wide")

st.title("📈 Real-Time Market Insights Dashboard")

# Simulated placeholders
latest_price = 1845.60
sentiment_score = 0.72
main_topic = "Adani Group - Short Seller Report"

col1, col2, col3 = st.columns(3)
col1.metric("Stock Price", f"${latest_price}", "+3.2%")
col2.metric("Sentiment Score", f"{sentiment_score:.2f}", "Positive")
col3.metric("Trending Topic", main_topic)

st.markdown("### 📊 Live Sentiment Over Time")
# Dummy data for line chart
df = pd.DataFrame({
    "time": pd.date_range(start=datetime.datetime.now(), periods=10, freq='T'),
    "sentiment": [0.3, 0.5, 0.6, 0.8, 0.72, 0.4, 0.35, 0.7, 0.65, 0.9]
})
st.line_chart(df.set_index("time"))

st.markdown("### 🧠 NLP Insights (from Twitter, Telegram, etc.)")
st.write("“Adani's response to short-seller claims appears to be calming the market. #Adani #StockMarket”")



# Load the model only once
sentiment_analyzer = pipeline("sentiment-analysis")

def run_sentiment_pipeline():
    sentiment_pipeline = pipeline("sentiment-analysis",
                                  model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_pipeline("Adani stock sentiment today")
    return result

def update_streamlit(price, sentiment_score):
    sentiment_label = sentiment_score[0]['label']
    sentiment_value = sentiment_score[0]['score']

    st.metric("Stock Price", f"₹{price:.2f}")
    st.metric("Sentiment", sentiment_label)
    st.progress(sentiment_value)



def get_price(ticker="ADANIENT.NS"):
    stock = yf.Ticker(ticker)
    todays_data = stock.history(period="1d")
    if not todays_data.empty:
        return todays_data["Close"].iloc[-1]
    else:
        return None

while True:
    latest_price = get_price("ADANIENT.NS")  # Use Yahoo Finance or yfinance
    sentiment_score = run_sentiment_pipeline()  # Your NLP model here
    update_streamlit(latest_price, sentiment_score)
    time.sleep(60)  # Update every 60s
