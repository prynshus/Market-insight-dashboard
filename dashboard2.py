import streamlit as st
import yfinance as yf
from transformers import pipeline
import ta

# --- Setup ---
st.set_page_config(page_title="📊 Quant + Sentiment Dashboard", layout="wide")
st.title("💹 Real-Time Market Signal Dashboard")

# --- Cache functions ---
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_data(ttl=60)
def get_stock_data(ticker):
    df = yf.Ticker(ticker).history(period="7d", interval="30m")
    df = df.dropna()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    return df

# --- Load Model and Data ---
sentiment_analyzer = load_sentiment_model()
ticker = st.sidebar.selectbox("Choose a Fintech Stock", ["PAYTM.NS", "ICICIBANK.NS", "HDFCBANK.NS"], index=0)
df = get_stock_data(ticker)

# --- Latest Values ---
latest_price = df['Close'].iloc[-1]
ma20 = df['MA20'].iloc[-1]
volatility = df['Volatility'].iloc[-1]
rsi = df['RSI'].iloc[-1]

# --- Sentiment Analysis ---
sentiment_input = st.sidebar.text_area("Enter News Headline or Tweet", "Paytm's financials look promising.")
sentiment_result = sentiment_analyzer(sentiment_input)
sentiment_label = sentiment_result[0]['label']
sentiment_score = sentiment_result[0]['score']

# --- Composite Signal ---
signal = "Hold"
if sentiment_score > 0.6 and latest_price > ma20 and df['Volume'].iloc[-1] > df['Volume'].mean():
    signal = "Strong Buy"
elif sentiment_score < 0.4 and rsi > 70:
    signal = "Potential Sell"

# --- Display Metrics ---
st.subheader(f"📍 Signal: {signal}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Price", f"₹{latest_price:.2f}")
col2.metric("RSI", f"{rsi:.2f}")
col3.metric("Sentiment", sentiment_label, f"{sentiment_score:.2%}")
col4.metric("Volatility (σ)", f"{volatility:.2f}")

# --- Charts ---
st.markdown("### 📈 Price with MA20")
st.line_chart(df[['Close', 'MA20']])

st.markdown("### 🔄 RSI Trend")
st.line_chart(df[['RSI']])

# --- Info Box ---
st.markdown("### 🧠 Composite Signal Insight")
st.info(f"Sentiment: {sentiment_label} | RSI: {rsi:.2f} | Signal: {signal}")

# --- Auto Refresh ---
from streamlit_autorefresh import st_autorefresh
st_autorefresh(interval=60 * 1000, key="refresh")
