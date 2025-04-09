import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

try:
    sentiment_analyzer = load_sentiment_model()
    st.success("Sentiment model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
