from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load IndicBERT fine-tuned for sentiment (you can choose from models on HuggingFace)
model_name = "ai4bharat/IndicBERTv2-MLM-only"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-xlm-roberta-base-sentiment")

# Build sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Sample texts in various languages
texts = [
    "Paytm का stock गिर गया है, investors दुखी हैं।",           # Hindi
    "Paytm stock ennai ketkavum illa, romba bad news.",         # Tamil + English
    "Paytm is crashing again. Total mess!"                     # English
]

# Run sentiment analysis
for text in texts:
    result = sentiment_pipeline(text)[0]
    print(f"Text: {text}")
    print(f"Sentiment: {result['label']} (Confidence: {result['score']:.2f})")
    print("-" * 60)
