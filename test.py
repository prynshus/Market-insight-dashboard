from transformers import pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
result = sentiment_pipeline("Adani stock is showing strong performance")
print(result)

sentiment = run_sentiment_pipeline("Adani stock is down!")
label = sentiment[0]['label']
score = sentiment[0]['score']

print(label, score)

