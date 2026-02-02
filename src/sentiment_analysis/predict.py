from src.config import SENTIMENT_MODEL_PATH, SENTIMENT_TOKENIZER_PATH
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

_sentiment_model = None
_sentiment_tokenizer = None

def get_sentiment_tokenizer():
    global _sentiment_tokenizer
    if _sentiment_tokenizer is None:
        _sentiment_tokenizer = AutoTokenizer.from_pretrained(str(SENTIMENT_TOKENIZER_PATH))
    return _sentiment_tokenizer


def get_sentiment_model():
    global _sentiment_model
    if _sentiment_model is None:
        _sentiment_model = AutoModelForSequenceClassification.from_pretrained(str(SENTIMENT_MODEL_PATH))
    return _sentiment_model

sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

def sentiment_analysis(text):
    tokenizer = get_sentiment_tokenizer()
    model = get_sentiment_model()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=-1).item()
    return sentiment_map[pred]
