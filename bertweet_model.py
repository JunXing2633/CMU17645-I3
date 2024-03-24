
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model and tokenizer (Do this outside of the function)
model_path = './model'
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

def predict_sentiment_bertweet(row):
    """
    Predicts sentiment of the given text. Returns 1 for Positive sentiment, and 0 for any other sentiment.

    Parameters:
    - text (str): The text to analyze.
    - model: The loaded BERTweet sentiment analysis model.
    - tokenizer: The tokenizer for the BERTweet model.

    Returns:
    - int: 1 if the sentiment is Positive, 0 otherwise.
    """
    # Tokenize text and prepare input for the model
    inputs = tokenizer(row['inputs'], return_tensors="pt")

    # Make a prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # Determine the sentiment (Positive -> 1, Others -> 0)
    predicted_class_id = logits.argmax().item()
    return 1 if predicted_class_id == 2 else 0

