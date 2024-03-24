from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "finiteautomata/bertweet-base-sentiment-analysis"

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

# Save model and tokenizer for offline use
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')
