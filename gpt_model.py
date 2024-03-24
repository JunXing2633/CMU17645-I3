
from openai import OpenAI


# Function to perform sentiment analysis using GPT-4
def analyze_sentiment_gpt(text, openai_api_key):
    """
    Performs sentiment analysis on the provided text using GPT-4.

    Parameters:
    - text (str): The text to analyze.
    - openai_api_key (str): Your OpenAI API key.

    Returns:
    - str: The sentiment of the text ('positive', 'negative', 'neutral').
    """
    # Configure the OpenAI API client
    client = OpenAI(
    # This is the default and can be omitted
    api_key=openai_api_key,
)

    # Construct the prompt to ask GPT-4 for sentiment analysis with structured output
    prompt = (f"Read the following text and classify its sentiment into one of the following categories only: "
              f"'Positive', 'Negative', or 'Neutral'.\n\nText: \"{text}\""
              f"\nSentiment:")
    

    # Send the prompt to the GPT-3.5-turbo model
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
       
    )

    # Extract the response text
    sentiment_raw = response.choices[0].message.content.strip().capitalize()

    # Validate and correct the sentiment output if necessary
    valid_sentiments = ["Positive", "Negative", "Neutral"]
    sentiment = sentiment_raw if sentiment_raw in valid_sentiments else "Neutral"

    return sentiment