"""
Test the sentiment model on fresh data
"""
from loguru import logger
import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from the model-training package
from model_training.dataset import load_fresh_dataset
from model_training.modeling.predict import predict_sentiment, get_sentiment_label, load_models

# Load the fresh dataset
logger.info("Loading fresh dataset...")
dataset = load_fresh_dataset()
reviews = dataset['Review'].tolist()[:5]  # Take first 5 reviews for testing

# Load models
logger.info("Loading models...")
vectorizer, classifier = load_models()

# Predict sentiment
logger.info("Predicting sentiment...")
predictions = predict_sentiment(reviews, vectorizer, classifier)

# Print results
logger.info("Results:")
for review, pred in zip(reviews, predictions):
    sentiment = get_sentiment_label(pred)
    logger.info(f"Review: {review}")
    logger.info(f"Sentiment: {sentiment}")
    logger.info("---")

logger.success("Test completed successfully!") 