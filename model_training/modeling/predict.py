import joblib
from pathlib import Path
import pandas as pd

from loguru import logger
import typer

from model_training.config import (
    MODELS_DIR,
    BOW_MODEL_FILE,
    CLASSIFIER_MODEL_FILE,
    PROCESSED_DATA_DIR,
)
from libml.data_preprocessing import preprocess_reviews

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
):
    logger.info("Loading test data...")
    # Check if the features_path is a CSV file with reviews or a processed features file
    if features_path.exists():
        if features_path.suffix in [".csv", ".tsv"]:
            try:
                # Try to load as a CSV/TSV file with reviews
                delimiter = "\t" if features_path.suffix == ".tsv" else ","
                df = pd.read_csv(features_path, delimiter=delimiter, quoting=3)
                if "Review" in df.columns:
                    reviews = df["Review"].tolist()
                    logger.info(f"Loaded {len(reviews)} reviews from {features_path}")
                else:
                    logger.error(f"File does not contain 'Review' column: {features_path}")
                    return
            except Exception as e:
                logger.error(f"Failed to load file: {features_path}, error: {e}")
                return
        else:
            logger.error(f"Unsupported file format: {features_path}")
            return
    else:
        logger.error(f"File not found: {features_path}")
        return

    logger.info("Loading models...")
    vectorizer, classifier = load_models()

    logger.info("Predicting sentiment...")
    predictions = predict_sentiment(reviews, vectorizer, classifier)

    # Create a DataFrame with the results
    results_df = pd.DataFrame(
        {
            "Review": reviews,
            "Prediction": predictions,
            "Sentiment": [get_sentiment_label(pred) for pred in predictions],
        }
    )

    # Save predictions
    logger.info(f"Saving predictions to {predictions_path}")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(predictions_path, index=False)

    # Print a summary
    positive_count = sum(predictions)
    negative_count = len(predictions) - positive_count
    logger.info(f"Prediction summary: {positive_count} positive, {negative_count} negative reviews")
    logger.success("Inference complete.")


def load_models(bow_path=None, classifier_path=None):
    """
    Load the trained vectorizer and classifier models

    Args:
        bow_path (Path, optional): Path to the Bag of Words vectorizer
        classifier_path (Path, optional): Path to the trained classifier

    Returns:
        tuple: Loaded vectorizer and classifier
    """
    # Use default paths if not provided
    if bow_path is None:
        bow_path = MODELS_DIR / BOW_MODEL_FILE
    if classifier_path is None:
        classifier_path = MODELS_DIR / CLASSIFIER_MODEL_FILE

    # Load models
    vectorizer = joblib.load(bow_path)
    classifier = joblib.load(classifier_path)

    return vectorizer, classifier


def predict_sentiment(reviews, vectorizer=None, classifier=None):
    """
    Predict sentiment for a list of reviews

    Args:
        reviews (list): List of review texts
        vectorizer: Trained CountVectorizer (if None, will be loaded)
        classifier: Trained classifier (if None, will be loaded)

    Returns:
        numpy.ndarray: Predicted sentiment (1 for positive, 0 for negative)
    """
    # Load models if not provided
    if vectorizer is None or classifier is None:
        vectorizer, classifier = load_models()

    # Create a DataFrame with the reviews
    df = pd.DataFrame({"Review": reviews})

    # Preprocess the reviews
    corpus = preprocess_reviews(df)

    # Transform reviews to feature vectors
    X = vectorizer.transform(corpus).toarray()

    # Make predictions
    predictions = classifier.predict(X)

    return predictions


def predict_sentiment_from_file(file_path, vectorizer=None, classifier=None):
    """
    Predict sentiment for reviews in a file

    Args:
        file_path (Path): Path to the file containing reviews
        vectorizer: Trained CountVectorizer (if None, will be loaded)
        classifier: Trained classifier (if None, will be loaded)

    Returns:
        tuple: Reviews and their predicted sentiments
    """
    # Load reviews from file
    df = pd.read_csv(file_path, delimiter="\t", quoting=3)
    reviews = df["Review"].tolist()

    # Predict sentiment
    predictions = predict_sentiment(reviews, vectorizer, classifier)

    return reviews, predictions


def get_sentiment_label(prediction):
    """
    Convert numeric prediction to text label

    Args:
        prediction (int): Predicted sentiment (1 or 0)

    Returns:
        str: Sentiment label ('Positive' or 'Negative')
    """
    return "Positive" if prediction == 1 else "Negative"


if __name__ == "__main__":
    app()
