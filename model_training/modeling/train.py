import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import json

from loguru import logger
from model_training.config import (
    RAW_DATA_DIR,
    MODELS_DIR,
    MAX_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    BOW_MODEL_FILE,
    CLASSIFIER_MODEL_FILE,
    REPORTS_DIR
)
from model_training.dataset import load_historic_dataset, download_dataset
from model_training.features import extract_features
from libml.data_preprocessing import preprocess_reviews


def train_model(X_train, y_train, model_path=None):
    """
    Train a Naive Bayes classifier

    Args:
        X_train (numpy.ndarray): Training features
        y_train (numpy.ndarray): Training labels
        model_path (Path, optional): Path to save the trained model

    Returns:
        GaussianNB: Trained classifier
    """
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    if model_path:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(classifier, model_path)
        logger.info(f"Model saved to {model_path}")

    return classifier


def evaluate_model(classifier, X_test, y_test):
    """
    Evaluate the trained model

    Args:
        classifier: Trained classifier
        X_test (numpy.ndarray): Test features
        y_test (numpy.ndarray): Test labels

    Returns:
        tuple: Confusion matrix and accuracy score
    """
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Accuracy: {acc:.4f}")
    
    # Save metrics
    metrics = {
        'accuracy': float(acc),
        'confusion_matrix': cm.tolist()
    }
    
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    return cm, acc


def run_training_pipeline():
    """
    Run the complete training pipeline
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Check if dataset exists before trying to download
    historic_file = RAW_DATA_DIR / "a1_RestaurantReviews_HistoricDump.tsv"

    if not historic_file.exists():
        logger.info("Dataset not found, attempting to download...")
        try:
            download_dataset()
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            logger.info("Trying to proceed with existing data...")
    else:
        logger.info("Dataset already exists, skipping download.")

    logger.info("Loading dataset...")
    dataset = load_historic_dataset()

    logger.info("Preprocessing text...")
    corpus = preprocess_reviews(dataset)

    logger.info("Extracting features...")
    bow_path = MODELS_DIR / BOW_MODEL_FILE
    X, cv = extract_features(corpus, max_features=MAX_FEATURES, vectorizer_path=bow_path)
    y = dataset["Liked"].values

    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info("Training model...")
    classifier_path = MODELS_DIR / "model.joblib"
    classifier = train_model(X_train, y_train, model_path=classifier_path)

    logger.info("Evaluating model...")
    evaluate_model(classifier, X_test, y_test)

    logger.success("Training pipeline completed successfully!")


if __name__ == "__main__":
    run_training_pipeline()
