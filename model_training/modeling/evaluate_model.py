"""Model evaluation script."""

import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import yaml
from loguru import logger
import typer

from model_training.config import MODELS_DIR, REPORTS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["evaluation"]


def evaluate_predictions(y_true, y_pred):
    """Calculate evaluation metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
    }
    return metrics


def save_metrics(metrics, output_path):
    """Save metrics to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {output_path}")


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.csv",
    model_path: Path = MODELS_DIR / "model.joblib",
    output_path: Path = REPORTS_DIR / "evaluation.json",
):
    """Evaluate the trained model."""
    logger.info("Loading parameters...")
    params = load_params()

    logger.info("Loading model...")
    model = joblib.load(model_path)

    logger.info("Loading test data...")
    df = pd.read_csv(features_path)
    X = df.drop("Liked", axis=1).values if "Liked" in df.columns else df.values
    y = df["Liked"].values if "Liked" in df.columns else None

    if y is None:
        logger.warning("No ground truth labels found in the dataset")
        return

    logger.info("Making predictions...")
    y_pred = model.predict(X)

    logger.info("Calculating metrics...")
    metrics = evaluate_predictions(y, y_pred)

    logger.info("Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    logger.info("Saving metrics...")
    save_metrics(metrics, output_path)

    logger.success("Evaluation complete!")


if __name__ == "__main__":
    app()
