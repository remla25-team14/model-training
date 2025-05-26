from pathlib import Path
import pandas as pd

from loguru import logger
import typer

from model_training.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
):
    logger.info("Processing dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.info("Attempting to download dataset...")
        try:
            download_dataset()
            if input_path.name == "dataset.csv":
                historic_path = RAW_DATA_DIR / "a1_RestaurantReviews_HistoricDump.tsv"
                if historic_path.exists():
                    input_path = historic_path
                    logger.info(f"Using historic dataset: {input_path}")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            return

    logger.info(f"Loading dataset from {input_path}")
    try:
        if input_path.suffix == ".tsv":
            df = pd.read_csv(input_path, delimiter="\t", quoting=3)
        else:
            df = pd.read_csv(input_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info(f"Loaded {len(df)} records")

    if "Review" not in df.columns:
        logger.error("Dataset missing 'Review' column")
        return

    if "Liked" not in df.columns:
        logger.warning(
            "Dataset missing 'Liked' column, this might be a fresh dataset for prediction"
        )

    logger.info(f"Saving processed dataset to {output_path}")
    df.to_csv(output_path, index=False)
    logger.success(f"Processing dataset complete. Saved to {output_path}")


def download_dataset(target_dir=None):
    """Simple fallback - DVC should handle this"""
    logger.info("Data should be managed by DVC. Run 'dvc repro' to get data.")
    raise FileNotFoundError("Run 'dvc pull' or 'dvc repro' to download data")


def load_historic_dataset(file_path=None):
    """
    Load the historic restaurant reviews dataset

    Args:
        file_path (Path, optional): Path to the dataset file

    Returns:
        pd.DataFrame: The loaded dataset
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "a1_RestaurantReviews_HistoricDump.tsv"

    return pd.read_csv(file_path, delimiter="\t", quoting=3)


def load_fresh_dataset(file_path=None):
    """
    Load the fresh restaurant reviews dataset

    Args:
        file_path (Path, optional): Path to the dataset file

    Returns:
        pd.DataFrame: The loaded dataset
    """
    if file_path is None:
        file_path = RAW_DATA_DIR / "a2_RestaurantReviews_FreshDump.tsv"

    return pd.read_csv(file_path, delimiter="\t", quoting=3)


if __name__ == "__main__":
    app()
