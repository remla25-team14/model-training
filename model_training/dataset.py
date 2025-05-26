<<<<<<< HEAD
"""Dataset handling utilities and download functionality."""
=======
>>>>>>> main
from pathlib import Path
import os
import pandas as pd
import requests
<<<<<<< HEAD
import yaml
=======

>>>>>>> main
from loguru import logger
from tqdm import tqdm
import typer

from model_training.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

<<<<<<< HEAD
def load_params():
    """Load parameters from params.yaml."""
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    return params["data"]["download"]

def download_file(url: str, destination: str):
    """Download a file from url to destination with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f, tqdm(
        desc=f"Downloading {os.path.basename(destination)}",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            pbar.update(size)

def download_dataset():
    """Download both training and test datasets using parameters from params.yaml."""
    params = load_params()
    raw_data_dir = Path("data/raw")
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download training data
    training_data = params["training_data"]
    training_dest = raw_data_dir / "a1_RestaurantReviews_HistoricDump.tsv"
    logger.info(f"Downloading training dataset to {training_dest}")
    download_file(training_data["url"], str(training_dest))
    
    # Download test data
    test_data = params["test_data"]
    test_dest = raw_data_dir / "a2_RestaurantReviews_FreshDump.tsv"
    logger.info(f"Downloading test dataset to {test_dest}")
    download_file(test_data["url"], str(test_dest))
=======

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
        if input_path.suffix == '.tsv':
            df = pd.read_csv(input_path, delimiter='\t', quoting=3)
        else:
            df = pd.read_csv(input_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    logger.info(f"Loaded {len(df)} records")
    
    if 'Review' not in df.columns:
        logger.error("Dataset missing 'Review' column")
        return
    
    if 'Liked' not in df.columns:
        logger.warning("Dataset missing 'Liked' column, this might be a fresh dataset for prediction")
    
    logger.info(f"Saving processed dataset to {output_path}")
    df.to_csv(output_path, index=False)
    logger.success(f"Processing dataset complete. Saved to {output_path}")


def download_dataset(target_dir=None):
    """Simple fallback - DVC should handle this"""
    logger.info("Data should be managed by DVC. Run 'dvc repro' to get data.")
    raise FileNotFoundError("Run 'dvc pull' or 'dvc repro' to download data")
>>>>>>> main

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
    
<<<<<<< HEAD
    if not file_path.exists():
        logger.warning(f"Dataset not found at {file_path}")
        try:
            download_dataset()
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise FileNotFoundError(f"Could not find or download dataset at {file_path}")
    
=======
>>>>>>> main
    return pd.read_csv(file_path, delimiter='\t', quoting=3)

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
    
<<<<<<< HEAD
    if not file_path.exists():
        logger.warning(f"Dataset not found at {file_path}")
        try:
            download_dataset()
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise FileNotFoundError(f"Could not find or download dataset at {file_path}")
    
    return pd.read_csv(file_path, delimiter='\t', quoting=3)

@app.command()
def main():
    """Command-line interface for dataset operations."""
    logger.info("Downloading datasets...")
    download_dataset()
    logger.success("Datasets downloaded successfully")
=======
    return pd.read_csv(file_path, delimiter='\t', quoting=3)

>>>>>>> main

if __name__ == "__main__":
    app()
