from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJECT_ROOT path is: {PROJECT_ROOT}")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Model parameters
MAX_FEATURES = 1420
TEST_SIZE = 0.2
RANDOM_STATE = 0

# File names
HISTORIC_DATA_FILE = "a1_RestaurantReviews_HistoricDump.tsv"
FRESH_DATA_FILE = "a2_RestaurantReviews_FreshDump.tsv"
BOW_MODEL_FILE = "bow_vectorizer.pkl"
CLASSIFIER_MODEL_FILE = "sentiment_classifier.pkl"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
