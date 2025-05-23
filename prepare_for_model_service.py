"""
Prepare model files for the model-service.

This script copies the trained model files to the format expected by the model-service.
"""
import os
import shutil
from pathlib import Path
from loguru import logger

def prepare_models_for_service():
    """
    Copy and rename the model files to match what model-service expects.
    """
    models_dir = Path("models")
    bow_src = models_dir / "bow_vectorizer.pkl"
    clf_src = models_dir / "sentiment_classifier.pkl"
    
    if not bow_src.exists() or not clf_src.exists():
        logger.error("Model files not found. Please train the model first.")
        logger.info("Run: python -m model_training.modeling.train")
        return False
    
    # Create a directory for model-service files
    output_dir = Path("model_service_artifacts")
    output_dir.mkdir(exist_ok=True)
    
    bow_dst = output_dir / "c1_BoW_v1.pkl"
    clf_dst = output_dir / "c2_Classifier_v1.pkl"
    
    shutil.copy2(bow_src, bow_dst)
    shutil.copy2(clf_src, clf_dst)
    
    logger.success(f"Model files prepared for model-service in {output_dir}")
    logger.info(f"Original vectorizer: {bow_src} → {bow_dst}")
    logger.info(f"Original classifier: {clf_src} → {clf_dst}")
    return True

if __name__ == "__main__":
    if prepare_models_for_service():
        logger.info("Now you can use these files with the model-service")
        logger.info("Options:")
        logger.info("1. Create a ZIP archive and upload it manually as a GitHub artifact")
        logger.info("2. Copy the files directly to the model-service's model_cache directory") 