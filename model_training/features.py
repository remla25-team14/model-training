from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer
import re
import nltk
import joblib
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

from model_training.config import PROCESSED_DATA_DIR

app = typer.Typer()

# Download NLTK data
nltk.download('stopwords', quiet=True)

def preprocess_reviews(dataset):
    """
    Preprocess the text reviews by:
    - Removing non-alphabetic characters
    - Converting to lowercase
    - Removing stopwords
    - Stemming
    
    Args:
        dataset (pd.DataFrame): Dataset containing reviews
        
    Returns:
        list: Preprocessed corpus
    """
    # Initialize stemmer
    ps = PorterStemmer()
    
    # Get stopwords but keep "not" as it's important for sentiment
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    
    corpus = []
    
    # Preprocess each review
    for i in range(len(dataset)):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)
    
    return corpus

def extract_features(corpus, max_features=1420, vectorizer_path=None):
    """
    Extract features from the preprocessed corpus using Bag of Words
    
    Args:
        corpus (list): Preprocessed text corpus
        max_features (int): Maximum number of features to extract
        vectorizer_path (Path, optional): Path to save the vectorizer
        
    Returns:
        tuple: Feature matrix and vectorizer
    """
    # Create and fit the CountVectorizer
    cv = CountVectorizer(max_features=max_features)
    X = cv.fit_transform(corpus).toarray()
    
    # Save the vectorizer if a path is provided
    if vectorizer_path:
        vectorizer_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cv, vectorizer_path)
        logger.info(f"Vectorizer saved to {vectorizer_path}")
    
    return X, cv

@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
):
    logger.info("Generating features from dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    logger.info(f"Loading dataset from {input_path}")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    corpus = preprocess_reviews(df)
    vectorizer_path = PROCESSED_DATA_DIR / "bow_vectorizer.pkl"
    X, cv = extract_features(corpus, vectorizer_path=vectorizer_path)
    feature_df = pd.DataFrame(X)
    if 'Liked' in df.columns:
        feature_df['Liked'] = df['Liked'].values
    logger.info(f"Saving features to {output_path}")
    feature_df.to_csv(output_path, index=False)
    logger.success(f"Features generation complete. Saved to {output_path}")


if __name__ == "__main__":
    app()
