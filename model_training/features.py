from model_training.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


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
    input_path: Path = RAW_DATA_DIR / "a1_RestaurantReviews_HistoricDump.tsv",
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    vectorizer_path: Path = PROCESSED_DATA_DIR / "bow_vectorizer.pkl"
):
    logger.info("Generating features from dataset...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    logger.info(f"Loading dataset from {input_path}")
    try:
        df = pd.read_csv(input_path, delimiter='\t', quoting=3)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    logger.info("Preprocessing text...")
    corpus = preprocess_reviews(df)
    
    logger.info("Extracting features...")
    X, cv = extract_features(corpus, vectorizer_path=vectorizer_path)
    
    logger.info("Creating feature DataFrame...")
    feature_df = pd.DataFrame(X)
    if 'Liked' in df.columns:
        feature_df['Liked'] = df['Liked'].values

    logger.info(f"Saving features to {output_path}")
    feature_df.to_csv(output_path, index=False)
    logger.success(f"Features generation complete. Saved to {output_path}")


if __name__ == "__main__":
    app()

