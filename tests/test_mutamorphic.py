import pytest
import joblib
from model_training.dataset import load_historic_dataset
from model_training.features import extract_features
from model_training.modeling.train import train_model

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def generate_mutants(corpus):
    synonyms = {
        "bad": "terrible",
        "okay": "fine",
        "good": "great",
        "quality": "standard"
    }
    mutants = []
    for sentence in corpus:
        mutated = sentence
        for k, v in synonyms.items():
            mutated = mutated.replace(k, v)
        mutants.append(mutated)
    return mutants

def repair_mutant(mutant, correct_label, model, vectorizer):
    mutant_vec = vectorizer.transform([mutant])
    original_vecs = vectorizer.transform(["bad", "terrible", "good", "great"])
    similarities = cosine_similarity(mutant_vec, original_vecs)
    # If at least one similar known word maps to correct label, accept
    if similarities.max() > 0.7:
        pred = model.predict(mutant_vec)[0]
        return pred == correct_label
    return False


def test_mutamorphic_synonym_consistency(tmp_path, monkeypatch):
    # 1) Setup raw and models dirs in tmp
    RAW = tmp_path / "raw"
    MODELS = tmp_path / "models"
    RAW.mkdir()
    MODELS.mkdir()

    # Write minimal TSV
    hist = RAW / "a1_RestaurantReviews_HistoricDump.tsv"
    hist.write_text(
        "Review\tLiked\n"
        "This is okay.\t1\n"
        "This is bad!\t0\n"
        "Good food.\t1\n"
        "Bad quality.\t0\n"
    )

    # Patch dataset loader and model output dir
    monkeypatch.setattr("model_training.dataset.RAW_DATA_DIR", RAW)
    monkeypatch.setattr("model_training.modeling.train.MODELS_DIR", MODELS)

    # 2) Load and extract features
    df = load_historic_dataset()
    corpus = df["Review"].tolist()
    X_sparse, vec = extract_features(corpus, vectorizer_path=MODELS / "vec.joblib")
    joblib.dump(vec, MODELS / "vec.joblib")
    X = X_sparse.toarray() if hasattr(X_sparse, "toarray") else X_sparse

    # Train model
    y = df["Liked"].to_numpy()
    mdl = train_model(X, y, model_path=MODELS / "model.joblib")
    joblib.dump(mdl, MODELS / "model.joblib")

    # 3) Original predictions
    orig_preds = mdl.predict(X)

    # 4) Mutant predictions
    mutants = generate_mutants(corpus)
    mX_sparse = vec.transform(mutants)
    mX = mX_sparse.toarray() if hasattr(mX_sparse, "toarray") else mX_sparse
    mut_preds = mdl.predict(mX)

    # 5) Repair mismatches if any
    mismatches = []
    for orig, o_lbl, m_lbl in zip(corpus, orig_preds, mut_preds):
        if o_lbl != m_lbl and not repair_mutant(orig, o_lbl, mdl, vec):
            mismatches.append(orig)

    assert not mismatches, f"Mutamorphic failures on: {mismatches}"
