import pytest
import time, tracemalloc, os, contextlib, socket, random
import numpy as np
from scipy.sparse import issparse

from libml.data_preprocessing import preprocess_reviews
from model_training.data_pipeline import load_and_gen_feats
from model_training.features import extract_features


@pytest.mark.feature_data
def test_duplicate():
    import pandas as pd
    from model_training.dataset import load_historic_dataset
    df: pd.DataFrame = load_historic_dataset()
    dup_ratio = df.duplicated(subset=["Review", "Liked"]).mean()
    assert dup_ratio < 0.02, f"Too many duplicated rows: {dup_ratio:.2%}"


@pytest.mark.feature_data
def test_feat_latency():
    import pandas as pd
    from model_training.dataset import load_historic_dataset

    df: pd.DataFrame = (
        load_historic_dataset()
        .sample(200, random_state=0)
        .reset_index(drop=True)
    )

    t0 = time.perf_counter()
    corpus = preprocess_reviews(df)
    t1 = time.perf_counter()
    _X, _ = extract_features(corpus, max_features=2000, vectorizer_path=None)
    t2 = time.perf_counter()

    prep_time, vector_time = t1 - t0, t2 - t1
    assert prep_time < 0.5, f"Text pre-processing {prep_time:.3f}s ≥ 0.5s"
    assert vector_time < 1.0, f"Vectorisation {vector_time:.3f}s ≥ 1.0s"


@pytest.mark.feature_data
def test_memory():
    tracemalloc.start()
    load_and_gen_feats()                   # historic 样本集
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < 200 * 1024 ** 2, (
        f"Peak memory {peak/1e6:.1f} MB ≥ 200 MB"
    )


@pytest.mark.feature_data
def test_largest_feat_memory():
    X, _ = load_and_gen_feats()
    col_nnz = np.diff(X.indptr)
    hottest_col = col_nnz.argmax()
    vec_sparse = X[:, hottest_col]
    dense_bytes = vec_sparse.toarray().nbytes
    assert dense_bytes < 10 * 1024 ** 2, \
        f"Single feature dense size {dense_bytes/1e6:.2f} MB > 10 MB"


@pytest.mark.feature_data
def test_basic():
    X, y = load_and_gen_feats("historic")
    assert issparse(X), "Feature matrix must be a scipy sparse"
    assert y is not None and len(y) == X.shape[0], "y should exist and match sample count"
    assert set(np.unique(y)).issubset({0, 1}), "Labels must be binary"
    assert X.shape[0] > 0 and X.shape[1] > 0, "Feature matrix is empty"


@pytest.mark.feature_data
def test_values():
    X, _ = load_and_gen_feats()
    dense = X.toarray()
    assert not np.isnan(dense).any(), "Features contain NaN"
    assert not np.isinf(dense).any(), "Features contain Inf"
    assert (dense >= 0).all(), "Features contain negatives"


@pytest.mark.feature_data
def test_distribution():
    _, y = load_and_gen_feats()
    assert len(np.unique(y)) > 1, "Labels contain only one class"


@pytest.mark.feature_data
def test_correlation():
    X, y = load_and_gen_feats()
    dense = X.toarray()
    diff = np.abs(dense[y == 1].mean(axis=0) - dense[y == 0].mean(axis=0))
    assert np.any(diff > 0.01), "No significant difference between samples"


@pytest.mark.feature_data
def test_sparsity():
    X, _ = load_and_gen_feats()
    assert 500 <= X.shape[1] <= 4000, "Unexpected vocabulary size"
    sparsity = X.nnz / (X.shape[0] * X.shape[1])
    assert 0.001 <= sparsity <= 0.05, f"Sparsity {sparsity:.4f} out of range"


@pytest.mark.feature_data
def test_speed():
    t0 = time.time()
    load_and_gen_feats()
    assert time.time() - t0 < 5, "Feature generation too slow"