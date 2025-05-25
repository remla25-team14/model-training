import time, tempfile, joblib, tracemalloc, numpy as np, pytest
from pathlib import Path
from sklearn.metrics import accuracy_score

from model_training.dataset import load_historic_dataset, load_fresh_dataset
from libml.data_preprocessing import preprocess_reviews
from model_training.features import extract_features
from model_training.modeling.train import train_model


TMP = Path(tempfile.gettempdir()) / "monitoring_tmp"
TMP.mkdir(exist_ok=True)


def _fit_shared_vectorizer():
    df = load_historic_dataset()
    corpus = preprocess_reviews(df)
    X_train, vec = extract_features(corpus, vectorizer_path=None)
    joblib.dump(vec, TMP / "shared_vec.joblib")
    return X_train, vec


def _transform_with_shared_vec(df):
    vec = joblib.load(TMP / "shared_vec.joblib")
    corpus = preprocess_reviews(df)
    X = vec.transform(corpus).toarray()
    return X


@pytest.mark.monitoring
def test_invariants():
    X_train, vec = _fit_shared_vectorizer()
    X_serv = _transform_with_shared_vec(load_fresh_dataset())

    assert X_train.shape[1] == X_serv.shape[1]

    dens_train = np.count_nonzero(X_train) / X_train.size
    dens_serv  = np.count_nonzero(X_serv)  / X_serv.size
    assert abs(dens_train - dens_serv) <= 0.005


@pytest.mark.monitoring
def test_latency_and_memory():
    X_train, _ = _fit_shared_vectorizer()
    y = load_historic_dataset()["Liked"].to_numpy()
    model = train_model(X_train, y)

    X_fresh = _transform_with_shared_vec(load_fresh_dataset())

    model.predict(X_fresh[:1])

    tracemalloc.start()
    t0 = time.perf_counter()
    _ = model.predict(X_fresh)
    dt = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert dt / len(X_fresh) <= 0.010
    assert peak < 10 * 1024**2


@pytest.mark.monitoring
def test_staleness():
    X_train, _ = _fit_shared_vectorizer()
    hist_df = load_historic_dataset()
    y = hist_df["Liked"].to_numpy()

    cutoff = int(len(X_train) * 0.75)
    old_model = train_model(X_train[:cutoff], y[:cutoff])
    new_model = train_model(X_train, y)

    X_fresh = _transform_with_shared_vec(load_fresh_dataset())

    old_preds = old_model.predict(X_fresh)
    new_preds = new_model.predict(X_fresh)

    def tv(p, q):
        return 0.5 * np.abs(p - q).sum()

    dist_old = np.bincount(old_preds, minlength=2) / len(old_preds)
    dist_new = np.bincount(new_preds, minlength=2) / len(new_preds)

    tv_dist = tv(dist_old, dist_new)

    assert tv_dist <= 0.05, f"Prediction drift TV={tv_dist:.3f} > 0.05"
    (TMP / "shared_vec.joblib").unlink(missing_ok=True)
