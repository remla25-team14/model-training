import time, joblib, shutil, tempfile, os, pytest
from pathlib import Path
from sklearn.metrics import accuracy_score

from model_training.data_pipeline import load_and_gen_feats
from model_training.modeling.train import train_model
from model_training.dataset import load_historic_dataset


TMP_DIR = Path(tempfile.gettempdir()) / "mltest_artifacts"


@pytest.mark.ml_infra
def test_step1_read_raw():
    df = load_historic_dataset()
    assert not df.empty and {"Review", "Liked"} <= set(df.columns)


@pytest.mark.ml_infra
def test_step2_preprocess_to_features():
    X_sparse, y = load_and_gen_feats()
    assert X_sparse.shape[0] == len(y) and X_sparse.nnz > 0


@pytest.mark.ml_infra
def test_step3_train_and_dump():
    X, y = load_and_gen_feats(as_dense=True)
    model = train_model(X, y)

    TMP_DIR.mkdir(exist_ok=True)
    model_path = TMP_DIR / "gnb_model.joblib"
    joblib.dump(model, model_path)

    assert model_path.exists() and model_path.stat().st_size > 1_000


@pytest.mark.ml_infra
def test_step4_load_and_predict():
    model_path = TMP_DIR / "gnb_model.joblib"
    if not model_path.exists():
        pytest.skip("model artifact missing, step3 failed")

    model = joblib.load(model_path)
    X, _ = load_and_gen_feats(as_dense=True)
    y_pred = model.predict(X[:100])

    assert len(y_pred) == 100


@pytest.mark.ml_infra
def test_step5_accuracy_and_latency():
    model = joblib.load(TMP_DIR / "gnb_model.joblib")
    X, y = load_and_gen_feats(as_dense=True)
    X_tr, X_te, y_tr, y_te = X[:800], X[800:], y[:800], y[800:]

    t0 = time.perf_counter()
    y_pred = model.predict(X_te)
    latency = time.perf_counter() - t0

    acc = accuracy_score(y_te, y_pred)
    assert acc >= 0.60, f"acc {acc:.2f} < 0.60"
    assert latency < 0.02, f"batch latency {latency:.3f}s ≥ 20 ms"

    shutil.rmtree(TMP_DIR, ignore_errors=True)
