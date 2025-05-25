import time, joblib, shutil, tempfile, os, pytest, json, pathlib, pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score

from model_training.data_pipeline import load_and_gen_feats
from model_training.modeling.train import train_model
from model_training.dataset import load_historic_dataset
from model_training import features as feat_mod
from model_training.modeling import train as train_mod
from model_training.modeling import predict as pred_mod


TMP_DIR = Path(tempfile.gettempdir()) / "mltest_artifacts"


@pytest.mark.ml_infra
def test_step1_read():
    df = load_historic_dataset()
    assert not df.empty and {"Review", "Liked"} <= set(df.columns)


@pytest.mark.ml_infra
def test_step2_preprocess():
    X_sparse, y = load_and_gen_feats()
    assert X_sparse.shape[0] == len(y) and X_sparse.nnz > 0


@pytest.mark.ml_infra
def test_step3_train():
    X, y = load_and_gen_feats(as_dense=True)
    model = train_model(X, y)

    TMP_DIR.mkdir(exist_ok=True)
    model_path = TMP_DIR / "gnb_model.joblib"
    joblib.dump(model, model_path)

    assert model_path.exists() and model_path.stat().st_size > 1_000


@pytest.mark.ml_infra
def test_step4_predict():
    model_path = TMP_DIR / "gnb_model.joblib"
    if not model_path.exists():
        pytest.skip("model artifact missing, step3 failed")

    model = joblib.load(model_path)
    X, _ = load_and_gen_feats(as_dense=True)
    y_pred = model.predict(X[:100])

    assert len(y_pred) == 100


@pytest.mark.ml_infra
def test_step5_performance():
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


@pytest.mark.ml_infra
def test_features_main(tmp_path, monkeypatch):
    hist_tsv = tmp_path / "raw" / "a1_RestaurantReviews_HistoricDump.tsv"
    hist_tsv.parent.mkdir(parents=True, exist_ok=True)
    hist_tsv.write_text("Review\tLiked\nGreat taste!\t1\nTerrible!\t0\n")

    monkeypatch.setattr(feat_mod, "RAW_DATA_DIR", tmp_path / "raw")
    monkeypatch.setattr(feat_mod, "PROCESSED_DATA_DIR", tmp_path / "proc")

    feat_mod.main(
        input_path=hist_tsv,
        output_path=feat_mod.PROCESSED_DATA_DIR / "features.csv",
        vectorizer_path=feat_mod.PROCESSED_DATA_DIR / "bow_vectorizer.pkl",
    )

    feat_csv = feat_mod.PROCESSED_DATA_DIR / "features.csv"
    vec_pkl  = feat_mod.PROCESSED_DATA_DIR / "bow_vectorizer.pkl"

    assert feat_csv.exists() and vec_pkl.exists()

    df = pd.read_csv(feat_csv)
    assert len(df) == 2
    if 'Liked' in df.columns:
        assert set(df['Liked']) == {0, 1}


@pytest.mark.ml_infra
def test_train_pipeline(tmp_path: pathlib.Path, monkeypatch):
    monkeypatch.setattr(train_mod, "RAW_DATA_DIR",   tmp_path / "raw")
    monkeypatch.setattr(train_mod, "MODELS_DIR",     tmp_path / "models")
    monkeypatch.setattr(train_mod, "REPORTS_DIR",    tmp_path / "reports")

    def _fake_download():
        dest = train_mod.RAW_DATA_DIR / "a1_RestaurantReviews_HistoricDump.tsv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("Review\tLiked\nGreat!\t1\nTerrible\t0\n")
    monkeypatch.setattr(train_mod, "download_dataset", _fake_download)

    train_mod.run_training_pipeline()

    model_file   = train_mod.MODELS_DIR  / "model.joblib"
    metrics_file = train_mod.REPORTS_DIR / "metrics.json"

    assert model_file.exists()   and model_file.stat().st_size > 1_000
    assert metrics_file.exists() and metrics_file.stat().st_size > 50

    with metrics_file.open() as f:
        acc = json.load(f)["accuracy"]
    assert acc >= 0.60, f"pipeline accuracy {acc:.2f} < 0.60"


@pytest.mark.ml_infra
def test_predict_pipeline(tmp_path: pathlib.Path, monkeypatch):
    reviews_tsv = tmp_path / "fresh.tsv"
    reviews_tsv.write_text("Review\tLiked\nGreat food\nAwful service\n")

    monkeypatch.setattr(train_mod, "RAW_DATA_DIR",   tmp_path / "raw")
    monkeypatch.setattr(train_mod, "MODELS_DIR",     tmp_path / "models")
    monkeypatch.setattr(train_mod, "REPORTS_DIR",    tmp_path / "reports")

    monkeypatch.setattr(pred_mod,  "PROCESSED_DATA_DIR", tmp_path / "proc")
    monkeypatch.setattr(pred_mod,  "MODELS_DIR",         tmp_path / "models")

    def _fake_download():
        dest = train_mod.RAW_DATA_DIR / "a1_RestaurantReviews_HistoricDump.tsv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("Review\tLiked\nNice!\t1\nBad!\t0\n")
    monkeypatch.setattr(train_mod, "download_dataset", _fake_download)

    train_mod.run_training_pipeline()

    (pred_mod.MODELS_DIR / "sentiment_classifier.pkl").write_bytes(
        (pred_mod.MODELS_DIR / "model.joblib").read_bytes()
    )

    pred_mod.main(
        features_path=reviews_tsv,
        model_path=pred_mod.MODELS_DIR / "model.joblib",
        predictions_path=pred_mod.PROCESSED_DATA_DIR / "preds.csv",
    )

    preds_csv = pred_mod.PROCESSED_DATA_DIR / "preds.csv"
    assert preds_csv.exists() and preds_csv.stat().st_size > 50

    df = pd.read_csv(preds_csv)
    assert len(df) == 2 and {"Review", "Prediction", "Sentiment"} <= set(df.columns)
