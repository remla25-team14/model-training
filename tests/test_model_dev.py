import random, numpy as np, pytest
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from model_training.data_pipeline import load_and_gen_feats
from model_training.dataset import load_historic_dataset
from model_training.modeling.train import train_model


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


@pytest.fixture(scope="module")
def split_data():
    X, y = load_and_gen_feats(as_dense=True)
    df = load_historic_dataset().reset_index(drop=True)
    X_tr, X_te, y_tr, y_te, df_tr, df_te = train_test_split(
        X, y, df, test_size=0.2, stratify=y, random_state=42
    )
    return X_tr, X_te, y_tr, y_te, df_te


@pytest.mark.model_dev
def test_against_baseline(split_data):
    X_tr, X_te, y_tr, y_te, _ = split_data

    base = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    base_acc = accuracy_score(y_te, base.predict(X_te))

    model = train_model(X_tr, y_tr)
    mdl_acc = accuracy_score(y_te, model.predict(X_te))

    assert mdl_acc >= base_acc + 0.1, (
        f"{mdl_acc:.3f} vs baseline {base_acc:.3f}"
    )


@pytest.mark.model_dev
def test_slices(split_data):
    X_tr, X_te, y_tr, y_te, df_te = split_data
    model = train_model(X_tr, y_tr)

    length = df_te["Review"].str.split().str.len()
    very_long = length > 25
    very_short = length < 5

    f1_long = f1_score(y_te[very_long],  model.predict(X_te[very_long]))
    f1_short = f1_score(y_te[very_short], model.predict(X_te[very_short]))

    assert min(f1_long, f1_short) >= 0.65, (
        f"Slice F1 too low long={f1_long:.2f} short={f1_short:.2f}"
    )


@pytest.mark.model_dev
# GaussNB has no nondeterminism!!! So this test will always pass no matter how small the difference set.
# If this is not desired, consider changing a model or introducing some randomness during training.
def test_nondeterminism(split_data):
    X_tr, X_te, y_tr, y_te, _ = split_data
    accs = []
    for seed in (1, 11):
        set_global_seed(seed)
        accs.append(
            accuracy_score(y_te, train_model(X_tr, y_tr).predict(X_te))
        )
    assert abs(accs[0] - accs[1]) <= 0.01
