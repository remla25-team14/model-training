from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from scipy import sparse
from loguru import logger
import pandas as pd
from model_training import config, dataset, features
from libml.data_preprocessing import preprocess_reviews


def _load_dataframe(which: str = "historic") -> "pd.DataFrame":
    if which == "historic":
        return dataset.load_historic_dataset()
    elif which == "fresh":
        return dataset.load_fresh_dataset()
    else:
        raise ValueError("which must be 'historic' or 'fresh'")


def load_and_gen_feats(
    which: str = "historic",
    max_features: Optional[int] = None,
    vectorizer_path: Optional[Path] = None,
    as_dense: bool = False,
) -> Tuple[sparse.csr_matrix | np.ndarray, Optional[np.ndarray]]:

    import pandas as pd

    # 1. load dataframe
    df: pd.DataFrame = _load_dataframe(which)
    logger.info(f"{which.capitalize()} dataset loaded: {df.shape}")

    # 2. preprocess text → list[str]
    corpus = preprocess_reviews(df)        # uses default 'Review' col
    logger.info("Text preprocessing finished")

    # 3. BOW vectorisation → dense ndarray
    X_dense, _ = features.extract_features(
        corpus,
        max_features=max_features or config.MAX_FEATURES,
        vectorizer_path=vectorizer_path,
    )

    # 4. labels (may not exist for 'fresh')
    y = df["Liked"].to_numpy(dtype=np.int8) if "Liked" in df.columns else None

    # 5. sparse or dense out
    X = X_dense if as_dense else sparse.csr_matrix(X_dense)
    logger.info(f"Generated X={X.shape}, y={None if y is None else y.shape}")

    return X, y


if __name__ == "__main__":
    X, y = load_and_gen_feats()
    print("Feature matrix", X.shape, "label vector", None if y is None else y.shape)
