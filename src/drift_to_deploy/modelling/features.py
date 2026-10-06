"""Feature engineering, and the full scikit-learn pipelines that contain it.

The engineered features live inside the logged model, so serving runs exactly the code
training ran: there is no second implementation to drift out of sync.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from drift_to_deploy.data.schema import BILL_AMOUNTS, FEATURES, PAY_AMOUNTS, PAY_STATUS

CATEGORICAL = ["education", "marriage"]
UTILISATION = [f"utilisation_{i}" for i in range(1, 7)]
ENGINEERED = [*UTILISATION, "mean_utilisation", "max_utilisation", "payment_ratio", "months_late", "max_delay"]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Raw model inputs (FEATURES) -> raw inputs plus engineered ratios and delinquency counts."""
    X = X[FEATURES].copy()
    limit = X["limit_bal"].clip(lower=1)
    for i, bill in enumerate(BILL_AMOUNTS, 1):
        # Clipped: a few customers are far over their limit, and the tail adds noise, not signal.
        X[f"utilisation_{i}"] = (X[bill] / limit).clip(-1, 5)
    X["mean_utilisation"] = X[UTILISATION].mean(axis=1)
    X["max_utilisation"] = X[UTILISATION].max(axis=1)
    billed = X[BILL_AMOUNTS].clip(lower=0).sum(axis=1)
    paid = X[PAY_AMOUNTS].sum(axis=1)
    X["payment_ratio"] = np.where(billed > 0, paid / billed.clip(lower=1), 1.0).clip(0, 5)
    X["months_late"] = (X[PAY_STATUS] > 0).sum(axis=1)
    X["max_delay"] = X[PAY_STATUS].max(axis=1)
    return X


def _feature_names(transformer: FunctionTransformer, input_features: object) -> list[str]:
    # A named function rather than a lambda, so the pipeline pickles with the standard library.
    return [*FEATURES, *ENGINEERED]


def _numeric() -> list[str]:
    return [c for c in [*FEATURES, *ENGINEERED] if c not in CATEGORICAL]


def build(kind: str, settings: dict, seed: int) -> Pipeline:
    """A fitted-ready pipeline that takes raw FEATURES columns and outputs default probabilities."""
    engineer = FunctionTransformer(add_features, feature_names_out=_feature_names)
    if kind == "lightgbm":
        prep = ColumnTransformer(
            [
                ("num", "passthrough", _numeric()),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ],
            verbose_feature_names_out=False,
        )
        model = LGBMClassifier(random_state=seed, verbose=-1, **settings)
    elif kind == "logistic":
        prep = ColumnTransformer(
            [
                ("num", StandardScaler(), _numeric()),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ],
            verbose_feature_names_out=False,
        )
        model = LogisticRegression(max_iter=2000, **settings)
    else:
        raise ValueError(f"Unknown model kind '{kind}'.")
    return Pipeline([("features", engineer), ("prep", prep), ("model", model)]).set_output(transform="pandas")


CANDIDATES = ("lightgbm", "logistic")

# MLflow saves models with skops, which refuses to load any type that isn't explicitly
# trusted (unlike pickle, which runs whatever the file contains). These are the only
# non-scikit-learn types our pipelines contain; anything else fails loudly.
TRUSTED_TYPES = [
    "collections.OrderedDict",
    "drift_to_deploy.modelling.features._feature_names",
    "drift_to_deploy.modelling.features.add_features",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
]
