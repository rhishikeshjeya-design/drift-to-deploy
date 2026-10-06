"""Synthetic data that satisfies the schema, so tests never need the network or the real dataset."""

import numpy as np
import pandas as pd
import pytest

from drift_to_deploy.data.schema import BILL_AMOUNTS, ID, PAY_AMOUNTS, PAY_STATUS, TARGET
from drift_to_deploy.params import Params


def make_clean(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    limit = rng.choice([10_000, 50_000, 100_000, 200_000, 500_000], size=n).astype("float64")
    frame = {
        ID: np.arange(1, n + 1),
        "limit_bal": limit,
        "sex": rng.integers(1, 3, n),
        "education": rng.integers(1, 5, n),
        "marriage": rng.integers(1, 4, n),
        # Roughly a third under 30, like the real data.
        "age": np.where(rng.random(n) < 0.32, rng.integers(21, 30, n), rng.integers(30, 70, n)),
        **{c: rng.integers(-2, 4, n) for c in PAY_STATUS},
        **{c: (limit * rng.uniform(-0.05, 1.1, n)).round() for c in BILL_AMOUNTS},
        **{c: rng.uniform(0, 20_000, n).round() for c in PAY_AMOUNTS},
    }
    df = pd.DataFrame(frame)
    # Defaults depend on repayment status, so models have real signal to learn.
    risk = 0.12 + 0.25 * (df["pay_1"] > 0) + 0.1 * (df["pay_2"] > 0)
    df[TARGET] = (rng.random(n) < risk).astype(int)
    return df


@pytest.fixture
def tracking(tmp_path):
    """A throwaway MLflow tracking store and registry per test."""
    from drift_to_deploy.modelling import registry

    return registry.configure(
        registry.Tracking(f"sqlite:///{tmp_path / 'mlflow.db'}", (tmp_path / "artifacts").as_uri())
    )


@pytest.fixture
def clean() -> pd.DataFrame:
    return make_clean()


@pytest.fixture
def params() -> Params:
    return Params.model_validate(
        {
            "seed": 7,
            "source": {"url": "https://example.invalid/data.zip", "sha256": "0" * 64},
            "batches": {"reference_size": 2000, "months": 6, "month_size": 400},
            "scenarios": [
                {
                    "name": "squeeze",
                    "kind": "covariate",
                    "start_month": 2,
                    "limit_bal_factor": 0.7,
                    "bill_amt_factor": 1.25,
                },
                {"name": "young", "kind": "population", "start_month": 4, "age_below": 30, "share": 0.55},
                {"name": "shock", "kind": "concept", "start_month": 5, "min_utilisation": 0.8, "flip_probability": 0.3},
            ],
            "model": {
                "name": "test-model",
                "validation_size": 0.2,
                "lightgbm": {"n_estimators": 40, "num_leaves": 7, "min_child_samples": 40},
                "logistic": {"C": 0.5},
            },
        }
    )
