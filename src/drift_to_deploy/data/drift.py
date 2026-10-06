"""Reproducible drift scenarios applied to monthly batches.

Each kind changes something different, which matters for how it can be detected:

- covariate: inputs move but the input-to-outcome relationship doesn't. Visible to input
  drift detection immediately; a good model may not need replacing at all.
- population: the customer mix changes. Real rows with real labels, so the outcomes stay
  genuine; only the proportions shift.
- concept: the relationship itself changes. Inputs look unremarkable, so this only shows
  up once outcomes (labels) arrive and performance drops.
"""

import numpy as np
import pandas as pd

from drift_to_deploy.data.schema import BILL_AMOUNTS, PAY_STATUS, TARGET
from drift_to_deploy.params import ConceptScenario, CovariateScenario


def apply_covariate(df: pd.DataFrame, scenario: CovariateScenario) -> pd.DataFrame:
    out = df.copy()
    out["limit_bal"] = (out["limit_bal"] * scenario.limit_bal_factor).round(-1).clip(lower=10)
    out[BILL_AMOUNTS] = (out[BILL_AMOUNTS] * scenario.bill_amt_factor).round()
    return out


def utilisation(df: pd.DataFrame) -> pd.Series:
    return df["bill_amt_1"] / df["limit_bal"]


def apply_concept(df: pd.DataFrame, scenario: ConceptScenario, rng: np.random.Generator) -> pd.DataFrame:
    """Customers near their limit who pay on time start defaulting with some probability."""
    out = df.copy()
    pays_on_time = (out[PAY_STATUS[0]] <= 0) & (out[TARGET] == 0)
    exposed = pays_on_time & (utilisation(out) >= scenario.min_utilisation)
    flips = exposed & (rng.random(len(out)) < scenario.flip_probability)
    out.loc[flips, TARGET] = 1
    return out
