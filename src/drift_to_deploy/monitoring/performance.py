"""Performance once outcomes arrive: logged predictions joined with what actually happened.

Outcomes and demographics come from the lender's records, joined by customer ID. The scoring
API never had them; this is the only place protected attributes meet predictions, and only
to check that no group is served worse.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from drift_to_deploy.data.schema import ID, TARGET
from drift_to_deploy.modelling import metrics


@dataclass
class PerformanceResult:
    rows: int
    auc: float
    predicted_rate: float
    observed_rate: float
    segments: dict[str, dict[str, float]]
    model_versions: list[str]

    @property
    def calibration_gap(self) -> float:
        return abs(self.predicted_rate - self.observed_rate)

    def summary(self) -> dict:
        return {
            "rows": self.rows,
            "auc": round(self.auc, 4),
            "predicted_rate": round(self.predicted_rate, 4),
            "observed_rate": round(self.observed_rate, 4),
            "calibration_gap": round(self.calibration_gap, 4),
            "model_versions": self.model_versions,
            "segments": {k: {"auc": round(v["auc"], 4), "n": v["n"]} for k, v in self.segments.items()},
        }


def join_outcomes(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Latest prediction per customer, with their outcome and the attributes used for auditing."""
    latest = predictions.drop_duplicates(ID, keep="last")[[ID, "probability", "model_version"]]
    records = outcomes[[ID, TARGET, "sex", "age", "education"]]
    return latest.merge(records, on=ID, how="inner", validate="one_to_one")


def measure(predictions: pd.DataFrame, outcomes: pd.DataFrame) -> PerformanceResult | None:
    """None if fewer than two outcome classes are present (AUC is undefined)."""
    joined = join_outcomes(predictions, outcomes)
    y, p = joined[TARGET].to_numpy(), joined["probability"].to_numpy()
    if len(np.unique(y)) < 2:
        return None
    return PerformanceResult(
        rows=len(joined),
        auc=metrics.overall(y, p)["auc"],
        predicted_rate=float(p.mean()),
        observed_rate=float(y.mean()),
        segments=metrics.by_segment(joined, y, p),
        model_versions=sorted(joined["model_version"].unique().tolist()),
    )
