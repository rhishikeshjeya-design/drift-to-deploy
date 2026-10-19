"""Input and output drift, measured with Evidently against the reference set.

The statistical methods are pinned rather than left to Evidently's defaults, which switch
between p-value tests and distance metrics depending on sample size. Pinned, a threshold
means the same thing every month.
"""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import DataDriftPreset

from drift_to_deploy.data.schema import FEATURES

CATEGORICAL = ["education", "marriage"]
NUMERICAL = [c for c in FEATURES if c not in CATEGORICAL]
SCORE = "probability"


@dataclass
class DriftResult:
    drifted_share: float
    drifted_columns: list[str]
    column_distance: dict[str, float]
    score_distance: float
    rows: int
    snapshot: object = field(default=None, repr=False)

    def summary(self) -> dict:
        return {
            "rows": self.rows,
            "drifted_share": round(self.drifted_share, 4),
            "drifted_columns": self.drifted_columns,
            "score_distance": round(self.score_distance, 4),
            "column_distance": {k: round(v, 4) for k, v in sorted(self.column_distance.items())},
        }


def _dataset(frame: pd.DataFrame) -> Dataset:
    definition = DataDefinition(numerical_columns=[*NUMERICAL, SCORE], categorical_columns=CATEGORICAL)
    return Dataset.from_pandas(frame[[*FEATURES, SCORE]], data_definition=definition)


def measure(
    reference: pd.DataFrame, current: pd.DataFrame, column_threshold: float, score_threshold: float
) -> DriftResult:
    """Both frames need the FEATURES columns plus `probability` (the serving model's scores)."""
    report = Report(
        [
            DataDriftPreset(
                columns=FEATURES,
                num_method="wasserstein",
                cat_method="jensenshannon",
                threshold=column_threshold,
            ),
            ValueDrift(column=SCORE, method="wasserstein", threshold=score_threshold),
        ]
    )
    snapshot = report.run(_dataset(current), _dataset(reference))

    distances, score = {}, None
    for metric in snapshot.dict()["metrics"]:
        column = metric["config"].get("column")
        if column == SCORE:
            score = float(metric["value"])
        elif column in FEATURES:
            distances[column] = float(metric["value"])
    if score is None or set(distances) != set(FEATURES):
        raise RuntimeError("Evidently did not report a drift value for every column.")

    drifted = sorted(c for c, d in distances.items() if d >= column_threshold)
    return DriftResult(
        drifted_share=len(drifted) / len(FEATURES),
        drifted_columns=drifted,
        column_distance=distances,
        score_distance=score,
        rows=len(current),
        snapshot=snapshot,
    )


def save_html(result: DriftResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    result.snapshot.save_html(str(path))
