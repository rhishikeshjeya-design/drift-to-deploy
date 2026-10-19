"""Send a month's customers to the scoring API, as production traffic would arrive."""

from typing import Protocol

import pandas as pd

from drift_to_deploy.data.schema import FEATURES, ID

CHUNK = 500


class Poster(Protocol):
    def post(self, url: str, json: dict) -> object: ...


def send(batch: pd.DataFrame, client: Poster, chunk: int = CHUNK) -> int:
    """Post every customer in `batch` (inputs only: no outcomes, no protected attributes)."""
    sent = 0
    for start in range(0, len(batch), chunk):
        rows = batch.iloc[start : start + chunk][[ID, *FEATURES]].to_dict(orient="records")
        response = client.post("/predict", json={"customers": rows})
        response.raise_for_status()
        sent += len(rows)
    return sent
