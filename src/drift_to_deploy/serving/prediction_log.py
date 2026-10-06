"""Every prediction, with the exact inputs the model saw. Monitoring reads from here.

Drift is measured on what the model was actually given in production, not on a copy of the
data that might have been prepared differently.
"""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from drift_to_deploy.data.schema import FEATURES

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT    NOT NULL,
    request_id    TEXT    NOT NULL,
    customer_id   INTEGER NOT NULL,
    model_version TEXT    NOT NULL,
    probability   REAL    NOT NULL,
    features      TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_ts ON predictions(ts);
"""


class PredictionLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=10)
        try:
            with con:
                yield con
        finally:
            con.close()

    def write(self, request_id: str, model_version: str, rows: list[dict], probabilities: list[float]) -> None:
        ts = datetime.now(UTC).isoformat()
        records = [
            (
                ts,
                request_id,
                row["customer_id"],
                model_version,
                p,
                json.dumps({k: row[k] for k in FEATURES}),
            )
            for row, p in zip(rows, probabilities, strict=True)
        ]
        with self._connect() as con:
            con.executemany(
                "INSERT INTO predictions (ts, request_id, customer_id, model_version, probability, features) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                records,
            )

    def read(self, since: str | None = None) -> pd.DataFrame:
        """Logged predictions with their features expanded into columns."""
        query = "SELECT ts, request_id, customer_id, model_version, probability, features FROM predictions"
        params: tuple = ()
        if since:
            query += " WHERE ts >= ?"
            params = (since,)
        with self._connect() as con:
            frame = pd.read_sql_query(query + " ORDER BY id", con, params=params)
        if frame.empty:
            return frame.drop(columns="features")
        features = pd.DataFrame([json.loads(f) for f in frame.pop("features")])
        return pd.concat([frame, features], axis=1)
