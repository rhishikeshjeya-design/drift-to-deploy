"""Holds the model being served and swaps it, without downtime, when the champion alias moves.

A swap only happens after the new version has been loaded and has scored a known-good input.
If anything fails, the current model keeps serving: a broken release must not take scoring down.
"""

import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

import mlflow
import pandas as pd
from loguru import logger

from drift_to_deploy.data.schema import FEATURES
from drift_to_deploy.modelling import registry
from drift_to_deploy.serving.schemas import EXAMPLE


@dataclass(frozen=True)
class LoadedModel:
    pipeline: object
    version: str
    loaded_at: datetime
    tags: dict[str, str] = field(default_factory=dict)

    def predict(self, frame: pd.DataFrame) -> list[float]:
        return self.pipeline.predict_proba(frame[FEATURES])[:, 1].tolist()


class ModelStore:
    def __init__(self, model_name: str, alias: str = registry.CHAMPION) -> None:
        self.model_name = model_name
        self.alias = alias
        self._current: LoadedModel | None = None
        self._lock = threading.Lock()  # one refresh at a time; reads never wait

    @property
    def current(self) -> LoadedModel | None:
        return self._current

    def _load(self, version: str) -> LoadedModel:
        # Load by version, not alias, so the alias moving mid-load can't mix two versions.
        pipeline = mlflow.sklearn.load_model(f"models:/{self.model_name}/{version}")
        smoke = pipeline.predict_proba(pd.DataFrame([EXAMPLE])[FEATURES])[:, 1]
        if not (0.0 <= float(smoke[0]) <= 1.0):
            raise ValueError(f"version {version} returned an invalid probability on a known-good input")
        tags = dict(mlflow.MlflowClient().get_model_version(self.model_name, version).tags)
        return LoadedModel(pipeline, version, datetime.now(UTC), tags)

    def refresh(self) -> str:
        """Load the alias's version if it changed. Returns 'unchanged', 'swapped', 'missing' or 'failed'."""
        with self._lock:
            version = registry.version_for(self.model_name, self.alias)
            if version is None:
                return "missing"
            if self._current is not None and self._current.version == version:
                return "unchanged"
            try:
                loaded = self._load(version)
            except Exception as e:
                serving = self._current.version if self._current else "nothing"
                logger.error(f"Could not load {self.model_name} v{version}; still serving {serving}: {e}")
                return "failed"
            previous = self._current.version if self._current else None
            self._current = loaded  # a single reference swap: requests in flight keep their model
            logger.info(f"Now serving {self.model_name} v{version} (was {previous})")
            return "swapped"
