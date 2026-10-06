"""MLflow tracking and model-registry helpers.

Which model serves traffic is decided by registry aliases, never by version numbers in code:
`champion` is the live model, `challenger` the latest candidate. Promotion is an alias move,
which is atomic and keeps a complete history of versions to roll back to.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

CHAMPION = "champion"
CHALLENGER = "challenger"
EXPERIMENT = "credit-default"
DEFAULT_DIR = Path(".mlflow")


@dataclass(frozen=True)
class Tracking:
    uri: str
    artifacts: str | None = None  # None: the tracking server decides


def configure(tracking: Tracking | None = None) -> Tracking:
    """Point MLflow at a tracking store. Defaults to a local SQLite file, or MLFLOW_TRACKING_URI."""
    if tracking is None:
        uri = os.environ.get("MLFLOW_TRACKING_URI")
        if uri:
            tracking = Tracking(uri)
        else:
            DEFAULT_DIR.mkdir(exist_ok=True)
            tracking = Tracking(
                f"sqlite:///{DEFAULT_DIR / 'mlflow.db'}", (DEFAULT_DIR / "artifacts").resolve().as_uri()
            )
    mlflow.set_tracking_uri(tracking.uri)
    mlflow.set_registry_uri(tracking.uri)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=tracking.artifacts)
    mlflow.set_experiment(EXPERIMENT)
    return tracking


def version_for(model_name: str, alias: str) -> str | None:
    """The model version an alias points at, or None if it isn't set."""
    try:
        # str(): MLflow returns alias versions as ints but registered versions as strings.
        return str(MlflowClient().get_model_version_by_alias(model_name, alias).version)
    except MlflowException:
        return None


def set_alias(model_name: str, alias: str, version: str) -> None:
    MlflowClient().set_registered_model_alias(model_name, alias, version)


def model_uri(model_name: str, alias: str) -> str:
    return f"models:/{model_name}@{alias}"


def load(model_name: str, alias: str = CHAMPION):
    """The scikit-learn pipeline behind an alias."""
    return mlflow.sklearn.load_model(model_uri(model_name, alias))
