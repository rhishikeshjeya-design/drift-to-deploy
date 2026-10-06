"""Train candidate models, pick the best, and register it as the challenger.

Every run records what it was trained on (data hash, params, code version), so any model in
the registry can be traced back to exactly how it was made.
"""

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from drift_to_deploy.data.schema import FEATURES, TARGET
from drift_to_deploy.modelling import features, metrics, registry
from drift_to_deploy.params import Params


@dataclass
class CandidateResult:
    kind: str
    run_id: str
    metrics: dict[str, float]
    segments: dict[str, dict[str, float]]


@dataclass
class TrainResult:
    winner: CandidateResult
    candidates: list[CandidateResult]
    version: str
    bootstrapped_champion: bool
    lineage: dict[str, str] = field(default_factory=dict)


def data_fingerprint(df: pd.DataFrame) -> str:
    """A short, order-sensitive hash of the training data."""
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return digest.hexdigest()[:16]


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout
        return out.stdout.strip() + ("-dirty" if dirty.strip() else "") if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _fit_and_score(kind: str, settings: dict, X_train, y_train, X_val, y_val, val_frame, seed) -> tuple:
    pipeline = features.build(kind, settings, seed).fit(X_train, y_train)
    proba = pipeline.predict_proba(X_val)[:, 1]
    return pipeline, metrics.overall(y_val, proba), metrics.by_segment(val_frame, y_val, proba)


def train(data: pd.DataFrame, params: Params, reason: str = "initial training") -> TrainResult:
    """Fit every candidate on `data`, register the best as challenger.

    If no champion exists yet (the very first model), the winner also becomes champion.
    Replacing an existing champion is the promotion step's decision, not training's.
    """
    cfg = params.model
    train_df, val_df = train_test_split(
        data, test_size=cfg.validation_size, stratify=data[TARGET], random_state=params.seed
    )
    X_train, y_train = train_df[FEATURES], train_df[TARGET].to_numpy()
    X_val, y_val = val_df[FEATURES], val_df[TARGET].to_numpy()
    lineage = {
        "data_fingerprint": data_fingerprint(data),
        "data_rows": str(len(data)),
        "git_commit": _git_commit(),
        "reason": reason,
    }

    results: list[CandidateResult] = []
    pipelines = {}
    with mlflow.start_run(run_name=f"train: {reason}"):
        mlflow.set_tags(lineage)
        mlflow.log_params({"validation_size": cfg.validation_size, "seed": params.seed})
        for kind in features.CANDIDATES:
            settings = getattr(cfg, kind)
            with mlflow.start_run(run_name=kind, nested=True) as child:
                pipeline, overall, segs = _fit_and_score(
                    kind, settings, X_train, y_train, X_val, y_val, val_df, params.seed
                )
                mlflow.set_tags({**lineage, "candidate": kind})
                mlflow.log_params({f"{kind}.{k}": v for k, v in settings.items()})
                mlflow.log_metrics(overall)
                mlflow.log_metrics({f"segment.{name}.auc": s["auc"] for name, s in segs.items()})
                mlflow.log_dict(segs, "segments.json")
                pipelines[kind] = pipeline
                results.append(CandidateResult(kind, child.info.run_id, overall, segs))

        winner = max(results, key=lambda r: r.metrics["auc"])
        mlflow.set_tag("winner", winner.kind)
        mlflow.log_metrics({f"winner.{k}": v for k, v in winner.metrics.items()})

        example = X_val.head(5)
        info = mlflow.sklearn.log_model(
            pipelines[winner.kind],
            name="model",
            registered_model_name=cfg.name,
            signature=infer_signature(example, pipelines[winner.kind].predict_proba(example)[:, 1]),
            input_example=example,
            # Serving and `mlflow models serve` should return probabilities, not 0/1 labels.
            pyfunc_predict_fn="predict_proba",
            skops_trusted_types=features.TRUSTED_TYPES,
        )
        version = str(info.registered_model_version)
        mlflow.set_tag("registered_version", version)

    client = mlflow.MlflowClient()
    for key, value in {**lineage, "candidate": winner.kind, "validation_auc": f"{winner.metrics['auc']:.4f}"}.items():
        client.set_model_version_tag(cfg.name, version, key, value)

    registry.set_alias(cfg.name, registry.CHALLENGER, version)
    bootstrapped = registry.version_for(cfg.name, registry.CHAMPION) is None
    if bootstrapped:
        registry.set_alias(cfg.name, registry.CHAMPION, version)
    return TrainResult(winner, results, version, bootstrapped, lineage)


def write_report(result: TrainResult, path: Path) -> None:
    """A small JSON summary, kept in git so results are visible without opening MLflow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "registered_version": result.version,
        "became_champion": result.bootstrapped_champion,
        "winner": result.winner.kind,
        "lineage": result.lineage,
        "candidates": {c.kind: {k: round(v, 4) for k, v in c.metrics.items()} for c in result.candidates},
        "winner_segments": {k: {"auc": round(v["auc"], 4), "n": v["n"]} for k, v in result.winner.segments.items()},
    }
    path.write_text(json.dumps(report, indent=2) + "\n")
