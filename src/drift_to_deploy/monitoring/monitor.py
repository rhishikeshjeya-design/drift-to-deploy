"""Monitor one month: drift on its traffic now, performance on the month whose outcomes just arrived.

Each month's result is appended to a history file, which later decisions read: whether a
labelled month had drifted, and when the last retrain was.
"""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from drift_to_deploy.data.schema import FEATURES, ID
from drift_to_deploy.modelling import registry
from drift_to_deploy.monitoring import drift, performance, triggers
from drift_to_deploy.params import Params
from drift_to_deploy.serving.prediction_log import PredictionLog


class NoTraffic(RuntimeError):
    pass


@dataclass
class MonitorPaths:
    batches: Path = Path("data/batches/months")
    reports: Path = Path("reports/monitoring")

    def batch(self, month: int) -> Path:
        return self.batches / f"month_{month:02d}.parquet"

    @property
    def history(self) -> Path:
        return self.reports / "history.json"


def load_history(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    return {int(k): v for k, v in json.loads(path.read_text())["months"].items()}


def save_history(path: Path, history: dict[int, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"months": {str(k): history[k] for k in sorted(history)}}, indent=2) + "\n")


def traffic_for(log: PredictionLog, customers: pd.Series) -> pd.DataFrame:
    """The latest logged prediction for each of these customers."""
    logged = log.read()
    if logged.empty:
        return logged
    return logged[logged[ID].isin(customers)].drop_duplicates(ID, keep="last")


def _serving_version(traffic: pd.DataFrame) -> str:
    return Counter(traffic["model_version"]).most_common(1)[0][0]


def _reference_with_scores(model_name: str, version: str) -> pd.DataFrame:
    reference = registry.training_data(model_name, version)
    model = registry.load_version(model_name, version)
    return reference.assign(**{drift.SCORE: model.predict_proba(reference[FEATURES])[:, 1]})


def run_month(month: int, params: Params, log: PredictionLog, paths: MonitorPaths | None = None) -> dict:
    paths = paths or MonitorPaths()
    cfg, name = params.monitoring, params.model.name
    history = load_history(paths.history)

    batch = pd.read_parquet(paths.batch(month))
    traffic = traffic_for(log, batch[ID])
    if traffic.empty:
        raise NoTraffic(f"No predictions logged for month {month}. Send its traffic to the API first.")

    version = _serving_version(traffic)
    current = traffic[[*FEATURES]].assign(**{drift.SCORE: traffic["probability"].to_numpy()})
    drift_result = drift.measure(
        _reference_with_scores(name, version), current, cfg.column_drift_threshold, cfg.score_drift_alert
    )

    labelled_month = month - cfg.label_lag_months
    perf, baseline_auc = None, None
    if labelled_month >= 1 and paths.batch(labelled_month).exists():
        outcomes = pd.read_parquet(paths.batch(labelled_month))
        labelled_traffic = traffic_for(log, outcomes[ID])
        if not labelled_traffic.empty:
            perf = performance.measure(labelled_traffic, outcomes)
            baseline_auc = float(registry.tags(name, _serving_version(labelled_traffic))["validation_auc"])
    else:
        labelled_month = None

    labelled_record = history.get(labelled_month, {}) if labelled_month else {}
    last_retrain = max((m for m, r in history.items() if m < month and r["decision"]["retrain"]), default=None)
    decision = triggers.decide(
        month=month,
        drift=drift_result,
        labelled_month=labelled_month,
        perf=perf,
        labelled_month_drifted=labelled_record.get("decision", {}).get("status") in ("drift", "degraded"),
        baseline_auc=baseline_auc or 0.0,
        last_retrain_month=last_retrain,
        cfg=cfg,
    )

    record = {
        "month": month,
        "serving_version": version,
        "drift": drift_result.summary(),
        "labelled_month": labelled_month,
        "performance": perf.summary() if perf else None,
        "baseline_auc": baseline_auc,
        "decision": decision.summary(),
    }
    history[month] = record
    save_history(paths.history, history)
    drift.save_html(drift_result, paths.reports / f"drift_month_{month:02d}.html")
    return record
