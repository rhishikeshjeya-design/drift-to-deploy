"""When to retrain: pure rules over monitoring results, so every decision is testable and explained.

Drift alone is an alert, not a retrain: retraining needs outcomes from the changed world, and
those arrive late. A retrain is triggered when outcomes show the model doing worse, or when
outcomes have arrived for a month that had already drifted (so there is now labelled data
from the new world to learn from).
"""

from dataclasses import dataclass, field
from typing import Literal

from drift_to_deploy.monitoring.drift import DriftResult
from drift_to_deploy.monitoring.performance import PerformanceResult
from drift_to_deploy.params import MonitoringParams

Status = Literal["ok", "drift", "degraded"]


@dataclass
class Decision:
    status: Status
    retrain: bool
    reasons: list[str] = field(default_factory=list)
    blocked_by: str | None = None

    def summary(self) -> dict:
        return {"status": self.status, "retrain": self.retrain, "reasons": self.reasons, "blocked_by": self.blocked_by}


def drift_alerts(drift: DriftResult, cfg: MonitoringParams) -> list[str]:
    alerts = []
    if drift.drifted_share >= cfg.drifted_share_alert:
        cols = ", ".join(drift.drifted_columns[:4]) + ("…" if len(drift.drifted_columns) > 4 else "")
        alerts.append(f"{drift.drifted_share:.0%} of inputs drifted ({cols})")
    if drift.score_distance >= cfg.score_drift_alert:
        alerts.append(f"score distribution drifted (distance {drift.score_distance:.2f})")
    return alerts


def performance_alerts(perf: PerformanceResult, baseline_auc: float, cfg: MonitoringParams) -> list[str]:
    if perf.rows < cfg.min_labelled_rows:
        return []
    alerts = []
    drop = baseline_auc - perf.auc
    if drop >= cfg.auc_drop_alert:
        alerts.append(f"AUC fell {drop:.3f} below validation ({perf.auc:.3f} vs {baseline_auc:.3f})")
    if perf.calibration_gap >= cfg.calibration_gap_alert:
        direction = "over" if perf.predicted_rate > perf.observed_rate else "under"
        alerts.append(
            f"{direction}predicting defaults ({perf.predicted_rate:.1%} predicted vs {perf.observed_rate:.1%} actual)"
        )
    return alerts


def decide(
    month: int,
    drift: DriftResult,
    labelled_month: int | None,
    perf: PerformanceResult | None,
    labelled_month_drifted: bool,
    baseline_auc: float,
    last_retrain_month: int | None,
    cfg: MonitoringParams,
) -> Decision:
    current_drift = drift_alerts(drift, cfg)
    degraded = performance_alerts(perf, baseline_auc, cfg) if perf is not None else []

    reasons = [f"month {month}: {a}" for a in current_drift]
    reasons += [f"month {labelled_month} outcomes: {a}" for a in degraded]
    status: Status = "degraded" if degraded else "drift" if current_drift else "ok"

    want_retrain = bool(degraded)
    if (
        labelled_month is not None
        and labelled_month_drifted
        and perf is not None
        and perf.rows >= cfg.min_labelled_rows
    ):
        want_retrain = True
        reasons.append(f"month {labelled_month} outcomes arrived for a drifted month: labelled data from the new world")

    if want_retrain and last_retrain_month is not None and month - last_retrain_month < cfg.cooldown_months:
        return Decision(status, False, reasons, f"cooldown: last retrain was month {last_retrain_month}")
    return Decision(status, want_retrain, reasons)
