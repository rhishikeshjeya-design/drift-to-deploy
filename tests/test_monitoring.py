import pandas as pd
import pytest
from fastapi.testclient import TestClient

from drift_to_deploy.data import batches
from drift_to_deploy.data.schema import FEATURES, ID, TARGET
from drift_to_deploy.modelling.train import train
from drift_to_deploy.monitoring import drift, performance, traffic, triggers
from drift_to_deploy.monitoring.monitor import MonitorPaths, NoTraffic, load_history, run_month
from drift_to_deploy.serving.app import ServingSettings, create_app

# --- trigger rules (pure) ----------------------------------------------------------


def _drift(share=0.0, score=0.0) -> drift.DriftResult:
    cols = [f"c{i}" for i in range(int(share * 20))]
    return drift.DriftResult(share, cols, {}, score, rows=1000)


def _perf(auc=0.78, predicted=0.22, observed=0.22, rows=1000) -> performance.PerformanceResult:
    return performance.PerformanceResult(rows, auc, predicted, observed, {}, ["1"])


def _decide(params, *, month=6, drift_=None, perf=None, labelled_drifted=False, last_retrain=None):
    return triggers.decide(
        month=month,
        drift=drift_ or _drift(),
        labelled_month=month - 2 if perf else None,
        perf=perf,
        labelled_month_drifted=labelled_drifted,
        baseline_auc=0.78,
        last_retrain_month=last_retrain,
        cfg=params.monitoring,
    )


def test_quiet_month_is_ok(params):
    decision = _decide(params, perf=_perf())
    assert decision.status == "ok" and not decision.retrain and decision.reasons == []


def test_input_drift_alone_alerts_but_does_not_retrain(params):
    decision = _decide(params, drift_=_drift(share=0.35, score=0.2))
    assert decision.status == "drift" and not decision.retrain
    assert any("inputs drifted" in r for r in decision.reasons)
    assert any("score distribution" in r for r in decision.reasons)


def test_outcomes_for_a_drifted_month_trigger_retraining(params):
    decision = _decide(params, perf=_perf(), labelled_drifted=True)
    assert decision.retrain
    assert any("labelled data from the new world" in r for r in decision.reasons)


@pytest.mark.parametrize(
    ("perf", "phrase"),
    [
        (_perf(auc=0.72), "AUC fell"),
        (_perf(predicted=0.30, observed=0.22), "overpredicting"),
        (_perf(predicted=0.22, observed=0.30), "underpredicting"),
    ],
)
def test_degraded_performance_triggers_retraining(params, perf, phrase):
    decision = _decide(params, perf=perf)
    assert decision.status == "degraded" and decision.retrain
    assert any(phrase in r for r in decision.reasons)


def test_too_few_outcomes_are_not_acted_on(params):
    decision = _decide(params, perf=_perf(auc=0.60, rows=50), labelled_drifted=True)
    assert not decision.retrain and decision.status == "ok"


def test_cooldown_holds_back_a_retrain_but_keeps_the_reasons(params):
    decision = _decide(params, month=7, perf=_perf(auc=0.70), last_retrain=6)
    assert not decision.retrain and decision.blocked_by.startswith("cooldown")
    assert decision.reasons
    assert _decide(params, month=8, perf=_perf(auc=0.70), last_retrain=6).retrain


# --- drift and performance measurement ------------------------------------------------


def _scored(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[FEATURES].assign(probability=(frame["pay_1"].clip(0, 3) / 4 + 0.1).to_numpy())


def test_drift_is_low_for_the_same_population_and_high_after_a_shift(clean, params):
    reference, current = clean.iloc[:3000], clean.iloc[3000:]
    same = drift.measure(_scored(reference), _scored(current), 0.1, 0.1)
    assert same.drifted_share < 0.25 and same.score_distance < 0.1

    shifted = current.assign(limit_bal=current["limit_bal"] * 0.5, bill_amt_1=current["bill_amt_1"] * 2)
    moved = drift.measure(_scored(reference), _scored(shifted), 0.1, 0.1)
    assert {"limit_bal", "bill_amt_1"} <= set(moved.drifted_columns)
    assert set(moved.column_distance) == set(FEATURES)


def test_performance_joins_outcomes_and_keeps_the_latest_prediction(clean):
    outcomes = clean.head(400)
    preds = pd.DataFrame(
        {
            ID: [*outcomes[ID], outcomes[ID].iloc[0]],
            "probability": [*(outcomes[TARGET] * 0.6 + 0.2), 0.99],  # a later re-score for the first customer
            "model_version": "1",
        }
    )
    result = performance.measure(preds, outcomes)
    assert result.rows == 400
    joined = performance.join_outcomes(preds, outcomes).set_index(ID)
    assert joined.loc[outcomes[ID].iloc[0], "probability"] == 0.99
    assert result.auc > 0.9 and "age_band:30_to_44" in result.segments


# --- the monitor, end to end through the real API ---------------------------------------


@pytest.fixture
def world(tmp_path, clean, params, tracking):
    reference, monthly, summary = batches.build(clean, params)
    batches.write(tmp_path / "batches", reference, monthly, summary)
    train(reference, params)
    settings = ServingSettings(model_name=params.model.name, prediction_log=tmp_path / "log.db", reload_seconds=3600)
    paths = MonitorPaths(batches=tmp_path / "batches" / "months", reports=tmp_path / "reports")
    return monthly, settings, paths


def test_monitor_month_by_month(world, params):
    monthly, settings, paths = world
    app = create_app(settings)
    with TestClient(app) as client:
        with pytest.raises(NoTraffic):
            run_month(1, params, app.state.log, paths)

        records = []
        for month in (1, 2, 3):
            traffic.send(monthly[month - 1], client)
            records.append(run_month(month, params, app.state.log, paths))

    first, second, third = records
    assert first["decision"]["status"] == "ok" and first["performance"] is None
    # The squeeze starts in month 2 and is visible in the inputs straight away.
    assert second["decision"]["status"] in ("drift", "degraded")
    assert "limit_bal" in second["drift"]["drifted_columns"]
    # Month 2's outcomes arrive in month 3 (lag 1): labelled data from a drifted month.
    assert third["labelled_month"] == 2 and third["performance"]["rows"] == 400
    assert third["decision"]["retrain"]

    history = load_history(paths.history)
    assert sorted(history) == [1, 2, 3]
    assert (paths.reports / "drift_month_02.html").exists()
