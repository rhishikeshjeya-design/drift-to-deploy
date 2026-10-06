import json

import numpy as np
import pandas as pd
import pytest

from drift_to_deploy.data.schema import FEATURES, PROTECTED, TARGET
from drift_to_deploy.modelling import features, metrics, registry
from drift_to_deploy.modelling.train import data_fingerprint, train, write_report

# --- features ------------------------------------------------------------------


def test_engineered_features_are_finite_and_bounded(clean):
    extreme = clean.copy()
    extreme.loc[:5, "limit_bal"] = 10.0
    extreme.loc[:5, "bill_amt_1"] = 1e7
    extreme.loc[6:10, features.BILL_AMOUNTS] = 0.0
    X = features.add_features(extreme)
    assert np.isfinite(X[features.ENGINEERED].to_numpy()).all()
    assert X["utilisation_1"].between(-1, 5).all()
    assert (X.loc[6:10, "payment_ratio"] == 1.0).all()  # nothing billed: treated as fully paid


def test_add_features_ignores_extra_columns(clean):
    X = features.add_features(clean)  # includes id, sex, age, target
    assert not set(PROTECTED) & set(X.columns)
    assert TARGET not in X.columns


@pytest.mark.parametrize("kind", features.CANDIDATES)
def test_pipelines_learn_signal_and_never_see_protected_attributes(clean, kind):
    pipeline = features.build(kind, {"n_estimators": 40} if kind == "lightgbm" else {}, seed=0)
    pipeline.fit(clean[FEATURES], clean[TARGET])
    proba = pipeline.predict_proba(clean[FEATURES])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert metrics.overall(clean[TARGET], proba)["auc"] > 0.6
    seen = set(pipeline[:-1].get_feature_names_out())
    assert not seen & set(PROTECTED)


def test_unknown_model_kind():
    with pytest.raises(ValueError):
        features.build("random_forest", {}, seed=0)


# --- metrics -------------------------------------------------------------------


def test_ks_and_calibration_on_known_cases():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert metrics.ks_statistic(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == pytest.approx(1.0)
    assert metrics.ks_statistic(y, np.array([0.5] * 6)) == 0.0  # all tied: no separation
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20_000)
    assert metrics.calibration_error((rng.random(20_000) < p).astype(int), p) < 0.02  # calibrated
    assert metrics.calibration_error(np.zeros(1000, int), np.full(1000, 0.4)) == pytest.approx(0.4)


def test_segments_skip_small_or_single_class_groups():
    df = pd.DataFrame({"age": [25] * 150 + [50] * 20, "sex": [1] * 170, "education": [2] * 170})
    y = np.array([0, 1] * 75 + [0] * 20)
    out = metrics.by_segment(df, y, np.linspace(0, 1, 170))
    assert "age_band:under_30" in out
    assert "age_band:45_plus" not in out  # only 20 customers
    assert "sex:male" in out and "sex:female" not in out


# --- training and the registry ---------------------------------------------------


def test_fingerprint_changes_with_the_data(clean):
    assert data_fingerprint(clean) == data_fingerprint(clean.copy())
    changed = clean.copy()
    changed.loc[0, "limit_bal"] += 1
    assert data_fingerprint(changed) != data_fingerprint(clean)


def test_first_model_becomes_champion_later_ones_only_challenger(clean, params, tracking, tmp_path):
    first = train(clean, params)
    assert first.version == "1" and first.bootstrapped_champion
    assert registry.version_for("test-model", registry.CHAMPION) == "1"
    assert {c.kind for c in first.candidates} == set(features.CANDIDATES)
    assert first.winner.metrics["auc"] == max(c.metrics["auc"] for c in first.candidates)

    second = train(clean.sample(frac=0.8, random_state=1), params, reason="retrain")
    assert second.version == "2" and not second.bootstrapped_champion
    assert registry.version_for("test-model", registry.CHAMPION) == "1"  # training never replaces it
    assert registry.version_for("test-model", registry.CHALLENGER) == "2"

    write_report(second, tmp_path / "report.json")
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["registered_version"] == "2" and report["lineage"]["reason"] == "retrain"


def test_registered_model_loads_by_alias_and_carries_lineage(clean, params, tracking):
    result = train(clean, params)
    model = registry.load("test-model", registry.CHAMPION)
    proba = model.predict_proba(clean[FEATURES].head(10))[:, 1]
    assert proba.shape == (10,)
    import mlflow

    version = mlflow.MlflowClient().get_model_version("test-model", result.version)
    assert version.tags["data_fingerprint"] == result.lineage["data_fingerprint"]
    assert version.tags["candidate"] == result.winner.kind


def test_missing_alias_is_none(tracking):
    assert registry.version_for("no-such-model", registry.CHAMPION) is None
