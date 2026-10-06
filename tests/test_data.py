import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.errors
import pytest
import yaml

from drift_to_deploy.data import batches, download, schema
from drift_to_deploy.data.drift import apply_concept, apply_covariate, utilisation
from drift_to_deploy.params import Params

ROOT = Path(__file__).parent.parent


# --- params ------------------------------------------------------------------


def test_shipped_params_are_valid():
    params = Params.load(ROOT / "params.yaml")
    sizes = params.batches
    assert sizes.reference_size + sizes.months * sizes.month_size <= 30_000
    assert [s.kind for s in params.active(12)] == ["covariate", "population", "concept"]
    assert params.active(1) == []


def test_params_reject_unknown_scenarios():
    raw = yaml.safe_load((ROOT / "params.yaml").read_text())
    raw["scenarios"][0]["kind"] = "teleport"
    with pytest.raises(ValueError):
        Params.model_validate(raw)


# --- download ------------------------------------------------------------------


def test_checksum_mismatch_is_refused_before_parsing(tmp_path, params):
    out = tmp_path / "raw.parquet"
    with pytest.raises(download.ChecksumError):
        download.download(params.source, out, payload=b"not the real file")
    assert not out.exists()


def test_checksum_match_passes():
    payload = b"data"
    download.verify(payload, hashlib.sha256(payload).hexdigest())


# --- schema --------------------------------------------------------------------


def _raw(clean: pd.DataFrame) -> pd.DataFrame:
    """Turn clean data back into the source's column names and undocumented codes."""
    inverse = {v: k for k, v in schema._RENAME.items()}
    raw = clean.rename(columns=inverse)
    raw.loc[:9, "EDUCATION"] = [0, 5, 6, 0, 5, 6, 0, 5, 6, 0]
    raw.loc[:4, "MARRIAGE"] = 0
    return raw


def test_clean_folds_undocumented_codes(clean):
    cleaned = schema.clean(_raw(clean))
    assert set(cleaned["education"]) <= {1, 2, 3, 4}
    assert set(cleaned["marriage"]) <= {1, 2, 3}
    assert (cleaned.loc[:9, "education"] == 4).all()
    assert "pay_1" in cleaned.columns and "PAY_0" not in cleaned.columns
    schema.validate(cleaned)


def test_validation_reports_every_problem(clean):
    bad = clean.copy()
    bad.loc[0, "age"] = 150
    bad.loc[1, "pay_amt_1"] = -5
    bad.loc[2, schema.TARGET] = 2
    with pytest.raises(pandera.errors.SchemaErrors) as info:
        schema.validate(bad)
    failing = set(info.value.failure_cases["column"])
    assert {"age", "pay_amt_1", schema.TARGET} <= failing


def test_validation_rejects_unexpected_columns(clean):
    with pytest.raises(pandera.errors.SchemaErrors):
        schema.validate(clean.assign(surprise=1))


def test_protected_attributes_are_not_model_features():
    assert not set(schema.PROTECTED) & set(schema.FEATURES)


# --- drift scenarios -------------------------------------------------------------


def test_covariate_drift_moves_inputs_not_labels(clean, params):
    scenario = params.scenarios[0]
    drifted = apply_covariate(clean, scenario)
    assert drifted["limit_bal"].mean() == pytest.approx(clean["limit_bal"].mean() * 0.7, rel=0.01)
    assert drifted["bill_amt_1"].sum() == pytest.approx(clean["bill_amt_1"].sum() * 1.25, rel=0.01)
    pd.testing.assert_series_equal(drifted[schema.TARGET], clean[schema.TARGET])


def test_concept_drift_only_flips_exposed_non_defaulters(clean, params):
    scenario = params.scenarios[2]
    drifted = apply_concept(clean, scenario, np.random.default_rng(0))
    flipped = (clean[schema.TARGET] == 0) & (drifted[schema.TARGET] == 1)
    assert flipped.any()
    assert not ((clean[schema.TARGET] == 1) & (drifted[schema.TARGET] == 0)).any()  # never 1 -> 0
    assert (utilisation(clean[flipped]) >= scenario.min_utilisation).all()
    assert (clean.loc[flipped, "pay_1"] <= 0).all()
    pd.testing.assert_frame_equal(drifted.drop(columns=schema.TARGET), clean.drop(columns=schema.TARGET))


# --- batches -------------------------------------------------------------------


def test_batches_have_the_right_shape_and_never_reuse_a_customer(clean, params):
    reference, monthly, summary = batches.build(clean, params)
    assert len(reference) == 2000
    assert [len(b) for b in monthly] == [400] * 6
    ids = pd.concat([reference[schema.ID], *[b[schema.ID] for b in monthly]])
    assert ids.is_unique
    assert [s.month for s in summary] == [1, 2, 3, 4, 5, 6]
    assert summary[0].scenarios == [] and summary[-1].scenarios == ["squeeze", "young", "shock"]


def test_only_the_population_months_change_the_customer_mix(clean, params):
    reference, monthly, _ = batches.build(clean, params)
    young = lambda df: (df["age"] < 30).mean()  # noqa: E731
    natural = young(reference)
    for batch in monthly[:3]:  # before the population scenario
        assert young(batch) == pytest.approx(natural, abs=0.06)
    for batch in monthly[3:]:
        assert young(batch) == pytest.approx(0.55, abs=0.01)


def test_batches_are_deterministic(clean, params):
    _, first, _ = batches.build(clean, params)
    _, second, _ = batches.build(clean, params)
    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a, b)


def test_infeasible_segment_share_fails_clearly(clean, params):
    params.scenarios[1].share = 0.95
    with pytest.raises(batches.NotEnoughRows, match="Reduce `share`"):
        batches.build(clean, params)


def test_write_layout(tmp_path, clean, params):
    reference, monthly, summary = batches.build(clean, params)
    batches.write(tmp_path, reference, monthly, summary)
    assert (tmp_path / "reference.parquet").exists()
    assert sorted(p.name for p in (tmp_path / "months").iterdir())[0] == "month_01.parquet"
    assert (tmp_path / "summary.json").read_text().startswith("{")
