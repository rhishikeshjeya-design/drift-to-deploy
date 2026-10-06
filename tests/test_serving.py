import pytest
from fastapi.testclient import TestClient

from drift_to_deploy.data.schema import FEATURES, ID
from drift_to_deploy.modelling import registry
from drift_to_deploy.modelling.train import train
from drift_to_deploy.serving.app import ServingSettings, create_app
from drift_to_deploy.serving.schemas import EXAMPLE


@pytest.fixture
def settings(tmp_path, params):
    return ServingSettings(
        model_name=params.model.name, prediction_log=tmp_path / "predictions.db", reload_seconds=3600
    )


@pytest.fixture
def trained(clean, params, tracking):
    return train(clean, params)


def _payload(clean, n=5):
    return {"customers": clean[[ID, *FEATURES]].head(n).to_dict(orient="records")}


def test_not_ready_until_a_model_exists(settings, tracking):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"customers": [EXAMPLE]}).status_code == 503


def test_predict_scores_logs_and_counts(settings, trained, clean):
    app = create_app(settings)
    with TestClient(app) as client:
        response = client.post("/predict", json=_payload(clean))
        assert response.status_code == 200
        body = response.json()
        assert body["model_version"] == trained.version
        assert [p["customer_id"] for p in body["predictions"]] == clean[ID].head(5).tolist()
        assert all(0 <= p["default_probability"] <= 1 for p in body["predictions"])

        logged = app.state.log.read()
        assert len(logged) == 5 and set(FEATURES) <= set(logged.columns)
        assert (logged["model_version"] == trained.version).all()

        metrics = client.get("/metrics").text
        assert f'd2d_predictions_total{{model_version="{trained.version}"}} 5.0' in metrics
        assert f'd2d_model_version{{version="{trained.version}"}} 1.0' in metrics


@pytest.mark.parametrize(
    ("change", "field"),
    [
        ({"sex": 2}, "sex"),  # protected attributes are not accepted at all
        ({"pay_amt_1": -5}, "pay_amt_1"),
        ({"education": 7}, "education"),
        ({"limit_bal": 0}, "limit_bal"),
    ],
)
def test_invalid_customers_are_rejected_with_the_field_named(settings, trained, change, field):
    with TestClient(create_app(settings)) as client:
        response = client.post("/predict", json={"customers": [{**EXAMPLE, **change}]})
        assert response.status_code == 422
        assert field in {err["loc"][-1] for err in response.json()["detail"]}


def test_batch_size_limits(settings, trained):
    with TestClient(create_app(settings)) as client:
        assert client.post("/predict", json={"customers": []}).status_code == 422
        too_many = {"customers": [{**EXAMPLE, "customer_id": i} for i in range(1001)]}
        assert client.post("/predict", json=too_many).status_code == 422


def test_hot_swap_when_the_champion_moves(settings, trained, clean, params):
    app = create_app(settings)
    with TestClient(app) as client:
        assert client.get("/model").json()["version"] == "1"
        second = train(clean.sample(frac=0.7, random_state=3), params, reason="retrain")
        assert client.post("/admin/reload").json()["outcome"] == "unchanged"  # only challenger moved

        registry.set_alias(params.model.name, registry.CHAMPION, second.version)
        assert client.post("/admin/reload").json() == {"outcome": "swapped", "model_version": "2"}
        assert client.get("/model").json()["tags"]["reason"] == "retrain"
        assert client.post("/predict", json=_payload(clean)).json()["model_version"] == "2"

        metrics = client.get("/metrics").text
        assert 'd2d_model_version{version="2"} 1.0' in metrics
        assert 'd2d_model_version{version="1"}' not in metrics


def test_a_broken_release_keeps_the_old_model_serving(settings, trained, clean, params, monkeypatch):
    app = create_app(settings)
    with TestClient(app) as client:
        second = train(clean, params, reason="retrain")
        registry.set_alias(params.model.name, registry.CHAMPION, second.version)

        def broken(version):
            raise RuntimeError("corrupt artifact")

        monkeypatch.setattr(app.state.store, "_load", broken)
        assert client.post("/admin/reload").json() == {"outcome": "failed", "model_version": "1"}
        assert client.post("/predict", json=_payload(clean)).status_code == 200
        assert 'd2d_model_reloads_total{outcome="failed"} 1.0' in client.get("/metrics").text
