"""The scoring API.

uv run d2d serve            # http://127.0.0.1:8000/docs
"""

import asyncio
import contextlib
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from pydantic_settings import BaseSettings, SettingsConfigDict

from drift_to_deploy.modelling import registry
from drift_to_deploy.serving.model_store import ModelStore
from drift_to_deploy.serving.prediction_log import PredictionLog
from drift_to_deploy.serving.schemas import ModelInfo, Prediction, PredictRequest, PredictResponse


class ServingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="D2D_")

    model_name: str = "credit-default"
    alias: str = registry.CHAMPION
    prediction_log: Path = Path(".d2d/predictions.db")
    # How often to check whether the alias has moved to a new version.
    reload_seconds: float = 30.0


class Metrics:
    """Prometheus metrics, on a registry per app so tests don't share global state."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter("d2d_requests_total", "HTTP requests.", ["path", "status"], registry=self.registry)
        self.predictions = Counter(
            "d2d_predictions_total", "Customers scored.", ["model_version"], registry=self.registry
        )
        self.latency = Histogram(
            "d2d_predict_latency_seconds",
            "Time to score one request.",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
            registry=self.registry,
        )
        self.scores = Histogram(
            "d2d_default_probability",
            "Distribution of predicted default probabilities.",
            buckets=tuple(i / 10 for i in range(1, 11)),
            registry=self.registry,
        )
        self.model_version = Gauge(
            "d2d_model_version", "Version currently serving (1 for the live one).", ["version"], registry=self.registry
        )
        self.reloads = Counter(
            "d2d_model_reloads_total", "Model refresh outcomes.", ["outcome"], registry=self.registry
        )


def create_app(settings: ServingSettings | None = None) -> FastAPI:
    settings = settings or ServingSettings()
    store = ModelStore(settings.model_name, settings.alias)
    log = PredictionLog(settings.prediction_log)
    metrics = Metrics()

    def refresh() -> str:
        before = store.current.version if store.current else None
        outcome = store.refresh()
        metrics.reloads.labels(outcome).inc()
        if outcome == "swapped":
            if before is not None:
                metrics.model_version.remove(before)
            metrics.model_version.labels(store.current.version).set(1)
        return outcome

    async def poll() -> None:
        while True:
            await asyncio.sleep(settings.reload_seconds)
            await asyncio.to_thread(refresh)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        registry.connect()
        await asyncio.to_thread(refresh)
        task = asyncio.create_task(poll())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(
        title="drift-to-deploy scoring API",
        description="Scores credit-default risk with the current champion model.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.store, app.state.log, app.state.metrics, app.state.refresh = store, log, metrics, refresh

    @app.middleware("http")
    async def count_requests(request: Request, call_next):
        response = await call_next(request)
        route = request.scope.get("route")
        metrics.requests.labels(getattr(route, "path", "unmatched"), str(response.status_code)).inc()
        return response

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready() -> dict[str, str]:
        """Readiness: a model is loaded and can score."""
        if store.current is None:
            raise HTTPException(503, "No model loaded yet.")
        return {"status": "ready", "model_version": store.current.version}

    @app.get("/model", response_model=ModelInfo, tags=["ops"])
    def model_info() -> ModelInfo:
        current = store.current
        if current is None:
            raise HTTPException(503, "No model loaded yet.")
        return ModelInfo(
            name=settings.model_name,
            alias=settings.alias,
            version=current.version,
            loaded_at=current.loaded_at,
            tags=current.tags,
        )

    @app.post("/predict", response_model=PredictResponse, tags=["scoring"])
    def predict(body: PredictRequest) -> PredictResponse:
        # Take one reference: a swap during this request can't mix two model versions.
        current = store.current
        if current is None:
            raise HTTPException(503, "No model loaded yet.")
        rows = [c.model_dump() for c in body.customers]
        start = time.perf_counter()
        probabilities = current.predict(pd.DataFrame(rows))
        metrics.latency.observe(time.perf_counter() - start)
        log.write(uuid.uuid4().hex, current.version, rows, probabilities)
        metrics.predictions.labels(current.version).inc(len(rows))
        for p in probabilities:
            metrics.scores.observe(p)
        return PredictResponse(
            model_version=current.version,
            predictions=[
                Prediction(customer_id=r["customer_id"], default_probability=round(p, 6))
                for r, p in zip(rows, probabilities, strict=True)
            ],
        )

    @app.post("/admin/reload", tags=["ops"])
    async def reload() -> dict[str, str]:
        """Check the registry now instead of waiting for the next poll."""
        outcome = await asyncio.to_thread(refresh)
        return {"outcome": outcome, "model_version": store.current.version if store.current else ""}

    @app.get("/metrics", include_in_schema=False)
    def prometheus() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app
