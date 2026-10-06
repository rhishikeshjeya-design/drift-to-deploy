# drift-to-deploy

An automated credit-risk ML lifecycle: a default-prediction model that watches its own
inputs, detects drift, retrains, and is only replaced when a challenger is provably better
and no customer segment gets worse.

> Work in progress. Data pipeline, training and serving are done; monitoring and promotion
> are next.

## Data pipeline

```bash
uv sync
uv run dvc repro        # download → checksum → clean + validate → reference set + 12 monthly batches
uv run pytest
```

The source file is checksum-verified, every dataset is validated with a strict
[pandera](https://pandera.readthedocs.io/) schema, and the whole pipeline is reproducible:
re-running it produces byte-identical batches.

### Simulated drift

The data has no timestamps, so a stream of monthly batches is simulated from it, with drift
injected on a schedule set in [params.yaml](params.yaml). Customers are drawn without
replacement, so no one appears in more than one batch.

| From month | Scenario | Kind | What changes |
| --- | --- | --- | --- |
| 4 | income squeeze | covariate | credit limits ×0.7, balances ×1.25; outcomes unchanged |
| 7 | young customers | population | 55% of new customers are under 30 (vs 32% normally) |
| 10 | utilisation shock | concept | customers near their limit who pay on time start defaulting |

The resulting profile per month is in [data/batches/summary.json](data/batches/summary.json).

`sex` and `age` are never model inputs; they are kept only to audit performance per segment.

## Training and the model registry

```bash
uv run d2d train                 # train candidates on the reference set, register the best
uv run mlflow ui --backend-store-uri sqlite:///.mlflow/mlflow.db   # browse runs at :5000
```

Two candidates are trained on the same split and the better one (by validation AUC) is
registered in MLflow:

| Model | AUC | PR-AUC | KS | Brier |
| --- | --- | --- | --- | --- |
| **LightGBM** (registered) | **0.776** | **0.543** | **0.417** | **0.138** |
| Logistic regression (baseline) | 0.748 | 0.488 | 0.412 | 0.143 |

Full metrics, per-segment AUC and lineage are in [reports/training.json](reports/training.json).

- **The pipeline is the model.** Feature engineering (utilisation ratios, payment ratio,
  delinquency counts) sits inside the logged scikit-learn pipeline, so serving runs exactly the
  code training ran.
- **Aliases, not version numbers.** `champion` serves traffic and `challenger` is the latest
  candidate. Training only ever moves `challenger`; replacing the champion is the promotion
  step's job. The very first model becomes champion so there is something to serve.
- **Lineage.** Every registered version is tagged with a fingerprint of its training data, the
  git commit and the reason it was trained.
- **Safe serialisation.** Models are saved with skops rather than pickle, with an explicit
  allowlist of the types they may contain ([features.py](src/drift_to_deploy/modelling/features.py)).

How the first champion holds up as the simulated world drifts (scored per month):

| Months | AUC | Predicted vs actual default rate |
| --- | --- | --- |
| 1–3 (no drift) | 0.77–0.80 | about 21% vs 21% |
| 4–9 (income squeeze, young customers) | 0.73–0.79 | overpredicts: about 26% vs 22% |
| 10–12 (utilisation shock) | about 0.71 | underpredicts: about 25% vs 31% |

## Serving

```bash
uv run d2d serve        # http://127.0.0.1:8000/docs for the interactive API
```

| Endpoint | Purpose |
| --- | --- |
| `POST /predict` | score 1 to 1,000 customers; returns default probabilities and the model version |
| `GET /model` | which version is live, when it was loaded, and its lineage tags |
| `GET /health`, `GET /ready` | liveness, and readiness (a model is loaded) |
| `POST /admin/reload` | check the registry now rather than at the next poll |
| `GET /metrics` | Prometheus: requests, latency, score distribution, live version, reloads |

- **Hot-swap without downtime.** The API polls the registry and, when `champion` moves, loads
  the new version, checks it scores a known-good input, and only then swaps it in. A request
  already in flight finishes on the model it started with. If the new version fails to load,
  the old one keeps serving and the failure is counted in `/metrics`.
- **Protected attributes never reach the API.** Requests carry a customer ID and the model's
  inputs only; `sex` or `age` in a request is rejected. Outcomes and demographics are joined
  back by ID for monitoring, the way a lender keeps them out of the scoring path.
- **Every prediction is logged** with the exact inputs the model saw, so drift is measured on
  real traffic rather than on a separately prepared copy of the data.
- **Validation at the edge.** Inputs are checked against the same rules as the data schema, and
  a rejected request names the offending field.

## Data source

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C55S3H>. Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The data covers credit-card
customers in Taiwan in 2005; the drift scenarios above are synthetic.
