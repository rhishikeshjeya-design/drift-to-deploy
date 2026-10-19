# Roadmap

Steps 1–4 are done: data pipeline (DVC + pandera, simulated drift), training (LightGBM vs a
logistic baseline, MLflow registry with `champion` / `challenger` aliases), serving (FastAPI
with hot-swap, prediction log, Prometheus metrics) and monitoring (Evidently drift, lagged
performance checks, retrain triggers).

## 5. Retraining and gated promotion

- When the monitor decides to retrain, train a challenger on a recent window: labelled months
  since the drift began, plus a sample of the champion's training data so the model doesn't
  forget the old world. Window size and mix go in `params.yaml`.
- Evaluate challenger and champion on **the same holdout**: the most recent labelled month,
  which neither was trained on.
- Promote only if every gate passes, and record the decision and the reason either way:
  - AUC at least `+0.005` over the champion
  - no segment (age band, sex, education) loses more than `0.02` AUC
  - p95 scoring latency under 50 ms
- Promotion is an alias move; the API's poll (or `/admin/reload`) picks it up. Record the
  retrain in the monitoring history so the cooldown counts real retrains, not requests.
- Tests: each gate passing and failing, a promotion end to end, a rejected challenger leaves
  the champion untouched.

## 6. Orchestration and the demo

- Prefect flows: `monitor_month` → (if triggered) `retrain` → `evaluate_and_promote`.
- `make demo`: replay all 12 months through the API end to end, so the README can show drift
  being detected, a retrain, a gated promotion, and recovery.

## 7. Docker Compose and Grafana

- Services: MLflow server, scoring API, Prefect, Prometheus (with a Pushgateway for monitor
  metrics), Grafana with a provisioned dashboard: latency, volume, drift share, score drift,
  live model version, promotions.

## 8. Polish

- GitHub Actions CI (ruff, format check, pytest). Speed tests up by sharing one trained model
  across tests where possible.
- README: architecture diagram, screenshots (Grafana, MLflow, an Evidently report), and a
  "what I'd do in production" section: shadow and canary deploys, real label delay, retraining
  cost, model cards.
- Make the repository public.
