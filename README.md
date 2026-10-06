# drift-to-deploy

An automated credit-risk ML lifecycle: a default-prediction model that watches its own
inputs, detects drift, retrains, and is only replaced when a challenger is provably better
and no customer segment gets worse.

> Work in progress. The data pipeline is done; training, serving, monitoring and promotion
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

## Data source

Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning Repository.
<https://doi.org/10.24432/C55S3H>. Licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The data covers credit-card
customers in Taiwan in 2005; the drift scenarios above are synthetic.
