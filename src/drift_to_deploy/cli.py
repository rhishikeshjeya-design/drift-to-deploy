"""Command-line entry points. The DVC pipeline (dvc.yaml) calls these."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from drift_to_deploy.data import batches, download, schema
from drift_to_deploy.modelling import registry, train
from drift_to_deploy.params import Params

RAW = Path("data/raw/credit_default.parquet")
CLEAN = Path("data/clean/credit_default.parquet")
BATCHES = Path("data/batches")
REPORT = Path("reports/training.json")


def _download(params: Params, args: argparse.Namespace) -> None:
    frame = download.download(params.source, args.out)
    print(f"Downloaded {len(frame):,} rows to {args.out}")


def _clean(params: Params, args: argparse.Namespace) -> None:
    cleaned = schema.validate(schema.clean(pd.read_parquet(args.input)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_parquet(args.out, index=False)
    print(f"Cleaned and validated {len(cleaned):,} rows to {args.out}")


def _batches(params: Params, args: argparse.Namespace) -> None:
    reference, monthly, summary = batches.build(pd.read_parquet(args.input), params)
    batches.write(args.out, reference, monthly, summary)
    print(f"Wrote reference ({len(reference):,} rows) and {len(monthly)} monthly batches to {args.out}")
    for month in summary:
        tags = ", ".join(month.scenarios) or "-"
        print(
            f"  month {month.month:2d}: default rate {month.default_rate:.1%}, mean age {month.mean_age:.1f}  [{tags}]"
        )


def _train(params: Params, args: argparse.Namespace) -> None:
    registry.configure()
    result = train.train(pd.read_parquet(args.input), params, reason=args.reason)
    train.write_report(result, args.report)
    for c in result.candidates:
        m = c.metrics
        print(f"  {c.kind:<9} AUC {m['auc']:.4f}  PR-AUC {m['pr_auc']:.4f}  KS {m['ks']:.3f}  Brier {m['brier']:.4f}")
    role = "champion and challenger" if result.bootstrapped_champion else "challenger"
    print(f"Registered {params.model.name} v{result.version} ({result.winner.kind}) as {role}.")


def _serve(params: Params, args: argparse.Namespace) -> None:
    import uvicorn

    from drift_to_deploy.serving.app import ServingSettings, create_app

    uvicorn.run(create_app(ServingSettings(model_name=params.model.name)), host=args.host, port=args.port)


def _traffic(params: Params, args: argparse.Namespace) -> None:
    import httpx

    from drift_to_deploy.monitoring import traffic

    batch = pd.read_parquet(BATCHES / "months" / f"month_{args.month:02d}.parquet")
    with httpx.Client(base_url=args.url, timeout=60) as client:
        sent = traffic.send(batch, client)
    print(f"Sent {sent:,} month-{args.month} customers to {args.url}")


def _monitor(params: Params, args: argparse.Namespace) -> None:
    from drift_to_deploy.monitoring.monitor import run_month
    from drift_to_deploy.serving.app import ServingSettings
    from drift_to_deploy.serving.prediction_log import PredictionLog

    registry.connect()
    record = run_month(args.month, params, PredictionLog(ServingSettings().prediction_log))
    d, decision = record["drift"], record["decision"]
    line = f"month {args.month:2d} (v{record['serving_version']}): {d['drifted_share']:.0%} inputs drifted"
    line += f", score drift {d['score_distance']:.2f}"
    if record["performance"]:
        p = record["performance"]
        rates = f"{p['predicted_rate']:.0%} predicted vs {p['observed_rate']:.0%} actual"
        line += f" | month {record['labelled_month']} outcomes: AUC {p['auc']:.3f}, {rates}"
    print(f"{line}  ->  {decision['status'].upper()}{'  RETRAIN' if decision['retrain'] else ''}")
    for reason in decision["reasons"]:
        print(f"    - {reason}")
    if decision["blocked_by"]:
        print(f"    (retrain held back: {decision['blocked_by']})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="d2d", description="drift-to-deploy")
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    groups = parser.add_subparsers(dest="group", required=True)

    data = groups.add_parser("data", help="data pipeline steps").add_subparsers(dest="step", required=True)
    step = data.add_parser("download", help="fetch and checksum the source dataset")
    step.add_argument("--out", type=Path, default=RAW)
    step.set_defaults(run=_download)
    step = data.add_parser("clean", help="normalise and validate the raw data")
    step.add_argument("--input", type=Path, default=RAW)
    step.add_argument("--out", type=Path, default=CLEAN)
    step.set_defaults(run=_clean)
    step = data.add_parser("batches", help="build the reference set and drifted monthly batches")
    step.add_argument("--input", type=Path, default=CLEAN)
    step.add_argument("--out", type=Path, default=BATCHES)
    step.set_defaults(run=_batches)

    step = groups.add_parser("train", help="train candidates and register the best as challenger")
    step.add_argument("--input", type=Path, default=BATCHES / "reference.parquet")
    step.add_argument("--report", type=Path, default=REPORT)
    step.add_argument("--reason", default="initial training")
    step.set_defaults(run=_train)

    step = groups.add_parser("serve", help="run the scoring API")
    step.add_argument("--host", default="127.0.0.1")
    step.add_argument("--port", type=int, default=8000)
    step.set_defaults(run=_serve)

    step = groups.add_parser("traffic", help="send a month's customers to the scoring API")
    step.add_argument("--month", type=int, required=True)
    step.add_argument("--url", default="http://127.0.0.1:8000")
    step.set_defaults(run=_traffic)

    step = groups.add_parser("monitor", help="drift and performance checks for a month, and the retrain decision")
    step.add_argument("--month", type=int, required=True)
    step.set_defaults(run=_monitor)

    args = parser.parse_args(argv)
    args.run(Params.load(args.params), args)


if __name__ == "__main__":
    sys.exit(main())
