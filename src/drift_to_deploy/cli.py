"""Command-line entry points. The DVC pipeline (dvc.yaml) calls these."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from drift_to_deploy.data import batches, download, schema
from drift_to_deploy.params import Params

RAW = Path("data/raw/credit_default.parquet")
CLEAN = Path("data/clean/credit_default.parquet")
BATCHES = Path("data/batches")


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

    args = parser.parse_args(argv)
    args.run(Params.load(args.params), args)


if __name__ == "__main__":
    sys.exit(main())
