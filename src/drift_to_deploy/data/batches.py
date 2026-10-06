"""Split the cleaned data into a reference set and a stream of monthly batches, then inject drift.

Rows are drawn without replacement, so no customer appears in more than one place and a
model retrained on recent months can never be evaluated on rows it has already seen.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from drift_to_deploy.data.drift import apply_concept, apply_covariate
from drift_to_deploy.data.schema import TARGET, validate
from drift_to_deploy.params import ConceptScenario, CovariateScenario, Params, PopulationScenario

MONTH = "month"


@dataclass
class MonthSummary:
    month: int
    rows: int
    default_rate: float
    mean_age: float
    mean_limit_bal: float
    scenarios: list[str]


class NotEnoughRows(ValueError):
    pass


def _draw_month(
    pool: pd.DataFrame, size: int, population: PopulationScenario | None, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take `size` rows from the pool, optionally over-representing a segment. Returns (batch, rest)."""
    if population is None:
        idx = rng.choice(pool.index, size=size, replace=False)
    else:
        in_segment = pool.index[pool["age"] < population.age_below]
        others = pool.index[pool["age"] >= population.age_below]
        n_seg = round(size * population.share)
        if len(in_segment) < n_seg or len(others) < size - n_seg:
            raise NotEnoughRows(
                f"'{population.name}' needs {n_seg} customers under {population.age_below} "
                f"but only {len(in_segment)} remain. Reduce `share` or `month_size`."
            )
        idx = np.concatenate(
            [rng.choice(in_segment, size=n_seg, replace=False), rng.choice(others, size=size - n_seg, replace=False)]
        )
        rng.shuffle(idx)
    return pool.loc[idx], pool.drop(index=idx)


def build(clean: pd.DataFrame, params: Params) -> tuple[pd.DataFrame, list[pd.DataFrame], list[MonthSummary]]:
    """Returns (reference, monthly batches, per-month summary)."""
    cfg = params.batches
    needed = cfg.reference_size + cfg.months * cfg.month_size
    if needed > len(clean):
        raise NotEnoughRows(f"Batches need {needed} rows but the dataset has {len(clean)}.")

    rng = np.random.default_rng(params.seed)
    shuffled = clean.sample(frac=1.0, random_state=params.seed)
    reference = shuffled.iloc[: cfg.reference_size].reset_index(drop=True)
    pool = shuffled.iloc[cfg.reference_size :]

    population_by_month = {
        m: next((s for s in params.active(m) if isinstance(s, PopulationScenario)), None)
        for m in range(1, cfg.months + 1)
    }
    # Ordinary months draw first, at the natural customer mix. Drawing the segment-heavy
    # months first would starve the early months of that segment: unintended drift.
    order = sorted(range(1, cfg.months + 1), key=lambda m: population_by_month[m] is not None)
    drawn: dict[int, pd.DataFrame] = {}
    for month in order:
        drawn[month], pool = _draw_month(pool, cfg.month_size, population_by_month[month], rng)

    batches, summary = [], []
    for month in range(1, cfg.months + 1):
        batch = drawn[month]
        for scenario in params.active(month):
            if isinstance(scenario, CovariateScenario):
                batch = apply_covariate(batch, scenario)
            elif isinstance(scenario, ConceptScenario):
                batch = apply_concept(batch, scenario, rng)
        batch = batch.reset_index(drop=True).assign(**{MONTH: month})
        validate(batch, extra_columns=(MONTH,))
        batches.append(batch)
        summary.append(
            MonthSummary(
                month=month,
                rows=len(batch),
                default_rate=round(float(batch[TARGET].mean()), 4),
                mean_age=round(float(batch["age"].mean()), 2),
                mean_limit_bal=round(float(batch["limit_bal"].mean()), 1),
                scenarios=[s.name for s in params.active(month)],
            )
        )
    return reference, batches, summary


def write(out_dir: Path, reference: pd.DataFrame, batches: list[pd.DataFrame], summary: list[MonthSummary]) -> None:
    months_dir = out_dir / "months"
    months_dir.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(out_dir / "reference.parquet", index=False)
    for batch in batches:
        batch.to_parquet(months_dir / f"month_{int(batch[MONTH].iloc[0]):02d}.parquet", index=False)
    (out_dir / "summary.json").write_text(json.dumps({"months": [asdict(s) for s in summary]}, indent=2) + "\n")
