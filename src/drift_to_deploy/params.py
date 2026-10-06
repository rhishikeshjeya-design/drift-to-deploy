"""Typed access to params.yaml, the single source of truth for the data pipeline."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, Field


class Source(BaseModel):
    url: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Batches(BaseModel):
    reference_size: int = Field(gt=0)
    months: int = Field(gt=0)
    month_size: int = Field(gt=0)


class CovariateScenario(BaseModel):
    name: str
    kind: Literal["covariate"]
    start_month: int = Field(ge=1)
    limit_bal_factor: float = Field(gt=0)
    bill_amt_factor: float = Field(gt=0)


class PopulationScenario(BaseModel):
    name: str
    kind: Literal["population"]
    start_month: int = Field(ge=1)
    age_below: int
    share: float = Field(gt=0, lt=1)


class ConceptScenario(BaseModel):
    name: str
    kind: Literal["concept"]
    start_month: int = Field(ge=1)
    min_utilisation: float = Field(ge=0)
    flip_probability: float = Field(gt=0, le=1)


Scenario = Annotated[CovariateScenario | PopulationScenario | ConceptScenario, Field(discriminator="kind")]


class ModelParams(BaseModel):
    name: str
    validation_size: float = Field(gt=0, lt=1)
    lightgbm: dict[str, float | int] = Field(default_factory=dict)
    logistic: dict[str, float | int] = Field(default_factory=dict)


class Params(BaseModel):
    seed: int
    source: Source
    batches: Batches
    scenarios: list[Scenario] = Field(default_factory=list)
    model: ModelParams

    @classmethod
    def load(cls, path: Path = Path("params.yaml")) -> "Params":
        return cls.model_validate(yaml.safe_load(path.read_text()))

    def active(self, month: int) -> list[Scenario]:
        """Scenarios in effect in a given month (they persist once started)."""
        return [s for s in self.scenarios if s.start_month <= month]
