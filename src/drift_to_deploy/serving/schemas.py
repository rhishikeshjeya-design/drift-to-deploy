"""Request and response models. The constraints mirror the data schema, so invalid input is
rejected at the edge with a clear error instead of reaching the model."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from drift_to_deploy.data.schema import FEATURES

PayStatus = Field(ge=-2, le=9, description="Repayment status: -2/-1/0 paid duly or no use, 1-9 months late.")
BillAmount = Field(description="Statement balance. Negative means a credit balance.")
PayAmount = Field(ge=0, description="Amount paid that month.")


class Customer(BaseModel):
    """One customer to score. Sex and age are deliberately absent: the model never sees them."""

    model_config = ConfigDict(extra="forbid")

    customer_id: int = Field(description="Your identifier, echoed back and used to join outcomes later.")
    limit_bal: float = Field(gt=0, description="Credit limit.")
    education: int = Field(ge=1, le=4, description="1 graduate school, 2 university, 3 high school, 4 other.")
    marriage: int = Field(ge=1, le=3, description="1 married, 2 single, 3 other.")
    pay_1: int = PayStatus
    pay_2: int = PayStatus
    pay_3: int = PayStatus
    pay_4: int = PayStatus
    pay_5: int = PayStatus
    pay_6: int = PayStatus
    bill_amt_1: float = BillAmount
    bill_amt_2: float = BillAmount
    bill_amt_3: float = BillAmount
    bill_amt_4: float = BillAmount
    bill_amt_5: float = BillAmount
    bill_amt_6: float = BillAmount
    pay_amt_1: float = PayAmount
    pay_amt_2: float = PayAmount
    pay_amt_3: float = PayAmount
    pay_amt_4: float = PayAmount
    pay_amt_5: float = PayAmount
    pay_amt_6: float = PayAmount


EXAMPLE = {
    "customer_id": 1,
    "limit_bal": 50000.0,
    "education": 2,
    "marriage": 1,
    **{f"pay_{i}": 0 for i in range(1, 7)},
    **{f"bill_amt_{i}": 20000.0 for i in range(1, 7)},
    **{f"pay_amt_{i}": 2000.0 for i in range(1, 7)},
}
assert set(EXAMPLE) - {"customer_id"} == set(FEATURES), "the API schema must match the model's features"


class PredictRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)

    model_config = ConfigDict(json_schema_extra={"examples": [{"customers": [EXAMPLE]}]})


class Prediction(BaseModel):
    customer_id: int
    default_probability: float


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    name: str
    alias: str
    version: str
    loaded_at: datetime
    tags: dict[str, str]
