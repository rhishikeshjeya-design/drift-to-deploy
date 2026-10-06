"""Column names, cleaning rules and the validation schema every dataset must pass."""

import pandas as pd
import pandera.pandas as pa

PAY_STATUS = [f"pay_{i}" for i in range(1, 7)]
BILL_AMOUNTS = [f"bill_amt_{i}" for i in range(1, 7)]
PAY_AMOUNTS = [f"pay_amt_{i}" for i in range(1, 7)]

TARGET = "default"
ID = "customer_id"
# Used to audit the model per segment, never as model inputs.
PROTECTED = ["sex", "age"]
FEATURES = ["limit_bal", "education", "marriage", *PAY_STATUS, *BILL_AMOUNTS, *PAY_AMOUNTS]

_RENAME = {
    "ID": ID,
    "LIMIT_BAL": "limit_bal",
    "SEX": "sex",
    "EDUCATION": "education",
    "MARRIAGE": "marriage",
    "AGE": "age",
    # The source skips PAY_1: its repayment-status columns are PAY_0, PAY_2, ..., PAY_6.
    "PAY_0": "pay_1",
    **{f"PAY_{i}": f"pay_{i}" for i in range(2, 7)},
    **{f"BILL_AMT{i}": f"bill_amt_{i}" for i in range(1, 7)},
    **{f"PAY_AMT{i}": f"pay_amt_{i}" for i in range(1, 7)},
    "default payment next month": TARGET,
}


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and fold undocumented category codes into documented ones."""
    df = raw.rename(columns=_RENAME)[list(_RENAME.values())].copy()
    # The data dictionary defines education 1-4 (4 = other) and marriage 1-3 (3 = other);
    # the file also contains 0, 5 and 6, which are folded into "other".
    df["education"] = df["education"].where(df["education"].between(1, 4), 4)
    df["marriage"] = df["marriage"].where(df["marriage"].between(1, 3), 3)
    money = ["limit_bal", *BILL_AMOUNTS, *PAY_AMOUNTS]
    df[money] = df[money].astype("float64")
    return df


def _int_in(values: list[int]) -> pa.Column:
    return pa.Column(int, pa.Check.isin(values))


SCHEMA = pa.DataFrameSchema(
    {
        ID: pa.Column(int, unique=True),
        "limit_bal": pa.Column(float, pa.Check.gt(0)),
        "sex": _int_in([1, 2]),
        "education": _int_in([1, 2, 3, 4]),
        "marriage": _int_in([1, 2, 3]),
        "age": pa.Column(int, pa.Check.in_range(18, 100)),
        **{c: pa.Column(int, pa.Check.in_range(-2, 9)) for c in PAY_STATUS},
        # Negative bill amounts are legitimate: an overpayment leaves a credit balance.
        **{c: pa.Column(float) for c in BILL_AMOUNTS},
        **{c: pa.Column(float, pa.Check.ge(0)) for c in PAY_AMOUNTS},
        TARGET: _int_in([0, 1]),
    },
    strict=True,
    coerce=False,
)


def validate(df: pd.DataFrame, extra_columns: tuple[str, ...] = ()) -> pd.DataFrame:
    """Raise pandera.errors.SchemaErrors listing every failure, not just the first."""
    schema = SCHEMA
    if extra_columns:
        schema = SCHEMA.add_columns({c: pa.Column(nullable=False) for c in extra_columns})
    return schema.validate(df, lazy=True)
