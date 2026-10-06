"""Model metrics, overall and per customer segment."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

# Segments are evaluated, never used as model inputs.
AGE_BANDS = (0, 30, 45, 200)
AGE_LABELS = ("under_30", "30_to_44", "45_plus")
SEX = {1: "male", 2: "female"}
EDUCATION = {1: "graduate_school", 2: "university", 3: "high_school", 4: "other"}
# Below this, a segment's AUC is too noisy to act on.
MIN_SEGMENT_SIZE = 100


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Largest gap between the score distributions of defaulters and non-defaulters."""
    p = np.asarray(p)
    order = np.argsort(p, kind="stable")
    p, y = p[order], np.asarray(y)[order]
    cum_pos = np.cumsum(y) / max(y.sum(), 1)
    cum_neg = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    # Compare only where the score changes: customers with tied scores can't be told apart,
    # so splitting them would invent separation that isn't there.
    boundary = np.r_[p[1:] != p[:-1], True]
    return float(np.max(np.abs(cum_pos[boundary] - cum_neg[boundary])))


def calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: average gap between predicted and observed default rates."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    gaps = [abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()]
    return float(sum(gaps))


def overall(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "calibration_error": calibration_error(y, p),
        "predicted_default_rate": float(np.mean(p)),
        "observed_default_rate": float(np.mean(y)),
    }


def segments(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Segment label per row, for each audited attribute."""
    return {
        "age_band": pd.cut(df["age"], bins=AGE_BANDS, labels=AGE_LABELS, right=False).astype(str),
        "sex": df["sex"].map(SEX),
        "education": df["education"].map(EDUCATION),
    }


def by_segment(df: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> dict[str, dict[str, float]]:
    """AUC and size per segment, e.g. {"sex:female": {"auc": 0.77, "n": 1200}}.

    Segments that are too small, or contain only one outcome, are left out rather than
    reported with a misleading number.
    """
    y, p = np.asarray(y), np.asarray(p)
    out = {}
    for attribute, labels in segments(df).items():
        for label in sorted(labels.dropna().unique()):
            mask = (labels == label).to_numpy()
            if mask.sum() < MIN_SEGMENT_SIZE or len(np.unique(y[mask])) < 2:
                continue
            out[f"{attribute}:{label}"] = {"auc": float(roc_auc_score(y[mask], p[mask])), "n": int(mask.sum())}
    return out
