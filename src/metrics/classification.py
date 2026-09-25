"""Classification metrics computation.

Single entry point: compute_metrics(y_true, y_pred, y_score=None, metrics=...)

- Accepts numpy arrays, python lists, pandas Series, and torch.Tensors.
  Torch inputs are detached from the graph and moved to CPU before conversion.
- Always returns a flat dict[str, float].
- ROC-AUC / average_precision work with either probabilities or decision
  function scores (pass them via `y_score`).
- zero_division=0 for precision/recall/f1 to avoid crashes on degenerate
  predictions.
- Metrics that cannot be computed (e.g. ROC-AUC with a single class in y_true)
  are skipped with a warning instead of raising.
"""
from __future__ import annotations

import warnings
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


SUPPORTED_METRICS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "mcc",
)

DEFAULT_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def _as_1d(a: Any) -> np.ndarray | None:
    if a is None:
        return None
    if hasattr(a, "detach"):
        a = a.detach()
    if hasattr(a, "cpu"):
        a = a.cpu()
    if hasattr(a, "numpy"):
        a = a.numpy()
    return np.asarray(a).ravel()


def _safe(fn, *args, **kwargs) -> float | None:
    try:
        return float(fn(*args, **kwargs))
    except Exception as e:
        warnings.warn(f"[metrics] skipped: {e}", RuntimeWarning)
        return None


def compute_metrics(
    y_true: Sequence | np.ndarray | Any,
    y_pred: Sequence | np.ndarray | Any,
    y_score: Sequence | np.ndarray | Any | None = None,
    metrics: Iterable[str] = DEFAULT_METRICS,
    *,
    pos_label: int = 1,
) -> dict[str, float]:
    y_true = _as_1d(y_true)
    y_pred = _as_1d(y_pred)
    y_score = _as_1d(y_score)

    if y_true is None or y_pred is None:
        raise ValueError("y_true and y_pred must not be None")
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length: "
            f"{y_true.shape[0]} vs {y_pred.shape[0]}"
        )

    requested = list(metrics)
    unknown = [m for m in requested if m not in SUPPORTED_METRICS]
    if unknown:
        raise ValueError(
            f"Unknown metrics: {unknown}. Supported: {list(SUPPORTED_METRICS)}"
        )

    out: dict[str, float] = {}
    for name in requested:
        if name == "accuracy":
            v = _safe(accuracy_score, y_true, y_pred)
        elif name == "precision":
            v = _safe(precision_score, y_true, y_pred, pos_label=pos_label, zero_division=0)
        elif name == "recall":
            v = _safe(recall_score, y_true, y_pred, pos_label=pos_label, zero_division=0)
        elif name == "f1":
            v = _safe(f1_score, y_true, y_pred, pos_label=pos_label, zero_division=0)
        elif name == "mcc":
            v = _safe(matthews_corrcoef, y_true, y_pred)
        elif name == "roc_auc":
            if y_score is None:
                warnings.warn("[metrics] roc_auc requested but y_score is None", RuntimeWarning)
                v = None
            else:
                v = _safe(roc_auc_score, y_true, y_score)
        elif name == "average_precision":
            if y_score is None:
                warnings.warn("[metrics] average_precision requested but y_score is None", RuntimeWarning)
                v = None
            else:
                v = _safe(average_precision_score, y_true, y_score)
        else:
            v = None

        if v is not None:
            out[name] = v

    return out