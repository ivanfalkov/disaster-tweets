"""Classic sklearn models with input-type validation."""
from __future__ import annotations

from typing import Any

from scipy.sparse import spmatrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def build_classic_model(cfg: dict[str, Any]):
    mtype = cfg.get("type")
    params = cfg.get("params", {}) or {}

    if mtype == "logreg":
        return LogisticRegression(**params)
    if mtype == "linsvc":
        return LinearSVC(**params)
    if mtype == "gbm":
        return GradientBoostingClassifier(**params)
    raise ValueError(f"Unknown classic model type: {mtype!r}")


def validate_sparse_input(X, model_name: str) -> None:
    """Raise a clear error if the model can't handle this X."""
    if not isinstance(X, spmatrix):
        raise TypeError(
            f"{model_name} expects a scipy.sparse matrix, got {type(X).__name__}. "
            f"Likely all feature blocks are 'none'. Either enable text/keyword/location "
            f"encoding, or use a model that consumes raw columns."
        )