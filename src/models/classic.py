"""Classic sklearn models."""
from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

def build_classic_model(cfg: dict[str, Any]):
    """Build an unfitted sklearn classifier from cfg = {'type': ..., 'params': {...}}."""
    mtype = cfg.get("type")
    params = cfg.get("params", {}) or {}

    if mtype == "logreg":
        return LogisticRegression(**params)
    if mtype == "gbm":
        return GradientBoostingClassifier(**params)
    if mtype == "lgbm":
        return LGBMClassifier(**params)   

    raise ValueError(f"Unknown classic model type: {mtype!r}")