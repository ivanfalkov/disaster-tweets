"""Model factory: dispatch by model.type across classic, catboost, finetune."""
from __future__ import annotations

from typing import Any

from src.models.catboost import build_catboost_native, build_catboost_sparse
from src.models.classic import build_classic_model
from src.models.finetune import build_finetune_model

CLASSIC_TYPES = {"logreg", "gbm", "lgbm"}
CATBOOST_SPARSE_TYPES = {"catboost_sparse"}
CATBOOST_NATIVE_TYPES = {"catboost_native"}
FINETUNE_TYPES = {"finetune"}


def build_model(cfg: dict[str, Any]):
    """Build an unfitted model from cfg = {'type': ..., 'params': {...}}."""
    mtype = cfg.get("type")
    params = cfg.get("params", {}) or {}

    if mtype in CLASSIC_TYPES:
        return build_classic_model(cfg)
    if mtype in CATBOOST_SPARSE_TYPES:
        return build_catboost_sparse(params)
    if mtype in CATBOOST_NATIVE_TYPES:
        return build_catboost_native(params)
    if mtype in FINETUNE_TYPES:
        return build_finetune_model(params)
    raise ValueError(f"Unknown model type: {mtype!r}")
