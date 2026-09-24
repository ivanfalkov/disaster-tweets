"""Categorical encoder: normalize -> collapse rare -> one-hot.

Fit on train only. Unknown categories at transform time are mapped to
`__rare__`. Missing values are mapped to `__missing__`.

The rare threshold is computed on train: any category with frequency
strictly less than `min_freq` is collapsed to `__rare__`.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

MISSING = "__missing__"
RARE = "__rare__"

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9\s,]")


def _normalize_value(value: Any) -> str:
    if value is None:
        return MISSING
    if isinstance(value, float) and np.isnan(value):
        return MISSING
    s = str(value).strip().lower()
    if not s:
        return MISSING
    s = _NON_ALNUM.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    if not s:
        return MISSING
    return s


def _normalize_series(series: pd.Series, normalize: bool) -> pd.Series:
    if normalize:
        return series.apply(_normalize_value)
    return series.apply(
        lambda v: MISSING if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v)
    )


class CategoricalEncoder:
    """Fit on train, transform val/test. Unknown -> __rare__, NaN -> __missing__."""

    def __init__(
        self,
        *,
        normalize: bool = True,
        min_freq: int = 5,
        handle_missing: str = MISSING,
        handle_unknown: str = RARE,
    ) -> None:
        self.normalize = normalize
        self.min_freq = min_freq
        self.handle_missing = handle_missing
        self.handle_unknown = handle_unknown
        self._ohe: OneHotEncoder | None = None
        self._keep_categories: set[str] | None = None

    def fit(self, series: pd.Series) -> "CategoricalEncoder":
        values = _normalize_series(series, self.normalize)

        # frequency on train
        counts = values.value_counts()
        keep = set(counts[counts >= self.min_freq].index)
        # __missing__ всегда сохраняем как отдельную категорию, даже если редкая
        keep.add(self.handle_missing)

        # collapse to __rare__ / __missing__
        def _collapse(v: str) -> str:
            if v == self.handle_missing:
                return self.handle_missing
            if v not in keep:
                return self.handle_unknown
            return v

        collapsed = values.apply(_collapse)

        # обучаем OHE на схлопнутых категориях
        self._keep_categories = keep | {self.handle_unknown}
        self._ohe = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
        self._ohe.fit(collapsed.to_numpy().reshape(-1, 1))
        return self

    def transform(self, series: pd.Series) -> csr_matrix:
        if self._ohe is None or self._keep_categories is None:
            raise RuntimeError("CategoricalEncoder is not fitted")

        values = _normalize_series(series, self.normalize)

        def _map(v: str) -> str:
            if v == self.handle_missing:
                return self.handle_missing
            if v not in self._keep_categories:
                return self.handle_unknown
            return v

        mapped = values.apply(_map).to_numpy().reshape(-1, 1)
        return self._ohe.transform(mapped).tocsr()

        
    def fit_transform(self, series: pd.Series):
        return self.fit(series).transform(series)
    @property
    def n_features(self) -> int:
        if self._ohe is None:
            raise RuntimeError("CategoricalEncoder is not fitted")
        return len(self._ohe.get_feature_names_out())


def build_categorical_encoder(cfg: dict[str, Any]) -> CategoricalEncoder | None:
    """Return an unfitted CategoricalEncoder, or None if type == 'none'."""
    ctype = cfg.get("type", "none")
    if ctype == "none":
        return None
    if ctype == "onehot":
        return CategoricalEncoder(
            normalize=cfg.get("normalize", True),
            min_freq=cfg.get("min_freq", 5),
            handle_missing=cfg.get("handle_missing", MISSING),
            handle_unknown=cfg.get("handle_unknown", RARE),
        )
    raise ValueError(f"Unknown categorical encoder type: {ctype!r}")