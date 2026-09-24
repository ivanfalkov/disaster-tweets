"""CatBoost model factories.

Two modes:
    build_catboost_sparse : CatBoostClassifier fed with a sparse matrix
                            (from builder_output='sparse').
    build_catboost_native : CatBoostClassifier fed with a DataFrame,
                            using cat_features / text_features from model.params.
                            (from builder_output='dataframe').

Both return an unfitted CatBoostClassifier. Actual fit / predict is done
by the experiment runner.
"""
from __future__ import annotations

from typing import Any

from catboost import CatBoostClassifier


def build_catboost_sparse(params: dict[str, Any] | None = None) -> CatBoostClassifier:
    """CatBoost over a sparse feature matrix.

    `cat_features` and `text_features` are not applicable here
    (sparse input already contains numerical features).
    If present in params, they are removed with a warning.
    """
    params = dict(params or {})
    for forbidden in ("cat_features", "text_features"):
        if forbidden in params:
            params.pop(forbidden)
            print(f"[catboost] ignoring {forbidden!r}: not applicable to sparse input")
    return CatBoostClassifier(**params)


def build_catboost_native(params: dict[str, Any] | None = None) -> CatBoostClassifier:
    """CatBoost over a DataFrame with cat_features / text_features.

    Expects params to contain `cat_features` and/or `text_features`
    as lists of column names.
    """
    return CatBoostClassifier(**(params or {}))


def validate_native_columns(
    df,
    params: dict[str, Any],
) -> None:
    """Check that cat_features / text_features columns exist in df."""
    cat_features = params.get("cat_features", []) or []
    text_features = params.get("text_features", []) or []

    missing = [c for c in cat_features + text_features if c not in df.columns]
    if missing:
        raise KeyError(
            f"CatBoost native mode: columns not found in DataFrame: {missing}. "
            f"Available columns: {list(df.columns)}"
        )