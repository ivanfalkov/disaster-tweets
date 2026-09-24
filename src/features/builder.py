"""Build feature matrices for an experiment.

Reads `features` section of an experiment config.

builder_output:
    "sparse"    : all blocks -> numerical features -> csr_matrix
    "dataframe" : all blocks -> raw columns -> pd.DataFrame

Rules:
    sparse mode:
        text      : none | bow | tfidf | word2vec
        keyword   : none | onehot
        location  : none | onehot
        ("none" forbidden: nothing numerical to build)

    dataframe mode:
        text      : none
        keyword   : none
        location  : none
        (tfidf/bow/word2vec/onehot forbidden)

`cat_features` / `text_features` are not handled here — they live in
`model.params` and are read by the model itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix, hstack

from src.features.categorical import build_categorical_encoder
from src.features.vectorizer import build_text_vectorizer


VALID_BUILDER_OUTPUTS = ("sparse", "dataframe")
BLOCKS = ("text", "keyword", "location")
DEFAULT_COLUMNS = {"text": "text_pp", "keyword": "keyword", "location": "location"}
FORBIDDEN_IN_DATAFRAME = {"text": {"tfidf", "bow", "word2vec"}, "keyword": {"onehot"}, "location": {"onehot"}}


@dataclass
class FeaturesBundle:
    output: str
    blocks: list[str] = field(default_factory=list)
    vectorizers: dict[str, Any] = field(default_factory=dict)

    X_train: csr_matrix | None = None
    X_val: csr_matrix | None = None
    X_test: csr_matrix | None = None

    df_train: pd.DataFrame | None = None
    df_val: pd.DataFrame | None = None
    df_test: pd.DataFrame | None = None
    feature_names: list[str] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        if self.output == "sparse":
            if self.X_train is None:
                raise RuntimeError("X_train is None in sparse mode")
            return self.X_train.shape[1]
        if self.df_train is None:
            raise RuntimeError("df_train is None in dataframe mode")
        return self.df_train.shape[1]


def _build_sparse(train_df, val_df, test_df, features_cfg) -> FeaturesBundle:
    parts_train, parts_val, parts_test = [], [], []
    blocks, fitted = [], {}

    for block in BLOCKS:
        cfg = features_cfg.get(block, {}) or {}
        btype = cfg.get("type", "none")
        if btype == "none":
            raise ValueError(
                f"builder_output='sparse' does not support features.{block}.type='none'. "
                f"Use builder_output='dataframe' to pass raw columns to the model."
            )
        col = cfg.get("column", DEFAULT_COLUMNS[block])

        if block == "text":
            vec = build_text_vectorizer(cfg)
        else:
            vec = build_categorical_encoder(cfg)

        parts_train.append(vec.fit_transform(train_df[col]).tocsr())
        parts_val.append(vec.transform(val_df[col]).tocsr())
        parts_test.append(vec.transform(test_df[col]).tocsr())
        blocks.append(block)
        fitted[block] = vec

    return FeaturesBundle(
        output="sparse",
        blocks=blocks,
        vectorizers=fitted,
        X_train=hstack(parts_train).tocsr(),
        X_val=hstack(parts_val).tocsr(),
        X_test=hstack(parts_test).tocsr(),
    )


def _build_dataframe(train_df, val_df, test_df, features_cfg) -> FeaturesBundle:
    dfs = {
        "train": pd.DataFrame(index=train_df.index),
        "val": pd.DataFrame(index=val_df.index),
        "test": pd.DataFrame(index=test_df.index),
    }
    sources = {"train": train_df, "val": val_df, "test": test_df}
    blocks = []

    for block in BLOCKS:
        cfg = features_cfg.get(block, {}) or {}
        btype = cfg.get("type", "none")
        if btype in FORBIDDEN_IN_DATAFRAME[block]:
            raise ValueError(
                f"builder_output='dataframe' does not support features.{block}.type={btype!r}. "
                f"Use builder_output='sparse' for encoding, or set type='none' "
                f"and let the model handle the raw column (e.g. CatBoost cat_features)."
            )
        col = cfg.get("column", DEFAULT_COLUMNS[block])
        for split, src in sources.items():
            dfs[split][col] = src[col].values
        blocks.append(block)

    for split in dfs:
        dfs[split] = dfs[split].reset_index(drop=True)

    return FeaturesBundle(
        output="dataframe",
        blocks=blocks,
        df_train=dfs["train"],
        df_val=dfs["val"],
        df_test=dfs["test"],
        feature_names=list(dfs["train"].columns),
    )


def build_features(train_df, val_df, test_df, features_cfg: dict[str, Any]) -> FeaturesBundle:
    if not isinstance(features_cfg, dict):
        raise TypeError(f"features_cfg must be dict, got {type(features_cfg).__name__}")

    output = features_cfg.get("builder_output", "sparse")
    if output not in VALID_BUILDER_OUTPUTS:
        raise ValueError(f"Unknown builder_output: {output!r}. Expected {VALID_BUILDER_OUTPUTS}")

    if output == "sparse":
        return _build_sparse(train_df, val_df, test_df, features_cfg)
    return _build_dataframe(train_df, val_df, test_df, features_cfg)