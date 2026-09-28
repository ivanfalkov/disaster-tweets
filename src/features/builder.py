"""Feature builder as an sklearn-compatible transformer.

    builder = FeatureBuilder(features_cfg)
    X_train = builder.fit_transform(train_df)
    X_val   = builder.transform(val_df)
    X_test  = builder.transform(test_df)

Two output modes via features_cfg['builder_output']:

    "sparse"    : all blocks -> numerical features -> csr_matrix
    "dataframe" : all blocks -> raw columns -> pd.DataFrame

Rules:
    sparse mode:
        text      : none | bow | tfidf | word2vec
        keyword   : none | onehot
        location  : none | onehot

    dataframe mode:
        text      : none
        keyword   : none
        location  : none

`cat_features` / `text_features` are not handled here — they live in
model.params and are read by the model itself.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from scipy.sparse import csr_matrix, hstack

from src.features.categorical import build_categorical_encoder
from src.features.vectorizer import build_text_vectorizer


VALID_BUILDER_OUTPUTS = ("sparse", "dataframe")
BLOCKS = ("text", "keyword", "location")
DEFAULT_COLUMNS = {"text": "text_pp", "keyword": "keyword", "location": "location"}
FORBIDDEN_IN_DATAFRAME = {
    "text": {"tfidf", "bow", "word2vec"},
    "keyword": {"onehot"},
    "location": {"onehot"},
}


class FeatureBuilder:
    """Fit on train, transform any split. Stateless between calls to fit()."""

    def __init__(self, features_cfg: dict[str, Any]) -> None:
        if not isinstance(features_cfg, dict):
            raise TypeError(
                f"features_cfg must be dict, got {type(features_cfg).__name__}"
            )
        output = features_cfg.get("builder_output", "sparse")
        if output not in VALID_BUILDER_OUTPUTS:
            raise ValueError(
                f"Unknown builder_output: {output!r}. "
                f"Expected {VALID_BUILDER_OUTPUTS}"
            )
        self.cfg = features_cfg
        self.output = output
        self.transformers_: dict[str, Any] = {}
        self.blocks_: list[str] = []
        self.feature_names_: list[str] = []
        self._fitted = False

    # --- public API ---

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        if self.output == "sparse":
            self._fit_sparse(df)
        else:
            self._fit_dataframe(df)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame):
        if not self._fitted:
            raise RuntimeError("FeatureBuilder is not fitted")
        if self.output == "sparse":
            return self._transform_sparse(df)
        return self._transform_dataframe(df)

    def fit_transform(self, df: pd.DataFrame):
        return self.fit(df).transform(df)

    # --- helpers ---

    def _block_cfg(self, block: str) -> dict[str, Any]:
        return self.cfg.get(block, {}) or {}

    def _block_column(self, block: str) -> str:
        return self._block_cfg(block).get("column", DEFAULT_COLUMNS[block])

    # --- sparse ---

    def _fit_sparse(self, df: pd.DataFrame) -> None:
        for block in BLOCKS:
            cfg = self._block_cfg(block)
            btype = cfg.get("type", "none")
            if btype == "none":
                raise ValueError(
                    f"builder_output='sparse' does not support "
                    f"features.{block}.type='none'. "
                    f"Use builder_output='dataframe' to pass raw columns."
                )
            col = self._block_column(block)
            if block == "text":
                vec = build_text_vectorizer(cfg)
            else:
                vec = build_categorical_encoder(cfg)
            vec.fit(df[col])
            self.transformers_[block] = vec
            self.blocks_.append(block)

    def _transform_sparse(self, df: pd.DataFrame) -> csr_matrix:
        parts = []
        for block in self.blocks_:
            col = self._block_column(block)
            parts.append(self.transformers_[block].transform(df[col]).tocsr())
        return hstack(parts).tocsr()

    # --- dataframe ---

    def _fit_dataframe(self, df: pd.DataFrame) -> None:
        for block in BLOCKS:
            cfg = self._block_cfg(block)
            btype = cfg.get("type", "none")
            if btype in FORBIDDEN_IN_DATAFRAME[block]:
                raise ValueError(
                    f"builder_output='dataframe' does not support "
                    f"features.{block}.type={btype!r}. "
                    f"Use builder_output='sparse' for encoding, or set "
                    f"type='none' and let the model handle the raw column."
                )
            self.blocks_.append(block)
        self.feature_names_ = [
            self._block_column(block) for block in self.blocks_
        ]

    def _transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for block in self.blocks_:
            col = self._block_column(block)
            out[col] = df[col].values
        return out.reset_index(drop=True)

    # --- introspection ---

    @property
    def n_features(self) -> int:
        if not self._fitted:
            raise RuntimeError("FeatureBuilder is not fitted")
        if self.output == "sparse":
            # sum of feature counts from fitted transformers
            total = 0
            for block in self.blocks_:
                tr = self.transformers_[block]
                if hasattr(tr, "n_features"):
                    total += tr.n_features
                elif hasattr(tr, "get_feature_names_out"):
                    total += len(tr.get_feature_names_out())
            return total
        return len(self.feature_names_)