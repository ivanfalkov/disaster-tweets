"""Text vectorizers: bow, tfidf, word2vec, none.

Each vectorizer exposes sklearn-like `fit_transform` / `transform`,
so the builder stays agnostic of the underlying implementation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

try:
    from gensim.models import Word2Vec
    _HAS_GENSIM = True
except ImportError:
    _HAS_GENSIM = False


class GensimWord2VecVectorizer:
    """Average-pooled Word2Vec embeddings with sklearn-like API.

    Expects input texts to be pre-tokenized whitespace-separated strings
    (i.e. output of `preprocessing_text` / `text_pp` column).
    Splits on whitespace internally via `str.split()`.
    """

    def __init__(
        self,
        *,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        sg: int = 1,
        epochs: int = 10,
        seed: int = 42,
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.epochs = epochs
        self.seed = seed
        self._model: Word2Vec | None = None

    def _tokenize(self, text: str) -> list[str]:
        if not isinstance(text, str):
            return []
        return text.split()

    def fit(self, texts: pd.Series) -> "GensimWord2VecVectorizer":
        if not _HAS_GENSIM:
            raise ImportError(
                "gensim is not installed. Run: uv add gensim"
            )
        sentences = [self._tokenize(t) for t in texts]
        self._model = Word2Vec(
            sentences=sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=self.sg,
            epochs=self.epochs,
            seed=self.seed,
        )
        return self

    def _embed(self, text: str) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("GensimWord2VecVectorizer is not fitted")
        tokens = self._tokenize(text)
        if not tokens:
            return np.zeros(self.vector_size, dtype=np.float32)
        vecs = []
        for tok in tokens:
            if tok in self._model.wv:
                vecs.append(self._model.wv[tok])
        if not vecs:
            return np.zeros(self.vector_size, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    def transform(self, texts: pd.Series) -> csr_matrix:
        matrix = np.vstack([self._embed(t) for t in texts])
        return csr_matrix(matrix)

    def fit_transform(self, texts: pd.Series) -> csr_matrix:
        return self.fit(texts).transform(texts)

    @property
    def n_features(self) -> int:
        return self.vector_size

    @property
    def model(self) -> Word2Vec:
        if self._model is None:
            raise RuntimeError("GensimWord2VecVectorizer is not fitted")
        return self._model


def build_text_vectorizer(cfg: dict[str, Any]):
    """Return an unfitted vectorizer, or None if type == 'none'.

    Supported types: none, bow, tfidf, word2vec.
    """
    vtype = cfg.get("type", "none")
    params = cfg.get("params", {}) or {}

    if vtype == "none":
        return None
    if vtype == "bow":
        return CountVectorizer(**params)
    if vtype == "tfidf":
        return TfidfVectorizer(**params)
    if vtype == "word2vec":
        return GensimWord2VecVectorizer(**params)
    raise ValueError(f"Unknown text vectorizer type: {vtype!r}")