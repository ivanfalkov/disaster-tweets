"""Embedding extraction with a pretrained sentence encoder."""
from __future__ import annotations

from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingExtractor:
    """Encode texts into dense vectors with a pretrained sentence encoder."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts) -> np.ndarray:
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return emb.astype(np.float32)

    @property
    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


def build_encoder(cfg: dict[str, Any]) -> EmbeddingExtractor:
    return EmbeddingExtractor(
        model_name=cfg["name"],
        device=cfg.get("device", "cpu"),
        normalize=cfg.get("normalize", True),
        batch_size=cfg.get("batch_size", 32),
    )
