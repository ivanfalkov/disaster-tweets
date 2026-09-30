"""Fine-tune wrapper around HuggingFace sequence classification models.

FinetuneModel is NOT an nn.Module. It holds an internal AutoModel and delegates
forward/parameters/train/eval/to to it, so transformers.Trainer works with it
unchanged. The wrapper adds a sklearn-like predict / predict_proba interface
(numpy in, numpy out) that our evaluate_and_save expects.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class FinetuneModel:
    """Wrapper around AutoModelForSequenceClassification + AutoTokenizer."""

    def __init__(
        self,
        model_name: str,
        num_labels: int = 2,
        max_length: int = 64,
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.num_labels = num_labels
        self.max_length = max_length
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.model.to(self.device)

    # --- delegation to internal model (for Trainer) ---

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self.model.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self.model.named_parameters(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.device = device
        self.model.to(device)
        return self

    @property
    def config(self):
        return self.model.config

    # --- inference (sklearn-like, numpy in / numpy out) ---

    def _tokenize(self, texts: Iterable[str]):
        return self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )

    def _forward_logits(self, texts: Iterable[str]) -> torch.Tensor:
        self.model.eval()
        inputs = self._tokenize(texts).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        logits = self._forward_logits(texts)
        return logits.argmax(dim=-1).cpu().numpy()

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        logits = self._forward_logits(texts)
        proba = torch.softmax(logits, dim=-1).cpu().numpy()
        return proba

    # --- persistence ---

    def save_pretrained(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(cls, path: str | Path, device: str = "cuda") -> FinetuneModel:
        path = Path(path)
        obj = cls.__new__(cls)
        obj.model_name = str(path)
        obj.num_labels = 2
        obj.max_length = 64
        obj.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        obj.model.to(obj.device)
        return obj


def build_finetune_model(cfg: dict[str, Any]) -> FinetuneModel:
    """Build FinetuneModel from the `model.params` section of an experiment config.

    Expected keys: model_name, num_labels, max_length, device.
    """
    return FinetuneModel(
        model_name=cfg["model_name"],
        num_labels=cfg.get("num_labels", 2),
        max_length=cfg.get("max_length", 64),
        device=cfg.get("device", "cuda"),
    )
