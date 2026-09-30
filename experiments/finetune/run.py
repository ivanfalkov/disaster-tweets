"""End-to-end runner for fine-tuning a HuggingFace transformer.

Pipeline:
    setup_experiment -> prepare_data -> save_data_artifacts
    -> build FinetuneModel
    -> tokenize train/val/test into datasets.Dataset
    -> transformers.Trainer with early stopping on val F1
    -> evaluate_and_save on train/test (val used for early stopping)
    -> save_model (save_pretrained) + save tokenizer + save_config
    -> finish_experiment

Usage:
    uv run python experiments/finetune/run.py \
        --config configs/experiments/finetune/roberta_base.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import shutil
import numpy as np
from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from src.experiments.artifacts import evaluate_and_save, save_config
from src.experiments.runner import (
    finish_experiment,
    prepare_data,
    save_data_artifacts,
    setup_experiment,
)
from src.metrics.classification import compute_metrics
from src.models.factory import build_model
from src.utils.clearml_utils import log_artifact
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fine-tune experiment.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _to_dataset(df, tokenizer, text_col: str, target_col: str, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(df[[text_col, target_col]].rename(columns={target_col: "labels"}))
    ds = ds.map(
        lambda batch: tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
            padding=False,
        ),
        batched=True,
    )
    ds = ds.remove_columns([text_col])
    ds.set_format("torch")
    return ds


def _make_compute_metrics(metric_names: list[str]):
    """Wrap our compute_metrics into the HF EvalPrediction interface."""

    def _fn(eval_pred: EvalPrediction) -> dict[str, float]:
        logits = eval_pred.predictions
        labels = eval_pred.label_ids

        y_pred = np.argmax(logits, axis=-1)
        # softmax for probabilities of class 1
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        proba = exp / exp.sum(axis=-1, keepdims=True)
        y_score = proba[:, 1]

        return compute_metrics(labels, y_pred, y_score, metrics=metric_names)

    return _fn


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    task, artifacts_dir = setup_experiment(cfg, args.config)

    tr, va, te = prepare_data(cfg)
    save_data_artifacts(tr, va, te, artifacts_dir, task)

    text_col = cfg["data"]["text_col"]
    target_col = cfg["data"]["target_col"]
    max_length = cfg["model"]["params"].get("max_length", 64)
    metric_names = cfg.get("metrics") or ["accuracy", "f1", "roc_auc"]

    # --- model ---
    model = build_model(cfg["model"])
    print(
        f"[finetune] model={model.model_name}, "
        f"device={model.device}, max_length={max_length}"
    )

    # --- tokenize ---
    print("[finetune] tokenizing splits...")
    train_ds = _to_dataset(tr, model.tokenizer, text_col, target_col, max_length)
    val_ds = _to_dataset(va, model.tokenizer, text_col, target_col, max_length)
    test_ds = _to_dataset(te, model.tokenizer, text_col, target_col, max_length)

    # --- training args ---
    train_cfg = dict(cfg["training"])
    early_stop_patience = train_cfg.pop("early_stopping_patience", 1)
    checkpoints_dir = artifacts_dir / train_cfg.get("output_dir", "checkpoints")
    if checkpoints_dir.exists():
        print(f"[finetune] removing old checkpoints: {checkpoints_dir}")
        shutil.rmtree(checkpoints_dir, ignore_errors=True)
    train_cfg["output_dir"] = str(checkpoints_dir)

    training_args = TrainingArguments(**train_cfg)

    # --- trainer ---
    trainer = Trainer(
        model=model.model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer=model.tokenizer),
        compute_metrics=_make_compute_metrics(metric_names),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stop_patience)],
    )

    print("[finetune] starting training...")
    trainer.train()
    print("[finetune] training finished")

    # best checkpoint is already loaded (load_best_model_at_end=True)

    # --- evaluate on train and test (val was used for early stopping) ---
    splits = {
        "train": (tr, tr[text_col].values),
        "test": (te, te[text_col].values),
    }
    evaluate_and_save(model, splits, cfg, task, artifacts_dir)

    # --- save model + tokenizer ---
    ckpt_dir = artifacts_dir / "model"
    model.save_pretrained(ckpt_dir)
    log_artifact(task, ckpt_dir)
    print(f"[save] model -> {ckpt_dir}")

    save_config(cfg, args.config, artifacts_dir, task)

    finish_experiment(task)


if __name__ == "__main__":
    main()