"""Experiment artifacts: evaluate, save predictions, model, config.

Contract for any object passed as `model` to `evaluate_and_save`:
    - model.predict(X)        -> np.ndarray[n]
    - model.predict_proba(X)  -> np.ndarray[n, 2]

Works for sklearn-style models and for the future finetune model class.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import joblib
import pandas as pd
import numpy as np

from src.metrics.classification import compute_metrics
from src.utils.clearml_utils import log_artifact, log_dict_as_json, log_metrics


def evaluate_and_save(
    model: Any,
    splits: dict[str, tuple[pd.DataFrame, Any]],
    cfg: dict[str, Any],
    task: Any,
    artifacts_dir: Path,
) -> dict[str, dict[str, float]]:
    """Evaluate on each split, log metrics, save predictions and metrics.json.

    Parameters
    ----------
    model : fitted model with predict / predict_proba
    splits : {"train": (df_split, X_split), "val": (...), "test": (...)}
    cfg : experiment config (uses cfg["metrics"], cfg["data"])
    task : ClearML task or DummyTask
    artifacts_dir : where to save predictions and metrics.json

    Returns
    -------
    dict[str, dict[str, float]] : {split_name: metrics}
    """
    target_col = cfg["data"]["target_col"]
    text_col = cfg["data"]["text_col"]
    metric_names = cfg.get("metrics")  # None -> defaults from general_config

    metrics_payload: dict[str, dict[str, float]] = {}

    for split_name, (df_split, X_split) in splits.items():
        y_true = df_split[target_col].values

        y_pred = model.predict(X_split)
        y_proba = model.predict_proba(X_split)[:, 1]

        metrics = compute_metrics(y_true, y_pred, y_proba, metrics=metric_names)
        metrics_payload[split_name] = metrics
        print(f"[eval] {split_name:5s}: {metrics}")

        log_metrics(task, metrics, split=split_name)

        preds_df = pd.DataFrame(
            {
                "text": df_split[text_col].values,
                "target": y_true,
                "pred": y_pred,
                "proba": y_proba,
            }
        )
        preds_path = artifacts_dir / f"{split_name}_predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        log_artifact(task, preds_path)

    log_dict_as_json(task, metrics_payload, artifacts_dir / "metrics.json")
    return metrics_payload


def save_model(
    model: Any,
    artifacts_dir: Path,
    task: Any,
    *,
    filename: str = "model.joblib",
    save_fn: Callable[[Any, Path], None] | None = None,
) -> None:
    """Save model to artifacts_dir/filename and log it.

    Default: joblib.dump. Provide `save_fn` for models with custom
    serialization (e.g. HF save_pretrained in finetune).
    """
    path = artifacts_dir / filename
    if save_fn is None:
        joblib.dump(model, path)
    else:
        save_fn(model, path)
    log_artifact(task, path)
    print(f"[save] model -> {path}")


def save_config(
    cfg: dict[str, Any],
    config_path: str | Path,
    artifacts_dir: Path,
    task: Any,
) -> None:
    """Copy the original YAML config to artifacts_dir/config.yaml and log it."""
    config_path = Path(config_path)
    dest = artifacts_dir / "config.yaml"

    if config_path.exists():
        dest.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        # fallback: dump the loaded dict as JSON (loses comments, keeps content)
        dest.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")

    log_artifact(task, dest)
    print(f"[save] config -> {dest}")

def save_arrays(
    artifacts_dir: Path,
    task: Any,
    *,
    filename: str,
    **arrays: np.ndarray,
) -> Path:
    """Save named numpy arrays to artifacts_dir/filename (.npz) and log it.

    Example:
        save_arrays(artifacts_dir, task, filename="embeddings.npz",
                    train=X_train, val=X_val, test=X_test)
    """
    path = artifacts_dir / filename
    np.savez(path, **arrays)
    log_artifact(task, path)
    print(f"[save] arrays -> {path}")
    return path
