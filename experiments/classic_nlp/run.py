"""End-to-end runner for classic NLP experiments.

Pipeline:
    load_raw -> preprocess_dataframe -> stratified_split
    -> build_features -> build_classic_model -> fit / predict
    -> compute_metrics -> save artifacts -> log to ClearML

Usage:
    uv run python experiments/classic_nlp/run.py \
        --config configs/experiments/classic_nlp/lemma_stop_tfidf_logreg.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.loader import load_raw
from src.data.split import stratified_split
from src.features.builder import build_features
from src.metrics.classification import compute_metrics
from src.models.classic import build_classic_model, validate_sparse_input
from src.preprocessing.preprocessing import preprocess_dataframe
from src.utils.clearml_utils import (
    finish,
    init_task,
    log_artifact,
    log_dict_as_json,
    log_metrics,
    log_params,
)
from src.utils.config import load_config, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a classic NLP experiment.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _get_scores(model, X) -> np.ndarray | None:
    """Extract positive-class scores; returns None if not possible."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    return None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    exp_name = cfg.get("name") or args.config.stem
    print(f"[run] experiment: {exp_name}")

    task = init_task(cfg)
    log_params(task, cfg)

    artifacts_dir = resolve_path(f"artifacts/{exp_name}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # --- data ---
    train_df, _ = load_raw()
    df = preprocess_dataframe(train_df, pp_cfg=cfg["preprocess"])
    print(f"[run] after preprocess: {df.shape}")

    tr, va, te = stratified_split(
        df,
        target_col=cfg["data"]["target_col"],
        val_size=cfg["split"]["val_size"],
        test_size=cfg["split"]["test_size"],
        seed=cfg["split"]["seed"],
    )
    print(f"[run] splits: train={len(tr)}, val={len(va)}, test={len(te)}")

    # --- features ---
    bundle = build_features(tr, va, te, cfg["features"])
    print(f"[run] features: output={bundle.output}, blocks={bundle.blocks}, n_features={bundle.n_features}")

    if bundle.output != "sparse":
        raise ValueError(
            f"classic_nlp/run.py expects builder_output='sparse', got {bundle.output!r}"
        )

    X_train, X_val, X_test = bundle.X_train, bundle.X_val, bundle.X_test
    y_train = tr[cfg["data"]["target_col"]].values
    y_val = va[cfg["data"]["target_col"]].values
    y_test = te[cfg["data"]["target_col"]].values

    # --- model ---
    model = build_classic_model(cfg["model"])
    validate_sparse_input(X_train, type(model).__name__)
    model.fit(X_train, y_train)

    # --- metrics ---
    metric_names = cfg.get("metrics", ["accuracy", "f1", "roc_auc"])

    val_metrics = compute_metrics(
        y_val, model.predict(X_val), _get_scores(model, X_val), metrics=metric_names,
    )
    test_metrics = compute_metrics(
        y_test, model.predict(X_test), _get_scores(model, X_test), metrics=metric_names,
    )
    train_metrics = compute_metrics(
        y_train, model.predict(X_train), _get_scores(model, X_train), metrics=metric_names,
    )

    print(f"[run] train metrics: {train_metrics}")
    print(f"[run] val   metrics: {val_metrics}")
    print(f"[run] test  metrics: {test_metrics}")

    log_metrics(task, train_metrics, split="train")
    log_metrics(task, val_metrics, split="val")
    log_metrics(task, test_metrics, split="test")

    # --- artifacts ---
    metrics_path = artifacts_dir / "metrics.json"
    log_dict_as_json(task, {"train": train_metrics, "val": val_metrics, "test": test_metrics}, metrics_path)

    preds_path = artifacts_dir / "val_predictions.csv"
    val_pred_df = pd.DataFrame({
        "text": va[cfg["data"]["text_col"]].values,
        "target": y_val,
        "pred": model.predict(X_val),
        "score": _get_scores(model, X_val),
    })
    val_pred_df.to_csv(preds_path, index=False)
    log_artifact(task, preds_path)

    model_path = artifacts_dir / "model.joblib"
    joblib.dump(model, model_path)
    log_artifact(task, model_path)

    # save config copy for reproducibility
    cfg_path = artifacts_dir / "config.yaml"
    cfg_path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    log_artifact(task, cfg_path)

    print(f"[run] artifacts saved to {artifacts_dir}")
    finish(task)


if __name__ == "__main__":
    main()