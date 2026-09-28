"""End-to-end runner for classic NLP experiments.

Pipeline:
    load_raw -> preprocess_dataframe -> stratified_split
    -> build_features -> build_classic_model -> fit
    -> per-split metrics + predictions -> save artifacts -> log to ClearML

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
from src.features.builder import FeatureBuilder
from src.metrics.classification import compute_metrics
from src.models.classic import validate_sparse_input
from src.models.factory import build_model
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


def _positive_scores(model, X) -> np.ndarray | None:
    """Return positive-class scores, or None if the model exposes neither API."""
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
    builder = FeatureBuilder(cfg["features"])
    X_train = builder.fit_transform(tr)
    X_val = builder.transform(va)
    X_test = builder.transform(te)

    if builder.output != "sparse":
        raise ValueError(
            f"classic_nlp/run.py expects builder_output='sparse', "
            f"got {builder.output!r}"
        )

    print(
        f"[run] features: output={builder.output}, "
        f"blocks={builder.blocks_}, n_features={builder.n_features}"
    )

    target_col = cfg["data"]["target_col"]
    text_col = cfg["data"]["text_col"]
    splits = {
        "train": (tr, X_train),
        "val": (va, X_val),
        "test": (te, X_test),
    }

    model = build_model(cfg["model"])
    validate_sparse_input(X_train, type(model).__name__)
    model.fit(X_train, tr[target_col].values)

    # --- per-split evaluation + predictions ---
    metric_names = cfg.get("metrics", ["accuracy", "f1", "roc_auc"])
    metrics_payload: dict[str, dict] = {}

    for split_name, (df_split, X_split) in splits.items():
        y_split = df_split[target_col].values

        y_pred = model.predict(X_split)
        y_score = _positive_scores(model, X_split)
        metrics = compute_metrics(y_split, y_pred, y_score, metrics=metric_names)

        metrics_payload[split_name] = metrics
        print(f"[run] {split_name:5s} metrics: {metrics}")
        log_metrics(task, metrics, split=split_name)

        preds_df = pd.DataFrame(
            {
                "text": df_split[text_col].values,
                "target": y_split,
                "pred": y_pred,
                "score": y_score,
            }
        )
        preds_path = artifacts_dir / f"{split_name}_predictions.csv"
        preds_df.to_csv(preds_path, index=False)
        log_artifact(task, preds_path)

    log_dict_as_json(task, metrics_payload, artifacts_dir / "metrics.json")

    # --- model artifact ---
    model_path = artifacts_dir / "model.joblib"
    joblib.dump(model, model_path)
    log_artifact(task, model_path)

    # --- config snapshot for reproducibility ---
    cfg_path = artifacts_dir / "config.yaml"
    cfg_path.write_text(json.dumps(cfg, indent=2, default=str), encoding="utf-8")
    log_artifact(task, cfg_path)

    print(f"[run] artifacts saved to {artifacts_dir}")
    finish(task)


if __name__ == "__main__":
    main()