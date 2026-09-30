"""End-to-end runner for embeddings + classic ML experiments.

Pipeline:
    setup_experiment -> prepare_data -> save_data_artifacts
    -> build_encoder -> encode train/val/test
    -> save embeddings as artifact
    -> build_model (classic head) -> fit on train embeddings
    -> evaluate_and_save -> save_model -> save_config -> finish_experiment

Usage:
    uv run python experiments/embeddings/run.py \
        --config configs/experiments/embeddings/bge_small_logreg.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.artifacts import (
    evaluate_and_save,
    save_arrays,
    save_config,
    save_model,
)
from src.experiments.runner import (
    finish_experiment,
    prepare_data,
    save_data_artifacts,
    setup_experiment,
)
from src.features.embeddings import build_encoder
from src.models.factory import build_model
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an embeddings + classic ML experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    task, artifacts_dir = setup_experiment(cfg, args.config)

    tr, va, te = prepare_data(cfg)
    save_data_artifacts(tr, va, te, artifacts_dir, task)

    text_col = cfg["data"]["text_col"]
    encoder = build_encoder(cfg["encoder"])
    print(f"[encode] encoder: {encoder.model_name} (dim={encoder.dim})")

    X_train = encoder.encode(tr[text_col].values)
    X_val = encoder.encode(va[text_col].values)
    X_test = encoder.encode(te[text_col].values)
    print(
        f"[features] embeddings: "
        f"train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )

    save_arrays(
        artifacts_dir,
        task,
        filename="embeddings.npz",
        train=X_train,
        val=X_val,
        test=X_test,
    )

    target_col = cfg["data"]["target_col"]
    splits = {
        "train": (tr, X_train),
        "val": (va, X_val),
        "test": (te, X_test),
    }

    model = build_model(cfg["model"])
    model.fit(X_train, tr[target_col].values)

    evaluate_and_save(model, splits, cfg, task, artifacts_dir)

    save_model(model, artifacts_dir, task)
    save_config(cfg, args.config, artifacts_dir, task)

    finish_experiment(task)


if __name__ == "__main__":
    main()
