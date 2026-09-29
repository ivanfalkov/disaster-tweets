"""End-to-end runner for classic NLP experiments.

Pipeline:
    setup_experiment -> prepare_data -> save_data_artifacts
    -> FeatureBuilder -> build_model -> fit
    -> evaluate_and_save -> save_model -> save_config -> finish_experiment

Usage:
    uv run python experiments/classic_nlp/run.py \
        --config configs/experiments/classic_nlp/lemma_stop_tfidf_logreg.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.experiments.artifacts import evaluate_and_save, save_config, save_model
from src.experiments.runner import (
    finish_experiment,
    prepare_data,
    save_data_artifacts,
    setup_experiment,
)
from src.features.builder import FeatureBuilder
from src.models.factory import build_model
from src.utils.config import load_config
from src.models.catboost import validate_native_columns

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a classic NLP experiment.")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    task, artifacts_dir = setup_experiment(cfg, args.config)

    tr, va, te = prepare_data(cfg)
    save_data_artifacts(tr, va, te, artifacts_dir, task)

    builder = FeatureBuilder(cfg["features"])
    X_train = builder.fit_transform(tr)
    X_val = builder.transform(va)
    X_test = builder.transform(te)

    print(
        f"[features] output={builder.output}, "
        f"blocks={builder.blocks_}, n_features={builder.n_features}"
    )

    target_col = cfg["data"]["target_col"]
    splits = {
        "train": (tr, X_train),
        "val": (va, X_val),
        "test": (te, X_test),
    }

    model = build_model(cfg["model"])
    if builder.output == "dataframe" and cfg["model"]["type"] == "catboost_native":
        validate_native_columns(X_train, cfg["model"]["params"])
    model.fit(X_train, tr[target_col].values)

    evaluate_and_save(model, splits, cfg, task, artifacts_dir)

    save_model(model, artifacts_dir, task)
    save_config(cfg, args.config, artifacts_dir, task)

    finish_experiment(task)


if __name__ == "__main__":
    main()