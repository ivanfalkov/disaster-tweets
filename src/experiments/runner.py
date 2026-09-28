"""Experiment lifecycle helpers.

Used by experiments/*/run.py to avoid duplicating boilerplate:
    - setup_experiment   : init ClearML task, create artifacts dir, log config
    - prepare_data       : load_raw + preprocess + split
    - save_data_artifacts: dump train/val/test CSVs and log them
    - finish_experiment  : flush ClearML
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.loader import load_raw
from src.data.split import stratified_split
from src.preprocessing.preprocessing import preprocess_dataframe
from src.utils.clearml_utils import finish, init_task, log_artifact, log_params
from src.utils.config import resolve_path


def setup_experiment(
    cfg: dict[str, Any],
    config_path: str | Path,
) -> tuple[Any, Path]:
    """Create ClearML task, artifacts dir, log config.

    Returns (task, artifacts_dir).
    """
    config_path = Path(config_path)
    exp_name = cfg.get("name") or config_path.stem

    task = init_task(cfg)
    log_params(task, cfg)

    artifacts_dir = resolve_path(f"artifacts/{exp_name}")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"[setup] experiment: {exp_name}")
    print(f"[setup] artifacts dir: {artifacts_dir}")
    return task, artifacts_dir


def prepare_data(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """load_raw -> preprocess_dataframe -> stratified_split.

    Reads cfg['preprocess'], cfg['split'], cfg['data'].
    Returns (train, val, test).
    """
    train_df, _ = load_raw()
    df = preprocess_dataframe(train_df, pp_cfg=cfg["preprocess"])
    print(f"[data] after preprocess: {df.shape}")

    target_col = cfg["data"]["target_col"]
    tr, va, te = stratified_split(
        df,
        target_col=target_col,
        val_size=cfg["split"]["val_size"],
        test_size=cfg["split"]["test_size"],
        seed=cfg["split"]["seed"],
    )
    print(f"[data] splits: train={len(tr)}, val={len(va)}, test={len(te)}")
    print(
        f"[data] target rate: "
        f"train={tr[target_col].mean():.4f}, "
        f"val={va[target_col].mean():.4f}, "
        f"test={te[target_col].mean():.4f}"
    )
    return tr, va, te


def save_data_artifacts(
    tr: pd.DataFrame,
    va: pd.DataFrame,
    te: pd.DataFrame,
    artifacts_dir: Path,
    task: Any,
) -> None:
    """Save train/val/test CSVs to artifacts_dir/data and log them."""
    data_dir = artifacts_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for split_name, df_split in {"train": tr, "val": va, "test": te}.items():
        split_path = data_dir / f"{split_name}.csv"
        df_split.to_csv(split_path, index=False)
        log_artifact(task, split_path)

    print(f"[data] saved splits to {data_dir}")


def finish_experiment(task: Any) -> None:
    """Flush ClearML task."""
    finish(task)