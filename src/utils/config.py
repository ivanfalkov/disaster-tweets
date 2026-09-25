"""Config loading utilities.

All paths in YAML configs are resolved relative to the project root,
so configs work regardless of cwd.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]   # src/utils/config.py -> src/utils -> src -> project_root


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config. Relative paths resolved against project root."""
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


def resolve_path(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)
    