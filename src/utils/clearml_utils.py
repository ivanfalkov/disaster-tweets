"""ClearML logging helpers.

All logging goes through a Task object:
    - real clearml.Task when ClearML is installed and not disabled,
    - DummyTask otherwise (no-op).

Disable via env: CLEARML_OFFLINE=1
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from clearml import Task as _ClearMLTask
    _HAS_CLEARML = True
except ImportError:
    _ClearMLTask = None
    _HAS_CLEARML = False


class _DummyLogger:
    def report_scalar(self, *args, **kwargs) -> None:
        pass

    def flush(self) -> None:
        pass


class DummyTask:
    """No-op replacement when ClearML is unavailable or disabled."""

    def __init__(self) -> None:
        self.id = "dummy"
        self.logger = _DummyLogger()

    def connect(self, *args, **kwargs) -> None:
        pass

    def set_tags(self, *args, **kwargs) -> None:
        pass

    def upload_artifact(self, *args, **kwargs) -> None:
        pass


def _disabled() -> bool:
    return os.environ.get("CLEARML_OFFLINE", "").lower() in ("1", "true", "yes")


def init_task(cfg: dict[str, Any]) -> Any:
    """Create a ClearML task from cfg['clearml'], or DummyTask if disabled."""
    if _disabled() or not _HAS_CLEARML:
        print("[clearml] disabled or not installed -> DummyTask")
        return DummyTask()

    cm = cfg.get("clearml", {}) or {}
    task = _ClearMLTask.init(
        project_name=cm.get("project", "disaster-tweets"),
        task_name=cm.get("task", "experiment"),
        auto_connect_arg_parser=False,
        auto_connect_frameworks=True,
    )
    if cm.get("tags"):
        task.set_tags(cm["tags"])
    return task


def log_params(task: Any, params: dict[str, Any], *, name: str | None = None) -> None:
    payload = {name: params} if name else params
    task.connect(payload, name=name)


def log_metrics(task: Any, metrics: dict[str, float], *, split: str = "val", step: int = 0) -> None:
    for k, v in metrics.items():
        task.logger.report_scalar(title=split, series=k, value=float(v), iteration=step)


def log_artifact(task: Any, path: str | Path, *, name: str | None = None) -> None:
    path = Path(path)
    if not path.exists():
        print(f"[clearml] artifact not found, skipping: {path}")
        return
    task.upload_artifact(name=name or path.name, artifact_object=str(path))


def log_dict_as_json(task: Any, data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    log_artifact(task, path)


def finish(task: Any) -> None:
    task.logger.flush()