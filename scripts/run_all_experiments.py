"""Run all experiments defined in configs/experiments/.

For each YAML, resolves the matching runner based on the subdirectory name
and executes it as a subprocess. Supports filtering by block and by name
pattern, and skipping experiments whose artifacts already exist.

Usage:
    uv run python scripts/run_all_experiments.py
    uv run python scripts/run_all_experiments.py --block classic_nlp
    uv run python scripts/run_all_experiments.py --pattern "tfidf.*logreg"
    uv run python scripts/run_all_experiments.py --skip-existing
    uv run python scripts/run_all_experiments.py --dry-run
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = PROJECT_ROOT / "configs" / "experiments"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RUNNERS = {
    "classic_nlp": PROJECT_ROOT / "experiments" / "classic_nlp" / "run.py",
    "embeddings":  PROJECT_ROOT / "experiments" / "embeddings" / "run.py",
    "finetune":    PROJECT_ROOT / "experiments" / "finetune" / "run.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all experiments from configs.")
    parser.add_argument(
        "--block",
        choices=list(RUNNERS.keys()),
        default=None,
        help="Only run experiments from this block.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Regex to filter config file names (e.g. 'tfidf.*logreg').",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip experiments whose artifacts dir already contains metrics.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed, do not run.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with the next experiment if one fails.",
    )
    return parser.parse_args()


def discover_configs(block: str | None, pattern: str | None) -> list[tuple[str, Path]]:
    """Return list of (block_name, config_path) matching the filters."""
    blocks = [block] if block else list(RUNNERS.keys())
    regex = re.compile(pattern) if pattern else None
    found: list[tuple[str, Path]] = []

    for block_name in blocks:
        block_dir = CONFIGS_DIR / block_name
        if not block_dir.exists():
            continue
        for cfg_path in sorted(block_dir.glob("*.yaml")):
            if regex and not regex.search(cfg_path.stem):
                continue
            found.append((block_name, cfg_path))
    return found


def exp_name_from_config(cfg_path: Path) -> str:
    """Read `name` from YAML; fall back to file stem."""
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("name") or cfg_path.stem


def already_done(exp_name: str) -> bool:
    return (ARTIFACTS_DIR / exp_name / "metrics.json").exists()


def run_experiment(block: str, cfg_path: Path, dry_run: bool) -> int:
    runner = RUNNERS[block]
    cmd = [sys.executable, str(runner), "--config", str(cfg_path)]
    pretty = " ".join(cmd)
    print(f"\n{'=' * 70}")
    print(f"[run_all] block={block}  config={cfg_path.name}")
    print(f"[run_all] cmd: {pretty}")
    print(f"{'=' * 70}")

    if dry_run:
        return 0

    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main() -> None:
    args = parse_args()

    configs = discover_configs(args.block, args.pattern)
    if not configs:
        print("[run_all] no configs found for the given filters")
        return

    print(f"[run_all] found {len(configs)} config(s):")
    for block, cfg_path in configs:
        print(f"  - {block:12s} {cfg_path.name}")

    succeeded: list[str] = []
    skipped: list[str] = []
    failed: list[tuple[str, int]] = []

    for block, cfg_path in configs:
        exp_name = exp_name_from_config(cfg_path)

        if args.skip_existing and already_done(exp_name):
            print(f"\n[run_all] skip {exp_name}: metrics.json already exists")
            skipped.append(exp_name)
            continue

        code = run_experiment(block, cfg_path, args.dry_run)
        if code == 0:
            succeeded.append(exp_name)
        else:
            failed.append((exp_name, code))
            print(f"[run_all] FAILED {exp_name} (exit={code})")
            if not args.continue_on_error:
                print("[run_all] stopping. Use --continue-on-error to ignore failures.")
                break

    print(f"\n{'=' * 70}")
    print("[run_all] summary")
    print(f"  succeeded: {len(succeeded)}")
    for name in succeeded:
        print(f"    + {name}")
    if skipped:
        print(f"  skipped: {len(skipped)}")
        for name in skipped:
            print(f"    - {name}")
    if failed:
        print(f"  failed: {len(failed)}")
        for name, code in failed:
            print(f"    x {name} (exit={code})")
    print(f"{'=' * 70}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()