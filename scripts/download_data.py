"""Download a Kaggle competition dataset.

Token is read from .env (KAGGLE_API_TOKEN, or KAGGLE_USERNAME + KAGGLE_KEY).

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --competition nlp-getting-started
    uv run python scripts/download_data.py --out data/raw --force
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_COMPETITION = "nlp-getting-started"
# Файлы, которые ожидаем увидеть после распаковки для дефолтного соревнования.
EXPECTED_FILES = ("train.csv", "test.csv", "sample_submission.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Kaggle competition dataset.")
    parser.add_argument(
        "--competition",
        type=str,
        default=DEFAULT_COMPETITION,
        help=f"Kaggle competition slug (default: {DEFAULT_COMPETITION}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw"),
        help="Directory to store raw data (default: data/raw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist.",
    )
    return parser.parse_args()


def load_kaggle_env() -> None:
    """Load .env and validate that some Kaggle credentials exist."""
    load_dotenv()  # ищет .env начиная с cwd и выше

    token = os.getenv("KAGGLE_API_TOKEN")
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not token and not (username and key):
        sys.exit(
            "[download] Kaggle credentials not found.\n"
            "  Put into .env one of:\n"
            "    KAGGLE_API_TOKEN=KGAT_...\n"
            "  or legacy pair:\n"
            "    KAGGLE_USERNAME=...\n"
            "    KAGGLE_KEY=...\n"
            "  Token page: https://www.kaggle.com/settings -> API -> Create New Token"
        )

    # kaggle CLI принимает KAGGLE_API_TOKEN напрямую из окружения.
    # Если задан legacy-формат — он тоже поддерживается CLI.
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token
    if username:
        os.environ["KAGGLE_USERNAME"] = username
    if key:
        os.environ["KAGGLE_KEY"] = key


def ensure_kaggle_cli() -> None:
    if shutil.which("kaggle") is None:
        sys.exit(
            "[download] kaggle CLI not found.\n"
            "  Install it: uv add kaggle"
        )


def already_downloaded(out_dir: Path, competition: str) -> bool:
    # Для дефолтного соревнования проверяем конкретные файлы,
    # для остальных — просто наличие хоть одного csv/zip.
    if competition == DEFAULT_COMPETITION:
        return all((out_dir / f).exists() for f in EXPECTED_FILES)
    return any(out_dir.glob("*.csv")) or any(out_dir.glob("*.zip"))


def clean_output_dir(out_dir: Path, competition: str) -> None:
    if competition == DEFAULT_COMPETITION:
        for f in EXPECTED_FILES:
            p = out_dir / f
            if p.exists():
                print(f"[download] removing old {p}")
                p.unlink()
    for archive in out_dir.glob("*.zip"):
        print(f"[download] removing old {archive}")
        archive.unlink()


def download(out_dir: Path, competition: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "competitions",
        "download",
        "-c",
        competition,
        "-p",
        str(out_dir),
    ]
    print(f"[download] running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        sys.exit(
            f"[download] kaggle CLI failed with exit code {e.returncode}.\n"
            f"  Check: (1) you accepted the competition rules on kaggle.com, "
            f"(2) token is valid, (3) competition slug '{competition}' is correct."
        )

    archives = list(out_dir.glob("*.zip"))
    if not archives:
        print("[download] no .zip found (kaggle might have downloaded raw files).")
    for archive in archives:
        print(f"[download] unzipping {archive.name}")
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(out_dir)
        archive.unlink()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    print(f"[download] competition: {args.competition}")
    print(f"[download] target dir:  {out_dir.resolve()}")

    if already_downloaded(out_dir, args.competition) and not args.force:
        print(
            f"[download] files already exist in {out_dir}.\n"
            f"[download] nothing to do. Use --force to re-download."
        )
        return

    load_kaggle_env()
    ensure_kaggle_cli()

    if args.force:
        clean_output_dir(out_dir, args.competition)

    download(out_dir, args.competition)

    print(f"[download] done. Contents of {out_dir.resolve()}:")
    for p in sorted(out_dir.iterdir()):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()