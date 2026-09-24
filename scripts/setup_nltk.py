"""Download required NLTK resources. Run once after cloning the repo."""
import nltk


REQUIRED = [
    "stopwords",
    "wordnet",
    "omw-1.4",
]


def main() -> None:
    for pkg in REQUIRED:
        print(f"[nltk] downloading {pkg}...")
        ok = nltk.download(pkg, quiet=False)
        if not ok:
            raise SystemExit(f"[nltk] failed to download {pkg}")
    print("[nltk] done")


if __name__ == "__main__":
    main()