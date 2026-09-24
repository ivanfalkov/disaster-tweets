import nltk
import pytest


REQUIRED_NLTK = [
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet",   "wordnet"),
    ("corpora/omw-1.4",   "omw-1.4"),
]


@pytest.fixture(scope="session", autouse=True)
def _ensure_nltk_data():
    missing = []
    for find_path, pkg in REQUIRED_NLTK:
        try:
            nltk.data.find(find_path)
        except LookupError:
            missing.append(pkg)

    if not missing:
        return

    print(f"\n[conftest] downloading NLTK resources: {missing}")
    ok = nltk.download(missing, quiet=False)
    if not ok:
        pytest.exit(f"Failed to download NLTK resources: {missing}")

    for find_path, pkg in REQUIRED_NLTK:
        try:
            nltk.data.find(find_path)
        except LookupError:
            pytest.exit(f"NLTK resource still missing after download: {pkg}")