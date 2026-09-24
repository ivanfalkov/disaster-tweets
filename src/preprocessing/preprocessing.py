"""Preprocessing logic: from a single tweet to a full dataframe.

This module is pure: it does NOT read any config files.
It accepts a `pp_cfg` dict (the `preprocess` section of an experiment config).
If pp_cfg is None, DEFAULT_PREPROCESS is used as-is (no merging).

Layers (bottom-up):
    - filtering()             : regex-only cleanup (url, mentions, hashtags)
    - preprocessing_text()    : full single-string pipeline
    - duplicates_processing() : dedup before split
    - preprocess_dataframe()  : full dataframe pipeline

Duplicates handling lives here because it must happen BEFORE the split.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer


DEFAULT_PREPROCESS: dict[str, Any] = {
    "filtering": {
        "lower": True,
        "remove_url": True,
        "remove_mentions": True,
        "remove_hashtags_symbol": True,
        "keep_only_alpha": True,
        "min_token_len": 2,
    },
    "stopwords": {
        "enabled": False,
        "language": "english",
    },
    "normalization": "none",
    "duplicates": {
        "mode": "drop_conflict",
        "text_col": "text",
        "target_col": "target",
    },
    "columns": {
        "text": "text",
        "out": "text_pp",
    },
    "drop_empty": True,
}


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")
WORD_PATTERN = re.compile(r"^[a-z]+$")


@lru_cache(maxsize=4)
def _stopwords(language: str) -> frozenset[str]:
    return frozenset(stopwords.words(language))


@lru_cache(maxsize=1)
def _stemmer() -> PorterStemmer:
    return PorterStemmer()


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _tokenizer() -> TweetTokenizer:
    return TweetTokenizer(
        preserve_case=False,
        strip_handles=True,
        reduce_len=True,
    )


def filtering(
    text: str,
    *,
    remove_url: bool = True,
    remove_mentions: bool = True,
    remove_hashtags_symbol: bool = True,
) -> str:
    if remove_url:
        text = URL_PATTERN.sub(" ", text)
    if remove_mentions:
        text = MENTION_PATTERN.sub(" ", text)
    if remove_hashtags_symbol:
        text = HASHTAG_PATTERN.sub(r"\1", text)
    return text


def preprocessing_text(
    text: str,
    *,
    lower: bool = True,
    do_filtering: bool = True,
    remove_url: bool = True,
    remove_mentions: bool = True,
    remove_hashtags_symbol: bool = True,
    keep_only_alpha: bool = True,
    min_token_len: int = 2,
    remove_stopwords: bool = False,
    stopwords_language: str = "english",
    normalization: str = "none",
) -> str:
    """Full preprocessing for a single tweet. All steps switchable."""
    if not isinstance(text, str):
        return ""

    if lower:
        text = text.lower()

    if do_filtering:
        text = filtering(
            text,
            remove_url=remove_url,
            remove_mentions=remove_mentions,
            remove_hashtags_symbol=remove_hashtags_symbol,
        )

    tokens = _tokenizer().tokenize(text)

    if keep_only_alpha:
        tokens = [t for t in tokens if WORD_PATTERN.fullmatch(t)]
    else:
        tokens = [t for t in tokens if t.strip()]

    if min_token_len > 1:
        tokens = [t for t in tokens if len(t) >= min_token_len]

    if remove_stopwords:
        sw = _stopwords(stopwords_language)
        tokens = [t for t in tokens if t not in sw]

    if normalization == "lemma":
        lm = _lemmatizer()
        tokens = [lm.lemmatize(t) for t in tokens]
    elif normalization == "stem":
        st = _stemmer()
        tokens = [st.stem(t) for t in tokens]
    elif normalization != "none":
        raise ValueError(f"Unknown normalization: {normalization!r}")

    return " ".join(tokens)


def _preprocessing_text_from_cfg(text: str, cfg: dict[str, Any]) -> str:
    f = cfg["filtering"]
    sw = cfg["stopwords"]
    return preprocessing_text(
        text,
        lower=f["lower"],
        do_filtering=True,
        remove_url=f["remove_url"],
        remove_mentions=f["remove_mentions"],
        remove_hashtags_symbol=f["remove_hashtags_symbol"],
        keep_only_alpha=f["keep_only_alpha"],
        min_token_len=f["min_token_len"],
        remove_stopwords=sw["enabled"],
        stopwords_language=sw["language"],
        normalization=cfg["normalization"],
    )


def duplicates_processing(
    df: pd.DataFrame,
    *,
    pp_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Handle duplicates before split.

    mode:
        "off"           — return as is
        "drop_exact"    — drop exact duplicates by text (keep first)
        "drop_conflict" — drop rows where the same text has different targets,
                          then drop remaining exact duplicates
    """
    cfg = pp_cfg or DEFAULT_PREPROCESS
    dup = cfg["duplicates"]
    mode = dup["mode"]
    text_col = dup["text_col"]
    target_col = dup["target_col"]

    df = df.copy()

    if mode == "off":
        return df.reset_index(drop=True)

    if mode == "drop_exact":
        before = len(df)
        df = df.drop_duplicates(subset=text_col, keep="first").reset_index(drop=True)
        print(f"[preproc] drop_exact: {before} -> {len(df)}")
        return df

    if mode == "drop_conflict":
        before = len(df)
        conflict_mask = df.groupby(text_col)[target_col].transform("nunique") > 1
        df = df[~conflict_mask]
        df = df.drop_duplicates(subset=text_col, keep="first").reset_index(drop=True)
        print(f"[preproc] drop_conflict: {before} -> {len(df)}")
        return df

    raise ValueError(f"Unknown duplicates mode: {mode!r}")


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    pp_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Full dataframe pipeline before split:

        1) duplicates_processing
        2) build `columns.out` from `columns.text` via preprocessing
        3) optionally drop rows where the preprocessed text is empty

    If pp_cfg is None, DEFAULT_PREPROCESS is used as-is (no merging).
    Returns a new dataframe. The input is not mutated.
    """
    cfg = pp_cfg or DEFAULT_PREPROCESS

    text_col = cfg["columns"]["text"]
    out_col = cfg["columns"]["out"]

    df = duplicates_processing(df, pp_cfg=cfg)
    df[out_col] = df[text_col].apply(lambda t: _preprocessing_text_from_cfg(t, cfg))

    if cfg.get("drop_empty", True):
        before = len(df)
        df = df[df[out_col].str.len() > 0].reset_index(drop=True)
        if before != len(df):
            print(f"[preproc] dropped {before - len(df)} empty-after-preprocess rows")

    return df