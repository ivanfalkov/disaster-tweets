"""Preprocessing logic: from a single tweet to a full dataframe.

This module is pure except for regex patterns, which are read once from
configs/general_config.yaml (section `patterns`) and cached.

If pp_cfg is None, DEFAULT_PREPROCESS is used as-is (no merging).

Tokenizer modes:
    "tweet" : TweetTokenizer with configurable params
    "none"  : no tokenization at all. Intended for finetune pipelines where
              a HF tokenizer will handle the raw text later.
              In this mode, text-level transforms (stopwords, lemmatization,
              stemming, keep_only_alpha, min_token_len) must be disabled,
              otherwise a ValueError is raised.

Layers (bottom-up):
    - filtering()             : regex-only cleanup (url, mentions, hashtags)
    - tokenize()              : text -> list[str], switchable via cfg
    - preprocessing_text()    : full single-string pipeline
    - duplicates_processing() : dedup before split
    - preprocess_dataframe()  : full dataframe pipeline
"""
from __future__ import annotations

import re
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import TweetTokenizer

from src.preprocessing.defaults import DEFAULT_PREPROCESS
from src.utils.config import load_config

DEFAULT_general_config = "configs/general_config.yaml"


@lru_cache(maxsize=4)
def _patterns(config_path: str) -> dict[str, re.Pattern]:
    """Load and compile regex patterns from general_config.yaml."""
    cfg = load_config(config_path)
    p = cfg.get("patterns", {})
    missing = [k for k in ("url", "mention", "hashtag", "word") if k not in p]
    if missing:
        raise KeyError(
            f"general_config.yaml is missing patterns: {missing}. "
            f"Add a `patterns` section with keys url, mention, hashtag, word."
        )
    return {k: re.compile(p[k]) for k in ("url", "mention", "hashtag", "word")}


@lru_cache(maxsize=4)
def _stopwords(language: str) -> frozenset[str]:
    return frozenset(stopwords.words(language))


@lru_cache(maxsize=1)
def _stemmer() -> PorterStemmer:
    return PorterStemmer()


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    return WordNetLemmatizer()


def _tokenizer_tweet_factory(
    preserve_case: bool,
    strip_handles: bool,
    reduce_len: bool,
) -> Callable[[str], list[str]]:
    tok = TweetTokenizer(
        preserve_case=preserve_case,
        strip_handles=strip_handles,
        reduce_len=reduce_len,
    )
    return tok.tokenize


@lru_cache(maxsize=8)
def _build_tokenizer(
    tokenizer_type: str,
    params_key: tuple[tuple[str, Any], ...],
) -> Callable[[str], list[str]] | None:
    """Return tokenization callable, or None if type == 'none'."""
    if tokenizer_type == "none":
        return None
    if tokenizer_type == "tweet":
        params = dict(params_key)
        return _tokenizer_tweet_factory(
            preserve_case=params.get("preserve_case", False),
            strip_handles=params.get("strip_handles", True),
            reduce_len=params.get("reduce_len", True),
        )
    raise ValueError(f"Unknown tokenizer type: {tokenizer_type!r}")


def _freeze_params(params: dict[str, Any] | None) -> tuple[tuple[str, Any], ...]:
    if not params:
        return ()
    return tuple(sorted(params.items()))


def tokenize(
    text: str,
    *,
    tokenizer_type: str = "tweet",
    tokenizer_params: dict[str, Any] | None = None,
) -> list[str]:
    """Split text into tokens. Raises if tokenizer_type == 'none'."""
    fn = _build_tokenizer(tokenizer_type, _freeze_params(tokenizer_params))
    if fn is None:
        raise ValueError(
            "tokenize() called with tokenizer_type='none'. "
            "Callers must check for 'none' before invoking tokenize()."
        )
    return fn(text)


def filtering(
    text: str,
    *,
    remove_url: bool = True,
    remove_mentions: bool = True,
    remove_hashtags_symbol: bool = True,
    config_path: str = DEFAULT_general_config,
) -> str:
    pat = _patterns(config_path)
    if remove_url:
        text = pat["url"].sub(" ", text)
    if remove_mentions:
        text = pat["mention"].sub(" ", text)
    if remove_hashtags_symbol:
        text = pat["hashtag"].sub(r"\1", text)
    return text


def _validate_none_tokenizer_cfg(
    *,
    tokenizer_type: str,
    keep_only_alpha: bool,
    min_token_len: int,
    remove_stopwords: bool,
    normalization: str,
) -> None:
    if tokenizer_type != "none":
        return
    problems = []
    if remove_stopwords:
        problems.append("remove_stopwords=True")
    if normalization != "none":
        problems.append(f"normalization={normalization!r}")
    if keep_only_alpha:
        problems.append("keep_only_alpha=True")
    if min_token_len > 1:
        problems.append(f"min_token_len={min_token_len}")
    if problems:
        raise ValueError(
            "tokenizer.type='none' disables tokenization, so text-level "
            "transforms are not applicable. Disable them in the config: "
            + ", ".join(problems)
        )


def preprocessing_text(
    text: str,
    *,
    lower: bool = True,
    do_filtering: bool = True,
    remove_url: bool = True,
    remove_mentions: bool = True,
    remove_hashtags_symbol: bool = True,
    tokenizer_type: str = "tweet",
    tokenizer_params: dict[str, Any] | None = None,
    keep_only_alpha: bool = True,
    min_token_len: int = 2,
    remove_stopwords: bool = False,
    stopwords_language: str = "english",
    normalization: str = "none",
    config_path: str = DEFAULT_general_config,
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
            config_path=config_path,
        )

    if tokenizer_type == "none":
        _validate_none_tokenizer_cfg(
            tokenizer_type=tokenizer_type,
            keep_only_alpha=keep_only_alpha,
            min_token_len=min_token_len,
            remove_stopwords=remove_stopwords,
            normalization=normalization,
        )
        return text.strip()

    tokens = tokenize(
        text,
        tokenizer_type=tokenizer_type,
        tokenizer_params=tokenizer_params,
    )

    pat = _patterns(config_path)
    if keep_only_alpha:
        tokens = [t for t in tokens if pat["word"].fullmatch(t)]
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
    tok = cfg.get("tokenizer", {}) or {}
    return preprocessing_text(
        text,
        lower=f["lower"],
        do_filtering=True,
        remove_url=f["remove_url"],
        remove_mentions=f["remove_mentions"],
        remove_hashtags_symbol=f["remove_hashtags_symbol"],
        tokenizer_type=tok.get("type", "tweet"),
        tokenizer_params=tok.get("params"),
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
