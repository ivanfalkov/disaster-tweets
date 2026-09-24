"""Tests for src/preprocessing/preprocessing.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.preprocessing import (
    DEFAULT_PREPROCESS,
    duplicates_processing,
    filtering,
    preprocess_dataframe,
    preprocess_series,
    preprocessing_text,
)


class TestFiltering:
    def test_removes_url(self):
        out = filtering("check https://t.co/abc now", remove_url=True)
        assert "https" not in out
        assert "t.co" not in out
        assert "check" in out and "now" in out

    def test_keeps_url_when_disabled(self):
        out = filtering("check https://t.co/abc now", remove_url=False)
        assert "https" in out

    def test_removes_mention(self):
        out = filtering("hi @user how are you", remove_mentions=True)
        assert "@user" not in out
        assert "hi" in out and "how" in out

    def test_keeps_mention_when_disabled(self):
        out = filtering("hi @user", remove_mentions=False)
        assert "@user" in out

    def test_strips_hashtag_symbol(self):
        out = filtering("big #fire today", remove_hashtags_symbol=True)
        assert "#" not in out
        assert "fire" in out

    def test_keeps_hashtag_symbol_when_disabled(self):
        out = filtering("big #fire today", remove_hashtags_symbol=False)
        assert "#fire" in out


class TestPreprocessingText:
    def test_non_string_returns_empty(self):
        assert preprocessing_text(None) == ""
        assert preprocessing_text(123) == ""
        assert preprocessing_text(float("nan")) == ""

    def test_lower(self):
        out = preprocessing_text("FIRE In The HOLE", lower=True)
        assert out == "fire in the hole"

    def test_no_lower(self):
        out = preprocessing_text("FIRE In The HOLE", lower=False, keep_only_alpha=True)
        assert "In" not in out
        assert "HOLE" not in out

    def test_keeps_only_alpha(self):
        out = preprocessing_text("hello world 123 !!!", keep_only_alpha=True)
        assert out == "hello world"

    def test_min_token_len_drops_short(self):
        out = preprocessing_text("a bb ccc dddd", min_token_len=3)
        assert out == "ccc ddd"

    def test_min_token_len_one_keeps_all(self):
        out = preprocessing_text("a bb ccc", min_token_len=1)
        assert out == "a bb ccc"

    def test_remove_stopwords(self):
        out = preprocessing_text("the cat is on the mat", remove_stopwords=True)
        for sw in ["the", "is", "on"]:
            assert sw not in out.split()
        assert "cat" in out and "mat" in out

    def test_keeps_stopwords_when_disabled(self):
        out = preprocessing_text("the cat is on the mat", remove_stopwords=False)
        assert "the" in out.split()

    def test_normalization_none(self):
        out = preprocessing_text("running dogs", normalization="none")
        assert "running" in out and "dogs" in out

    def test_normalization_lemma(self):
        out = preprocessing_text("running dogs", normalization="lemma")
        assert "run" in out and "dog" in out

    def test_normalization_stem(self):
        out = preprocessing_text("running dogs", normalization="stem")
        assert "run" in out and "dog" in out

    def test_unknown_normalization_raises(self):
        with pytest.raises(ValueError, match="Unknown normalization"):
            preprocessing_text("hello", normalization="foo")

    def test_full_pipeline(self):
        text = "Check this! https://t.co/x @user #fire running dogs 123"
        out = preprocessing_text(text, remove_stopwords=True, normalization="lemma")
        assert "https" not in out
        assert "@user" not in out
        assert "#" not in out
        assert "check" in out
        assert "fire" in out
        assert "run" in out
        assert "dog" in out
        assert "123" not in out

    def test_empty_after_filtering(self):
        assert preprocessing_text("!!! 123 @user #") == ""


class TestPreprocessSeries:
    def test_returns_series_same_length(self):
        s = pd.Series(["hello world", "fire!", None, 123])
        out = preprocess_series(s)
        assert isinstance(out, pd.Series)
        assert len(out) == len(s)

    def test_applies_pp_cfg(self):
        s = pd.Series(["Running dogs", "the cat"])
        out = preprocess_series(
            s,
            pp_cfg={"normalization": "lemma", "stopwords": {"enabled": True}},
        )
        assert out.iloc[0] == "run dog"
        assert "the" not in out.iloc[1].split()

    def test_none_cfg_uses_defaults(self):
        s = pd.Series(["FIRE!"])
        out = preprocess_series(s, pp_cfg=None)
        assert out.iloc[0] == "fire"

    def test_partial_cfg_merges_with_defaults(self):
        s = pd.Series(["FIRE running"])
        out = preprocess_series(s, pp_cfg={"normalization": "lemma"})
        assert "fire" in out.iloc[0]
        assert "run" in out.iloc[0]


class TestDuplicatesProcessing:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "text": ["a", "a", "b", "b", "c", "c", "d"],
            "target": [1, 1, 0, 1, 1, 1, 0],
        })

    def test_mode_off_keeps_all(self, df):
        out = duplicates_processing(df, pp_cfg={"duplicates": {"mode": "off"}})
        assert len(out) == len(df)

    def test_drop_exact_keeps_first(self, df):
        out = duplicates_processing(df, pp_cfg={"duplicates": {"mode": "drop_exact"}})
        assert len(out) == 4
        assert out.loc[out["text"] == "b", "target"].iloc[0] == 0

    def test_drop_conflict_removes_conflicting_rows(self, df):
        out = duplicates_processing(df, pp_cfg={"duplicates": {"mode": "drop_conflict"}})
        assert "b" not in out["text"].values
        assert len(out) == 3

    def test_unknown_mode_raises(self, df):
        with pytest.raises(ValueError, match="Unknown duplicates mode"):
            duplicates_processing(df, pp_cfg={"duplicates": {"mode": "nope"}})

    def test_does_not_mutate_input(self, df):
        snapshot = df.copy()
        duplicates_processing(df, pp_cfg={"duplicates": {"mode": "drop_exact"}})
        pd.testing.assert_frame_equal(df, snapshot)

    def test_custom_column_names(self):
        df = pd.DataFrame({"t": ["a", "a"], "y": [0, 1]})
        out = duplicates_processing(
            df,
            pp_cfg={"duplicates": {"mode": "drop_conflict", "text_col": "t", "target_col": "y"}},
        )
        assert len(out) == 0


class TestPreprocessDataframe:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "text": ["Fire!!!", "fire", "@user hello", "the running dogs", ""],
            "target": [1, 1, 0, 1, 0],
        })

    def test_adds_text_pp_column(self, df):
        out = preprocess_dataframe(df)
        assert "text_pp" in out.columns
        assert "text" in out.columns
        assert len(out["text_pp"]) == len(out)

    def test_keeps_original_columns(self, df):
        out = preprocess_dataframe(df)
        for col in df.columns:
            assert col in out.columns

    def test_dedup_before_pp_removes_conflicts(self, df):
        out = preprocess_dataframe(df, pp_cfg={"duplicates": {"mode": "drop_exact"}})
        assert "text_pp" in out.columns
        assert len(out) == 3

    def test_drop_empty(self, df):
        out = preprocess_dataframe(df)
        assert (out["text_pp"].str.len() > 0).all()

    def test_drop_empty_disabled(self, df):
        out = preprocess_dataframe(df, pp_cfg={"drop_empty": False})
        assert (out["text_pp"] == "").any()

    def test_does_not_mutate_input(self, df):
        snapshot = df.copy()
        preprocess_dataframe(df)
        pd.testing.assert_frame_equal(df, snapshot)

    def test_custom_columns(self):
        df = pd.DataFrame({"t": ["Fire!", "fire"], "y": [1, 1]})
        out = preprocess_dataframe(
            df,
            pp_cfg={
                "columns": {"text": "t", "out": "clean"},
                "duplicates": {"mode": "drop_exact", "text_col": "t", "target_col": "y"},
            },
        )
        assert "clean" in out.columns
        assert "text_pp" not in out.columns

    def test_default_cfg_when_none(self, df):
        out = preprocess_dataframe(df, pp_cfg=None)
        assert "text_pp" in out.columns


class TestDefaultPreprocess:
    def test_has_expected_keys(self):
        for key in ["filtering", "stopwords", "normalization", "duplicates", "columns"]:
            assert key in DEFAULT_PREPROCESS

    def test_default_normalization_is_none(self):
        assert DEFAULT_PREPROCESS["normalization"] == "none"

    def test_default_stopwords_disabled(self):
        assert DEFAULT_PREPROCESS["stopwords"]["enabled"] is False

    def test_default_dedup_is_drop_conflict(self):
        assert DEFAULT_PREPROCESS["duplicates"]["mode"] == "drop_conflict"