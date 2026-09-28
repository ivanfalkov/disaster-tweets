"""Default preprocessing configuration.

Used when `pp_cfg=None` is passed to preprocessing functions.
Not merged with user config — either use defaults entirely, or provide
a full pp_cfg.
"""
from __future__ import annotations

from typing import Any


DEFAULT_PREPROCESS: dict[str, Any] = {
    "filtering": {
        "lower": True,
        "remove_url": True,
        "remove_mentions": True,
        "remove_hashtags_symbol": True,
        "keep_only_alpha": True,
        "min_token_len": 2,
    },
    "tokenizer": {
        "type": "tweet",
        "params": {
            "preserve_case": False,
            "strip_handles": True,
            "reduce_len": True,
        },
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