"""Text-level preprocessing helpers (T6 time-token enrichment).

Goal: prepend an explicit, normalized time hint to the input text so the
model sees a strong T2 (verification_timeline) signal that survives
tokenization. The transform is **deterministic** and applied to train +
val identically (not stochastic data augmentation).

Pattern coverage (audited on vpesg4k_train_1000 V1.csv):
  西元 4 位 (1900-2099): 47.1%
  中文「YYYY 年」      : 45.9%  (largely overlaps with above)
  民國 NN(N) 年        :  1.4%  (民國年 + 1911 = AD)
  中文「YYYY 年」 + 月  :  7.4%

Strategy: extract every year reference, take max() as the commitment
target year, compute delta vs CURRENT_YEAR, bucket into the 4 active T2
classes, and emit a Chinese-friendly prefix.
"""
from __future__ import annotations

import re
from typing import Callable

# Competition baseline year; vpesg4k corpus is sourced from 2024 reports
# but verification rubric is anchored to the 2025 release window.
CURRENT_YEAR = 2025

# Compiled regexes
_RE_AD_YEAR = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)")
_RE_AD_YEAR_CH = re.compile(r"(19\d{2}|20\d{2})\s*年")
_RE_ROC_YEAR = re.compile(r"民國\s*(\d{2,3})")


def _extract_years(text: str) -> list[int]:
    years: list[int] = []
    for m in _RE_AD_YEAR.finditer(text):
        y = int(m.group(1))
        if 1990 <= y <= 2099:
            years.append(y)
    for m in _RE_AD_YEAR_CH.finditer(text):
        y = int(m.group(1))
        if 1990 <= y <= 2099:
            years.append(y)
    for m in _RE_ROC_YEAR.finditer(text):
        years.append(int(m.group(1)) + 1911)
    return years


def _bucket(delta: int) -> str:
    if delta <= 0:
        return "already"
    if delta <= 2:
        return "within_2_years"
    if delta <= 5:
        return "between_2_and_5_years"
    return "longer_than_5_years"


def add_time_tokens(text: str) -> str:
    """Prepend a normalized time hint string. Idempotent on no-match."""
    if not text:
        return text
    years = _extract_years(text)
    if not years:
        return text
    target = max(years)
    delta = target - CURRENT_YEAR
    bucket = _bucket(delta)
    sign = "後" if delta >= 0 else "前"
    prefix = f"[時間 年份{target} 距今{abs(delta)}年{sign} {bucket}] "
    return prefix + text


# T6 v2 — dedicated bucket tokens (must be added to tokenizer vocab so each
# survives tokenization as a single id with its own learnable embedding).
BUCKET_TOKENS: dict[str, str] = {
    "already":               "[T_already]",
    "within_2_years":        "[T_within2]",
    "between_2_and_5_years": "[T_2to5]",
    "longer_than_5_years":   "[T_longer5]",
    "N/A":                   "[T_NA]",
}
ADDED_TOKENS_TIME_BUCKET: list[str] = list(BUCKET_TOKENS.values())


def add_time_bucket_token(text: str) -> str:
    """Prepend ONE dedicated bucket token to the input.

    Unlike add_time_tokens (which uses a multi-character Chinese prefix that
    macbert tokenizer fragments into 30+ sub-tokens), this transform emits a
    single added-vocab token like '[T_already] '. The token MUST be registered
    via tokenizer.add_tokens(ADDED_TOKENS_TIME_BUCKET) and the encoder embedding
    layer resized accordingly. Otherwise the bracketed string degenerates to
    sub-tokens and provides no benefit over T6 v1.
    """
    if not text:
        return text
    years = _extract_years(text)
    if not years:
        return BUCKET_TOKENS["N/A"] + " " + text
    target = max(years)
    delta = target - CURRENT_YEAR
    bucket = _bucket(delta)
    return BUCKET_TOKENS[bucket] + " " + text


# Registry used by train_kfold to look up named transforms from yaml config.
TEXT_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "time_tokens":       add_time_tokens,        # T6 v1 (Phase 10)
    "time_bucket_token": add_time_bucket_token,  # T6 v2 (Phase 11)
}


# Map transform name → list of vocab tokens that must be added before training.
TRANSFORM_ADDED_TOKENS: dict[str, list[str]] = {
    "time_bucket_token": ADDED_TOKENS_TIME_BUCKET,
}


def get_added_tokens(name: str | None) -> list[str]:
    if not name:
        return []
    return list(TRANSFORM_ADDED_TOKENS.get(name, []))


def get_text_transform(name: str | None) -> Callable[[str], str] | None:
    if not name:
        return None
    if name not in TEXT_TRANSFORMS:
        raise ValueError(f"Unknown text_transform: {name!r}. Known: {list(TEXT_TRANSFORMS)}")
    return TEXT_TRANSFORMS[name]
