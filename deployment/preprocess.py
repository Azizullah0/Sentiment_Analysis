"""Train-serve text cleaning for deployment (no Hazm).

Pipeline: strip URLs/mentions/emojis → orthography unify → usability gate
(Persian/Dari script). Non-usable comments are Excluded before the model runs.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional, Tuple

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")

_EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U0001FA00-\U0001FAFF"
    "\U00002700-\U000027BF"
    "\U00002600-\U000026FF"
    "\U0000FE00-\U0000FE0F"
    "\U0001F1E0-\U0001F1FF"
    "\U0000200D"
    "]+",
    flags=re.UNICODE,
)

_PERSIAN_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

DEFAULT_MIN_TOKENS = 4
DEFAULT_MIN_PERSIAN_RATIO = 0.5


def strip_emojis(text: str) -> str:
    if not text:
        return ""
    return _EMOJI_RE.sub(" ", text)


def normalize_orthography(text: str) -> str:
    if not text:
        return ""
    text = (
        text.replace("ي", "ی")
        .replace("ك", "ک")
        .replace("ۀ", "ه")
        .replace("\u200c", " ")
        .replace("،", ",")
    )
    return text


def clean_text(text: str, unify_orthography: bool = True) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = strip_emojis(text)
    if unify_orthography:
        text = normalize_orthography(text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def token_count(text: str) -> int:
    if not text:
        return 0
    return len(text.split())


def persian_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = [c for c in text if c.isalpha() or _PERSIAN_ARABIC_RE.match(c)]
    if not letters:
        non_space = [c for c in text if not c.isspace()]
        if not non_space:
            return 0.0
        arabic = sum(1 for c in non_space if _PERSIAN_ARABIC_RE.match(c))
        return arabic / len(non_space)
    arabic = sum(1 for c in letters if _PERSIAN_ARABIC_RE.match(c))
    return arabic / len(letters)


def exclusion_reason(
    original: str,
    cleaned: Optional[str] = None,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    min_persian_ratio: float = DEFAULT_MIN_PERSIAN_RATIO,
) -> Optional[str]:
    """Return reason if Excluded, else None: empty, emoji_only, too_short, non_persian."""
    raw = (original or "").strip()
    cleaned = clean_text(original) if cleaned is None else cleaned

    if not raw:
        return "empty"
    if not cleaned:
        return "emoji_only"
    if token_count(cleaned) < min_tokens:
        return "too_short"
    if persian_char_ratio(cleaned) < min_persian_ratio:
        return "non_persian"
    return None


def is_usable(
    text: str,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    min_persian_ratio: float = DEFAULT_MIN_PERSIAN_RATIO,
) -> bool:
    cleaned = clean_text(text) if text else ""
    return exclusion_reason(text, cleaned, min_tokens, min_persian_ratio) is None


def classify_text(
    text: str,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    min_persian_ratio: float = DEFAULT_MIN_PERSIAN_RATIO,
) -> Tuple[str, bool, Optional[str]]:
    """Return (cleaned, usable, exclusion_reason)."""
    cleaned = clean_text(text)
    reason = exclusion_reason(text, cleaned, min_tokens, min_persian_ratio)
    return cleaned, reason is None, reason
