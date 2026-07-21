"""Train-serve text cleaning for deployment (no Hazm)."""

from __future__ import annotations

import re
import unicodedata

# URL / mention / hashtag-ish tokens
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")

# Arabic / Persian letters (rough) + common marks
_PERSIAN_ARABIC_RE = re.compile(
    r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]"
)

DEFAULT_MIN_TOKENS = 4


def normalize_orthography(text: str) -> str:
    """Unify common Arabic/Persian orthographic variants."""
    if not text:
        return ""
    text = (
        text.replace("ي", "ی")
        .replace("ك", "ک")
        .replace("ۀ", "ه")
        .replace("\u200c", " ")  # ZWNJ → space for token count stability
        .replace("،", ",")
    )
    return text


def clean_text(text: str, unify_orthography: bool = True) -> str:
    """
    Lightweight clean aligned with stored `clean` column style:
    strip URLs/mentions, normalize whitespace, optional ي/ی ك/ک unify.
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", str(text))
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
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
        # fall back: any non-space chars
        non_space = [c for c in text if not c.isspace()]
        if not non_space:
            return 0.0
        arabic = sum(1 for c in non_space if _PERSIAN_ARABIC_RE.match(c))
        return arabic / len(non_space)
    arabic = sum(1 for c in letters if _PERSIAN_ARABIC_RE.match(c))
    return arabic / len(letters)


def is_usable(
    text: str,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    min_persian_ratio: float = 0.3,
) -> bool:
    """Return True if text is long enough and sufficiently Persian/Arabic-script."""
    cleaned = clean_text(text) if text else ""
    if token_count(cleaned) < min_tokens:
        return False
    if persian_char_ratio(cleaned) < min_persian_ratio:
        return False
    return True
