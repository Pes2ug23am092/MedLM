from __future__ import annotations

import re
import unicodedata

# Zero-width / invisible formatting characters commonly used to hide payloads.
_ZERO_WIDTH = re.compile(
    "[\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064"
    "\ufeff\u00ad\u180e\u202a-\u202e\u2066-\u2069]"
)


def normalize_text(text: str) -> str:
    """NFKC-normalize, then strip zero-width characters."""
    text = unicodedata.normalize("NFKC", text)
    return _ZERO_WIDTH.sub("", text)
