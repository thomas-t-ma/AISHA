"""Resolve only harmless, single-character quotation punctuation differences.

Always return the exact substring of the recorded USER message, not a model-
reconstructed quote. Never fuzzy-match words, spelling, case, or whitespace.
"""
from __future__ import annotations

QUOTE_PUNCTUATION = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u02bc": "'",
    "\u201c": '"', "\u201d": '"',
})


def original_source_quote(user_text: str, proposed: str) -> str | None:
    if not isinstance(proposed, str) or not proposed.strip():
        return None
    if proposed in user_text:
        return proposed
    normalized_user = user_text.translate(QUOTE_PUNCTUATION)
    normalized_quote = proposed.translate(QUOTE_PUNCTUATION)
    offset = normalized_user.find(normalized_quote)
    if offset < 0:
        return None
    # Translation is strictly one-codepoint-to-one-codepoint; offsets are safe.
    exact = user_text[offset:offset + len(proposed)]
    return exact if exact.strip() else None
