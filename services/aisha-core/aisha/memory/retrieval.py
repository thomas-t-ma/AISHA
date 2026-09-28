from __future__ import annotations

import re
from collections import Counter
from math import log

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "she", "that", "the", "their",
    "them", "they", "this", "to", "user", "was", "were", "will", "with",
    "you", "your",
}

# These words are too conversationally common to justify recalling a personal
# memory on their own. Keep this separate from STOPWORDS so diagnostics can
# distinguish structural stop words from low-information content words.
LOW_INFORMATION_RECALL_TOKENS = {
    "about", "again", "away", "current", "day", "enjoy", "feel", "get",
    "getting", "help", "helping", "just", "little", "lot", "make", "more",
    "much", "really", "something", "supposed", "think", "thinking", "thing",
    "tired", "want", "wish",
}


def _tokens(text: str) -> list[str]:
    raw = [token.lower() for token in TOKEN_RE.findall(text.replace("_", " "))]
    normalized: list[str] = []
    for token in raw:
        if token in STOPWORDS or len(token) < 3:
            continue
        # Precision-first normalization: only collapse simple plurals. Earlier
        # suffix chopping turned "something" -> "someth" and "getting" -> "gett",
        # creating misleading overlaps between unrelated memories.
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        normalized.append(token)
    return normalized


def _belief_tokens(belief: dict) -> tuple[set[str], set[str], set[str]]:
    topic = set(_tokens(str(belief.get("topic_key", ""))))
    text = set(_tokens(str(belief.get("text", ""))))
    question = set(_tokens(str(belief.get("open_question") or "")))
    return topic, text, question


def explain_relevant_beliefs(
    user_text: str,
    beliefs: list[dict],
    *,
    limit: int = 4,
) -> list[dict]:
    """Return selected beliefs plus transparent lexical match diagnostics."""
    query = set(_tokens(user_text))
    if not query:
        return []

    verified = [b for b in beliefs if b.get("evidence_status") == "verified"]
    if not verified:
        return []

    docs: list[tuple[dict, set[str], set[str], set[str]]] = []
    df: Counter[str] = Counter()
    for belief in verified:
        topic, text, question = _belief_tokens(belief)
        all_tokens = topic | text | question
        for token in all_tokens:
            df[token] += 1
        docs.append((belief, topic, text, question))

    n_docs = max(1, len(docs))
    ranked: list[tuple[float, str, dict]] = []
    for belief, topic, text, question in docs:
        overlap = query & (topic | text | question)
        informative_overlap = overlap - LOW_INFORMATION_RECALL_TOKENS
        if not informative_overlap:
            continue

        score = 0.0
        for token in informative_overlap:
            rarity = 1.0 + log((n_docs + 1) / (df[token] + 1))
            if token in topic:
                score += 2.5 * rarity
            if token in text:
                score += 1.0 * rarity
            if token in question:
                score += 1.25 * rarity

        if len(informative_overlap) >= 2:
            score += 1.5

        ranked.append((
            score,
            str(belief.get("updated_at", "")),
            {
                "belief": belief,
                "method": "lexical",
                "score": round(score, 4),
                "matched_tokens": sorted(informative_overlap),
                "ignored_low_information_tokens": sorted(
                    overlap & LOW_INFORMATION_RECALL_TOKENS
                ),
            },
        ))

    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [detail for _score, _updated, detail in ranked[:max(1, limit)]]


def select_relevant_beliefs(
    user_text: str,
    beliefs: list[dict],
    *,
    limit: int = 4,
) -> list[dict]:
    """Return only verified beliefs with lexical evidence of current relevance."""
    return [
        detail["belief"]
        for detail in explain_relevant_beliefs(user_text, beliefs, limit=limit)
    ]
