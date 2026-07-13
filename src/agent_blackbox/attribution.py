"""
documents_referenced — work out which retrieved documents a model ACTUALLY
used vs. explicitly ignored, from its own reasoning/answer text.

Retrieval ranking is not the same as what the model relied on: two near-
identical documents can tie in score, so the "top" chunk may be the outdated
one the model correctly discarded. This reads the model's own words instead,
so the decision's `choice` reflects reality.

    used, ignored = documents_referenced(reasoning_plus_answer, doc_ids)

A document mentioned next to an ignore-cue ("superseded", "outdated",
"I'll ignore that", ...) is classified ignored; a plainly-mentioned one is
used; a document never mentioned is in neither list. Pure stdlib, no API call.
"""

from __future__ import annotations

import re
from collections import Counter

# Phrases that clearly mean the model set a document aside. Deliberately narrow
# and unambiguous: "superseded" (passive) not the "supersed" stem, so "X
# supersedes Y" does not flag X; no generic words like "old policy" that name a
# document rather than discard it.
IGNORE_CUES = (
    "superseded", "outdated", "obsolete", "no longer valid", "no longer applies",
    "ignore", "disregard", "deprecated", "not relevant", "archived",
    "old version", "don't use", "do not use",
)

# Strong reliance phrases — the model is clearly using the document, not merely
# naming it. Excludes ambiguous doc-references like "current policy" or "in
# effect" (those describe WHICH document, and leak across clauses).
USE_CUES = (
    "based on", "according to", "confirms", "rely on", "relied on", "per ",
    "use this", "i'll use", "i will use", "as stated in", "citing",
)

# Fallback span (chars) when the mention's sentence has no trailing terminator.
_FORWARD = 120


# Filename tokens too generic or ambiguous to identify a document on their own
# ("the old policy", "the current one" could mean either refund policy).
_TOKEN_STOP = {
    "old", "new", "current", "latest", "all", "one", "doc", "docs", "file",
    "files", "policy", "policies", "data", "info", "page", "note", "notes",
    "the", "and", "for", "refund", "return", "version", "v1", "v2", "v3",
}


def _variants(doc_id: str) -> list:
    d = doc_id.lower()
    base = d[:-3] if d.endswith(".md") else d
    out = {d, base, base.replace("-", " "), base.replace("_", " ")}
    return sorted(out, key=len, reverse=True)


def _filename_tokens(doc_id: str) -> list:
    base = doc_id.lower()
    base = base[:-3] if base.endswith(".md") else base
    return [t for t in re.split(r"[^a-z0-9]+", base) if t]


def _distinctive_tokens(doc_ids) -> list:
    """Per-doc set of filename tokens unique to that doc (so 'faq' -> the FAQ,
    but not the shared/ambiguous 'policy' or 'refund')."""
    token_sets = [set(_filename_tokens(d)) for d in doc_ids]
    df: Counter = Counter(t for s in token_sets for t in s)
    return [{t for t in s if df[t] == 1 and len(t) >= 3 and t not in _TOKEN_STOP}
            for s in token_sets]


def classify_references(text: str, doc_ids) -> dict:
    """Classify how `text` refers to each document.

    Returns {doc_id: status} for documents that are mentioned, where status is:
      "endorsed"  — mentioned with a use-cue ("the FAQ confirms this",
                    "based on the current policy") → the model relied on it
      "ignored"   — mentioned with an ignore-cue ("superseded", "outdated")
      "mentioned" — named/keyword-matched but neither endorsed nor discarded
                    (discussed, but not necessarily used)
    Documents not referred to at all are absent from the dict. Recognition is by
    filename (exact/spaced) or a distinctive filename keyword, and cues are read
    only within the mention's own sentence (no cross-sentence leak).
    """
    low = (text or "").lower()
    doc_ids = list(doc_ids)
    distinctive = _distinctive_tokens(doc_ids)
    result: dict = {}
    for doc_id, tokens in zip(doc_ids, distinctive):
        pos = matched = None
        for v in _variants(doc_id):           # exact / spaced filename first
            i = low.find(v)
            if i != -1:
                pos, matched = i, v
                break
        if pos is None:                       # then a distinctive keyword
            for tok in sorted(tokens, key=len, reverse=True):
                m = re.search(rf"\b{re.escape(tok)}\b", low)
                if m:
                    pos, matched = m.start(), tok
                    break
        if pos is None:
            continue
        end_rel = pos + len(matched)
        prev = max(low.rfind(c, 0, pos) for c in ".!?")
        nxt = re.search(r"[.!?]", low[end_rel:])
        sentence = low[prev + 1: end_rel + (nxt.start() if nxt else _FORWARD)]
        has_ignore = any(cue in sentence for cue in IGNORE_CUES)
        has_use = any(cue in sentence for cue in USE_CUES)
        if has_use:
            result[doc_id] = "endorsed"
        elif has_ignore:
            result[doc_id] = "ignored"
        else:
            result[doc_id] = "mentioned"
    return result


def documents_referenced(text: str, doc_ids) -> tuple:
    """Return (used, ignored) lists of doc_ids based on how `text` refers to them.

    `used` = documents endorsed or plainly mentioned (not discarded); `ignored`
    = documents named with an ignore-cue. Ordered by first mention. For the
    finer endorsed/mentioned distinction, use `classify_references`.
    """
    low = (text or "").lower()
    cls = classify_references(text, doc_ids)
    ordered = sorted(cls, key=lambda d: min(
        [low.find(v) for v in _variants(d) if low.find(v) != -1] or [10 ** 9]))
    used = [d for d in ordered if cls[d] in ("endorsed", "mentioned")]
    ignored = [d for d in ordered if cls[d] == "ignored"]
    return used, ignored
