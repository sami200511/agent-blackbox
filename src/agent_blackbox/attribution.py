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

# Cues, appearing shortly AFTER a document mention, that mean the model set it
# aside. Kept forward-biased and local so a cue about one document doesn't leak
# onto an adjacent mention of another.
IGNORE_CUES = (
    "ignore", "supersed", "outdated", "obsolet", "archiv", "no longer",
    "replaced", "disregard", "don't use", "do not use", "won't use",
    "not use", "old version", "old policy", "deprecated", "not relevant",
    "irrelevant", "skip",
)

# Positive cues that confirm reliance even if an ignore word is nearby (e.g.
# "current.md, which supersedes the old one" — current.md is USED).
USE_CUES = (
    "in effect", "current policy", "use this", "based on", "according to",
    "confirms", "rely on", "relied on", "indicates", "states", "per ",
    "the one in effect", "most recent", "i'll use", "i will use", "apply",
)

_FORWARD = 55   # chars after a mention to scan for cues
_BACK = 0       # forward-only: a cue ("X is superseded") describes the doc it
                # follows, so scanning backwards leaks one doc's cue onto the
                # next doc's mention ("...ignore OLD. current.md says...")


def _variants(doc_id: str) -> list:
    d = doc_id.lower()
    base = d[:-3] if d.endswith(".md") else d
    out = {d, base, base.replace("-", " "), base.replace("_", " ")}
    return sorted(out, key=len, reverse=True)


def documents_referenced(text: str, doc_ids) -> tuple:
    """Return (used, ignored) lists of doc_ids based on how `text` refers to them.

    Each list is ordered by first mention in the text, so `used[0]` is the
    document the model brought up first — its primary source.
    """
    low = (text or "").lower()
    used, ignored = [], []
    for doc_id in doc_ids:
        pos = matched = None
        for v in _variants(doc_id):
            i = low.find(v)
            if i != -1:
                pos, matched = i, v
                break
        if pos is None:
            continue
        window = low[max(0, pos - _BACK): pos + len(matched) + _FORWARD]
        has_ignore = any(cue in window for cue in IGNORE_CUES)
        has_use = any(cue in window for cue in USE_CUES)
        (ignored if has_ignore and not has_use else used).append((pos, doc_id))
    used.sort()
    ignored.sort()
    return [d for _, d in used], [d for _, d in ignored]
