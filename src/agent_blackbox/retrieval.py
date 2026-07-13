"""
Small, dependency-free information-retrieval utilities:

  * bm25(query, docs)          — principled relevance scores (replaces naive
                                 term-frequency counting, which ties documents
                                 that share vocabulary).
  * distinctive_terms(docs)    — each document's high-TF-IDF terms (what makes
                                 it different from the others).
  * grounding_scores(text, docs) — how much of each document's DISTINCTIVE
                                 content appears in `text` (e.g. the model's
                                 answer). This is the strongest signal for
                                 which document the answer is actually built on,
                                 and it works for any model without the model
                                 having to name the file.
  * attribute(text, docs)      — combine grounding with explicit ignore-cues to
                                 label each retrieved doc used / ignored / unused.

Pure stdlib. No API calls.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .attribution import classify_references

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "and", "for", "are", "was", "with", "you", "your", "have", "this",
    "that", "from", "will", "can", "our", "it", "its", "a", "an", "of", "to",
    "in", "on", "is", "be", "or", "we", "us", "if", "as", "at", "by", "not",
    "may", "any", "all", "but", "they", "them", "their", "were", "has", "had",
}


def tokenize(text: str) -> list:
    return [t for t in _TOKEN.findall((text or "").lower())
            if t not in _STOP and len(t) > 1]


def _idf(docs_tokens: list) -> dict:
    n_docs = len(docs_tokens)
    df: Counter = Counter()
    for toks in docs_tokens:
        for t in set(toks):
            df[t] += 1
    return {t: math.log(1 + (n_docs - n + 0.5) / (n + 0.5)) for t, n in df.items()}


def bm25(query: str, doc_texts: list, k1: float = 1.5, b: float = 0.75) -> list:
    """Okapi BM25 relevance score of `query` against each document."""
    docs_tokens = [tokenize(d) for d in doc_texts]
    if not docs_tokens:
        return []
    idf = _idf(docs_tokens)
    avgdl = sum(len(t) for t in docs_tokens) / len(docs_tokens) or 1.0
    q_terms = tokenize(query)
    scores = []
    for toks in docs_tokens:
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if not f:
                continue
            s += idf.get(term, 0.0) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(round(s, 3))
    return scores


def distinctive_terms(doc_texts: list, top: int = 20) -> list:
    """Per-document set of its most distinctive (high TF-IDF) terms."""
    docs_tokens = [tokenize(d) for d in doc_texts]
    idf = _idf(docs_tokens)
    out = []
    for toks in docs_tokens:
        tf = Counter(toks)
        ranked = sorted(tf, key=lambda t: tf[t] * idf.get(t, 0.0), reverse=True)
        # keep only terms that actually distinguish this doc (idf above the
        # trivial floor — terms present in every doc carry idf ~ 0.4 and below)
        ranked = [t for t in ranked if idf.get(t, 0.0) > 0.6]
        out.append(set(ranked[:top]))
    return out


def grounding_scores(text: str, doc_texts: list, top: int = 20) -> list:
    """Fraction of each document's distinctive terms that appear in `text`."""
    text_tokens = set(tokenize(text))
    scores = []
    for terms in distinctive_terms(doc_texts, top):
        if not terms:
            scores.append(0.0)
        else:
            scores.append(round(len(terms & text_tokens) / len(terms), 3))
    return scores


def attribute(answer: str, docs: list, cue_text: str = None,
              min_ground: float = 0.12) -> tuple:
    """Label each retrieved doc used / ignored / unused from the model's output.

    `docs` is a list of {"doc_id", "text"}. Grounding is measured against
    `answer` (the final output) — NOT the reasoning, because a model quotes the
    documents it rejects while thinking, which would falsely ground them.
    `cue_text` (default: the answer) is scanned for explicit ignore-cues; pass
    the reasoning here so "I'll ignore the old policy" is picked up.

    Returns (statuses, grounding): `statuses` is a list of
    {"doc_id", "grounding", "status"} ordered by grounding (highest first);
    `grounding` maps doc_id -> score.

      used    — the answer reflects the doc's distinctive content (grounded),
                OR the model endorsed it ("the FAQ confirms this")
      ignored — the model named it with an ignore-cue ("superseded")
      unused  — retrieved but not reflected in the answer (incl. bare mentions
                the model discussed but did not rely on)
    """
    ids = [d["doc_id"] for d in docs]
    texts = [d.get("text", "") for d in docs]
    g = grounding_scores(answer, texts)
    gmap = dict(zip(ids, g))
    top = max(g) if g else 0.0
    # A doc is grounded only if its answer-grounding is a clear fraction of the
    # best one — so among near-identical policies, incidental overlap from the
    # superseded twin (which shares most vocabulary) stays below the line while
    # the document the answer really reflects grounds well above it. Documents
    # the model explicitly relied on are still credited via the endorsement path.
    cut = max(min_ground, 0.6 * top)
    cls = classify_references(cue_text if cue_text is not None else answer, ids)

    statuses = []
    for did in ids:
        score = gmap[did]
        grounded = score > 0 and score >= cut
        ref = cls.get(did)
        # Grounding (answer content) and an explicit endorsement both mean used;
        # an explicit ignore-cue means ignored. A BARE mention is not usage — a
        # model discusses the documents it rejects, so that stays "unused".
        if grounded or ref == "endorsed":
            status = "used"
        elif ref == "ignored":
            status = "ignored"
        else:
            status = "unused"
        statuses.append({"doc_id": did, "grounding": score, "status": status})

    # used first (best grounding first), so statuses[0] is the primary source;
    # then ignored, then unused.
    order = {"used": 0, "ignored": 1, "unused": 2}
    statuses.sort(key=lambda s: (order[s["status"]], -s["grounding"]))
    return statuses, gmap
