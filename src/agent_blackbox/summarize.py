"""
condense_reason — reduce a block of model reasoning to its single most
informative line, for use as a short "stated reason" in a decision while the
full reasoning stays in the recording. Pure stdlib, deterministic, no API call.

    from agent_blackbox import condense_reason
    short = condense_reason(long_thinking_text)

The heuristic strips markdown/quotes, drops filler and restatement openers,
then picks the sentence richest in domain + decision signal (optionally adding
the next sentence if the pick is short). It is intentionally simple: it never
invents text — the result is always a verbatim (cleaned) excerpt of the input,
so it stays honest testimony.
"""

from __future__ import annotations

import re

_FILLER = re.compile(
    r"^(okay|ok|alright|hmm+|so|well|now|first(?:\s+things\s+first)?|"
    r"let me|let's|right|actually|wait|therefore|thus|ah|but)\b[,:!.]?\s*", re.I)
_SKIP = re.compile(
    r"^(the customer (stated|said|asked|wrote)|they('re| are)? (asking|want|stated)|"
    r"i need to|i should|my job|my (first )?step|i have)\b", re.I)

# Signal words. Tuned for support agents but harmless elsewhere; override via
# the `domain`/`decision` args for another domain.
DOMAIN_WORDS = ("refund", "warrant", "repair", "policy", "shipping", "ship",
                "order", "ticket", "return", "day", "window", "deliver",
                "escalat", "price", "eligible")
DECISION_WORDS = ("qualif", "eligible", "not ", "no longer", "doesn't", "don't",
                  "cannot", "can't", "offer", "instead", "appl", "within",
                  "after", "because", "since")

# Words that specifically mark the OUTCOME of the decision — a sentence with
# one of these is far more likely to be the real conclusion than mere context,
# so it gets an extra bonus when picking the single best line.
CONCLUSION_MARKERS = ("instead", "no longer", "eligible", "qualif",
                      "you are covered", "we can offer", "we offer",
                      "not able", "unable")


def _clean(text: str) -> str:
    text = re.sub(r"\*\*.*?\*\*", " ", text)          # **markdown headers**
    text = text.replace('"', " ").replace("`", " ")   # quotes / code ticks
    text = re.sub(r"[*_#>]+", " ", text)             # markdown punctuation
    text = re.sub(r"[–—-]+", " ", text)     # en/em dashes, hyphens
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)    # collapse "broke broke"
    return re.sub(r"\s+", " ", text).strip()


def condense_reason(
    text: str,
    max_chars: int = 260,
    domain: tuple = DOMAIN_WORDS,
    decision: tuple = DECISION_WORDS,
) -> str:
    """Return a short (~1-2 sentence) excerpt capturing the reasoning's point."""
    cleaned = _clean(text or "")
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned)
             if len(s.strip()) >= 25 and not s.strip().endswith("?")
             and not _SKIP.match(s.strip())]
    if not sents:
        return cleaned[:max_chars]

    def score(s: str) -> int:
        low = s.lower()
        d = sum(low.count(k) for k in domain)
        c = sum(low.count(k) for k in decision)
        concl = 3 if any(k in low for k in CONCLUSION_MARKERS) else 0
        return d + c + concl - (2 if _FILLER.match(s) else 0)

    best = max(range(len(sents)), key=lambda i: (score(sents[i]), -i))
    out = _FILLER.sub("", sents[best]).strip()
    out = (out[:1].upper() + out[1:]) if out else out
    if len(out) < 120 and best + 1 < len(sents):
        nxt = _FILLER.sub("", sents[best + 1]).strip()
        if any(k in nxt.lower() for k in decision) and len(out) + len(nxt) < max_chars:
            out = out.rstrip(".") + ". " + (nxt[:1].upper() + nxt[1:])
    return (out[:max_chars - 1].rstrip() + "…") if len(out) > max_chars else out
