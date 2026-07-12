"""
Built-in redaction for agent-blackbox.

Recordings capture everything the agent saw — which in a real company
includes customer emails, phone numbers, card numbers, and (if someone is
careless) API keys pasted into prompts. The default redactor scrubs the
obvious classes before anything is hashed or written, so the recording is
tamper-evident over the *redacted* content and the raw secrets never touch
disk.

Redaction is opt-in: `BlackBox(redact=True)` or pass any `redactor`
callable (str -> str) for custom rules.
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Pattern

# Order matters: more specific first.
DEFAULT_PATTERNS: list[tuple[str, Pattern[str]]] = [
    ("API_KEY", re.compile(r"\b(sk-[A-Za-z0-9_\-]{16,}|sk-ant-[A-Za-z0-9_\-]{16,})\b")),
    ("BEARER_TOKEN", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}\b")),
    ("AWS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("CARD_NUMBER", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")),
    ("PHONE", re.compile(r"(?<![\w/])\+?\d{1,3}[ \-.]?\(?\d{2,4}\)?[ \-.]?\d{3,4}[ \-.]?\d{3,4}\b")),
]


def _luhn_ok(digits: str) -> bool:
    ds = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(ds) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(ds)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def make_redactor(
    patterns: Iterable[tuple[str, Pattern[str]]] = DEFAULT_PATTERNS,
) -> Callable[[str], str]:
    """Build a str -> str redactor replacing matches with [REDACTED:<LABEL>]."""

    def redact(text: str) -> str:
        for label, pattern in patterns:
            if label == "CARD_NUMBER":
                # Only redact digit runs that pass Luhn, to avoid eating IDs.
                def repl(m: re.Match) -> str:
                    return f"[REDACTED:{label}]" if _luhn_ok(m.group(0)) else m.group(0)
                text = pattern.sub(repl, text)
            else:
                text = pattern.sub(f"[REDACTED:{label}]", text)
        return text

    return redact


default_redactor = make_redactor()
