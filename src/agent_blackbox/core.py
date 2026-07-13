"""
Core of agent-blackbox: the FlightRecorder and the hash-chained event format.

Like an aircraft's FDR/CVR, the recorder captures everything an agent sees,
thinks, and does in an append-only, tamper-evident log. After an incident
(a wrong answer to a customer), you replay the recording to find out WHY.

Event format (identical to v0 demo recordings, so old .jsonl files verify):

    {
      "seq": 1, "session_id": "...", "type": "user_input",
      "ts": "...iso...", "t_rel_ms": 12.3,
      "payload": {...}, "prev_hash": "<hex>", "hash": "<hex>"
    }

Every event's hash covers the previous event's hash, so editing any line
breaks the chain and verification pinpoints the exact event.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

GENESIS_HASH = "0" * 64

# Event types written by the recorder. Integrations and apps may also write
# custom types via FlightRecorder.custom().
EVENT_TYPES = (
    "session_start", "user_input", "system_prompt", "retrieval", "llm_call",
    "tool_call", "decision", "final_output", "error", "annotation",
    "session_end",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(obj: Any) -> str:
    """Deterministic JSON so hashes are reproducible."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def event_hash(prev_hash: str, event_without_hash: dict) -> str:
    return hashlib.sha256(
        (prev_hash + canonical(event_without_hash)).encode("utf-8")
    ).hexdigest()


def _truncate_strings(obj: Any, max_chars: int) -> Any:
    """Cap huge strings so a single giant document can't bloat the log.

    The replaced text keeps a sha256 of the full content so an investigator
    can still prove what was there if they have the original.
    """
    if isinstance(obj, str):
        if len(obj) > max_chars:
            digest = hashlib.sha256(obj.encode("utf-8")).hexdigest()
            return (
                obj[:max_chars]
                + f"…[TRUNCATED by blackbox: {len(obj)} chars total, sha256={digest}]"
            )
        return obj
    if isinstance(obj, dict):
        return {k: _truncate_strings(v, max_chars) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_truncate_strings(v, max_chars) for v in obj]
    return obj


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of arbitrary objects into JSON-safe values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    # pydantic models (openai/anthropic SDK objects)
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except Exception:
                pass
    return repr(obj)


class FlightRecorder:
    """Append-only, hash-chained recorder for one agent session.

    Storage is pluggable: pass a `store` (see agent_blackbox.storage). If you
    pass `directory` instead, a JsonlStore is created for you — this matches
    the original single-file version of this library.
    """

    def __init__(
        self,
        session_id: Optional[str] = None,
        pilot: str = "agent",
        store=None,
        directory: Optional[str] = None,
        metadata: Optional[dict] = None,
        redactor: Optional[Callable[[str], str]] = None,
        max_field_chars: int = 50_000,
    ):
        if store is None:
            from .storage import JsonlStore
            store = JsonlStore(directory or "recordings")
        self.store = store
        self.session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"
        self.pilot = pilot
        self.redactor = redactor
        self.max_field_chars = max_field_chars
        self._seq = 0
        self._prev_hash = GENESIS_HASH
        self._t0 = time.monotonic()
        self._closed = False
        self._lock = threading.Lock()
        self._seen_system_prompts: set[str] = set()
        self._last_by_type: dict[str, dict] = {}
        self._write_event("session_start", {"pilot": pilot, "metadata": metadata or {}})

    # Path of the recording when backed by files (None for DB stores).
    @property
    def path(self) -> Optional[str]:
        getter = getattr(self.store, "path_for", None)
        return getter(self.session_id) if getter else None

    # ------------------------------------------------------------------ #
    # Core append + hash chain
    # ------------------------------------------------------------------ #
    def _apply_redaction(self, obj: Any) -> Any:
        if self.redactor is None:
            return obj
        if isinstance(obj, str):
            return self.redactor(obj)
        if isinstance(obj, dict):
            return {k: self._apply_redaction(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._apply_redaction(v) for v in obj]
        return obj

    def _write_event(self, event_type: str, payload: dict) -> int:
        with self._lock:
            if self._closed:
                raise RuntimeError("FlightRecorder is closed; open a new session.")
            payload = _jsonable(payload)
            payload = self._apply_redaction(payload)
            payload = _truncate_strings(payload, self.max_field_chars)
            self._seq += 1
            event = {
                "seq": self._seq,
                "session_id": self.session_id,
                "type": event_type,
                "ts": _now_iso(),
                "t_rel_ms": round((time.monotonic() - self._t0) * 1000, 2),
                "payload": payload,
                "prev_hash": self._prev_hash,
            }
            event["hash"] = event_hash(self._prev_hash, event)
            self._prev_hash = event["hash"]
            self.store.append(event)
            self._last_by_type[event_type] = event
            return self._seq

    def last(self, event_type: str) -> Optional[dict]:
        """The most recent event of `event_type` written this session (or None).

        Lets an agent reference an auto-recorded event — e.g. the seq of the
        `llm_call` the instrumentation just wrote — when building a decision.
        """
        return self._last_by_type.get(event_type)

    @property
    def current_seq(self) -> int:
        """Sequence number of the most recently written event."""
        return self._seq

    # ------------------------------------------------------------------ #
    # Recording API — one method per kind of thing an agent does
    # ------------------------------------------------------------------ #
    def user_input(self, text: str, user_id: str = "anonymous", channel: str = "chat") -> int:
        return self._write_event("user_input", {
            "text": text, "user_id": user_id, "channel": channel,
        })

    def system_prompt(self, text: str, version: str = "unversioned") -> int:
        return self._write_event("system_prompt", {"text": text, "version": version})

    def system_prompt_once(self, text: str, version: str = "unversioned") -> Optional[int]:
        """Record a system prompt only the first time this exact text is seen.

        Used by auto-instrumentation so multi-turn loops don't re-log the
        same prompt on every call.
        """
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key in self._seen_system_prompts:
            return None
        self._seen_system_prompts.add(key)
        return self.system_prompt(text, version=version)

    def retrieval(self, query: str, chunks: list) -> int:
        """chunks: [{"doc_id":..., "text":..., "score":..., "source":...}, ...]"""
        return self._write_event("retrieval", {"query": query, "chunks": chunks})

    def llm_call(
        self,
        model: str,
        prompt: Any,
        response: str,
        reasoning: Optional[str] = None,
        params: Optional[dict] = None,
        usage: Optional[dict] = None,
        provider: Optional[str] = None,
        tool_calls: Optional[list] = None,
        stop_reason: Optional[str] = None,
        latency_ms: Optional[float] = None,
        streamed: bool = False,
    ) -> int:
        """The CVR (cockpit voice recorder) part: the model's own words,
        including its reasoning/thinking text if available. NOTE: stated
        reasoning is evidence of what the model SAID, not proof of why it
        decided."""
        return self._write_event("llm_call", {
            "model": model, "prompt": prompt, "reasoning": reasoning,
            "response": response, "params": params or {}, "usage": usage or {},
            "provider": provider, "tool_calls": tool_calls or [],
            "stop_reason": stop_reason, "latency_ms": latency_ms,
            "streamed": streamed,
        })

    def tool_call(self, name: str, arguments: Any, result: Any,
                  ok: bool = True, latency_ms: Optional[float] = None) -> int:
        return self._write_event("tool_call", {
            "name": name, "arguments": arguments,
            "result": result, "ok": ok, "latency_ms": latency_ms,
        })

    def decision(self, choice: str, alternatives: list,
                 stated_reason: str, evidence_refs: Optional[list] = None,
                 confidence: Optional[float] = None) -> int:
        """A decision point. `stated_reason` = the agent's own claimed why.
        `evidence_refs` = seq numbers of prior events (retrievals, tool
        results) that were actually in front of the agent at that moment.
        The investigator's job is to compare the two."""
        return self._write_event("decision", {
            "choice": choice, "alternatives": alternatives,
            "stated_reason": stated_reason,
            "evidence_refs": evidence_refs or [],
            "confidence": confidence,
        })

    def final_output(self, text: str) -> int:
        return self._write_event("final_output", {"text": text})

    def error(self, message: str, exception_type: str = "", traceback_str: str = "") -> int:
        return self._write_event("error", {
            "message": message, "exception_type": exception_type,
            "traceback": traceback_str,
        })

    def annotation(self, text: str, author: str = "system") -> int:
        """Post-hoc notes (e.g., 'customer complained', 'flagged by QA')."""
        return self._write_event("annotation", {"text": text, "author": author})

    def custom(self, event_type: str, payload: dict) -> int:
        return self._write_event(event_type, payload)

    def close(self) -> None:
        if not self._closed:
            self._write_event("session_end", {"total_events": self._seq + 1})
            self._closed = True

    @property
    def closed(self) -> bool:
        return self._closed

    # Context-manager sugar: exceptions inside the block are recorded with a
    # full traceback before the session closes — the last seconds on the CVR.
    def __enter__(self) -> "FlightRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc is not None and not self._closed:
            import traceback as tb_mod
            self.error(str(exc), exception_type=exc_type.__name__,
                       traceback_str="".join(tb_mod.format_exception(exc_type, exc, tb)))
        self.close()


# ---------------------------------------------------------------------- #
# Verification & loading — the crash-investigation side
# ---------------------------------------------------------------------- #
def load_recording(path: str) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def verify_events(events: list[dict]) -> tuple[bool, str]:
    """Recompute the hash chain over in-memory events."""
    prev = GENESIS_HASH
    for ev in events:
        expected = event_hash(prev, {k: v for k, v in ev.items() if k != "hash"})
        if ev.get("prev_hash") != prev:
            return False, f"Chain broken at seq {ev.get('seq')}: prev_hash mismatch."
        if ev.get("hash") != expected:
            return False, f"Tampering detected at seq {ev.get('seq')}: hash mismatch."
        prev = ev["hash"]
    return True, f"Chain intact: {len(events)} events verified."


def verify_recording(path: str) -> tuple[bool, str]:
    """Recompute the hash chain of a .jsonl recording file."""
    return verify_events(load_recording(path))
