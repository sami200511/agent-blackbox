"""
BlackBox: the top-level object that ties recording together.

    from agent_blackbox import BlackBox
    from openai import OpenAI

    bb = BlackBox("recordings")          # or BlackBox(db="blackbox.db")
    client = bb.watch(OpenAI())          # every LLM call is now recorded

    with bb.session(pilot="support-agent-v1", user_id="cust_42") as rec:
        rec.user_input(question, user_id="cust_42")
        answer = run_my_agent(client, question)   # llm/tool calls auto-recorded
        rec.final_output(answer)

If an instrumented client is used with no session open, a fallback session is
started automatically (named auto-YYYYMMDD-xxxx) and closed at process exit —
so "just watch the client" already gives you a usable recording.
"""

from __future__ import annotations

import atexit
import contextlib
import contextvars
import functools
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .core import FlightRecorder
from .storage import JsonlStore, SqliteStore, open_store

_current_recorder: contextvars.ContextVar[Optional[FlightRecorder]] = contextvars.ContextVar(
    "agent_blackbox_recorder", default=None
)


def current_recorder() -> Optional[FlightRecorder]:
    """The recorder for the active session in this context, if any."""
    return _current_recorder.get()


class BlackBox:
    def __init__(
        self,
        directory: Optional[str] = None,
        db: Optional[str] = None,
        store=None,
        redact: bool = False,
        redactor: Optional[Callable[[str], str]] = None,
        max_field_chars: int = 50_000,
    ):
        if store is None:
            store = open_store(directory=directory, db=db)
        self.store = store
        if redactor is None and redact:
            from .redact import default_redactor
            redactor = default_redactor
        self.redactor = redactor
        self.max_field_chars = max_field_chars
        self._auto_recorder: Optional[FlightRecorder] = None

    # ------------------------------------------------------------------ #
    # Sessions
    # ------------------------------------------------------------------ #
    def new_recorder(self, session_id: Optional[str] = None, pilot: str = "agent",
                     metadata: Optional[dict] = None) -> FlightRecorder:
        return FlightRecorder(
            session_id=session_id, pilot=pilot, store=self.store,
            metadata=metadata, redactor=self.redactor,
            max_field_chars=self.max_field_chars,
        )

    @contextlib.contextmanager
    def session(self, session_id: Optional[str] = None, pilot: str = "agent",
                metadata: Optional[dict] = None, **extra_metadata):
        """Open a recording session and make it the current one.

        Any keyword arguments beyond the named ones (e.g. user_id="cust_42")
        are merged into the session metadata.
        """
        md = dict(metadata or {})
        md.update(extra_metadata)
        rec = self.new_recorder(session_id=session_id, pilot=pilot, metadata=md)
        token = _current_recorder.set(rec)
        try:
            with rec:  # records exceptions with traceback, then closes
                yield rec
        finally:
            _current_recorder.reset(token)

    def _resolve_recorder(self) -> FlightRecorder:
        """Recorder used by auto-instrumentation: the active session, or a
        lazily created fallback session that lives until process exit."""
        rec = current_recorder()
        if rec is not None and not rec.closed:
            return rec
        if self._auto_recorder is None or self._auto_recorder.closed:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            self._auto_recorder = self.new_recorder(
                session_id=f"auto-{stamp}-{uuid.uuid4().hex[:6]}",
                pilot="unnamed-agent",
                metadata={"note": "auto-created session: no bb.session() was active"},
            )
            atexit.register(self._auto_recorder.close)
        return self._auto_recorder

    # ------------------------------------------------------------------ #
    # Instrumentation
    # ------------------------------------------------------------------ #
    def watch(self, client: Any) -> Any:
        """Instrument an LLM client in place so its calls are auto-recorded.

        Detects OpenAI and Anthropic clients by module. Returns the same
        client for chaining: `client = bb.watch(OpenAI())`.
        """
        module = type(client).__module__ or ""
        if module.startswith("openai"):
            # Also covers OpenRouter and any other OpenAI-compatible endpoint
            # (the adapter labels the provider from the client's base_url).
            from .integrations.openai import instrument_openai
            return instrument_openai(client, self)
        if module.startswith("anthropic"):
            from .integrations.anthropic import instrument_anthropic
            return instrument_anthropic(client, self)
        if module.startswith("google.genai"):
            from .integrations.gemini import instrument_gemini
            return instrument_gemini(client, self)
        raise TypeError(
            f"Don't know how to watch {type(client)!r}. Supported: openai.OpenAI "
            "(incl. OpenRouter via base_url), anthropic.Anthropic, and "
            "google.genai.Client (sync clients). For other frameworks record "
            "manually via the FlightRecorder API."
        )

    def tool(self, fn: Optional[Callable] = None, *, name: Optional[str] = None):
        """Decorator: record every call to an agent tool function.

            @bb.tool
            def lookup_order(order_id: str) -> dict: ...
        """
        def decorate(func: Callable) -> Callable:
            tool_name = name or func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                rec = self._resolve_recorder()
                t0 = time.monotonic()
                try:
                    result = func(*args, **kwargs)
                except Exception as exc:
                    rec.tool_call(
                        tool_name, {"args": list(args), "kwargs": kwargs},
                        result=f"{type(exc).__name__}: {exc}", ok=False,
                        latency_ms=round((time.monotonic() - t0) * 1000, 2),
                    )
                    raise
                rec.tool_call(
                    tool_name, {"args": list(args), "kwargs": kwargs},
                    result=result, ok=True,
                    latency_ms=round((time.monotonic() - t0) * 1000, 2),
                )
                return result

            return wrapper

        return decorate(fn) if fn is not None else decorate

    # ------------------------------------------------------------------ #
    # Investigation helpers
    # ------------------------------------------------------------------ #
    def list_sessions(self):
        return self.store.list_sessions()

    def load(self, session_id: str) -> list[dict]:
        return self.store.load(session_id)

    def verify(self, session_id: str) -> tuple[bool, str]:
        return self.store.verify(session_id)

    def report(self, session_id: str, out_path: Optional[str] = None) -> str:
        """Render the incident-investigation HTML report; returns the path."""
        from .viewer import write_report
        return write_report(self.store, session_id, out_path)
