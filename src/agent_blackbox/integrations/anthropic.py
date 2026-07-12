"""
Auto-instrumentation for the Anthropic Python SDK (sync client).

Wraps `client.messages.create` (and the `client.messages.stream` helper) on a
specific client instance — no global monkey-patching — so every call is
recorded into the active blackbox session:

  * the full request (system prompt logged once per session, messages, tools)
  * the response text and tool_use requests
  * thinking blocks -> the recording's `reasoning` field. This is the CVR:
    the model's own stated reasoning, treated as testimony, not truth.
  * stop_reason (including "refusal"), token usage, latency, errors

Usage:
    bb = BlackBox("recordings")
    client = bb.watch(anthropic.Anthropic())
"""

from __future__ import annotations

import time
from typing import Any

from ..core import _jsonable

_INSTRUMENTED_FLAG = "_blackbox_instrumented"

_PARAM_KEYS = ("max_tokens", "thinking", "output_config", "tool_choice", "stop_sequences")


def _extract_params(kwargs: dict) -> dict:
    params = {k: _jsonable(kwargs[k]) for k in _PARAM_KEYS if k in kwargs}
    if kwargs.get("tools"):
        try:
            params["tools"] = [t.get("name") or t.get("type") for t in kwargs["tools"]]
        except (TypeError, AttributeError):
            params["tools"] = _jsonable(kwargs["tools"])
    return params


def _system_text(system: Any) -> str:
    if isinstance(system, str):
        return system
    if isinstance(system, list):  # list of text blocks
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
            for b in system
        )
    return ""


def _parse_message(message: Any) -> dict:
    """Pull text / thinking / tool_use out of an anthropic Message."""
    text_parts, thinking_parts, tool_calls = [], [], []
    for block in getattr(message, "content", []) or []:
        btype = getattr(block, "type", None)
        if btype == "text":
            text_parts.append(getattr(block, "text", "") or "")
        elif btype == "thinking":
            t = getattr(block, "thinking", "") or ""
            if t:
                thinking_parts.append(t)
        elif btype == "tool_use":
            tool_calls.append({
                "id": getattr(block, "id", None),
                "name": getattr(block, "name", None),
                "arguments": _jsonable(getattr(block, "input", None)),
            })
    usage = getattr(message, "usage", None)
    usage_d = {}
    if usage is not None:
        usage_d = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
        }
    return {
        "text": "".join(text_parts),
        "reasoning": "\n".join(thinking_parts) or None,
        "tool_calls": tool_calls,
        "stop_reason": getattr(message, "stop_reason", None),
        "usage": usage_d,
    }


def _record_message(rec, message: Any, *, model, prompt, params, t0, streamed=False) -> None:
    parsed = _parse_message(message)
    rec.llm_call(
        model=getattr(message, "model", None) or model,
        prompt=prompt,
        response=parsed["text"],
        reasoning=parsed["reasoning"],
        provider="anthropic",
        params=params,
        usage=parsed["usage"],
        tool_calls=parsed["tool_calls"],
        stop_reason=parsed["stop_reason"],
        latency_ms=round((time.monotonic() - t0) * 1000, 2),
        streamed=streamed,
    )


class _RecordingRawStream:
    """Wraps a raw `stream=True` event iterator; records once consumed/closed."""

    def __init__(self, inner, rec, model, prompt, params, t0):
        self._inner = inner
        self._rec = rec
        self._model = model
        self._prompt = prompt
        self._params = params
        self._t0 = t0
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._stop_reason = None
        self._usage: dict = {}
        self._recorded = False

    def __iter__(self):
        try:
            for event in self._inner:
                self._absorb(event)
                yield event
        except Exception as exc:
            self._rec.error(f"Anthropic stream failed: {exc}", exception_type=type(exc).__name__)
            self._flush()
            raise
        self._flush()

    def _absorb(self, event) -> None:
        etype = getattr(event, "type", None)
        if etype == "content_block_delta":
            delta = getattr(event, "delta", None)
            dtype = getattr(delta, "type", None)
            if dtype == "text_delta":
                self._text.append(getattr(delta, "text", "") or "")
            elif dtype == "thinking_delta":
                self._thinking.append(getattr(delta, "thinking", "") or "")
        elif etype == "message_delta":
            self._stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None) or self._stop_reason
            usage = getattr(event, "usage", None)
            if usage is not None and getattr(usage, "output_tokens", None) is not None:
                self._usage["output_tokens"] = usage.output_tokens
        elif etype == "message_start":
            usage = getattr(getattr(event, "message", None), "usage", None)
            if usage is not None:
                self._usage["input_tokens"] = getattr(usage, "input_tokens", None)

    def _flush(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        self._rec.llm_call(
            model=self._model,
            prompt=self._prompt,
            response="".join(self._text),
            reasoning="".join(self._thinking) or None,
            provider="anthropic",
            params=self._params,
            usage=self._usage,
            stop_reason=self._stop_reason,
            latency_ms=round((time.monotonic() - self._t0) * 1000, 2),
            streamed=True,
        )

    def __getattr__(self, item):
        return getattr(self._inner, item)


class _RecordingStreamManager:
    """Wraps the `client.messages.stream(...)` context-manager helper.

    On exit, records whatever the SDK accumulated (the full message when the
    caller consumed the stream or called get_final_message()).
    """

    def __init__(self, inner_manager, rec, model, prompt, params, t0):
        self._manager = inner_manager
        self._rec = rec
        self._model = model
        self._prompt = prompt
        self._params = params
        self._t0 = t0
        self._stream = None

    def __enter__(self):
        self._stream = self._manager.__enter__()
        return self._stream

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc is not None:
                self._rec.error(f"Anthropic stream failed: {exc}",
                                exception_type=exc_type.__name__)
            snapshot = getattr(self._stream, "current_message_snapshot", None)
            if snapshot is not None:
                _record_message(
                    self._rec, snapshot, model=self._model, prompt=self._prompt,
                    params=self._params, t0=self._t0, streamed=True,
                )
        except Exception as record_exc:  # never let recording break the app
            try:
                self._rec.error(f"blackbox failed to record stream: {record_exc}",
                                exception_type=type(record_exc).__name__)
            except Exception:
                pass
        return self._manager.__exit__(exc_type, exc, tb)


def instrument_anthropic(client: Any, blackbox) -> Any:
    """Patch `client.messages.create` / `.stream` on this client instance."""
    messages = client.messages
    if getattr(messages, _INSTRUMENTED_FLAG, False):
        return client
    original_create = messages.create
    original_stream = getattr(messages, "stream", None)

    def _prepare(kwargs):
        rec = blackbox._resolve_recorder()
        model = kwargs.get("model", "unknown")
        sys_text = _system_text(kwargs.get("system"))
        if sys_text:
            rec.system_prompt_once(sys_text)
        prompt = {
            "system": sys_text or None,
            "messages": _jsonable(kwargs.get("messages") or []),
        }
        return rec, model, prompt, _extract_params(kwargs)

    def create(*args, **kwargs):
        rec, model, prompt, params = _prepare(kwargs)
        t0 = time.monotonic()
        try:
            response = original_create(*args, **kwargs)
        except Exception as exc:
            rec.error(f"Anthropic call failed: {exc}", exception_type=type(exc).__name__)
            raise
        if kwargs.get("stream"):
            return _RecordingRawStream(response, rec, model, prompt, params, t0)
        _record_message(rec, response, model=model, prompt=prompt, params=params, t0=t0)
        return response

    messages.create = create

    if callable(original_stream):
        def stream(*args, **kwargs):
            rec, model, prompt, params = _prepare(kwargs)
            t0 = time.monotonic()
            manager = original_stream(*args, **kwargs)
            return _RecordingStreamManager(manager, rec, model, prompt, params, t0)

        messages.stream = stream

    setattr(messages, _INSTRUMENTED_FLAG, True)
    return client
