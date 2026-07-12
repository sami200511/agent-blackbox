"""
Auto-instrumentation for the OpenAI Python SDK (sync client).

Wraps `client.chat.completions.create` on a specific client instance
(no global monkey-patching) so every call — streaming or not — is recorded
into the active blackbox session:

  * the full request messages (system prompt logged once per session)
  * the response text, tool-call requests, finish reason
  * token usage and wall-clock latency
  * errors, with exception type

Usage:
    bb = BlackBox("recordings")
    client = bb.watch(OpenAI())        # or instrument_openai(OpenAI(), bb)
"""

from __future__ import annotations

import time
from typing import Any

from ..core import _jsonable

_INSTRUMENTED_FLAG = "_blackbox_instrumented"

# Request params worth keeping in the recording (not the whole kwargs blob).
_PARAM_KEYS = (
    "temperature", "max_tokens", "max_completion_tokens", "top_p", "n",
    "stop", "seed", "response_format", "tool_choice", "reasoning_effort",
)


def _extract_params(kwargs: dict) -> dict:
    params = {k: _jsonable(kwargs[k]) for k in _PARAM_KEYS if k in kwargs}
    if "tools" in kwargs and kwargs["tools"]:
        try:
            params["tools"] = [t["function"]["name"] for t in kwargs["tools"]]
        except (TypeError, KeyError):
            params["tools"] = _jsonable(kwargs["tools"])
    return params


def _record_system_prompt(rec, messages: list) -> None:
    for m in messages or []:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        if role in ("system", "developer"):
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if isinstance(content, str) and content:
                rec.system_prompt_once(content)


def _parse_completion(response: Any) -> dict:
    """Pull text / tool calls / finish reason out of a ChatCompletion."""
    text_parts, tool_calls, finish = [], [], None
    for choice in getattr(response, "choices", []) or []:
        finish = getattr(choice, "finish_reason", finish)
        msg = getattr(choice, "message", None)
        if msg is None:
            continue
        if getattr(msg, "content", None):
            text_parts.append(msg.content)
        for tc in getattr(msg, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            tool_calls.append({
                "id": getattr(tc, "id", None),
                "name": getattr(fn, "name", None),
                "arguments": getattr(fn, "arguments", None),
            })
    usage = getattr(response, "usage", None)
    usage_d = {}
    if usage is not None:
        usage_d = {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
        }
    return {
        "text": "".join(text_parts),
        "tool_calls": tool_calls,
        "stop_reason": finish,
        "usage": usage_d,
    }


class _RecordingStream:
    """Wraps an OpenAI streaming response; records once fully consumed/closed."""

    def __init__(self, inner, rec, model, prompt, params, t0, provider="openai"):
        self._inner = inner
        self._rec = rec
        self._model = model
        self._prompt = prompt
        self._params = params
        self._t0 = t0
        self._provider = provider
        self._text: list[str] = []
        self._tool_calls: dict[int, dict] = {}
        self._finish = None
        self._usage = {}
        self._recorded = False

    def __iter__(self):
        try:
            for chunk in self._inner:
                self._absorb(chunk)
                yield chunk
        except Exception as exc:
            self._rec.error(f"OpenAI stream failed: {exc}", exception_type=type(exc).__name__)
            self._flush()
            raise
        self._flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self._flush()
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()

    def _absorb(self, chunk) -> None:
        for choice in getattr(chunk, "choices", []) or []:
            if getattr(choice, "finish_reason", None):
                self._finish = choice.finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            if getattr(delta, "content", None):
                self._text.append(delta.content)
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = self._tool_calls.setdefault(
                    getattr(tc, "index", 0) or 0,
                    {"id": None, "name": None, "arguments": ""},
                )
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self._usage = {
                "input_tokens": getattr(usage, "prompt_tokens", None),
                "output_tokens": getattr(usage, "completion_tokens", None),
            }

    def _flush(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        self._rec.llm_call(
            model=self._model,
            prompt=self._prompt,
            response="".join(self._text),
            provider=self._provider,
            params=self._params,
            usage=self._usage,
            tool_calls=[self._tool_calls[i] for i in sorted(self._tool_calls)],
            stop_reason=self._finish,
            latency_ms=round((time.monotonic() - self._t0) * 1000, 2),
            streamed=True,
        )

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _provider_label(client: Any) -> str:
    """Label recordings by the actual endpoint, not just the SDK.

    The OpenAI SDK is also the client for OpenAI-compatible providers
    (OpenRouter, local servers, ...); the base_url tells them apart.
    """
    base = str(getattr(client, "base_url", "") or "")
    if "openrouter" in base:
        return "openrouter"
    if base and "api.openai.com" not in base:
        return f"openai-compatible ({base})"
    return "openai"


def instrument_openai(client: Any, blackbox) -> Any:
    """Patch `client.chat.completions.create` on this client instance."""
    completions = client.chat.completions
    if getattr(completions, _INSTRUMENTED_FLAG, False):
        return client
    original = completions.create
    provider = _provider_label(client)

    def create(*args, **kwargs):
        rec = blackbox._resolve_recorder()
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages") or []
        _record_system_prompt(rec, messages)
        prompt = _jsonable(messages)
        params = _extract_params(kwargs)
        t0 = time.monotonic()
        try:
            response = original(*args, **kwargs)
        except Exception as exc:
            rec.error(f"OpenAI call failed: {exc}", exception_type=type(exc).__name__)
            raise
        if kwargs.get("stream"):
            return _RecordingStream(response, rec, model, prompt, params, t0,
                                    provider=provider)
        parsed = _parse_completion(response)
        rec.llm_call(
            model=getattr(response, "model", model),
            prompt=prompt,
            response=parsed["text"],
            provider=provider,
            params=params,
            usage=parsed["usage"],
            tool_calls=parsed["tool_calls"],
            stop_reason=parsed["stop_reason"],
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )
        return response

    completions.create = create
    setattr(completions, _INSTRUMENTED_FLAG, True)
    return client
