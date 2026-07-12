"""
Auto-instrumentation for Google's Gemini SDK (`google-genai`, sync client).

Wraps `client.models.generate_content` and `client.models.generate_content_stream`
on a specific client instance — no global monkey-patching — so every call is
recorded into the active blackbox session:

  * the full request contents (system_instruction logged once per session)
  * the response text and function-call requests
  * thought parts (thinking summaries) -> the recording's `reasoning` field.
    This is the CVR: the model's own stated reasoning, testimony not truth.
  * finish reason, token usage, latency, errors

Usage:
    from google import genai
    bb = BlackBox("recordings")
    client = bb.watch(genai.Client())      # api key from GEMINI_API_KEY env
"""

from __future__ import annotations

import time
from typing import Any

from ..core import _jsonable

_INSTRUMENTED_FLAG = "_blackbox_instrumented"

_CONFIG_KEYS = (
    "temperature", "max_output_tokens", "top_p", "top_k", "stop_sequences",
    "response_mime_type", "thinking_config", "tool_config", "seed",
)


def _cfg_get(config: Any, key: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _extract_params(config: Any) -> dict:
    params = {}
    for k in _CONFIG_KEYS:
        v = _cfg_get(config, k)
        if v is not None:
            params[k] = _jsonable(v)
    tools = _cfg_get(config, "tools")
    if tools:
        names = []
        for t in tools:
            decls = t.get("function_declarations") if isinstance(t, dict) \
                else getattr(t, "function_declarations", None)
            for d in decls or []:
                name = d.get("name") if isinstance(d, dict) else getattr(d, "name", None)
                if name:
                    names.append(name)
        params["tools"] = names or _jsonable(tools)
    return params


def _system_text(config: Any) -> str:
    si = _cfg_get(config, "system_instruction")
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    # types.Content or dict with parts
    parts = si.get("parts") if isinstance(si, dict) else getattr(si, "parts", None)
    if parts:
        out = []
        for p in parts:
            text = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
            if text:
                out.append(text)
        return "\n".join(out)
    return str(si)


def _absorb_response(resp: Any, acc: dict) -> None:
    """Merge one GenerateContentResponse (full or stream chunk) into `acc`."""
    for cand in getattr(resp, "candidates", None) or []:
        fr = getattr(cand, "finish_reason", None)
        if fr is not None:
            acc["stop_reason"] = getattr(fr, "name", None) or str(fr)
        content = getattr(cand, "content", None)
        for part in (getattr(content, "parts", None) or []):
            text = getattr(part, "text", None)
            if text:
                if getattr(part, "thought", False):
                    acc["thinking"].append(text)
                else:
                    acc["text"].append(text)
            fc = getattr(part, "function_call", None)
            if fc is not None:
                acc["tool_calls"].append({
                    "id": getattr(fc, "id", None),
                    "name": getattr(fc, "name", None),
                    "arguments": _jsonable(getattr(fc, "args", None)),
                })
    um = getattr(resp, "usage_metadata", None)
    if um is not None:
        usage = {
            "input_tokens": getattr(um, "prompt_token_count", None),
            "output_tokens": getattr(um, "candidates_token_count", None),
            "thinking_tokens": getattr(um, "thoughts_token_count", None),
        }
        acc["usage"] = {k: v for k, v in usage.items() if v is not None}
    mv = getattr(resp, "model_version", None)
    if mv:
        acc["model"] = mv


def _new_acc(model: str) -> dict:
    return {"text": [], "thinking": [], "tool_calls": [], "stop_reason": None,
            "usage": {}, "model": model}


def _record(rec, acc: dict, *, prompt, params, t0, streamed: bool) -> None:
    rec.llm_call(
        model=acc["model"],
        prompt=prompt,
        response="".join(acc["text"]),
        reasoning="\n".join(acc["thinking"]) or None,
        provider="gemini",
        params=params,
        usage=acc["usage"],
        tool_calls=acc["tool_calls"],
        stop_reason=acc["stop_reason"],
        latency_ms=round((time.monotonic() - t0) * 1000, 2),
        streamed=streamed,
    )


def instrument_gemini(client: Any, blackbox) -> Any:
    """Patch generate_content / generate_content_stream on this client."""
    models = client.models
    if getattr(models, _INSTRUMENTED_FLAG, False):
        return client
    original = models.generate_content
    original_stream = getattr(models, "generate_content_stream", None)

    def _prepare(kwargs):
        rec = blackbox._resolve_recorder()
        model = kwargs.get("model", "unknown")
        config = kwargs.get("config")
        sys_text = _system_text(config)
        if sys_text:
            rec.system_prompt_once(sys_text)
        prompt = {
            "system": sys_text or None,
            "contents": _jsonable(kwargs.get("contents")),
        }
        return rec, model, prompt, _extract_params(config)

    def generate_content(*args, **kwargs):
        rec, model, prompt, params = _prepare(kwargs)
        t0 = time.monotonic()
        try:
            response = original(*args, **kwargs)
        except Exception as exc:
            rec.error(f"Gemini call failed: {exc}", exception_type=type(exc).__name__)
            raise
        acc = _new_acc(model)
        _absorb_response(response, acc)
        _record(rec, acc, prompt=prompt, params=params, t0=t0, streamed=False)
        return response

    models.generate_content = generate_content

    if callable(original_stream):
        def generate_content_stream(*args, **kwargs):
            rec, model, prompt, params = _prepare(kwargs)
            t0 = time.monotonic()
            try:
                inner = original_stream(*args, **kwargs)
            except Exception as exc:
                rec.error(f"Gemini call failed: {exc}", exception_type=type(exc).__name__)
                raise

            def wrapper():
                acc = _new_acc(model)
                try:
                    for chunk in inner:
                        _absorb_response(chunk, acc)
                        yield chunk
                except Exception as exc:
                    rec.error(f"Gemini stream failed: {exc}",
                              exception_type=type(exc).__name__)
                    _record(rec, acc, prompt=prompt, params=params, t0=t0, streamed=True)
                    raise
                _record(rec, acc, prompt=prompt, params=params, t0=t0, streamed=True)

            return wrapper()

        models.generate_content_stream = generate_content_stream

    setattr(models, _INSTRUMENTED_FLAG, True)
    return client
