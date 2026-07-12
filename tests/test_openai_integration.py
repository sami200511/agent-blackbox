"""Instrumentation tests against the REAL OpenAI SDK, with HTTP mocked at the
transport layer — proves the adapter works with the actual client classes."""

import json

import httpx
import pytest

openai = pytest.importorskip("openai")

from agent_blackbox import BlackBox  # noqa: E402


COMPLETION_BODY = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{
        "index": 0,
        "finish_reason": "stop",
        "message": {
            "role": "assistant",
            "content": "You can get a refund within 30 days.",
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "create_ticket", "arguments": "{\"days\": 45}"},
            }],
        },
    }],
    "usage": {"prompt_tokens": 42, "completion_tokens": 12, "total_tokens": 54},
}


def _client(handler):
    return openai.OpenAI(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_chat_completion_recorded(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        return httpx.Response(200, json=COMPLETION_BODY)

    bb = BlackBox(directory=str(tmp_path))
    client = bb.watch(_client(handler))

    with bb.session(session_id="oai-1", pilot="support-agent"):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a support agent."},
                {"role": "user", "content": "Refund after 45 days?"},
            ],
        )
    assert resp.choices[0].message.content.startswith("You can get")

    events = bb.load("oai-1")
    ok, msg = bb.verify("oai-1")
    assert ok, msg

    sys_events = [e for e in events if e["type"] == "system_prompt"]
    assert len(sys_events) == 1
    assert sys_events[0]["payload"]["text"] == "You are a support agent."

    llm = next(e for e in events if e["type"] == "llm_call")
    p = llm["payload"]
    assert p["provider"] == "openai"
    assert p["model"] == "gpt-4o-mini"
    assert p["response"] == "You can get a refund within 30 days."
    assert p["usage"] == {"input_tokens": 42, "output_tokens": 12}
    assert p["stop_reason"] == "stop"
    assert p["params"]["temperature"] == 0.2
    assert p["tool_calls"] == [
        {"id": "call_1", "name": "create_ticket", "arguments": "{\"days\": 45}"}]
    assert p["latency_ms"] is not None
    assert p["prompt"][1]["content"] == "Refund after 45 days?"


def test_streaming_recorded(tmp_path):
    chunks = [
        {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini",
         "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hel"},
                      "finish_reason": None}]},
        {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini",
         "choices": [{"index": 0, "delta": {"content": "lo!"}, "finish_reason": None}]},
        {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini",
         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=sse.encode(),
                              headers={"content-type": "text/event-stream"})

    bb = BlackBox(directory=str(tmp_path))
    client = bb.watch(_client(handler))

    with bb.session(session_id="oai-2"):
        stream = client.chat.completions.create(
            model="gpt-4o-mini", stream=True,
            messages=[{"role": "user", "content": "hi"}],
        )
        received = "".join(
            c.choices[0].delta.content or ""
            for c in stream if c.choices and c.choices[0].delta
        )
    assert received == "Hello!"

    events = bb.load("oai-2")
    llm = next(e for e in events if e["type"] == "llm_call")
    assert llm["payload"]["response"] == "Hello!"
    assert llm["payload"]["streamed"] is True
    assert llm["payload"]["stop_reason"] == "stop"
    ok, _ = bb.verify("oai-2")
    assert ok


def test_api_error_recorded(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "kaput"}})

    bb = BlackBox(directory=str(tmp_path))
    client = bb.watch(_client(handler))
    client = openai.OpenAI(  # rebuild with 0 retries to keep the test fast
        api_key="k", max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    bb.watch(client)

    with bb.session(session_id="oai-3"):
        with pytest.raises(Exception):
            client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    events = bb.load("oai-3")
    err = next(e for e in events if e["type"] == "error")
    assert "OpenAI call failed" in err["payload"]["message"]
    ok, _ = bb.verify("oai-3")
    assert ok


def test_auto_session_when_none_active(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=COMPLETION_BODY)

    bb = BlackBox(directory=str(tmp_path))
    client = bb.watch(_client(handler))

    # No bb.session() open — the call must still be recorded somewhere.
    client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    sessions = bb.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].session_id.startswith("auto-")
    events = bb.load(sessions[0].session_id)
    assert any(e["type"] == "llm_call" for e in events)


def test_watch_is_idempotent(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=COMPLETION_BODY)

    bb = BlackBox(directory=str(tmp_path))
    client = _client(handler)
    bb.watch(client)
    bb.watch(client)  # second watch must not double-wrap

    with bb.session(session_id="oai-4"):
        client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])

    events = bb.load("oai-4")
    assert sum(1 for e in events if e["type"] == "llm_call") == 1
