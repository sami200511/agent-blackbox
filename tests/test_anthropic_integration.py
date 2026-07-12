"""Instrumentation tests against the REAL Anthropic SDK, with HTTP mocked at
the transport layer. The thinking block -> reasoning mapping is the CVR."""

import json

import httpx
import pytest

anthropic = pytest.importorskip("anthropic")

from agent_blackbox import BlackBox  # noqa: E402


MESSAGE_BODY = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-4-8",
    "content": [
        {"type": "thinking",
         "thinking": "The top document says 90 days, customer is at 45 days.",
         "signature": "sig"},
        {"type": "text", "text": "You qualify for a refund."},
        {"type": "tool_use", "id": "toolu_1", "name": "create_ticket",
         "input": {"days": 45}},
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 100, "output_tokens": 25},
}


def _client(handler, **kwargs):
    return anthropic.Anthropic(
        api_key="test-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kwargs,
    )


def test_message_recorded_with_reasoning(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/messages")
        return httpx.Response(200, json=MESSAGE_BODY)

    bb = BlackBox(directory=str(tmp_path))
    client = bb.watch(_client(handler))

    with bb.session(session_id="ant-1", pilot="support-agent"):
        resp = client.messages.create(
            model="claude-opus-4-8",
            max_tokens=1024,
            system="You are a support agent.",
            messages=[{"role": "user", "content": "Refund after 45 days?"}],
        )
    assert resp.stop_reason == "tool_use"

    events = bb.load("ant-1")
    ok, msg = bb.verify("ant-1")
    assert ok, msg

    sys_events = [e for e in events if e["type"] == "system_prompt"]
    assert len(sys_events) == 1 and sys_events[0]["payload"]["text"] == "You are a support agent."

    llm = next(e for e in events if e["type"] == "llm_call")
    p = llm["payload"]
    assert p["provider"] == "anthropic"
    assert p["model"] == "claude-opus-4-8"
    assert p["response"] == "You qualify for a refund."
    assert p["reasoning"] == "The top document says 90 days, customer is at 45 days."
    assert p["stop_reason"] == "tool_use"
    assert p["usage"]["input_tokens"] == 100
    assert p["usage"]["output_tokens"] == 25
    assert p["tool_calls"] == [
        {"id": "toolu_1", "name": "create_ticket", "arguments": {"days": 45}}]
    assert p["prompt"]["system"] == "You are a support agent."
    assert p["prompt"]["messages"][0]["content"] == "Refund after 45 days?"


def _sse_body() -> bytes:
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_s", "type": "message", "role": "assistant",
            "model": "claude-opus-4-8", "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 80, "output_tokens": 1}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "thinking", "thinking": "",
                                                   "signature": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "thinking_delta",
                                           "thinking": "Considering the policy."}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("content_block_start", {"type": "content_block_start", "index": 1,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                 "delta": {"type": "text_delta", "text": "Refund "}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 1,
                                 "delta": {"type": "text_delta", "text": "approved."}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 1}),
        ("message_delta", {"type": "message_delta",
                           "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 14}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(
        f"event: {name}\ndata: {json.dumps(payload)}\n\n" for name, payload in events
    ).encode()


def test_raw_stream_recorded(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse_body(),
                              headers={"content-type": "text/event-stream"})

    bb = BlackBox(directory=str(tmp_path))
    client = bb.watch(_client(handler))

    with bb.session(session_id="ant-2"):
        stream = client.messages.create(
            model="claude-opus-4-8", max_tokens=64, stream=True,
            messages=[{"role": "user", "content": "hi"}],
        )
        list(stream)  # consume

    llm = next(e for e in bb.load("ant-2") if e["type"] == "llm_call")
    p = llm["payload"]
    assert p["response"] == "Refund approved."
    assert p["reasoning"] == "Considering the policy."
    assert p["streamed"] is True
    assert p["stop_reason"] == "end_turn"
    assert p["usage"] == {"input_tokens": 80, "output_tokens": 14}
    ok, _ = bb.verify("ant-2")
    assert ok


def test_stream_helper_recorded(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=_sse_body(),
                              headers={"content-type": "text/event-stream"})

    bb = BlackBox(directory=str(tmp_path))
    client = bb.watch(_client(handler))

    with bb.session(session_id="ant-3"):
        with client.messages.stream(
            model="claude-opus-4-8", max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
        ) as stream:
            text = "".join(stream.text_stream)
    assert text == "Refund approved."

    llm = next(e for e in bb.load("ant-3") if e["type"] == "llm_call")
    p = llm["payload"]
    assert p["response"] == "Refund approved."
    assert p["streamed"] is True
    ok, _ = bb.verify("ant-3")
    assert ok


def test_api_error_recorded(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "bad params"}})

    bb = BlackBox(directory=str(tmp_path))
    client = bb.watch(_client(handler, max_retries=0))

    with bb.session(session_id="ant-4"):
        with pytest.raises(Exception):
            client.messages.create(model="claude-opus-4-8", max_tokens=10,
                                   messages=[{"role": "user", "content": "hi"}])

    err = next(e for e in bb.load("ant-4") if e["type"] == "error")
    assert "Anthropic call failed" in err["payload"]["message"]
    ok, _ = bb.verify("ant-4")
    assert ok
