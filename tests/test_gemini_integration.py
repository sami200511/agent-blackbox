"""Instrumentation tests against the REAL google-genai SDK classes.

The underlying API method is stubbed at the class level (no network), but the
adapter parses genuine typed GenerateContentResponse objects — the same ones
the SDK builds from real API responses."""

import pytest

genai = pytest.importorskip("google.genai")

from google.genai import types  # noqa: E402

from agent_blackbox import BlackBox  # noqa: E402


def _response(**overrides) -> types.GenerateContentResponse:
    body = {
        "candidates": [{
            "content": {"role": "model", "parts": [
                {"text": "Top doc says 90 days; customer at 45.", "thought": True},
                {"text": "You qualify for a refund."},
                {"functionCall": {"name": "create_ticket", "args": {"days": 45}}},
            ]},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 10,
                          "thoughtsTokenCount": 5},
        "modelVersion": "gemini-2.5-flash-001",
    }
    body.update(overrides)
    return types.GenerateContentResponse.model_validate(body)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(genai.models.Models, "generate_content",
                        lambda self, **kwargs: _response())

    def fake_stream(self, **kwargs):
        def gen():
            yield types.GenerateContentResponse.model_validate({
                "candidates": [{"content": {"role": "model",
                                            "parts": [{"text": "Refund "}]}}]})
            yield types.GenerateContentResponse.model_validate({
                "candidates": [{"content": {"role": "model",
                                            "parts": [{"text": "approved."}]},
                                "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 4}})
        return gen()

    monkeypatch.setattr(genai.models.Models, "generate_content_stream", fake_stream)
    return genai.Client(api_key="test-key")


def test_generate_content_recorded(tmp_path, client):
    bb = BlackBox(directory=str(tmp_path))
    client = bb.watch(client)

    with bb.session(session_id="gem-1", pilot="support-agent"):
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Refund after 45 days?",
            config=types.GenerateContentConfig(
                system_instruction="You are a support agent.",
                temperature=0.4,
                tools=[types.Tool(function_declarations=[
                    types.FunctionDeclaration(name="create_ticket")])],
            ),
        )
    assert resp.text == "You qualify for a refund."

    events = bb.load("gem-1")
    ok, msg = bb.verify("gem-1")
    assert ok, msg

    sys_events = [e for e in events if e["type"] == "system_prompt"]
    assert len(sys_events) == 1
    assert sys_events[0]["payload"]["text"] == "You are a support agent."

    llm = next(e for e in events if e["type"] == "llm_call")
    p = llm["payload"]
    assert p["provider"] == "gemini"
    assert p["model"] == "gemini-2.5-flash-001"      # from response model_version
    assert p["response"] == "You qualify for a refund."
    assert p["reasoning"] == "Top doc says 90 days; customer at 45."
    assert p["stop_reason"] == "STOP"
    assert p["usage"] == {"input_tokens": 30, "output_tokens": 10,
                          "thinking_tokens": 5}
    assert p["tool_calls"][0]["name"] == "create_ticket"
    assert p["tool_calls"][0]["arguments"] == {"days": 45}
    assert p["params"]["temperature"] == 0.4
    assert p["params"]["tools"] == ["create_ticket"]
    assert p["prompt"]["contents"] == "Refund after 45 days?"
    assert p["latency_ms"] is not None


def test_stream_recorded(tmp_path, client):
    bb = BlackBox(directory=str(tmp_path))
    client = bb.watch(client)

    with bb.session(session_id="gem-2"):
        chunks = list(client.models.generate_content_stream(
            model="gemini-2.5-flash", contents="hi"))
    assert len(chunks) == 2

    llm = next(e for e in bb.load("gem-2") if e["type"] == "llm_call")
    p = llm["payload"]
    assert p["response"] == "Refund approved."
    assert p["streamed"] is True
    assert p["stop_reason"] == "STOP"
    assert p["usage"] == {"input_tokens": 20, "output_tokens": 4}
    ok, _ = bb.verify("gem-2")
    assert ok


def test_error_recorded(tmp_path, monkeypatch):
    def boom(self, **kwargs):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(genai.models.Models, "generate_content", boom)
    client = genai.Client(api_key="test-key")

    bb = BlackBox(directory=str(tmp_path))
    client = bb.watch(client)

    with bb.session(session_id="gem-3"):
        with pytest.raises(RuntimeError):
            client.models.generate_content(model="m", contents="hi")

    err = next(e for e in bb.load("gem-3") if e["type"] == "error")
    assert "Gemini call failed" in err["payload"]["message"]
    ok, _ = bb.verify("gem-3")
    assert ok


def test_watch_is_idempotent(tmp_path, client):
    bb = BlackBox(directory=str(tmp_path))
    bb.watch(client)
    bb.watch(client)

    with bb.session(session_id="gem-4"):
        client.models.generate_content(model="m", contents="hi")

    events = bb.load("gem-4")
    assert sum(1 for e in events if e["type"] == "llm_call") == 1
