"""Tests for env-based client construction and the .env loader."""

import httpx
import pytest

from agent_blackbox.clients import (
    OPENROUTER_DEFAULT_BASE_URL,
    load_env_file,
    make_client,
)


# ------------------------------------------------------------- .env file ----
def test_load_env_file(tmp_path, monkeypatch):
    monkeypatch.delenv("FOO_KEY", raising=False)
    monkeypatch.setenv("ALREADY", "from-shell")
    env = tmp_path / ".env"
    env.write_text(
        "# comment line\n"
        "\n"
        "FOO_KEY=abc123\n"
        "QUOTED='with quotes'\n"
        "ALREADY=from-file\n"
        "not a valid line\n"
    )
    loaded = load_env_file(str(env))
    import os
    assert loaded["FOO_KEY"] == "abc123"
    assert os.environ["FOO_KEY"] == "abc123"
    assert os.environ["QUOTED"] == "with quotes"
    assert os.environ["ALREADY"] == "from-shell"   # shell wins by default

    load_env_file(str(env), override=True)
    assert os.environ["ALREADY"] == "from-file"    # unless override=True


def test_load_env_file_missing_is_fine(tmp_path):
    assert load_env_file(str(tmp_path / "nope.env")) == {}


# ------------------------------------------------------------ make_client ----
def test_make_client_openrouter(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "my-agent")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    client = make_client("openrouter")
    assert "openrouter.ai" in str(client.base_url)
    assert OPENROUTER_DEFAULT_BASE_URL.endswith(str(client.base_url.path).rstrip("/"))
    assert client.default_headers.get("X-Title") == "my-agent"


def test_make_client_requires_key(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_client("openrouter")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    pytest.importorskip("google.genai")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        make_client("gemini")


def test_make_client_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        make_client("hal9000")


def test_make_client_gemini(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    client = make_client("gemini")
    assert type(client).__module__.startswith("google.genai")


# ------------------------------------- openrouter recording (provider tag) ----
def test_openrouter_calls_recorded_with_provider_label(tmp_path, monkeypatch):
    openai = pytest.importorskip("openai")
    from agent_blackbox import BlackBox

    body = {
        "id": "gen-1", "object": "chat.completion", "created": 1,
        "model": "openai/gpt-4o-mini",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "Hi from OpenRouter"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "openrouter.ai"
        return httpx.Response(200, json=body)

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    client = make_client(
        "openrouter",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    bb = BlackBox(directory=str(tmp_path))
    client = bb.watch(client)

    with bb.session(session_id="or-1"):
        client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}])

    llm = next(e for e in bb.load("or-1") if e["type"] == "llm_call")
    assert llm["payload"]["provider"] == "openrouter"
    assert llm["payload"]["model"] == "openai/gpt-4o-mini"
    assert llm["payload"]["response"] == "Hi from OpenRouter"
    ok, _ = bb.verify("or-1")
    assert ok
