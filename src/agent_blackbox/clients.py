"""
Build LLM clients from environment variables — no API keys in code.

    from agent_blackbox import BlackBox
    from agent_blackbox.clients import load_env_file, make_client

    load_env_file()                       # optional: read a local .env file
    bb = BlackBox("recordings")
    client = bb.watch(make_client("openrouter"))

Environment variables per provider (see .env.example):

    openai      OPENAI_API_KEY                         (SDK reads it itself)
    anthropic   ANTHROPIC_API_KEY                      (SDK reads it itself)
    gemini      GEMINI_API_KEY or GOOGLE_API_KEY       (SDK reads it itself)
    openrouter  OPENROUTER_API_KEY  [required]
                OPENROUTER_BASE_URL [default https://openrouter.ai/api/v1]
                OPENROUTER_SITE_URL, OPENROUTER_APP_NAME [optional, for
                openrouter.ai rankings headers]

`load_env_file` is a tiny stdlib .env reader (KEY=VALUE lines, # comments,
optional quotes). Existing environment variables always win, so exported
shell vars are never overridden by the file.
"""

from __future__ import annotations

import os
from typing import Any, Optional

OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

PROVIDERS = ("openai", "anthropic", "gemini", "openrouter")


def load_env_file(path: str = ".env", override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file into os.environ.

    Returns the pairs that were read. Missing file is fine (returns {}).
    By default existing environment variables are NOT overridden.
    """
    loaded: dict[str, str] = {}
    if not os.path.exists(path):
        return loaded
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if not key:
                continue
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return loaded


def _require_env(name: str, provider: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Export it (export {name}=...) or put it in a "
            f".env file and call load_env_file() before make_client({provider!r})."
        )
    return value


def make_client(provider: str, **kwargs: Any) -> Any:
    """Construct a sync client for `provider`, configured from env vars.

    Extra kwargs are passed to the SDK client constructor. Wrap the result
    with bb.watch(...) to record its calls.
    """
    provider = provider.lower().strip()

    if provider == "openai":
        from openai import OpenAI
        _require_env("OPENAI_API_KEY", provider)
        return OpenAI(**kwargs)  # SDK reads OPENAI_API_KEY itself

    if provider == "anthropic":
        import anthropic
        _require_env("ANTHROPIC_API_KEY", provider)
        return anthropic.Anthropic(**kwargs)

    if provider == "gemini":
        from google import genai
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Export it or put "
                "it in a .env file and call load_env_file() first."
            )
        return genai.Client(**kwargs)

    if provider == "openrouter":
        from openai import OpenAI
        api_key = _require_env("OPENROUTER_API_KEY", provider)
        base_url = os.environ.get("OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE_URL)
        # Optional attribution headers, used by openrouter.ai leaderboards.
        headers = dict(kwargs.pop("default_headers", {}) or {})
        if os.environ.get("OPENROUTER_SITE_URL"):
            headers.setdefault("HTTP-Referer", os.environ["OPENROUTER_SITE_URL"])
        if os.environ.get("OPENROUTER_APP_NAME"):
            headers.setdefault("X-Title", os.environ["OPENROUTER_APP_NAME"])
        return OpenAI(api_key=api_key, base_url=base_url,
                      default_headers=headers or None, **kwargs)

    raise ValueError(f"Unknown provider {provider!r}. Choose one of {PROVIDERS}.")


def watched_client(provider: str, blackbox, **kwargs: Any) -> Any:
    """make_client + bb.watch in one call."""
    return blackbox.watch(make_client(provider, **kwargs))
