"""
agent-blackbox — a flight recorder for AI agents.

Quick start:

    from agent_blackbox import BlackBox
    bb = BlackBox("recordings")
    client = bb.watch(OpenAI())            # or anthropic.Anthropic()

    with bb.session(pilot="my-agent-v1") as rec:
        rec.user_input(question)
        answer = run_agent(client, question)   # LLM calls auto-recorded
        rec.final_output(answer)

Then investigate:  `blackbox list`, `blackbox report <session>`, `blackbox serve`.
"""

from .clients import load_env_file, make_client, watched_client
from .attribution import classify_references, documents_referenced
from .context import BlackBox, current_recorder
from .core import (
    GENESIS_HASH,
    FlightRecorder,
    load_recording,
    verify_events,
    verify_recording,
)
from .redact import default_redactor, make_redactor
from .replay import AblationResult, CounterfactualReplay
from .retrieval import attribute, bm25, freshness, grounding_scores, rank
from .summarize import condense_reason
from .storage import JsonlStore, SessionInfo, SqliteStore, open_store

__version__ = "0.1.0"

__all__ = [
    "BlackBox",
    "FlightRecorder",
    "current_recorder",
    "load_recording",
    "verify_recording",
    "verify_events",
    "GENESIS_HASH",
    "JsonlStore",
    "SqliteStore",
    "SessionInfo",
    "open_store",
    "CounterfactualReplay",
    "AblationResult",
    "condense_reason",
    "documents_referenced",
    "classify_references",
    "attribute",
    "bm25",
    "rank",
    "freshness",
    "grounding_scores",
    "make_redactor",
    "default_redactor",
    "load_env_file",
    "make_client",
    "watched_client",
    "__version__",
]
