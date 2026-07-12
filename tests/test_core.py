import json
import threading

import pytest

from agent_blackbox import (
    BlackBox,
    FlightRecorder,
    default_redactor,
    load_recording,
    verify_events,
    verify_recording,
)
from agent_blackbox.storage import JsonlStore


def test_chain_verifies(tmp_path):
    rec = FlightRecorder(session_id="s1", pilot="t", directory=str(tmp_path))
    rec.user_input("hello")
    rec.final_output("world")
    rec.close()
    ok, msg = verify_recording(rec.path)
    assert ok, msg
    assert "4 events" in msg  # start, input, output, end


def test_tampering_detected(tmp_path):
    rec = FlightRecorder(session_id="s2", directory=str(tmp_path))
    rec.user_input("the customer asked about refunds")
    rec.final_output("90 days")
    rec.close()
    path = rec.path

    lines = open(path).read().splitlines()
    ev = json.loads(lines[1])
    ev["payload"]["text"] = "the customer asked about UPGRADES"  # tamper
    lines[1] = json.dumps(ev, ensure_ascii=False)
    open(path, "w").write("\n".join(lines) + "\n")

    ok, msg = verify_recording(path)
    assert not ok
    assert "seq 2" in msg


def test_deleted_line_detected(tmp_path):
    rec = FlightRecorder(session_id="s3", directory=str(tmp_path))
    rec.user_input("a")
    rec.user_input("b")
    rec.close()
    lines = open(rec.path).read().splitlines()
    del lines[1]  # remove an event entirely
    open(rec.path, "w").write("\n".join(lines) + "\n")
    ok, msg = verify_recording(rec.path)
    assert not ok


def test_exception_recorded_and_session_closed(tmp_path):
    with pytest.raises(ValueError):
        with FlightRecorder(session_id="s4", directory=str(tmp_path)) as rec:
            rec.user_input("hi")
            raise ValueError("boom")
    events = load_recording(str(tmp_path / "s4.jsonl"))
    types = [e["type"] for e in events]
    assert "error" in types and types[-1] == "session_end"
    err = next(e for e in events if e["type"] == "error")
    assert err["payload"]["exception_type"] == "ValueError"
    assert "boom" in err["payload"]["traceback"]
    ok, _ = verify_events(events)
    assert ok


def test_closed_recorder_refuses_writes(tmp_path):
    rec = FlightRecorder(session_id="s5", directory=str(tmp_path))
    rec.close()
    with pytest.raises(RuntimeError):
        rec.user_input("too late")


def test_redaction_applied_before_hashing(tmp_path):
    rec = FlightRecorder(session_id="s6", directory=str(tmp_path),
                         redactor=default_redactor)
    rec.user_input("email me at alice@example.com, key sk-abcdefghijklmnop1234")
    rec.close()
    events = load_recording(rec.path)
    text = events[1]["payload"]["text"]
    assert "alice@example.com" not in text
    assert "sk-abcdefghijklmnop1234" not in text
    assert "[REDACTED:EMAIL]" in text and "[REDACTED:API_KEY]" in text
    ok, _ = verify_events(events)  # chain is over the redacted content
    assert ok


def test_truncation_of_huge_strings(tmp_path):
    rec = FlightRecorder(session_id="s7", directory=str(tmp_path), max_field_chars=100)
    rec.user_input("x" * 5000)
    rec.close()
    events = load_recording(rec.path)
    text = events[1]["payload"]["text"]
    assert len(text) < 400
    assert "TRUNCATED by blackbox" in text and "5000 chars" in text
    ok, _ = verify_events(events)
    assert ok


def test_thread_safety(tmp_path):
    rec = FlightRecorder(session_id="s8", directory=str(tmp_path))
    n_threads, per_thread = 8, 50

    def work(i):
        for j in range(per_thread):
            rec.annotation(f"t{i}-{j}")

    threads = [threading.Thread(target=work, args=(i,)) for i in range(n_threads)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    rec.close()
    events = load_recording(rec.path)
    assert len(events) == n_threads * per_thread + 2
    ok, msg = verify_events(events)
    assert ok, msg
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))


def test_system_prompt_once(tmp_path):
    rec = FlightRecorder(session_id="s9", directory=str(tmp_path))
    assert rec.system_prompt_once("You are helpful.") is not None
    assert rec.system_prompt_once("You are helpful.") is None
    assert rec.system_prompt_once("Different prompt.") is not None
    rec.close()
    events = load_recording(rec.path)
    assert sum(1 for e in events if e["type"] == "system_prompt") == 2


def test_nonjson_payloads_are_captured(tmp_path):
    class Weird:
        def __repr__(self):
            return "<Weird thing>"

    rec = FlightRecorder(session_id="s10", directory=str(tmp_path))
    rec.tool_call("t", arguments={"obj": Weird()}, result={1, 2})
    rec.close()
    events = load_recording(rec.path)
    payload = events[1]["payload"]
    assert payload["arguments"]["obj"] == "<Weird thing>"
    assert sorted(payload["result"]) == [1, 2]


def test_blackbox_session_contextvar(tmp_path):
    from agent_blackbox import current_recorder

    bb = BlackBox(directory=str(tmp_path))
    assert current_recorder() is None
    with bb.session(session_id="s11", pilot="p", user_id="u42") as rec:
        assert current_recorder() is rec
    assert current_recorder() is None
    events = bb.load("s11")
    assert events[0]["payload"]["metadata"]["user_id"] == "u42"


def test_blackbox_tool_decorator(tmp_path):
    bb = BlackBox(directory=str(tmp_path))

    @bb.tool
    def add(a, b):
        return a + b

    @bb.tool(name="explode")
    def bad():
        raise RuntimeError("nope")

    with bb.session(session_id="s12"):
        assert add(2, 3) == 5
        with pytest.raises(RuntimeError):
            bad()

    events = bb.load("s12")
    calls = [e for e in events if e["type"] == "tool_call"]
    assert calls[0]["payload"]["name"] == "add"
    assert calls[0]["payload"]["result"] == 5
    assert calls[0]["payload"]["ok"] is True
    assert calls[1]["payload"]["name"] == "explode"
    assert calls[1]["payload"]["ok"] is False
    assert "RuntimeError" in calls[1]["payload"]["result"]


def test_v0_recording_still_verifies():
    """Recordings made by the original single-file blackbox.py must verify."""
    import os
    legacy = os.path.join(os.path.dirname(__file__), "data", "legacy_v0.jsonl")
    if not os.path.exists(legacy):
        pytest.skip("no legacy fixture")
    ok, msg = verify_recording(legacy)
    assert ok, msg
