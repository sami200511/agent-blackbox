from agent_blackbox import BlackBox, FlightRecorder
from agent_blackbox.storage import JsonlStore, SqliteStore, open_store


def _record_two_sessions(store):
    r1 = FlightRecorder(session_id="alpha", pilot="agent-a", store=store)
    r1.user_input("q1")
    r1.final_output("a1")
    r1.close()
    r2 = FlightRecorder(session_id="beta", pilot="agent-b", store=store)
    r2.user_input("q2")
    r2.error("it broke")
    # left open on purpose (no close)


def test_sqlite_roundtrip_and_verify(tmp_path):
    store = SqliteStore(str(tmp_path / "bb.db"))
    _record_two_sessions(store)

    events = store.load("alpha")
    assert [e["type"] for e in events] == [
        "session_start", "user_input", "final_output", "session_end"]
    ok, msg = store.verify("alpha")
    assert ok, msg
    ok, msg = store.verify("beta")
    assert ok, msg


def test_sqlite_tamper_detected(tmp_path):
    import json
    import sqlite3

    db = str(tmp_path / "bb.db")
    store = SqliteStore(db)
    _record_two_sessions(store)

    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT event_json FROM events WHERE session_id='alpha' AND seq=2").fetchone()
    ev = json.loads(row[0])
    ev["payload"]["text"] = "tampered"
    conn.execute("UPDATE events SET event_json=? WHERE session_id='alpha' AND seq=2",
                 (json.dumps(ev),))
    conn.commit()
    conn.close()

    store2 = SqliteStore(db)
    ok, msg = store2.verify("alpha")
    assert not ok and "seq 2" in msg


def test_list_sessions_both_stores(tmp_path):
    for store in (JsonlStore(str(tmp_path / "j")), SqliteStore(str(tmp_path / "s.db"))):
        _record_two_sessions(store)
        infos = {i.session_id: i for i in store.list_sessions()}
        assert set(infos) == {"alpha", "beta"}
        assert infos["alpha"].closed and not infos["alpha"].has_error
        assert not infos["beta"].closed and infos["beta"].has_error
        assert infos["alpha"].pilot == "agent-a"
        assert infos["alpha"].n_events == 4


def test_open_store_resolution(tmp_path):
    s = open_store(directory=str(tmp_path / "x"))
    assert isinstance(s, JsonlStore)
    s = open_store(directory=str(tmp_path / "x"), db=str(tmp_path / "y.db"))
    assert isinstance(s, SqliteStore)


def test_blackbox_with_sqlite(tmp_path):
    bb = BlackBox(db=str(tmp_path / "bb.db"))
    with bb.session(session_id="gamma", pilot="p") as rec:
        rec.user_input("hi")
    ok, _ = bb.verify("gamma")
    assert ok
    assert bb.list_sessions()[0].session_id == "gamma"
