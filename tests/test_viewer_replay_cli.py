import json
import os

import pytest

from agent_blackbox import BlackBox, CounterfactualReplay
from agent_blackbox.cli import main as cli_main
from agent_blackbox.demo import run_demo
from agent_blackbox.viewer import build_index, build_report


@pytest.fixture
def demo_bb(tmp_path):
    bb = BlackBox(directory=str(tmp_path / "rec"))
    sid = run_demo(bb, session_id="INC-TEST-1")
    return bb, sid


# ---------------------------------------------------------------- viewer ----
def test_report_contains_key_sections(demo_bb, tmp_path):
    bb, sid = demo_bb
    out = bb.report(sid, str(tmp_path / "r.html"))
    html_doc = open(out).read()
    assert "CHAIN INTACT" in html_doc
    assert "Can I still get a refund?" in html_doc
    assert "90 days" in html_doc
    assert "STATED REASON" in html_doc
    assert "policy_2026_v3" in html_doc  # the ignored evidence is visible
    assert "INCIDENT: Finance rejected" in html_doc
    assert "cockpit voice recorder" in html_doc.lower()


def test_report_shows_tampering(demo_bb, tmp_path):
    bb, sid = demo_bb
    path = os.path.join(bb.store.directory, f"{sid}.jsonl")
    lines = open(path).read().splitlines()
    ev = json.loads(lines[3])
    ev["payload"]["query"] = "changed"
    lines[3] = json.dumps(ev, ensure_ascii=False)
    open(path, "w").write("\n".join(lines) + "\n")

    events = bb.load(sid)
    html_doc = build_report(events)
    assert "TAMPERING DETECTED" in html_doc


def test_report_shows_retrieval_rerank_and_decision(tmp_path):
    """An auto-recorded style session (retrieval + rerank + llm + decision)
    must render the retrieval, rerank and decision sections — the data the
    real agent produces, matching the demo's richness."""
    bb = BlackBox(directory=str(tmp_path))
    with bb.session(session_id="rich-1", pilot="web-agent") as rec:
        rec.user_input("refund after 30 days?")
        r_seq = rec.retrieval("refund policy", chunks=[
            {"doc_id": "policy-OLD.md", "source": "docs/OLD", "score": 1.14,
             "text": "90 day refunds"},
            {"doc_id": "policy-current.md", "source": "docs/current", "score": 1.10,
             "text": "30 day window"},
        ])
        rr_seq = rec.custom("rerank", {
            "provider": "openrouter", "model": "nvidia/rerank",
            "results": [
                {"doc_id": "policy-current.md", "source": "docs/current",
                 "relevance_score": 0.91},
                {"doc_id": "policy-OLD.md", "source": "docs/OLD",
                 "relevance_score": 0.42},
            ],
        })
        l_seq = rec.llm_call(model="gemini-3.5-flash", provider="gemini",
                             prompt="...", response="30 days.",
                             reasoning="The current policy says 30 days.")
        rec.decision(choice="answered_from_document:policy-current.md",
                     alternatives=["use_document:policy-OLD.md", "escalate_to_human"],
                     stated_reason="The current policy says 30 days.",
                     evidence_refs=[r_seq, rr_seq, l_seq])
        rec.final_output("You can get a refund within 30 days.")

    html_doc = build_report(bb.load("rich-1"))
    # dedicated retrieval/rerank section
    assert "Retrieval &amp; reranking" in html_doc
    assert "score 1.14" in html_doc
    assert "RERANK" in html_doc and "relevance 0.91" in html_doc
    # model reasoning (CVR) present
    assert "Model reasoning" in html_doc
    assert "The current policy says 30 days." in html_doc
    # decision analysis populated with evidence, not the empty placeholder
    assert "answered_from_document:policy-current.md" in html_doc
    assert "No decision events recorded." not in html_doc
    # rerank rendered inside the decision evidence too
    assert html_doc.count("relevance 0.91") >= 2


def test_report_posthoc_self_explanation(tmp_path):
    """A non-thinking model's stated reason comes from a self-explanation and
    must render, clearly flagged as post-hoc, with the event in the timeline."""
    bb = BlackBox(directory=str(tmp_path))
    with bb.session(session_id="se-1", pilot="web-agent") as rec:
        rec.user_input("do you ship to the UK?")
        r_seq = rec.retrieval("shipping", chunks=[
            {"doc_id": "shipping.md", "source": "docs/ship", "score": 0.9,
             "text": "We ship to the UK in 3-5 days."}])
        l_seq = rec.llm_call(model="openai/gpt-4o-mini", provider="openrouter",
                             prompt="...", response="Yes, 3-5 days.", reasoning=None)
        rec.custom("self_explanation", {
            "provider": "openrouter", "model": "openai/gpt-4o-mini",
            "asked": "which document did you rely on, and why?",
            "response": "I relied on shipping.md because it lists UK delivery times."})
        rec.decision(
            choice="answered_from_document:shipping.md",
            alternatives=["escalate_to_human"],
            stated_reason="[post-hoc self-explanation — the agent was asked why "
                          "AFTER answering, so treat it as testimony] I relied on "
                          "shipping.md because it lists UK delivery times.",
            evidence_refs=[r_seq, l_seq])
        rec.final_output("Yes, 3-5 days.")

    html_doc = build_report(bb.load("se-1"))
    assert "post-hoc self-explanation" in html_doc
    assert "I relied on shipping.md" in html_doc
    assert "self_explanation" in html_doc  # event visible in the timeline
    assert "No decision events recorded." not in html_doc
    ok, _ = bb.verify("se-1")
    assert ok


def test_report_rerank_failure_shown(tmp_path):
    bb = BlackBox(directory=str(tmp_path))
    with bb.session(session_id="rich-2") as rec:
        rec.retrieval("q", chunks=[{"doc_id": "d", "text": "t", "score": 1.0}])
        rec.custom("rerank", {"provider": "openrouter", "model": "nvidia/rerank",
                              "error": "401 Unauthorized",
                              "note": "rerank failed; falling back to keyword ranking"})
    html_doc = build_report(bb.load("rich-2"))
    assert "FAILED" in html_doc and "401 Unauthorized" in html_doc


def test_index_lists_sessions(demo_bb):
    bb, sid = demo_bb
    html_doc = build_index(bb.list_sessions())
    assert sid in html_doc
    assert "flagged" in html_doc  # demo has a qa-review annotation


# ---------------------------------------------------------------- replay ----
def test_counterfactual_replay_identifies_causal_chunk(tmp_path):
    bb = BlackBox(directory=str(tmp_path))
    old_policy = "Refunds within 90 days for electronics."
    new_policy = "Refund window is 30 days from purchase."
    with bb.session(session_id="rp-1") as rec:
        r_seq = rec.retrieval("refund", chunks=[
            {"doc_id": "old", "text": old_policy, "score": 0.9},
            {"doc_id": "new", "text": new_policy, "score": 0.8},
        ])
        l_seq = rec.llm_call(
            model="m",
            prompt={"system": None,
                    "messages": [{"role": "user",
                                  "content": f"{old_policy}\n{new_policy}\nRefund at 45 days?"}]},
            response="Yes, you qualify (90-day window).",
        )

    def fake_call_fn(prompt):
        text = json.dumps(prompt)
        if "90 days" in text:
            return "Yes, you qualify (90-day window)."
        return "No — the refund window is 30 days."

    replay = CounterfactualReplay(bb.load("rp-1"))
    results = replay.ablate_retrieval_chunks(r_seq, l_seq, call_fn=fake_call_fn)
    by_doc = {r.removed_doc_id: r for r in results}
    assert by_doc["old"].answer_changed is True      # removing old policy flips it
    assert by_doc["new"].answer_changed is False     # new policy was being ignored
    summary = replay.summarize(results)
    assert "likely causal" in summary


def test_replay_skips_chunks_not_in_prompt(tmp_path):
    bb = BlackBox(directory=str(tmp_path))
    with bb.session(session_id="rp-2") as rec:
        r_seq = rec.retrieval("q", chunks=[{"doc_id": "d", "text": "never in prompt"}])
        l_seq = rec.llm_call(model="m", prompt={"messages": []}, response="x")
    replay = CounterfactualReplay(bb.load("rp-2"))
    results = replay.ablate_retrieval_chunks(
        r_seq, l_seq, call_fn=lambda p: pytest.fail("must not be called"))
    assert results[0].note.startswith("chunk text not found")


# ------------------------------------------------------------------- cli ----
def test_cli_demo_list_verify_report(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rec_dir = str(tmp_path / "recordings")

    assert cli_main(["--dir", rec_dir, "demo"]) == 0
    out = capsys.readouterr().out
    assert "INC-DEMO-0001" in out

    assert cli_main(["--dir", rec_dir, "list"]) == 0
    out = capsys.readouterr().out
    assert "INC-DEMO-0001" in out and "flagged" in out

    assert cli_main(["--dir", rec_dir, "verify", "INC-DEMO-0001"]) == 0
    assert "Chain intact" in capsys.readouterr().out

    out_html = str(tmp_path / "rep.html")
    assert cli_main(["--dir", rec_dir, "report", "INC-DEMO-0001", "-o", out_html]) == 0
    assert "CHAIN INTACT" in open(out_html).read()


def test_cli_verify_detects_tampering(tmp_path, capsys):
    rec_dir = str(tmp_path / "recordings")
    assert cli_main(["--dir", rec_dir, "demo"]) == 0
    capsys.readouterr()

    path = os.path.join(rec_dir, "INC-DEMO-0001.jsonl")
    lines = open(path).read().splitlines()
    ev = json.loads(lines[4])
    ev["payload"]["reasoning"] = "edited after the fact"
    lines[4] = json.dumps(ev, ensure_ascii=False)
    open(path, "w").write("\n".join(lines) + "\n")

    assert cli_main(["--dir", rec_dir, "verify", "INC-DEMO-0001"]) == 1
    assert "Tampering detected at seq 5" in capsys.readouterr().out


def test_cli_report_from_jsonl_path(tmp_path, capsys):
    rec_dir = str(tmp_path / "recordings")
    cli_main(["--dir", rec_dir, "demo"])
    capsys.readouterr()
    path = os.path.join(rec_dir, "INC-DEMO-0001.jsonl")
    assert cli_main(["report", path]) == 0
    out_path = path.replace(".jsonl", "_report.html")
    assert os.path.exists(out_path)
