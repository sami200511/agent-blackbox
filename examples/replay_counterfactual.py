"""
Counterfactual replay: PROVE which retrieved document caused a wrong answer.

Point it at a recorded session that has a `retrieval` event followed by an
`llm_call` (e.g. one produced by examples/anthropic_support_agent.py). For
each retrieved chunk it re-runs the recorded LLM call with that chunk's text
removed, and reports whether the answer changed.

Run:  ANTHROPIC_API_KEY=... python examples/replay_counterfactual.py <session_id>
"""

import sys

import anthropic

from agent_blackbox import BlackBox, CounterfactualReplay

bb = BlackBox("recordings")
client = anthropic.Anthropic()          # NOT watched: replays aren't incidents


def call_fn(prompt) -> str:
    """Re-issue the recorded prompt (possibly mutated) against the same model."""
    kwargs = dict(model="claude-opus-4-8", max_tokens=16000,
                  messages=prompt["messages"])
    if prompt.get("system"):
        kwargs["system"] = prompt["system"]
    resp = client.messages.create(**kwargs)
    return "".join(b.text for b in resp.content if b.type == "text")


def main() -> None:
    session_id = sys.argv[1]
    events = bb.load(session_id)
    retrieval = next(e for e in events if e["type"] == "retrieval")
    llm = next(e for e in events if e["type"] == "llm_call")

    replay = CounterfactualReplay(events)
    results = replay.ablate_retrieval_chunks(
        retrieval["seq"], llm["seq"], call_fn=call_fn)

    print(replay.summarize(results))
    print()
    for r in results:
        print(f"--- without {r.removed_doc_id} ---")
        print((r.replayed_response or "(skipped)")[:300])
        print()


if __name__ == "__main__":
    main()
