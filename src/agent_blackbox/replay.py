"""
Counterfactual replay: turn a recording from "story" into "evidence".

The investigation report shows the agent's stated reason next to the evidence
it had. Replay goes one step further: re-run a recorded LLM call with one
piece of evidence removed and see whether the answer changes. If removing the
outdated policy chunk flips the answer, you have proof that chunk caused the
wrong output.

The recording stores the exact request the agent made (the instrumented
adapters record full messages). You supply a `call_fn` that takes the mutated
prompt and returns the model's text — usually a thin wrapper around the same
client the agent used:

    def call_fn(prompt):
        # prompt is the recorded llm_call "prompt" payload (messages, system)
        resp = client.messages.create(model=..., max_tokens=1024,
                                      system=prompt.get("system") or NOT_GIVEN,
                                      messages=prompt["messages"])
        return "".join(b.text for b in resp.content if b.type == "text")

Then:

    replay = CounterfactualReplay(events)
    results = replay.ablate_retrieval_chunks(retrieval_seq=3, llm_seq=4,
                                             call_fn=call_fn)
    for r in results:
        print(r.removed_doc_id, "-> answer changed:", r.answer_changed)
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class AblationResult:
    removed_doc_id: str
    removed_text: str
    original_response: str
    replayed_response: str
    answer_changed: bool
    note: str = ""


def _strip_text_everywhere(obj: Any, needle: str) -> Any:
    """Remove occurrences of `needle` from every string in a prompt payload."""
    if isinstance(obj, str):
        return obj.replace(needle, "[EVIDENCE REMOVED FOR REPLAY]")
    if isinstance(obj, dict):
        return {k: _strip_text_everywhere(v, needle) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_text_everywhere(v, needle) for v in obj]
    return obj


class CounterfactualReplay:
    def __init__(self, events: list[dict]):
        self.events = events
        self.by_seq = {e["seq"]: e for e in events}

    def _get(self, seq: int, expected_type: str) -> dict:
        ev = self.by_seq.get(seq)
        if ev is None:
            raise KeyError(f"No event with seq {seq}")
        if ev["type"] != expected_type:
            raise ValueError(f"Event seq {seq} is {ev['type']!r}, expected {expected_type!r}")
        return ev

    def run(self, llm_seq: int,
            mutate_fn: Callable[[Any], Any],
            call_fn: Callable[[Any], str]) -> str:
        """Generic replay: mutate the recorded prompt, call the model, return text."""
        llm = self._get(llm_seq, "llm_call")
        prompt = copy.deepcopy(llm["payload"]["prompt"])
        return call_fn(mutate_fn(prompt))

    def ablate_retrieval_chunks(
        self,
        retrieval_seq: int,
        llm_seq: int,
        call_fn: Callable[[Any], str],
        compare_fn: Optional[Callable[[str, str], bool]] = None,
    ) -> list[AblationResult]:
        """Re-run the LLM call once per retrieved chunk, with that chunk's
        text removed from the prompt. Returns one result per chunk.

        compare_fn(original, replayed) -> True if the answer CHANGED.
        Default: exact-string inequality after whitespace normalisation.
        """
        retrieval = self._get(retrieval_seq, "retrieval")
        llm = self._get(llm_seq, "llm_call")
        original_response = llm["payload"].get("response", "")

        if compare_fn is None:
            def compare_fn(a: str, b: str) -> bool:
                return " ".join(a.split()) != " ".join(b.split())

        results = []
        for chunk in retrieval["payload"].get("chunks", []):
            text = chunk.get("text", "")
            doc_id = chunk.get("doc_id", "?")
            if not text:
                results.append(AblationResult(
                    doc_id, text, original_response, "", False,
                    note="chunk has no text; skipped",
                ))
                continue
            prompt = copy.deepcopy(llm["payload"]["prompt"])
            mutated = _strip_text_everywhere(prompt, text)
            if json.dumps(mutated, sort_keys=True) == json.dumps(prompt, sort_keys=True):
                results.append(AblationResult(
                    doc_id, text, original_response, "", False,
                    note="chunk text not found verbatim in recorded prompt; "
                         "nothing removed — replay skipped",
                ))
                continue
            replayed = call_fn(mutated)
            results.append(AblationResult(
                removed_doc_id=doc_id,
                removed_text=text,
                original_response=original_response,
                replayed_response=replayed,
                answer_changed=compare_fn(original_response, replayed),
            ))
        return results

    def summarize(self, results: list[AblationResult]) -> str:
        lines = ["Counterfactual replay — which evidence caused the answer?", ""]
        for r in results:
            if r.note:
                lines.append(f"  - {r.removed_doc_id}: SKIPPED ({r.note})")
            elif r.answer_changed:
                lines.append(f"  - {r.removed_doc_id}: REMOVING THIS CHANGED THE ANSWER "
                             f"→ likely causal")
            else:
                lines.append(f"  - {r.removed_doc_id}: answer unchanged without it")
        return "\n".join(lines)
