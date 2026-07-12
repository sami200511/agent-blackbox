# Agent Black Box 🟠

A **flight recorder for AI agents**. When an agent in your company gives a
customer a wrong answer, you open the recording and investigate exactly what
happened and why — like aviation crash investigators do.

| Aviation | This tool |
|---|---|
| Flight Data Recorder (FDR) | Every input, retrieval, LLM call, tool call, output — with timestamps |
| Cockpit Voice Recorder (CVR) | The model's own reasoning text (its *stated* why) |
| Tamper-proof casing | SHA-256 hash chain — any edit to the log is detected and located |
| Crash investigation | HTML incident report: stated reason vs the evidence actually available |
| Accident reconstruction | Counterfactual replay: remove one piece of evidence, re-run, prove causality |

The core is **pure stdlib** — zero required dependencies. The OpenAI/Anthropic
adapters activate only if those SDKs are installed.

## Install

```bash
pip install -e .                    # from this directory
# with SDK extras for auto-instrumentation:
pip install -e ".[openai]"  or  ".[anthropic]"  or  ".[dev]"
```

## 60-second start (no API key needed)

```bash
blackbox demo          # records a realistic wrong-answer incident
blackbox list          # see it
blackbox report INC-DEMO-0001
blackbox serve         # browse everything at http://127.0.0.1:8686/
```

## Recording a real agent

Wrap your LLM client **once** — every call is then recorded by itself
(prompts, responses, model reasoning, tool-call requests, token usage,
latency, errors, streaming included):

```python
from agent_blackbox import BlackBox
import anthropic                       # or: from openai import OpenAI

bb = BlackBox("recordings")            # or BlackBox(db="blackbox.db") for SQLite
client = bb.watch(anthropic.Anthropic())

@bb.tool                               # tool executions recorded too
def create_refund_ticket(customer_id: str, days: int) -> dict:
    ...

with bb.session(pilot="support-agent-v3", user_id="cust_88213") as rec:
    rec.user_input(question, user_id="cust_88213")
    rec.retrieval(query=q, chunks=chunks)     # what the agent could see
    answer = run_my_agent(client, question)   # LLM + tool calls auto-recorded
    rec.final_output(answer)
```

Notes:

- **No session open?** Instrumented calls still get recorded into an
  auto-created `auto-YYYYMMDD-…` session, so `bb.watch(client)` alone is
  already useful.
- **Exceptions** inside `bb.session(...)` are recorded with a full traceback
  before the session closes — the last seconds on the CVR.
- **Anthropic thinking blocks** are captured as the call's `reasoning`
  (request `thinking={"type": "adaptive", "display": "summarized"}` to get
  readable summaries).
- Supported today: sync `openai.OpenAI` (`chat.completions.create`, incl.
  `stream=True`) and sync `anthropic.Anthropic` (`messages.create` incl.
  `stream=True`, and the `messages.stream()` helper). Anything else: call the
  `FlightRecorder` methods directly — they're the same events.

## Investigating an incident

```bash
blackbox list                        # find the session (error/flagged badges)
blackbox verify INC-2026-0712        # recompute the hash chain
blackbox report INC-2026-0712       # single-file HTML investigation report
blackbox serve                       # or browse everything in one place
```

The report deliberately splits **the agent's stated reason** from **the
evidence that was actually in front of it**. Models' explanations don't always
reflect their real decision process — so the stated reason is treated as
testimony, not truth. In the demo, the stated reason says "policy allows 90
days", but the evidence panel shows the *current* 30-day policy was also
retrieved, ranked second, and ignored. That gap is the finding: fix the
retrieval ranking / deprecate the stale document — don't blame the model.

Post-incident, mark the session from code:

```python
rec.annotation("INCIDENT: finance rejected ticket RF-55190", author="qa-review")
```

## Proving the cause: counterfactual replay

```python
from agent_blackbox import BlackBox, CounterfactualReplay

bb = BlackBox("recordings")
replay = CounterfactualReplay(bb.load("INC-2026-0712"))
results = replay.ablate_retrieval_chunks(retrieval_seq=3, llm_seq=4, call_fn=call_fn)
print(replay.summarize(results))
#   - policy_2023_v1: REMOVING THIS CHANGED THE ANSWER → likely causal
#   - policy_2026_v3: answer unchanged without it
```

`call_fn` re-issues the (mutated) recorded prompt against your model — see
`examples/replay_counterfactual.py`.

## Tamper evidence

Every event's SHA-256 hash covers the previous event's hash. Edit one
character anywhere — or delete a line — and verification pinpoints the event:

```
$ sed -i 's/90 days/900 days/' recordings/INC-DEMO-0001.jsonl
$ blackbox verify INC-DEMO-0001
✗ Tampering detected at seq 4: hash mismatch.     (exit code 1 — CI-friendly)
```

This makes the log **tamper-evident, not tamper-proof**: someone with write
access can destroy history, but cannot silently rewrite it. For stronger
guarantees, periodically anchor the latest hash externally (append it to a
WORM bucket, a signed git tag, or a ticket).

## Redaction (PII / secrets)

```python
bb = BlackBox("recordings", redact=True)   # emails, phones, Luhn-valid card
                                           # numbers, sk-/AKIA/bearer tokens
bb = BlackBox("recordings", redactor=my_str_to_str_fn)   # your own rules
```

Redaction runs **before hashing and before disk**, so raw secrets never touch
storage and the chain is valid over the redacted content. Regex-based
redaction is best-effort — review the patterns in `agent_blackbox/redact.py`
against your data. Giant fields are truncated at 50 KB (configurable) with the
full content's sha256 kept for provenance.

## Storage

| Backend | Use when | Select with |
|---|---|---|
| JSONL (default) | dev, small deployments, greppable files, one file per session | `BlackBox("recordings")` / `--dir` |
| SQLite (WAL) | production single-node, many sessions, fast listing | `BlackBox(db="blackbox.db")` / `--db` |

Both use the identical event format; `blackbox` CLI subcommands accept
`--dir`/`--db`. Recordings from the original single-file prototype verify
unchanged. To add your own backend (Postgres, ClickHouse, S3), implement four
methods: `append`, `load`, `list_sessions`, `verify` — see
`agent_blackbox/storage.py`.

## Event reference

`session_start`, `user_input`, `system_prompt`, `retrieval`, `llm_call`
(prompt, response, reasoning, tool_calls, usage, stop_reason, latency),
`tool_call`, `decision` (choice, alternatives, stated_reason, evidence_refs,
confidence), `final_output`, `error`, `annotation`, `session_end` — plus
`rec.custom(type, payload)` for anything else.

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest        # 35 tests; the SDK integration tests mock HTTP only —
                        # they run the real openai/anthropic client classes
```

## Roadmap / production next steps

- async client support (`AsyncOpenAI`, `AsyncAnthropic`)
- OpenTelemetry span export for Langfuse / Phoenix / Datadog interop
- LangChain callback handler
- external hash anchoring helper (WORM/S3 object-lock)
