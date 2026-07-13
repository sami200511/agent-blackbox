# DemoShop — a recorded multi-model agent with a website

A complete, working proof of the black box: an agent that answers from a
folder of company documents and can DO things (open tickets, check orders),
behind a chat website. Every question becomes one tamper-evident black-box
session with its own investigation report.

## Run it

```bash
cd agent-blackbox                       # project root
.venv/bin/python demo_site/server.py
```

Open **http://127.0.0.1:8700/**.

## What the website gives you

- **Model switcher** — Gemini (native), OpenRouter models (including a custom
  model box for any of the hundreds on openrouter.ai), OpenAI direct, and
  Anthropic direct. Models whose provider has no key show a 🔑 marker.
- **Ask-for-key flow** — pick a model with no API key and the site pops a
  dialog; the key you paste is stored in the project's `.env` (gitignored)
  and the question retries automatically.
- **Flight log panel** — the recorded-session history sits next to the chat;
  every entry links to its investigation report (`/sessions` has the full
  index).
- **Retrieval reranker (OpenRouter)** — when an OpenRouter key is present,
  the keyword-retrieved documents are reranked by
  `nvidia/llama-nemotron-rerank-vl-1b-v2:free` via OpenRouter's `/rerank`
  API. The rerank call is recorded as its own black-box event — result or
  failure — so investigations show exactly how the final ranking happened.
  If the rerank fails, the agent falls back to keyword order and keeps going.

Try:

- `I bought a laptop 45 days ago and it broke. Can I get a refund?`
  (the knowledge base contains BOTH an outdated 90-day policy and the current
  30-day one — check the report to see which evidence the model used)
- `Where is my order DS-1002?` (calls the `check_order_status` tool)
- `Open a support ticket for me, I'm Sam and my charger melted.`
  (calls the `create_support_ticket` tool)

## What gets recorded (per question)

```
session_start → user_input → retrieval (BM25 scores)
→ rerank (OpenRouter relevance scores, or the failure)
→ grounding (which document the ANSWER actually reflects) → system_prompt
→ tool_call(s) (name, arguments, result, latency)
→ llm_call (prompt, response, model reasoning, tokens, provider, model)
→ decision (choice, alternatives, stated reason, evidence refs)
→ final_output → session_end          (errors + tracebacks recorded too)
```

### Three scores, from weakest to strongest signal

The report shows all three side by side so you can see where retrieval and the
model disagree:

1. **BM25 retrieval score** — principled keyword relevance (replaces naive
   term-frequency, which tied documents that share vocabulary). Still ties the
   two refund policies, because they genuinely share query words — a real,
   visible retrieval weakness.
2. **OpenRouter rerank relevance** — a purpose-built relevance model; better,
   but for near-identical documents it can still get them wrong.
3. **Answer grounding** — how much of each document's *distinctive* content
   actually appears in the model's answer. This is the strongest signal for
   what the model really used, and it **works for any model without it naming
   the file**. It cleanly separates the two policies: the answer reflects the
   current 30-day policy, so `current.md` grounds high and the superseded
   `OLD.md` grounds ~0.

The decision's **choice** is set from the highest-grounded document, so a model
that correctly discards a higher-*ranked* but outdated document is credited for
it — and the outdated one is labelled *"retrieved but not reflected in the
answer"* under PATHS NOT TAKEN.

Because the agent records a `decision` that links the retrieval, the rerank
and the LLM call as `evidence_refs`, every real session's report is as rich
as `blackbox demo`: a **Retrieval & reranking** section (keyword vs. relevance
scores), the **Model reasoning (CVR)**, and a **Decision analysis** panel that
puts the model's stated reason next to *all* the evidence it had — including
the documents it ignored.

**The stated reason is always the model's own words:**

- **Thinking models** — Gemini (asked with `thinking_level="high"`, so even
  flash-lite returns a full thought summary) and Anthropic (adaptive thinking)
  expose their reasoning directly; it becomes the stated reason.
- **Non-thinking models** — OpenAI and most OpenRouter models expose no
  thinking, so the agent makes one short follow-up call asking the model which
  document it used and why. That `self_explanation` is recorded as its own
  event and shown as the stated reason, clearly flagged as *post-hoc
  testimony* (a self-explanation given after the fact may not reflect the real
  decision — the report says so).

Recordings live in `demo_site/recordings/`:

```bash
.venv/bin/blackbox --dir demo_site/recordings list
.venv/bin/blackbox --dir demo_site/recordings verify <session-id>
```

## Files

- `documents/` — dummy knowledge base (policies, catalog, FAQ) with a
  deliberate trap: an outdated refund policy next to the current one.
- `agent.py` — model registry, per-provider agent loops (Gemini AFC /
  OpenAI-compatible / Anthropic), keyword retriever + OpenRouter reranker,
  two `@bb.tool`-recorded tools, key management.
- `server.py` — stdlib HTTP server: chat UI, `/api/models`, `/api/ask`,
  `/api/set_key`, `/api/history`, plus the investigation viewer
  (`/sessions`, `/session/<id>`). No web framework.

## Notes & gotchas

- **Gemini free tier**: ~20 requests/day per model. If a model returns a 429
  quota error (recorded as an incident!), switch to another model in the
  dropdown — that's what it's for.
- Do **not** use `from __future__ import annotations` in the module that
  defines Gemini tool functions — google-genai's automatic function calling
  validates arguments with `isinstance(value, annotation)` and PEP 563 string
  annotations make every tool call fail. The black box caught this: the
  model's recorded reasoning quoted the exact error.
- The server binds to 127.0.0.1 (local only). Add auth before exposing it.
