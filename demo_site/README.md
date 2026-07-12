# DemoShop — a recorded Gemini agent with a website

A complete, working proof of the black box: a Gemini agent that answers from
a folder of company documents and can DO things (open tickets, check orders),
behind a small chat website. Every question you ask becomes one tamper-evident
black-box session with its own investigation report.

## Run it

```bash
cd agent-blackbox                       # project root (needs GEMINI_API_KEY in .env)
.venv/bin/python demo_site/server.py
```

Open **http://127.0.0.1:8700/** and chat. Try:

- `I bought a laptop 45 days ago and it broke. Can I get a refund?`
  (the knowledge base contains BOTH an outdated 90-day policy and the current
  30-day one — watch which one the agent uses, then check its reasoning)
- `Where is my order DS-1002?` (calls the `check_order_status` tool)
- `Open a support ticket for me, I'm Sam and my charger melted.`
  (calls the `create_support_ticket` tool)

Under each answer is a link to that session's investigation report. The full
index is at **http://127.0.0.1:8700/sessions**.

## What gets recorded (per question)

```
session_start → user_input → retrieval (docs + scores) → system_prompt
→ tool_call(s) (name, arguments, result, latency)
→ llm_call (prompt, response, Gemini's thinking as `reasoning`, tokens)
→ final_output → session_end
```

Recordings are JSONL files in `demo_site/recordings/` — verify any of them:

```bash
.venv/bin/blackbox --dir demo_site/recordings list
.venv/bin/blackbox --dir demo_site/recordings verify <session-id>
```

Edit one character in a recording file and `verify` will name the exact event.

## Files

- `documents/` — the dummy knowledge base (policies, catalog, FAQ). Includes a
  deliberate trap: an outdated refund policy next to the current one.
- `agent.py` — the agent: tiny keyword retriever over `documents/`, two tools
  (`@bb.tool`-recorded), one watched Gemini client. Swap the retriever and the
  documents folder for your real ones.
- `server.py` — stdlib HTTP server: chat page, `/api/ask`, and the built-in
  investigation viewer (`/sessions`, `/session/<id>`). No web framework needed.

## Gotcha we hit (kept for reference)

Do **not** use `from __future__ import annotations` in the module that defines
Gemini tool functions: google-genai's automatic function calling validates
arguments with `isinstance(value, annotation)`, and PEP 563 turns annotations
into strings, which makes every tool call fail with
`isinstance() arg 2 must be a type`. The black box caught this — the model's
recorded reasoning quoted the exact error.
