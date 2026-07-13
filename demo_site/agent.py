"""
DemoShop web agent — multi-model, fully recorded by the black box.

Providers: Gemini (native SDK), OpenRouter (any of its models), OpenAI,
Anthropic. The website lets you switch models per question; if the chosen
provider has no API key in the environment, the site asks for one.

Retrieval: tiny keyword search over documents/, optionally RERANKED by
OpenRouter's rerank endpoint (nvidia/llama-nemotron-rerank-vl-1b-v2:free)
when OPENROUTER_API_KEY is set. The rerank call is recorded in the black box
as its own event, so an investigator sees keyword scores AND rerank scores.

NOTE: no `from __future__ import annotations` in this file! google-genai's
automatic function calling reads tool annotations with isinstance(), which
breaks when PEP 563 turns them into strings.
"""

import json
import os
import re

import httpx

from agent_blackbox import BlackBox, load_env_file, make_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
ENV_PATH = os.path.abspath(os.path.join(BASE_DIR, os.pardir, ".env"))

load_env_file(ENV_PATH)
load_env_file()  # also allow a local .env

bb = BlackBox(RECORDINGS_DIR, redact=True)

SYSTEM_PROMPT = (
    "You are DemoShop's customer support agent on the website. Answer using "
    "ONLY the provided policy documents; when documents conflict, follow the "
    "most recent effective one. Use your tools when the customer asks you to "
    "DO something (open a ticket, check an order). Be concise and friendly. "
    "If the documents don't cover the question, say so and offer to escalate."
)

# ------------------------------------------------------------- providers ---
PROVIDER_KEYS = {
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}

MODELS = [
    {"id": "gemini-3.5-flash", "provider": "gemini",
     "label": "Gemini 3.5 Flash — Google"},
    {"id": "gemini-3.1-flash-lite", "provider": "gemini",
     "label": "Gemini 3.1 Flash Lite — Google"},
    {"id": "openai/gpt-4o-mini", "provider": "openrouter",
     "label": "GPT-4o mini — via OpenRouter"},
    {"id": "anthropic/claude-sonnet-4.5", "provider": "openrouter",
     "label": "Claude Sonnet 4.5 — via OpenRouter"},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "provider": "openrouter",
     "label": "Llama 3.3 70B (free) — via OpenRouter"},
    {"id": "__custom__", "provider": "openrouter",
     "label": "Custom OpenRouter model…"},
    {"id": "gpt-4o-mini", "provider": "openai", "label": "GPT-4o mini — OpenAI direct"},
    {"id": "claude-opus-4-8", "provider": "anthropic",
     "label": "Claude Opus 4.8 — Anthropic direct"},
]

RERANK_URL = "https://openrouter.ai/api/v1/rerank"
RERANK_MODEL = "nvidia/llama-nemotron-rerank-vl-1b-v2:free"


class MissingKeyError(Exception):
    def __init__(self, provider: str, env_var: str):
        self.provider = provider
        self.env_var = env_var
        super().__init__(f"{provider} needs {env_var}")


def has_key(provider: str) -> bool:
    return any(os.environ.get(v) for v in PROVIDER_KEYS[provider])


def models_status() -> dict:
    return {
        "models": MODELS,
        "keys": {p: has_key(p) for p in PROVIDER_KEYS},
        "rerank": {"enabled": has_key("openrouter"), "model": RERANK_MODEL},
    }


_clients: dict = {}


def get_client(provider: str):
    """Watched client for a provider; raises MissingKeyError if no key."""
    if not has_key(provider):
        raise MissingKeyError(provider, PROVIDER_KEYS[provider][0])
    if provider not in _clients:
        _clients[provider] = bb.watch(make_client(provider))
    return _clients[provider]


def set_api_key(env_var: str, value: str) -> str:
    """Store a key entered on the website: process env + project .env file.

    Returns the provider it unlocks. Raises ValueError for unknown vars.
    """
    provider = next((p for p, vars_ in PROVIDER_KEYS.items() if env_var in vars_), None)
    if provider is None:
        raise ValueError(f"Unknown key name {env_var!r}")
    value = value.strip()
    if not value:
        raise ValueError("Empty key")
    os.environ[env_var] = value
    _clients.pop(provider, None)  # rebuild the client with the new key

    lines = []
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH, encoding="utf-8") as f:
            lines = f.read().splitlines()
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{env_var}="):
            lines[i] = f"{env_var}={value}"
            replaced = True
            break
    if not replaced:
        lines.append(f"{env_var}={value}")
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return provider


# ------------------------------------------------------------------ tools ---
FAKE_ORDERS = {
    "DS-1001": {"status": "delivered", "delivered_days_ago": 45, "item": "Nimbus 14\" laptop"},
    "DS-1002": {"status": "shipped", "carrier": "DHL", "eta_days": 2, "item": "Pulse 5G phone"},
    "DS-1003": {"status": "processing", "item": "Slate 11\" tablet"},
}

_ticket_counter = {"n": 55190}


@bb.tool
def create_support_ticket(customer_name: str, issue_summary: str) -> dict:
    """Open a support ticket for a customer issue (refund, repair, complaint).

    Args:
        customer_name: The customer's name.
        issue_summary: One-sentence summary of the issue.
    """
    _ticket_counter["n"] += 1
    return {"ticket_id": f"RF-{_ticket_counter['n']}", "status": "created",
            "customer": customer_name, "summary": issue_summary}


@bb.tool
def check_order_status(order_id: str) -> dict:
    """Look up the status of an order by its id (format DS-XXXX).

    Args:
        order_id: The order id, e.g. DS-1001.
    """
    order = FAKE_ORDERS.get(order_id.strip().upper())
    if order is None:
        return {"error": f"No order found with id {order_id!r}"}
    return {"order_id": order_id.upper(), **order}


TOOL_FUNCS = {"create_support_ticket": create_support_ticket,
              "check_order_status": check_order_status}

_TOOL_SCHEMAS = {
    "create_support_ticket": {
        "description": "Open a support ticket for a customer issue "
                       "(refund, repair, complaint).",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_name": {"type": "string", "description": "Customer's name"},
                "issue_summary": {"type": "string",
                                  "description": "One-sentence issue summary"},
            },
            "required": ["customer_name", "issue_summary"],
        },
    },
    "check_order_status": {
        "description": "Look up the status of an order by its id (format DS-XXXX).",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "Order id, e.g. DS-1001"},
            },
            "required": ["order_id"],
        },
    },
}

OPENAI_TOOLS = [
    {"type": "function",
     "function": {"name": name, "description": s["description"],
                  "parameters": s["parameters"]}}
    for name, s in _TOOL_SCHEMAS.items()
]

ANTHROPIC_TOOLS = [
    {"name": name, "description": s["description"], "input_schema": s["parameters"]}
    for name, s in _TOOL_SCHEMAS.items()
]


# -------------------------------------------------------------- retrieval ---
def _load_documents() -> list:
    docs = []
    for name in sorted(os.listdir(DOCS_DIR)):
        if name.endswith((".md", ".txt")):
            with open(os.path.join(DOCS_DIR, name), encoding="utf-8") as f:
                docs.append({"doc_id": name, "source": f"documents/{name}",
                             "text": f.read()})
    return docs


DOCUMENTS = _load_documents()

_STOPWORDS = {"the", "and", "for", "can", "what", "how", "with", "you",
              "your", "have", "this", "that", "please"}


def retrieve(question: str, top_k: int = 5) -> list:
    """Tiny keyword retriever — swap for your vector store in a real system."""
    words = [w for w in re.findall(r"[a-z]{3,}", question.lower())
             if w not in _STOPWORDS]
    scored = []
    for doc in DOCUMENTS:
        text = doc["text"].lower()
        score = sum(text.count(w) for w in words)
        if score > 0:
            scored.append({**doc, "score": round(score / (len(words) or 1), 2)})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:top_k]


def rerank_chunks(question: str, chunks: list, rec):
    """Rerank retrieved chunks with OpenRouter's rerank API (if key present).

    Recorded as a custom `rerank` black-box event either way (result or
    failure), so investigations show exactly how the final ranking happened.
    Returns (reranked_chunks, rerank_event_seq_or_None).
    """
    if not chunks or not has_key("openrouter"):
        return chunks, None
    try:
        resp = httpx.post(
            RERANK_URL,
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                     "Content-Type": "application/json"},
            json={"model": RERANK_MODEL, "query": question,
                  "documents": [{"text": c["text"]} for c in chunks],
                  "top_n": min(3, len(chunks))},
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        reranked = [{**chunks[r["index"]],
                     "rerank_score": round(r["relevance_score"], 4)}
                    for r in results]
        seq = rec.custom("rerank", {
            "provider": "openrouter", "model": RERANK_MODEL, "query": question,
            "results": [{"doc_id": chunks[r["index"]]["doc_id"],
                         "source": chunks[r["index"]].get("source"),
                         "relevance_score": round(r["relevance_score"], 4)}
                        for r in results],
        })
        return reranked, seq
    except Exception as exc:
        seq = rec.custom("rerank", {
            "provider": "openrouter", "model": RERANK_MODEL, "error": str(exc),
            "note": "rerank failed; falling back to keyword ranking",
        })
        return chunks, seq


# ------------------------------------------------------- per-provider ask ---
def _ask_gemini(client, model: str, context: str, question: str) -> str:
    from google.genai import types
    response = client.models.generate_content(
        model=model,
        contents=f"Company documents:\n{context}\n\nCustomer message: {question}",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[create_support_ticket, check_order_status],
            # thinking_level="high" makes Gemini 3 return a real thought
            # summary — without it, flash-lite produces no reasoning at all.
            thinking_config=types.ThinkingConfig(
                include_thoughts=True, thinking_level="high"),
        ),
    )
    return response.text or "(the model returned no text)"


def _ask_openai_compat(client, model: str, context: str, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",
         "content": f"Company documents:\n{context}\n\nCustomer message: {question}"},
    ]
    for _ in range(4):  # agent loop, recorded call by call
        resp = client.chat.completions.create(
            model=model, messages=messages, tools=OPENAI_TOOLS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or "(the model returned no text)"
        messages.append(msg)
        for tc in msg.tool_calls:
            fn = TOOL_FUNCS.get(tc.function.name)
            try:
                out = fn(**json.loads(tc.function.arguments)) if fn else \
                    {"error": f"unknown tool {tc.function.name}"}
            except Exception as exc:
                out = {"error": str(exc)}
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(out)})
    return "(the agent stopped after too many tool rounds)"


def _ask_anthropic(client, model: str, context: str, question: str) -> str:
    messages = [{"role": "user",
                 "content": f"Company documents:\n{context}\n\n"
                            f"Customer message: {question}"}]
    for _ in range(4):
        resp = client.messages.create(
            model=model, max_tokens=16000,
            thinking={"type": "adaptive", "display": "summarized"},
            system=SYSTEM_PROMPT, tools=ANTHROPIC_TOOLS, messages=messages)
        if resp.stop_reason != "tool_use":
            return "".join(b.text for b in resp.content if b.type == "text") \
                or "(the model returned no text)"
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for block in resp.content:
            if block.type == "tool_use":
                fn = TOOL_FUNCS.get(block.name)
                try:
                    out = fn(**block.input) if fn else \
                        {"error": f"unknown tool {block.name}"}
                except Exception as exc:
                    out = {"error": str(exc)}
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": json.dumps(out)})
        messages.append({"role": "user", "content": results})
    return "(the agent stopped after too many tool rounds)"


# ------------------------------------------------------------------ agent ---
_meta_clients: dict = {}


def _meta_client(provider: str):
    """An UNWATCHED client for meta-calls (self-explanation) so they don't
    show up as extra LLM calls in the recording's cockpit-voice section."""
    if provider not in _meta_clients:
        _meta_clients[provider] = make_client(provider)
    return _meta_clients[provider]


def _self_explain(provider, model, context, question, answer, rec):
    """Ask the model to state, in 1-2 sentences, which document it used and why.

    A post-hoc self-explanation for models that don't expose thinking (OpenAI,
    most OpenRouter models). It's testimony, not proof — recorded as its own
    `self_explanation` event and clearly labeled in the report.
    """
    prompt = (
        "You just answered a customer support question using the documents "
        "below. In 1-2 sentences, state which specific document you based your "
        "answer on and why. Name the document.\n\n"
        f"Documents:\n{context}\n\nQuestion: {question}\n\nYour answer: {answer}\n\n"
        "Which document did you rely on, and why?"
    )
    try:
        client = _meta_client(provider)
        if provider == "gemini":
            from google.genai import types
            r = client.models.generate_content(
                model=model, contents=prompt,
                config=types.GenerateContentConfig(max_output_tokens=512))
            text = r.text
        elif provider == "anthropic":
            r = client.messages.create(
                model=model, max_tokens=400,
                messages=[{"role": "user", "content": prompt}])
            text = "".join(b.text for b in r.content if b.type == "text")
        else:  # openai / openrouter
            r = client.chat.completions.create(
                model=model, max_tokens=250,
                messages=[{"role": "user", "content": prompt}])
            text = r.choices[0].message.content
        text = (text or "").strip()
        if text:
            rec.custom("self_explanation", {
                "provider": provider, "model": model,
                "asked": "which document did you rely on, and why?",
                "response": text})
            return text
    except Exception as exc:
        rec.custom("self_explanation", {
            "provider": provider, "model": model, "error": str(exc),
            "note": "self-explanation call failed"})
    return None


def _record_decision(rec, all_chunks, used_chunks, retrieval_seq, rerank_seq,
                     llm_seq, reasoning, reason_source):
    """Build a decision event from what actually happened this turn.

    `stated_reason` is the model's own words — its thinking during generation
    if it exposed any, otherwise a post-hoc self-explanation (clearly flagged).
    Testimony, not proof. `evidence_refs` point at the retrieval, the rerank,
    and the LLM call so the report shows what the agent saw vs what it said.
    A tool_call this turn means the agent acted, not just answered.
    """
    tool_event = rec.last("tool_call")
    top_doc = used_chunks[0]["doc_id"] if used_chunks else "none"

    if tool_event is not None:
        choice = f"used_tool:{tool_event['payload'].get('name')} (grounded on {top_doc})"
    else:
        choice = f"answered_from_document:{top_doc}"

    alternatives = [f"use_document:{c['doc_id']}" for c in all_chunks
                    if c["doc_id"] != top_doc]
    alternatives += ["ask_clarifying_question", "escalate_to_human"]

    if reasoning and reason_source == "posthoc":
        stated_reason = ("[post-hoc self-explanation — the agent was asked why "
                         "AFTER answering, so treat it as testimony] " + reasoning)
    elif reasoning:
        stated_reason = reasoning
    else:
        stated_reason = (
            "The model exposed no reasoning and no self-explanation was "
            "available. Compare the response against the retrieved evidence on "
            "the right to judge whether it used the correct (most recent) document."
        )
    evidence_refs = [s for s in (retrieval_seq, rerank_seq, llm_seq) if s is not None]
    rec.decision(choice=choice, alternatives=alternatives,
                 stated_reason=stated_reason, evidence_refs=evidence_refs)


def ask(question: str, model_id: str = "gemini-3.5-flash",
        custom_model: str = "", user_id: str = "web-visitor"):
    """Answer one question with the chosen model.

    Returns (answer, session_id). Raises MissingKeyError before any session
    is opened if the provider's API key is absent.
    """
    entry = next((m for m in MODELS if m["id"] == model_id), None)
    if entry is None:
        raise ValueError(f"Unknown model {model_id!r}")
    provider = entry["provider"]
    model = custom_model.strip() if entry["id"] == "__custom__" else entry["id"]
    if not model:
        raise ValueError("Custom OpenRouter model name is empty")

    client = get_client(provider)  # MissingKeyError propagates to the server

    session_id = None
    try:
        with bb.session(pilot=f"demoshop-web-agent ({provider}:{model})",
                        user_id=user_id,
                        metadata={"provider": provider, "model": model}) as rec:
            session_id = rec.session_id
            rec.user_input(question, user_id=user_id, channel="web")

            all_chunks = retrieve(question)
            retrieval_seq = rec.retrieval(query=question, chunks=all_chunks)
            reranked, rerank_seq = rerank_chunks(question, all_chunks, rec)
            chunks = reranked[:3]
            context = "\n\n".join(
                f"--- {c['doc_id']} ---\n{c['text']}" for c in chunks
            ) or "(no relevant documents found)"

            if provider == "gemini":
                answer = _ask_gemini(client, model, context, question)
            elif provider == "anthropic":
                answer = _ask_anthropic(client, model, context, question)
            else:  # openai / openrouter share the OpenAI-compatible path
                answer = _ask_openai_compat(client, model, context, question)

            # The stated reason is the model's own words: its thinking if it
            # exposed any, else a short post-hoc self-explanation.
            main_llm = rec.last("llm_call")
            llm_seq = main_llm["seq"] if main_llm else None
            reasoning = (main_llm or {}).get("payload", {}).get("reasoning")
            reason_source = "thinking"
            if not reasoning:
                reasoning = _self_explain(provider, model, context, question,
                                          answer, rec)
                reason_source = "posthoc"

            # Record the decision so the report shows the agent's stated reason
            # side-by-side with ALL the evidence it had (used and ignored).
            _record_decision(rec, all_chunks, chunks, retrieval_seq, rerank_seq,
                             llm_seq, reasoning, reason_source)

            rec.final_output(answer)
            return answer, session_id
    except Exception as exc:
        # The session recorded the error + traceback before closing.
        answer = (f"Sorry — the agent hit an error and the incident was "
                  f"recorded ({type(exc).__name__}: {exc}). Session: {session_id}")
        return answer, session_id or "?"


if __name__ == "__main__":
    answer, sid = ask("I bought a laptop 45 days ago and it broke. Refund?")
    print(f"[{sid}] {answer}")
