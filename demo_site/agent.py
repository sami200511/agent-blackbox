"""
DemoShop web agent: Gemini over a folder of company documents, with tools,
fully recorded by the black box.

Every ask() call is one black-box session containing:
  user_input -> retrieval (docs + scores) -> tool_call(s) -> llm_call
  (with Gemini's thinking as `reasoning`) -> final_output
"""

# NOTE: no `from __future__ import annotations` in this file! google-genai's
# automatic function calling reads tool annotations with isinstance(), which
# breaks when PEP 563 turns them into strings (the SDK raises
# "isinstance() arg 2 must be a type" while invoking the tool).

import os
import re

from google.genai import types

from agent_blackbox import BlackBox, load_env_file, make_client

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")

MODEL = "gemini-3.5-flash"

# Key from .env at the project root (or exported in the shell).
load_env_file(os.path.join(BASE_DIR, os.pardir, ".env"))
load_env_file()  # also allow a local .env

bb = BlackBox(RECORDINGS_DIR, redact=True)
client = bb.watch(make_client("gemini"))

SYSTEM_PROMPT = (
    "You are DemoShop's customer support agent on the website. Answer using "
    "ONLY the provided policy documents; when documents conflict, follow the "
    "most recent effective one. Use your tools when the customer asks you to "
    "DO something (open a ticket, check an order). Be concise and friendly. "
    "If the documents don't cover the question, say so and offer to escalate."
)

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


# -------------------------------------------------------------- retrieval ---
def _load_documents() -> list[dict]:
    docs = []
    for name in sorted(os.listdir(DOCS_DIR)):
        if name.endswith((".md", ".txt")):
            with open(os.path.join(DOCS_DIR, name), encoding="utf-8") as f:
                docs.append({"doc_id": name, "source": f"documents/{name}",
                             "text": f.read()})
    return docs


DOCUMENTS = _load_documents()


def retrieve(question: str, top_k: int = 3) -> list[dict]:
    """Tiny keyword retriever — swap for your vector store in a real system."""
    words = [w for w in re.findall(r"[a-z]{3,}", question.lower())
             if w not in {"the", "and", "for", "can", "what", "how", "with",
                          "you", "your", "have", "this", "that", "please"}]
    scored = []
    for doc in DOCUMENTS:
        text = doc["text"].lower()
        score = sum(text.count(w) for w in words)
        if score > 0:
            scored.append({**doc, "score": round(score / (len(words) or 1), 2)})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:top_k]


# ------------------------------------------------------------------ agent ---
def ask(question: str, user_id: str = "web-visitor") -> tuple[str, str]:
    """Answer one question. Returns (answer, black-box session id)."""
    session_id = None
    try:
        with bb.session(pilot="demoshop-web-agent-v1", user_id=user_id) as rec:
            session_id = rec.session_id
            rec.user_input(question, user_id=user_id, channel="web")

            chunks = retrieve(question)
            rec.retrieval(query=question, chunks=chunks)
            context = "\n\n".join(
                f"--- {c['doc_id']} (score {c['score']}) ---\n{c['text']}"
                for c in chunks
            ) or "(no relevant documents found)"

            response = client.models.generate_content(   # auto-recorded
                model=MODEL,
                contents=f"Company documents:\n{context}\n\nCustomer message: {question}",
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    # tools run through @bb.tool, so executions are recorded
                    tools=[create_support_ticket, check_order_status],
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                ),
            )
            answer = response.text or "(the model returned no text)"
            rec.final_output(answer)
            return answer, session_id
    except Exception as exc:
        # The session already recorded the error + traceback before closing.
        return (f"Sorry — the agent hit an error and the incident was recorded "
                f"({type(exc).__name__}). Session: {session_id}"), session_id or "?"


if __name__ == "__main__":
    answer, sid = ask("I bought a laptop 45 days ago and it broke. Refund?")
    print(f"[{sid}] {answer}")
