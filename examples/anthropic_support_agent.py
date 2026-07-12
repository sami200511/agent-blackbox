"""
A real support agent on the Anthropic API, fully recorded by the black box.

Everything the agent does lands in the recording automatically:
  - the LLM calls (bb.watch wraps the client)
  - the model's thinking blocks -> the recording's `reasoning` (the CVR)
  - the tool executions (@bb.tool decorator)
Plus the manual context events you choose to add (user_input, retrieval,
decision, final_output).

Run:  ANTHROPIC_API_KEY=... python examples/anthropic_support_agent.py
Then: blackbox list && blackbox report <session> && blackbox serve
"""

import json

import anthropic

from agent_blackbox import BlackBox

bb = BlackBox("recordings", redact=True)   # redact=True scrubs emails/keys/cards
client = bb.watch(anthropic.Anthropic())   # every messages.create is now recorded

MODEL = "claude-opus-4-8"

KNOWLEDGE_BASE = [
    {"doc_id": "policy_2023_v1", "source": "kb/refund-policy-OLD.md", "score": 0.91,
     "text": "Refund Policy: full refund within 90 days of purchase for electronics."},
    {"doc_id": "policy_2026_v3", "source": "kb/refund-policy-current.md", "score": 0.84,
     "text": "Refund Policy (effective Jan 2026): refund window is 30 days. "
             "After 30 days, offer repair under warranty instead."},
]

TOOLS = [{
    "name": "create_refund_ticket",
    "description": "Open a refund ticket for a customer order. Call this only "
                   "after confirming the customer is eligible under the current policy.",
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string"},
            "order_days_old": {"type": "integer"},
        },
        "required": ["customer_id", "order_days_old"],
        "additionalProperties": False,
    },
}]


@bb.tool
def create_refund_ticket(customer_id: str, order_days_old: int) -> dict:
    # your real ticketing system call goes here
    return {"ticket_id": "RF-55190", "status": "created"}


def retrieve(query: str) -> list[dict]:
    # your real vector search goes here
    return KNOWLEDGE_BASE


def run_agent(question: str, user_id: str) -> str:
    with bb.session(pilot="support-agent-v3", user_id=user_id) as rec:
        rec.user_input(question, user_id=user_id, channel="web_chat")

        chunks = retrieve(question)
        rec.retrieval(query=question, chunks=chunks)

        context = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in chunks)
        messages = [{"role": "user",
                     "content": f"Policy documents:\n{context}\n\nCustomer: {question}"}]

        # simple agentic loop — every iteration is recorded automatically
        while True:
            response = client.messages.create(
                model=MODEL,
                max_tokens=16000,
                thinking={"type": "adaptive", "display": "summarized"},
                system="You are DemoShop's support agent. Answer using ONLY the "
                       "most recent policy document. Be concise and friendly.",
                tools=TOOLS,
                messages=messages,
            )
            if response.stop_reason != "tool_use":
                break
            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    out = create_refund_ticket(**block.input)   # recorded by @bb.tool
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": json.dumps(out)})
            messages.append({"role": "user", "content": results})

        answer = "".join(b.text for b in response.content if b.type == "text")
        rec.final_output(answer)
        print(f"session: {rec.session_id}")
        return answer


if __name__ == "__main__":
    print(run_agent("I bought a laptop 45 days ago and it broke. Refund?",
                    user_id="cust_88213"))
