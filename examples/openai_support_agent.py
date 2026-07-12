"""
The same support agent on the OpenAI SDK, fully recorded by the black box.

Run:  OPENAI_API_KEY=... python examples/openai_support_agent.py
Then: blackbox list && blackbox report <session> && blackbox serve
"""

import json

from openai import OpenAI

from agent_blackbox import BlackBox

bb = BlackBox("recordings", redact=True)
client = bb.watch(OpenAI())                # every chat.completions.create recorded

KNOWLEDGE_BASE = [
    {"doc_id": "policy_2023_v1", "score": 0.91,
     "text": "Refund Policy: full refund within 90 days of purchase for electronics."},
    {"doc_id": "policy_2026_v3", "score": 0.84,
     "text": "Refund Policy (effective Jan 2026): refund window is 30 days."},
]

TOOLS = [{
    "type": "function",
    "function": {
        "name": "create_refund_ticket",
        "description": "Open a refund ticket for an eligible customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string"},
                "order_days_old": {"type": "integer"},
            },
            "required": ["customer_id", "order_days_old"],
        },
    },
}]


@bb.tool
def create_refund_ticket(customer_id: str, order_days_old: int) -> dict:
    return {"ticket_id": "RF-55190", "status": "created"}


def run_agent(question: str, user_id: str) -> str:
    with bb.session(pilot="support-agent-v3-oai", user_id=user_id) as rec:
        rec.user_input(question, user_id=user_id)
        chunks = KNOWLEDGE_BASE
        rec.retrieval(query=question, chunks=chunks)

        context = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in chunks)
        messages = [
            {"role": "system", "content": "You are DemoShop's support agent. Answer "
             "using ONLY the most recent policy document."},
            {"role": "user", "content": f"Policies:\n{context}\n\nCustomer: {question}"},
        ]

        while True:
            resp = client.chat.completions.create(
                model="gpt-4o", messages=messages, tools=TOOLS)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                break
            messages.append(msg)
            for tc in msg.tool_calls:
                out = create_refund_ticket(**json.loads(tc.function.arguments))
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(out)})

        answer = msg.content or ""
        rec.final_output(answer)
        print(f"session: {rec.session_id}")
        return answer


if __name__ == "__main__":
    print(run_agent("I bought a laptop 45 days ago and it broke. Refund?",
                    user_id="cust_88213"))
