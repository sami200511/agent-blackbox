"""
The support agent through OpenRouter — one API for many models.

OpenRouter is OpenAI-compatible, so recording works through the same OpenAI
adapter; the black box labels these calls provider="openrouter".

API key comes from the environment: OPENROUTER_API_KEY (shell export or .env
file). Pick any model from https://openrouter.ai/models via MODEL below.

Run:  python examples/openrouter_support_agent.py
"""

from agent_blackbox import BlackBox
from agent_blackbox.clients import load_env_file, make_client

load_env_file()
bb = BlackBox("recordings", redact=True)
client = bb.watch(make_client("openrouter"))

MODEL = "openai/gpt-4o-mini"          # or e.g. "google/gemini-2.5-flash", ...

KNOWLEDGE_BASE = [
    {"doc_id": "policy_2023_v1", "score": 0.91,
     "text": "Refund Policy: full refund within 90 days of purchase for electronics."},
    {"doc_id": "policy_2026_v3", "score": 0.84,
     "text": "Refund Policy (effective Jan 2026): refund window is 30 days."},
]


def run_agent(question: str, user_id: str) -> str:
    with bb.session(pilot="support-agent-v3-openrouter", user_id=user_id) as rec:
        rec.user_input(question, user_id=user_id)
        rec.retrieval(query=question, chunks=KNOWLEDGE_BASE)

        context = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in KNOWLEDGE_BASE)
        resp = client.chat.completions.create(       # recorded automatically
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are DemoShop's support agent. "
                 "Answer using ONLY the most recent policy document."},
                {"role": "user", "content": f"Policies:\n{context}\n\nCustomer: {question}"},
            ],
        )
        answer = resp.choices[0].message.content or ""
        rec.final_output(answer)
        print(f"session: {rec.session_id}")
        return answer


if __name__ == "__main__":
    print(run_agent("I bought a laptop 45 days ago and it broke. Refund?",
                    user_id="cust_88213"))
