"""
The support agent on Google Gemini (google-genai SDK), fully recorded.

API key comes from the environment: GEMINI_API_KEY (or GOOGLE_API_KEY),
either exported in the shell or placed in a .env file next to this project.

Run:  python examples/gemini_support_agent.py
Then: blackbox list && blackbox serve
"""

from google import genai
from google.genai import types

from agent_blackbox import BlackBox
from agent_blackbox.clients import load_env_file, make_client

load_env_file()                       # reads .env if present; env vars win
bb = BlackBox("recordings", redact=True)
client = bb.watch(make_client("gemini"))

MODEL = "gemini-3.5-flash"

KNOWLEDGE_BASE = [
    {"doc_id": "policy_2023_v1", "score": 0.91,
     "text": "Refund Policy: full refund within 90 days of purchase for electronics."},
    {"doc_id": "policy_2026_v3", "score": 0.84,
     "text": "Refund Policy (effective Jan 2026): refund window is 30 days. "
             "After 30 days, offer repair under warranty instead."},
]


def run_agent(question: str, user_id: str) -> str:
    with bb.session(pilot="support-agent-v3-gemini", user_id=user_id) as rec:
        rec.user_input(question, user_id=user_id, channel="web_chat")
        chunks = KNOWLEDGE_BASE
        rec.retrieval(query=question, chunks=chunks)

        context = "\n\n".join(f"[{c['doc_id']}] {c['text']}" for c in chunks)
        response = client.models.generate_content(   # recorded automatically
            model=MODEL,
            contents=f"Policy documents:\n{context}\n\nCustomer: {question}",
            config=types.GenerateContentConfig(
                system_instruction="You are DemoShop's support agent. Answer "
                                   "using ONLY the most recent policy document. "
                                   "Be concise and friendly.",
                # include_thoughts=True records the model's reasoning summary
                # into the black box — the cockpit voice recorder.
                thinking_config=types.ThinkingConfig(include_thoughts=True),
            ),
        )
        answer = response.text or ""
        rec.final_output(answer)
        print(f"session: {rec.session_id}")
        return answer


if __name__ == "__main__":
    print(run_agent("I bought a laptop 45 days ago and it broke. Refund?",
                    user_id="cust_88213"))
