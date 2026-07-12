"""
Built-in demo incident (`blackbox demo`): a support agent gives a customer a
WRONG answer, and the black box records everything needed to investigate why.

The scenario (very common in real RAG systems):
  - Customer asks: "Can I still get a refund? I bought 45 days ago."
  - The knowledge base contains BOTH an outdated policy (90 days) and the
    current policy (30 days).
  - Retrieval ranks the OUTDATED doc higher (more keyword overlap).
  - The agent trusts the top chunk and answers "yes, 90 days" — wrong.

No network or API key needed — the model call is simulated.
"""

from __future__ import annotations

from .context import BlackBox


def run_demo(bb: BlackBox, session_id: str = "INC-DEMO-0001") -> str:
    with bb.session(session_id=session_id, pilot="support-agent-v2.3",
                    metadata={"company": "DemoShop", "environment": "demo"}) as rec:
        rec.user_input(
            "Hi, I bought a laptop 45 days ago and it stopped working. "
            "Can I still get a refund?",
            user_id="cust_88213", channel="web_chat",
        )
        rec.system_prompt(
            "You are DemoShop's support agent. Answer customer questions using "
            "the retrieved policy documents. Be concise and friendly.",
            version="prompt-v14",
        )
        retrieval_seq = rec.retrieval(
            query="refund policy time limit laptop",
            chunks=[
                {
                    "doc_id": "policy_2023_v1",
                    "source": "kb/returns/refund-policy-OLD.md",
                    "score": 0.91,
                    "text": "Refund Policy: Customers may request a full refund within "
                            "90 days of purchase for any electronics, including laptops.",
                },
                {
                    "doc_id": "policy_2026_v3",
                    "source": "kb/returns/refund-policy-current.md",
                    "score": 0.84,
                    "text": "Refund Policy (effective Jan 2026): Refund window is 30 days "
                            "from purchase. After 30 days, offer repair under warranty instead.",
                },
            ],
        )
        llm_seq = rec.llm_call(
            model="demo-model",
            prompt="[system prompt] + [2 retrieved chunks] + [customer question]",
            reasoning="The top-ranked policy document says refunds are allowed within "
                      "90 days. The customer is at 45 days, which is inside the window, "
                      "so they qualify for a refund.",
            response="Good news! Our refund policy covers purchases within 90 days, and "
                     "you're at 45 days — you qualify for a full refund.",
            params={"temperature": 0.3, "max_tokens": 300},
            usage={"input_tokens": 512, "output_tokens": 64},
            provider="demo",
        )
        rec.decision(
            choice="answer_using_policy_2023_v1",
            alternatives=[
                "answer_using_policy_2026_v3",
                "ask_clarifying_question",
                "escalate_to_human",
            ],
            stated_reason="Top retrieval result (score 0.91) states a 90-day refund window.",
            evidence_refs=[retrieval_seq, llm_seq],
            confidence=0.88,
        )
        rec.tool_call(
            name="create_refund_ticket",
            arguments={"customer_id": "cust_88213", "order_days_old": 45,
                       "policy_cited": "90-day"},
            result={"ticket_id": "RF-55190", "status": "created"},
            ok=True, latency_ms=142.7,
        )
        rec.final_output(
            "Good news! Our refund policy covers purchases within 90 days, and you're "
            "at 45 days — you qualify for a full refund. I've opened ticket RF-55190."
        )
        rec.annotation(
            "INCIDENT: Finance rejected ticket RF-55190 — current policy is 30 days. "
            "Customer was promised a refund they are not eligible for.",
            author="qa-review",
        )
    return session_id
