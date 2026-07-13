from agent_blackbox import attribute, bm25, grounding_scores
from agent_blackbox.retrieval import distinctive_terms, tokenize

OLD = ("Refund Policy 2023 STATUS SUPERSEDED. Customers may request a full "
       "refund within 90 days of purchase for electronics including laptops.")
CURRENT = ("Refund Policy effective January 2026. The refund window is 30 days "
           "from delivery for items in original condition. After 30 days offer "
           "a repair under warranty instead.")
SHIP = ("Shipping Policy. Standard shipping 3 to 5 business days free over $50. "
        "Express shipping 1 to 2 days. We ship to the EU UK and US.")
FAQ = ("FAQ. Can I change my delivery address? Do you price match? How do I "
       "reach a human? Refunds follow the current 30 day policy.")

DOCS = [{"doc_id": "refund-OLD.md", "text": OLD},
        {"doc_id": "refund-current.md", "text": CURRENT},
        {"doc_id": "shipping.md", "text": SHIP},
        {"doc_id": "faq.md", "text": FAQ}]
TEXTS = [d["text"] for d in DOCS]


def test_bm25_scores_are_differentiated():
    scores = bm25("shipping to the UK how many days", TEXTS)
    assert len(scores) == 4
    ranked = sorted(zip([d["doc_id"] for d in DOCS], scores), key=lambda x: -x[1])
    assert ranked[0][0] == "shipping.md"        # shipping doc wins its query
    assert len(set(scores)) > 1                  # not all tied


def test_distinctive_terms_separate_similar_docs():
    terms = distinctive_terms(TEXTS)
    old_terms, cur_terms = terms[0], terms[1]
    assert "90" in old_terms and "90" not in cur_terms
    assert "30" in cur_terms and "30" not in old_terms


def test_grounding_prefers_the_reflected_document():
    # An answer that reflects the CURRENT policy should ground it, not OLD
    answer = ("Under our current policy the refund window is 30 days from "
              "delivery; after that we offer a repair under warranty.")
    g = grounding_scores(answer, TEXTS)
    gmap = dict(zip([d["doc_id"] for d in DOCS], g))
    assert gmap["refund-current.md"] > gmap["refund-OLD.md"]


def test_attribute_credits_current_not_superseded():
    answer = ("Since it has been 45 days you are not eligible for a refund "
              "under our 30 day policy, but the warranty covers a repair.")
    reasoning = ("The old policy says 90 days but it is superseded so I ignore "
                 "refund-OLD.md. The current policy says 30 days.")
    statuses, gmap = attribute(answer, DOCS, cue_text=reasoning)
    by_id = {s["doc_id"]: s for s in statuses}
    assert by_id["refund-current.md"]["status"] == "used"
    assert by_id["refund-OLD.md"]["status"] != "used"       # not credited
    # highest grounding is the current policy → primary
    assert statuses[0]["doc_id"] == "refund-current.md"


def test_attribute_works_without_filenames():
    # model never names a file — grounding still attributes correctly
    answer = "We ship to the UK. Standard shipping is 3 to 5 business days."
    statuses, gmap = attribute(answer, DOCS)
    assert statuses[0]["doc_id"] == "shipping.md"
    assert statuses[0]["status"] == "used"


def test_bare_mention_is_not_credited_as_used():
    # The model discusses the OLD policy while rejecting it; a bare mention in
    # the reasoning must NOT make it "used" (answer reflects the current policy).
    answer = ("Under our 30 day policy you are not eligible for a refund after "
              "45 days; a warranty repair is available.")
    reasoning = ("There is an old policy at refund-OLD.md with a 90 day window, "
                 "but the current policy says 30 days.")
    statuses, _ = attribute(answer, DOCS, cue_text=reasoning)
    by_id = {s["doc_id"]: s for s in statuses}
    assert by_id["refund-current.md"]["status"] == "used"
    assert by_id["refund-OLD.md"]["status"] != "used"     # discussed, not used


def test_endorsed_document_is_used_even_if_grounding_low():
    answer = "Refunds are within 30 days; after that it is a warranty repair."
    reasoning = "The FAQ confirms this: refunds follow the 30 day policy."
    statuses, _ = attribute(answer, DOCS, cue_text=reasoning)
    by_id = {s["doc_id"]: s for s in statuses}
    assert by_id["faq.md"]["status"] == "used"            # endorsed via "confirms"


def test_tokenize_drops_stopwords():
    toks = tokenize("The refund is within 30 days of delivery")
    assert "the" not in toks and "is" not in toks and "of" not in toks
    assert "refund" in toks and "30" in toks
