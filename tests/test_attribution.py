from agent_blackbox import documents_referenced

DOCS = ["refund-policy-OLD.md", "refund-policy-current.md", "product-catalog.md",
        "company-faq.md", "shipping-policy.md"]


def test_ignores_superseded_uses_current():
    text = ("refund-policy-OLD.md seems to be superseded, so I'll ignore that. "
            "refund-policy-current.md indicates a 30-day refund window. "
            "The company-faq.md confirms this.")
    used, ignored = documents_referenced(text, DOCS)
    assert used == ["refund-policy-current.md", "company-faq.md"]
    assert ignored == ["refund-policy-OLD.md"]


def test_primary_is_first_mentioned():
    text = ("Based on company-faq.md and refund-policy-current.md, the window "
            "is 30 days.")
    used, ignored = documents_referenced(text, DOCS)
    assert used[0] == "company-faq.md"        # mentioned first
    assert "refund-policy-current.md" in used
    assert ignored == []


def test_unmentioned_documents_in_neither_list():
    text = "The shipping-policy.md says we ship to the UK in 3-5 days."
    used, ignored = documents_referenced(text, DOCS)
    assert used == ["shipping-policy.md"]
    assert ignored == []


def test_use_cue_overrides_nearby_ignore_word():
    # current.md is USED even though 'supersedes' appears right after it
    text = "refund-policy-current.md is the one in effect; it supersedes the old one."
    used, ignored = documents_referenced(text, DOCS)
    assert "refund-policy-current.md" in used
    assert "refund-policy-current.md" not in ignored


def test_matches_spaced_and_extensionless_names():
    text = "According to the refund policy current doc, the window is 30 days."
    used, ignored = documents_referenced(text, DOCS)
    assert "refund-policy-current.md" in used


def test_empty_text():
    assert documents_referenced("", DOCS) == ([], [])
    assert documents_referenced(None, DOCS) == ([], [])
