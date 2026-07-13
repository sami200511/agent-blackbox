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


def test_matches_distinctive_keyword_without_filename():
    # "FAQ" and "warranty" identify their docs even though the filename is not written
    text = "The FAQ confirms this, and the warranty covers a repair."
    used, ignored = documents_referenced(text, DOCS)
    assert "company-faq.md" in used
    # (no warranty doc in DOCS here, so just check faq is picked up by keyword)


def test_ignore_cue_before_the_filename():
    text = ("The old policy is superseded, so I ignore refund-policy-OLD.md. "
            "The current policy says 30 days.")
    used, ignored = documents_referenced(text, DOCS)
    assert ignored == ["refund-policy-OLD.md"]
    assert "refund-policy-OLD.md" not in used


def test_no_cross_sentence_cue_leak():
    # 'current policy' (a use-cue) in the NEXT sentence must not rescue OLD
    text = ("refund-policy-OLD.md is superseded so I ignore it. "
            "The current policy says 30 days.")
    used, ignored = documents_referenced(text, DOCS)
    assert ignored == ["refund-policy-OLD.md"]


def test_ambiguous_word_old_does_not_false_match():
    # "the old one" must NOT be attributed to refund-policy-OLD.md
    text = "refund-policy-current.md is in effect; it replaces the old one."
    used, ignored = documents_referenced(text, DOCS)
    assert "refund-policy-current.md" in used
    assert "refund-policy-OLD.md" not in ignored
    assert "refund-policy-OLD.md" not in used


def test_empty_text():
    assert documents_referenced("", DOCS) == ([], [])
    assert documents_referenced(None, DOCS) == ([], [])
