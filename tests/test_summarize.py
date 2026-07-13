from agent_blackbox import condense_reason


def test_picks_informative_line_from_thinking():
    text = (
        "**Analyzing the Refund Request**\n\n"
        "Okay, so a customer bought a laptop 45 days ago, and it's broken. "
        "They want a refund. My job is to figure out if they qualify. "
        "The current refund policy states refunds are only available within 30 "
        "days of delivery. Since it's been 45 days, the customer doesn't qualify "
        "for a refund, but the warranty applies, so it's a repair instead."
    )
    short = condense_reason(text)
    assert len(short) <= 260
    # the chosen line carries the actual decision, not the filler opener
    assert "Okay, so a customer" not in short
    assert "doesn't qualify" in short or "within 30" in short
    assert "**" not in short  # markdown stripped


def test_strips_markdown_quotes_and_dedupes():
    text = ('The customer stated, "I bought a laptop and it broke broke". '
            "After 30 days, defective products should be offered a repair under "
            "warranty instead.")
    short = condense_reason(text)
    assert '"' not in short and "`" not in short
    assert "broke broke" not in short
    assert "warranty" in short.lower()


def test_short_input_passes_through_cleaned():
    text = "Refund window is 30 days; the order is within it."
    short = condense_reason(text)
    assert "30 days" in short


def test_empty_input():
    assert condense_reason("") == ""
    assert condense_reason(None) == ""


def test_truncates_to_max_chars():
    text = ("The refund policy applies within thirty days and after that the "
            "warranty covers defective products for a full twelve months, "
            "which means the customer should be offered a repair rather than a "
            "refund because the return window has already closed for this order.")
    short = condense_reason(text, max_chars=80)
    assert len(short) <= 80
    assert short.endswith("…")


def test_no_sentence_qualifies_falls_back():
    # no domain/decision words, but still returns cleaned text (not empty)
    text = "Hello there. This is a friendly note with nothing useful in it here."
    short = condense_reason(text)
    assert short  # non-empty
    assert "**" not in short
