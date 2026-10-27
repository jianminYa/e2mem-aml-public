"""Unit tests for PairGuard (Contradiction and Temporal Update detection)."""
import pytest
from service.pair_guard import EvidenceItem, PairGuard, has_negation, planning_status


def test_polarity_and_planning_detection():
    tokens_pos = ["i", "am", "feeling", "anxious", "about", "grammar"]
    tokens_neg = ["i", "have", "never", "felt", "anxious", "about", "grammar"]
    tokens_plan = ["user", "plans", "to", "visit", "dr", "smith", "on", "march", "27"]
    tokens_done = ["user", "already", "visited", "dr", "smith", "yesterday"]

    assert not has_negation(tokens_pos)
    assert has_negation(tokens_neg)
    assert planning_status(tokens_plan) == "planned"
    assert planning_status(tokens_done) == "completed"

    # Test contractions
    from service.pair_guard import tokenize_words
    assert has_negation(tokenize_words("I didn't receive the package"))
    assert has_negation(tokenize_words("We haven't met attorney Stephanie"))
    assert has_negation(tokenize_words("They aren't attending the workshop"))


def test_pair_guard_detects_contradiction():
    guard = PairGuard(min_shared_keywords=2)

    item1 = EvidenceItem(
        id="ev_pos",
        content="I am feeling very anxious about my grammar accuracy after receiving feedback.",
        score=0.92,
        kind="event",
    )
    item2 = EvidenceItem(
        id="ev_neg",
        content="I've never felt anxious about grammar accuracy in any writing session.",
        score=0.45,  # Low score because query words matched item1 more closely
        kind="event",
    )
    item_unrelated = EvidenceItem(
        id="ev_other",
        content="I love cooking pasta with fresh tomatoes and basil.",
        score=0.70,
        kind="event",
    )

    items = [item1, item_unrelated, item2]
    pairs = guard.detect_pairs(items)

    assert len(pairs) == 1
    p = pairs[0]
    assert p.relation_type == "contradiction"
    assert "grammar" in p.shared_keywords or "anxious" in p.shared_keywords
    assert p.confidence > 0.5


def test_pair_guard_promotes_counter_evidence():
    guard = PairGuard(min_shared_keywords=2)

    item1 = EvidenceItem(
        id="ev_pos",
        content="I am feeling anxious about my grammar accuracy.",
        score=0.95,
        kind="event",
    )
    item_filler_1 = EvidenceItem(id="ev_f1", content="Weather is sunny today.", score=0.85, kind="event")
    item_filler_2 = EvidenceItem(id="ev_f2", content="Going for a quick morning run.", score=0.80, kind="event")
    item2 = EvidenceItem(
        id="ev_neg",
        content="I have never felt anxious about my grammar accuracy.",
        score=0.40,  # Would normally be cut or pushed far down
        kind="event",
    )

    ranked = [item1, item_filler_1, item_filler_2, item2]
    reordered, pairs = guard.apply_pair_guard(ranked, max_promotions=2)

    # item2 should be promoted immediately after item1
    ids = [x.id for x in reordered]
    assert ids[0] == "ev_pos"
    assert ids[1] == "ev_neg"  # Promoted right next to its paired contradiction!
    assert "ev_f1" in ids
    assert "ev_f2" in ids
