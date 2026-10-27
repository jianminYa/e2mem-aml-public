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


def test_top_k_boundary_pair_guard_never_overflows(tmp_path):
    """Verify that Search top_k is strictly obeyed even when atomic pairs are at the boundary."""
    from service.core import MemoryService
    db_file = tmp_path / "test_overflow.sqlite3"
    svc = MemoryService(str(db_file), mode="simulator")
    svc.db.execute("CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, user_id TEXT, session_id TEXT, body_hash TEXT)")
    svc.db.execute("CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, user_id TEXT, session_id TEXT, request_id TEXT, kind TEXT, parent_id TEXT, turn_index INT, role TEXT, source_time TEXT, content TEXT, evidence TEXT, vector BLOB, created_at TEXT, source_refs TEXT, aliases TEXT, keywords TEXT, event_type TEXT, time_anchor TEXT, topic_labels TEXT)")
    
    # insert 2 paired items
    svc.db.execute("INSERT INTO memories (id, user_id, session_id, request_id, kind, turn_index, role, source_time, content, evidence, vector, created_at, source_refs, aliases, keywords, event_type, time_anchor, topic_labels) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ev_pos", "u1", "s1", "r1", "event", 0, "user", "2024", "I felt very anxious about exams", "anxious about exams", b"", "now", "[]", "[]", "[]", "", "", ""))
    svc.db.execute("INSERT INTO memories (id, user_id, session_id, request_id, kind, turn_index, role, source_time, content, evidence, vector, created_at, source_refs, aliases, keywords, event_type, time_anchor, topic_labels) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ev_neg", "u1", "s1", "r1", "event", 1, "user", "2024", "I have never felt anxious about exams", "never felt anxious", b"", "now", "[]", "[]", "[]", "", "", ""))
    svc.db.commit()

    # When top_k=1, len(output) must be <= 1 (never 2!)
    res, _ = svc.search({"user_id": "u1", "query": "anxious exams", "top_k": 1})
    assert len(res["data"]) <= 1

    # When top_k=2, both fit
    res2, _ = svc.search({"user_id": "u1", "query": "anxious exams", "top_k": 2})
    assert len(res2["data"]) == 2

    # When top_k=10, len(output) must be <= 10
    res10, _ = svc.search({"user_id": "u1", "query": "anxious exams", "top_k": 10})
    assert len(res10["data"]) <= 10

    # When top_k=100, len(output) must be <= 100
    res100, _ = svc.search({"user_id": "u1", "query": "anxious exams", "top_k": 100})
    assert len(res100["data"]) <= 100

