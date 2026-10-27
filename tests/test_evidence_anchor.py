"""Unit tests for EvidenceAnchorEngine and Token budgeting."""
import sqlite3
import pytest
from service.evidence_anchor import EvidenceAnchorEngine, token_count


def test_token_counter_o200k():
    text = "User visited Dr. Smith on March 27 for annual health checkup."
    count = token_count(text)
    assert 10 <= count <= 20
    assert token_count("") == 0


def test_resolve_raw_sources_for_event():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE memories(
            id TEXT PRIMARY KEY,
            user_id TEXT,
            kind TEXT,
            turn_index INTEGER,
            request_id TEXT,
            parent_id TEXT,
            role TEXT,
            source_time TEXT,
            content TEXT,
            source_refs TEXT
        )
    """)

    # Insert raw turns
    conn.execute("""
        INSERT INTO memories VALUES
        ('raw_1', 'user_1', 'raw', 0, 'req_1', NULL, 'user', '2026-03-01',
         'Here are the options: 5. Freelance writer 6. Proofreader 7. Transcriptionist', '["ref_turn_1"]'),
        ('raw_2', 'user_1', 'raw', 1, 'req_1', NULL, 'assistant', '2026-03-01',
         'Got it, transcriptionist is an excellent flexible remote career.', '["ref_turn_2"]')
    """)
    conn.commit()

    engine = EvidenceAnchorEngine(conn)
    # Given an Event that references ref_turn_1
    event = {
        "id": "ev_1",
        "parent_id": "ep_1",
        "source_refs": ["ref_turn_1"],
    }

    resolved = engine.resolve_raw_sources_for_event(event, user_id="user_1")
    assert len(resolved) == 1
    assert "7. Transcriptionist" in resolved[0]["content"]
    assert resolved[0]["id"] == "raw_1"


def test_budget_cutoff_protection():
    engine = EvidenceAnchorEngine(None)
    # Create several mock hits
    hits = [
        {"id": f"ev_{i}", "content": f"Memory statement {i} with some detailed context text.", "score": 0.9 - i * 0.05}
        for i in range(50)
    ]

    # Test small budget (e.g. 500 tokens)
    assembled = engine.assemble_anchored_context(hits, user_id="user_1", budget=500, enable_pair_guard=False)
    assert len(assembled) < len(hits)
    total_tokens = sum(token_count(item["content"]) for item in assembled)
    assert total_tokens <= 500


def test_atomic_pair_budget_admission():
    engine = EvidenceAnchorEngine(None)
    # item1 and item2 are contradictory
    hits = [
        {
            "id": "ev_pos",
            "content": "I am feeling very anxious about my grammar accuracy after receiving feedback.",
            "score": 0.95,
            "kind": "event",
            "evidence": "anxious about grammar",
        },
        {
            "id": "ev_neg",
            "content": "I have never felt anxious about grammar accuracy in any writing session.",
            "score": 0.85,
            "kind": "event",
            "evidence": "never felt anxious about grammar",
        },
    ]

    assembled = engine.assemble_anchored_context(hits, user_id="user_1", budget=400, enable_pair_guard=True)
    ids = {x["id"] for x in assembled}
    # Atomic guarantee: both must be present together
    assert "ev_pos" in ids
    assert "ev_neg" in ids

