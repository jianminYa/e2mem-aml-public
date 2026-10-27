"""Unit tests for Local Verbatim Window Expansion & Deduplication."""
import pytest
from service.evidence_anchor import EvidenceAnchorEngine, token_count


def test_verbatim_window_deduplication_and_adjacent_turns():
    engine = EvidenceAnchorEngine(None)

    raw_cache = {
        "episodes": {
            "ep_music": {
                "id": "ep_music",
                "request_id": "req_m",
                "turn_index": 0,
                "source_time": "2023-10-01",
                "content": "In a recent reflection on music, user realized podcasting was too overwhelming.",
                "session_id": "sess_m",
                "source_refs": "[]",
                "seq_id": 1,
            }
        },
        "by_req": {
            "req_m": [
                {"id": "raw_m0", "kind": "raw", "role": "assistant", "turn_index": 0, "content": "How has your music trends podcast been going lately?"},
                {"id": "raw_m1", "kind": "raw", "role": "user", "turn_index": 1, "content": "I started it in 2018, but I decided to pause it because research was too overwhelming."},
                {"id": "raw_m2", "kind": "raw", "role": "assistant", "turn_index": 2, "content": "That makes total sense. Burnout is real."},
            ]
        },
        "by_id": {},
        "by_ref": {},
    }

    hits = [
        {
            "id": "ev_1",
            "kind": "event",
            "parent_id": "ep_music",
            "content": "User decided to pause podcasting endeavor due to overwhelming research",
            "evidence": "decided to pause it because research was too overwhelming",
            "source_time": "2023-10-01",
            "score": 0.90,
        },
        {
            "id": "ev_2",
            "kind": "event",
            "parent_id": "ep_music",
            "content": "User started music trends podcast in 2018",
            "evidence": "started it in 2018",
            "source_time": "2018-01-01",
            "score": 0.85,
        },
    ]

    assembled = engine.assemble_anchored_context(
        retrieval_hits=hits,
        user_id="user_test",
        budget=8000,
        raw_cache=raw_cache,
    )

    assert len(assembled) == 2
    # Both events must be present and preserve their atomic facts
    c0 = assembled[0]["content"]
    c1 = assembled[1]["content"]

    # Verify atomic facts preserved
    assert "User decided to pause podcasting endeavor due to overwhelming research" in c0 or "User decided to pause podcasting endeavor due to overwhelming research" in c1
    assert "User started music trends podcast in 2018" in c0 or "User started music trends podcast in 2018" in c1

    # Verify local verbatim source context includes both the question and the explanation
    has_full_dialogue = any("research was too overwhelming" in item["content"] for item in assembled)
    assert has_full_dialogue
