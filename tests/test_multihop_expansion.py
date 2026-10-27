"""Unit tests for Zero-LLM Multi-Hop Entity Relation Expansion."""
import pytest
from service.core import MemoryService, extract_salient_entities


def test_extract_salient_entities():
    # Test named entities extraction from text
    content = "Alice visited Google headquarters with Bob in San Francisco."
    entities = extract_salient_entities(content, query="Where did Alice go?")
    # Bridge entities not in the query must be prioritized first
    assert "San Francisco" in entities
    assert "Google" in entities
    assert "Bob" in entities
    assert "Alice" in entities
    assert entities.index("San Francisco") < entities.index("Alice")

    # Test possessive stripping ('s)
    content2 = "Sarah's brother Michael went to Stanford University."
    entities2 = extract_salient_entities(content2, query="Who is Sarah?")
    assert "Stanford University" in entities2 or "Stanford" in entities2
    assert "Michael" in entities2
    assert "Sarah" in entities2

    # Test filtering of common pronouns and words
    content3 = "They decided to visit the local market yesterday."
    entities3 = extract_salient_entities(content3)
    assert "They" not in entities3
    assert "yesterday" not in entities3


def test_multihop_entity_expansion_retrieval(tmp_path):
    db_file = tmp_path / "test_multihop.sqlite3"
    svc = MemoryService(str(db_file), mode="simulator")

    # Create schema
    svc.db.execute("CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, user_id TEXT, session_id TEXT, body_hash TEXT)")
    svc.db.execute("CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, user_id TEXT, session_id TEXT, request_id TEXT, kind TEXT, parent_id TEXT, turn_index INT, role TEXT, source_time TEXT, content TEXT, evidence TEXT, vector BLOB, created_at TEXT, source_refs TEXT, aliases TEXT, keywords TEXT, event_type TEXT, time_anchor TEXT, topic_labels TEXT)")

    # Insert multi-hop chain:
    # Event 1: Alice's brother is named Bob (directly answers who the brother is)
    # Event 2: Bob completed his PhD at MIT in 2018 (second hop answering where he went to university)
    # Event 3: Charlie works at Apple (unrelated)
    svc.db.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ev_1", "user_1", "s1", "r1", "event", "", 0, "user", "2020-01-01", "Alice has a younger brother named Bob.", "younger brother named Bob", b"", "now", "[]", "[]", "[]", "", "", "")
    )
    svc.db.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ev_2", "user_1", "s2", "r2", "event", "", 0, "user", "2020-02-01", "Bob completed his PhD at MIT in 2018.", "PhD at MIT in 2018", b"", "now", "[]", "[]", "[]", "", "", "")
    )
    svc.db.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ev_3", "user_1", "s3", "r3", "event", "", 0, "user", "2020-03-01", "Charlie works at Apple in Cupertino.", "works at Apple", b"", "now", "[]", "[]", "[]", "", "", "")
    )
    svc.db.commit()

    # Query asking about Alice's brother's university
    # Primary query will hit ev_1 strongly on "Alice", but has 0 word overlap with ev_2 except via entity "Bob"
    res, stats = svc.search({"user_id": "user_1", "query": "Where did Alice's brother go to university?", "top_k": 5})

    returned_ids = [item["id"] for item in res["data"]]
    # ev_1 should be retrieved
    assert "ev_1" in returned_ids
    # ev_2 must be successfully retrieved via multi-hop expansion on entity "Bob"
    assert "ev_2" in returned_ids
    # Unrelated ev_3 should not be in the top results or ranked below
    assert returned_ids.index("ev_1") < returned_ids.index("ev_2")


def test_multihop_strict_quota_and_contract(tmp_path):
    db_file = tmp_path / "test_multihop_quota.sqlite3"
    svc = MemoryService(str(db_file), mode="simulator")

    svc.db.execute("CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, user_id TEXT, session_id TEXT, body_hash TEXT)")
    svc.db.execute("CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, user_id TEXT, session_id TEXT, request_id TEXT, kind TEXT, parent_id TEXT, turn_index INT, role TEXT, source_time TEXT, content TEXT, evidence TEXT, vector BLOB, created_at TEXT, source_refs TEXT, aliases TEXT, keywords TEXT, event_type TEXT, time_anchor TEXT, topic_labels TEXT)")

    # Seed event mentioning David
    svc.db.execute(
        "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("ev_seed", "u_quota", "s0", "r0", "event", "", 0, "user", "2020", "David is leading the Apollo project.", "David", b"", "now", "[]", "[]", "[]", "", "", "")
    )
    # Insert 10 events mentioning David
    for i in range(1, 11):
        svc.db.execute(
            "INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (f"ev_bridge_{i}", "u_quota", f"s{i}", f"r{i}", "event", "", 0, "user", "2020", f"David milestone #{i} completed.", "David", b"", "now", "[]", "[]", "[]", "", "", "")
        )
    svc.db.commit()

    # Search with top_k = 3 -> must NEVER return more than 3 items
    res3, _ = svc.search({"user_id": "u_quota", "query": "Apollo project", "top_k": 3})
    assert len(res3["data"]) <= 3

    # Search with top_k = 1 -> must return exactly 1 item
    res1, _ = svc.search({"user_id": "u_quota", "query": "Apollo project", "top_k": 1})
    assert len(res1["data"]) == 1

    # Verify expand_multihop_entities directly enforces bridge_quota = 4
    bridges = svc.expand_multihop_entities(
        user_id="u_quota",
        seed_events=[{"id": "ev_seed", "content": "David is leading the Apollo project.", "score": 0.9}],
        seen_event_ids={"ev_seed"},
        bridge_quota=4,
    )
    assert len(bridges) == 4, f"Expected 4 bridge events, got {len(bridges)}"
    assert all(b.get("_is_bridge") for b in bridges)
    assert all(b.get("score") == pytest.approx(0.85 * 0.9, 0.01) for b in bridges)
