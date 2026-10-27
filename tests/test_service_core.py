"""Unit tests for MemoryService end-to-end integration."""
import pytest
from service.core import MemoryService


def test_add_and_search_basic():
    service = MemoryService(":memory:", mode="simulator")

    add_payload = {
        "request_id": "req_001",
        "user_id": "alice",
        "session_id": "sess_01",
        "messages": [
            {"role": "user", "content": "I adopted a golden retriever named Max in 2021."},
            {"role": "assistant", "content": "Congratulations on adopting Max!"},
        ]
    }

    res, stats = service.add(add_payload)
    assert res["success"] is True
    assert stats["messages"] == 2

    # Query
    search_payload = {
        "query": "What pet do I have?",
        "user_id": "alice",
        "top_k": 5,
    }
    s_res, s_stats = service.search(search_payload)
    assert len(s_res["data"]) > 0
    # Must retrieve Max the golden retriever
    combined_content = " ".join(item["content"] for item in s_res["data"])
    assert "Max" in combined_content or "golden retriever" in combined_content


def test_user_isolation():
    service = MemoryService(":memory:", mode="simulator")

    service.add({
        "request_id": "req_alice",
        "user_id": "alice",
        "session_id": "s1",
        "messages": [{"role": "user", "content": "Alice secret password is xyz123."}]
    })

    service.add({
        "request_id": "req_bob",
        "user_id": "bob",
        "session_id": "s2",
        "messages": [{"role": "user", "content": "Bob likes mountain biking."}]
    })

    # Bob searches
    b_res, _ = service.search({"query": "secret password", "user_id": "bob", "top_k": 5})
    # Bob should NEVER see Alice's content!
    bob_text = " ".join(item["content"] for item in b_res["data"])
    assert "xyz123" not in bob_text
    assert "Alice" not in bob_text


def test_request_id_idempotency():
    service = MemoryService(":memory:", mode="simulator")

    payload = {
        "request_id": "req_idem",
        "user_id": "alice",
        "session_id": "s1",
        "messages": [{"role": "user", "content": "Testing idempotency."}]
    }

    res1, stats1 = service.add(payload)
    assert res1["success"] is True
    assert stats1.get("duplicate") is None

    # Duplicate call with identical content
    res2, stats2 = service.add(payload)
    assert res2["success"] is True
    assert stats2.get("duplicate") is True


def test_uncovered_user_turn_factlet_extraction():
    from service.core import extract_uncovered_user_factlets

    chunk = [
        {
            "role": "user",
            "content": "I've allocated $4,000 for initial patent filing fees and $5,500 for attorney fees by July 2024, can you help me create a budget plan? ->-> 1,21",
            "timestamp": "2024-03-10",
            "source_ref": "s1:1",
        },
        {"role": "assistant", "content": "Here is a budget plan for your patent application."},
        {
            "role": "user",
            "content": "I've never attended any production meetings related to my invention, how can I get started?",
            "timestamp": "2024-03-10",
            "source_ref": "s1:3",
        },
    ]
    # Simulate LLM extractor only extracting from the last turn
    extracted_events = [
        {
            "event_text": "User has never attended production meetings for their invention.",
            "evidence": ["I've never attended any production meetings related to my invention"],
            "fact_date_resolved": "2024-03-10",
        }
    ]
    factlets = extract_uncovered_user_factlets(chunk, extracted_events, session_id="s1", chunk_base_idx=0)
    assert len(factlets) == 1
    assert "$4,000" in factlets[0]["content"]
    assert "$5,500" in factlets[0]["content"]
    assert "can you help" not in factlets[0]["content"].lower()


def test_pin_verbatim_evidence_and_refs():
    from service.core import pin_verbatim_evidence_and_refs

    chunk = [
        {"role": "user", "content": "Hello, good morning!", "timestamp": "2024-01-01", "source_ref": "s1:0"},
        {
            "role": "user",
            "content": "I adopted a golden retriever named Max in 2021, and we go hiking together.",
            "timestamp": "2024-01-01",
            "source_ref": "s1:1",
        },
        {"role": "assistant", "content": "That sounds lovely!", "timestamp": "2024-01-01", "source_ref": "s1:2"},
    ]

    # Paraphrased evidence from LLM
    ev_text = "User has a golden retriever named Max"
    ev_list = ["adopted a retriever Max in 2021"]
    pin_ev, refs, kws, msg_date, matched_off = pin_verbatim_evidence_and_refs(
        event_text=ev_text,
        evidence_list=ev_list,
        chunk=chunk,
        chunk_base_idx=0,
        session_id="s1",
    )

    assert matched_off == 1
    assert refs == ["s1:1"]
    assert "Max" in pin_ev and "retriever" in pin_ev
    assert "2021" in kws
    assert msg_date == "2024-01-01"

    # Test safety: completely unrelated event should NOT fabricate attribution to chunk[0]
    pin_ev2, refs2, kws2, msg_date2, matched_off2 = pin_verbatim_evidence_and_refs(
        event_text="Bought a motorcycle in Paris",
        evidence_list=["motorcycle in Paris"],
        chunk=chunk,
        chunk_base_idx=0,
        session_id="s1",
    )
    assert matched_off2 == -1
    assert refs2 == []
    assert msg_date2 == ""


def test_add_with_sliding_context():
    service = MemoryService(":memory:", mode="simulator")
    # Send 10 messages so it creates 2 chunks (8 + 2)
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message turn {i} about project planning"}
        for i in range(10)
    ]
    res, stats = service.add({
        "request_id": "req_slide",
        "user_id": "bob",
        "session_id": "sess_slide",
        "messages": messages,
    })
    assert res["success"] is True
    # Verify that raw count is exactly 10 and episodes is 2
    raw_count = service.db.execute("SELECT count(*) FROM memories WHERE kind='raw'").fetchone()[0]
    ep_count = service.db.execute("SELECT count(*) FROM memories WHERE kind='episode'").fetchone()[0]
    assert raw_count == 10
    assert ep_count == 2

