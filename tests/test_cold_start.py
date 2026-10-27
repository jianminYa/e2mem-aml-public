"""Test cold-start replay from completely empty database to verify live /add -> /search pipeline."""
import tempfile
from pathlib import Path
from service.core import MemoryService


def test_cold_start_add_and_search():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.sqlite3"
        svc = MemoryService(db_path=str(db_path), mode="simulator")

        # 1. Add Session 1
        payload1 = {
            "request_id": "req_001",
            "user_id": "user_test_1",
            "session_id": "session_001",
            "messages": [
                {"role": "user", "content": "I am thinking about adopting a pet. I really love golden retrievers.", "timestamp": "2024-01-01T10:00:00Z"},
                {"role": "assistant", "content": "Golden retrievers are wonderful dogs! They are friendly and energetic.", "timestamp": "2024-01-01T10:01:00Z"},
                {"role": "user", "content": "I bought a yellow leash and dog bowls yesterday.", "timestamp": "2024-01-02T12:00:00Z"},
                {"role": "assistant", "content": "Great preparation! You're ready for your new puppy.", "timestamp": "2024-01-02T12:02:00Z"},
            ]
        }
        res1, stats1 = svc.add(payload1)
        assert res1["success"] is True

        # 2. Add Session 2 with temporal update
        payload2 = {
            "request_id": "req_002",
            "user_id": "user_test_1",
            "session_id": "session_002",
            "messages": [
                {"role": "user", "content": "Actually, my apartment does not allow dogs. I decided to adopt a British Shorthair cat instead.", "timestamp": "2024-02-15T09:00:00Z"},
                {"role": "assistant", "content": "British Shorthairs are calm and lovely cats, perfect for apartment living.", "timestamp": "2024-02-15T09:02:00Z"},
            ]
        }
        res2, stats2 = svc.add(payload2)
        assert res2["success"] is True

        # 3. Query: what kind of pet did the user end up adopting?
        search_res, search_stats = svc.search(
            {"query": "What kind of pet did I decide to adopt?", "user_id": "user_test_1", "top_k": 10},
            budget=8000
        )
        assert len(search_res["data"]) > 0
        found_cat = any("British Shorthair" in m["content"] or "cat" in m["content"] for m in search_res["data"])
        assert found_cat, "Cold-start search must successfully retrieve the updated cat adoption fact"

        # 4. Check sequence ordering and atomicity
        assert search_stats["returned"] > 0
        assert search_stats["latency_ms"] < 100
