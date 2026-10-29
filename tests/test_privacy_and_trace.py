"""Search-output privacy masking and the search trace (v2.6.3)."""
import json

from fastapi.testclient import TestClient

from service.core import MemoryService
from service.privacy import redact_text


def test_redaction_masks_credentials_but_keeps_ordinary_numbers():
    masked = {
        "My card number is 4111 1111 1111 1111": "[REDACTED CARD]",
        "password: hunter2!": "password: [REDACTED]",
        "key AKIAIOSFODNN7EXAMPLE": "[REDACTED ACCESS KEY]",
        "ssn 123-45-6789": "[REDACTED SSN]",
        "the pin is 4821": "[REDACTED]",
    }
    for text, marker in masked.items():
        out, n = redact_text(text)
        assert n == 1 and marker in out, (text, out)
    kept = [
        "16 -93 6 10.5 2 7 4 11 4 4 12 7 5 18 15 33",
        "years 2009 2010 2011 2012",
        "order 4111111111111111 shipped in 2013",
        "password = request.form['password']",
        'cfg.password = "secret";',
        "cvv: M; status: DECLINED",
        "call 415-555-0199 on 2023-05-08",
    ]
    for text in kept:
        assert redact_text(text) == (text, 0), text


def _service():
    svc = MemoryService(":memory:", mode="simulator")
    svc.add({"request_id": "r1", "user_id": "alice", "session_id": "s1", "messages": [
        {"role": "user", "content": "My visa card is 4111 1111 1111 1111 and I adopted a cat named Miso."},
        {"role": "assistant", "content": "Noted, Miso sounds lovely."}]})
    return svc


def test_search_returns_masked_text_and_memory_is_unchanged(monkeypatch):
    monkeypatch.delenv("E2MEM_REDACT", raising=False)
    svc = _service()
    res, stats = svc.search({"query": "What is my card and my cat's name?", "user_id": "alice", "top_k": 100})
    text = " ".join(item["content"] for item in res["data"])
    assert "4111 1111 1111 1111" not in text and "Miso" in text
    assert stats["redactions"] >= 1
    stored = " ".join(r[0] for r in svc.db.execute("SELECT content FROM memories WHERE kind='raw'"))
    assert "4111 1111 1111 1111" in stored


def test_redaction_can_be_disabled(monkeypatch):
    monkeypatch.setenv("E2MEM_REDACT", "0")
    res, stats = _service().search({"query": "card", "user_id": "alice", "top_k": 100})
    assert stats["redactions"] == 0
    assert any("4111 1111 1111 1111" in item["content"] for item in res["data"])


def test_search_trace_records_request_and_exact_response(tmp_path, monkeypatch):
    trace = tmp_path / "search_trace.jsonl"
    monkeypatch.setenv("E2MEM_SEARCH_TRACE_FILE", str(trace))
    svc = _service()
    payload = {"query": "cat name?", "user_id": "alice", "top_k": 5, "options": ["(a) Miso", "(b) Tofu"]}
    res, _ = svc.search(payload)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["query"] == "cat name?" and row["user_id"] == "alice" and row["top_k"] == 5
    assert row["options"] == ["(a) Miso", "(b) Tofu"]
    assert row["data"] == res["data"] and row["returned"] == len(res["data"])


def test_trace_failure_never_breaks_search(tmp_path, monkeypatch):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    monkeypatch.setenv("E2MEM_SEARCH_TRACE_FILE", str(blocker / "trace.jsonl"))
    res, _ = _service().search({"query": "cat", "user_id": "alice", "top_k": 100})
    assert res["data"]


def test_api_accepts_options_of_any_type(monkeypatch):
    monkeypatch.setenv("E2MEM_MODE", "simulator")
    monkeypatch.delenv("MEMORY_SYSTEM_KEY", raising=False)
    import service.api as api
    api._memory_service = _service()
    client = TestClient(api.app)
    r = client.post("/search", json={"query": "cat", "user_id": "alice", "top_k": 10, "options": ["(a) x", 2, None]})
    assert r.status_code == 200 and len(r.json()["data"]) <= 10
    assert client.get("/health").json()["version"] == "2.6.3"
