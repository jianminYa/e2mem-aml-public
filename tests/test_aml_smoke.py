"""Comprehensive AML Protocol Smoke & Compliance Test Suite.

Verifies:
1. Health endpoint reachability (/health).
2. MEMORY_SYSTEM_KEY authentication & rejection of unauthorized requests.
3. Top-k boundary enforcement (top_k <= 100, rejection of top_k > 100).
4. Add request idempotency and duplicate request suppression.
5. User isolation (no cross-tenant leakage between user A and user B).
6. Search response contract compliance (id, content, score fields).
7. High-concurrency stress testing (zero deadlocks, zero 500 errors).
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

# Set test environment
os.environ["E2MEM_MODE"] = "simulator"
os.environ["MEMORY_SYSTEM_KEY"] = "aml_secret_token_12345"

tmp_db = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
os.environ["E2MEM_DB"] = tmp_db.name

from service.api import app, get_service

client = TestClient(app)
AUTH_HEADERS = {"Authorization": "Bearer aml_secret_token_12345"}
INVALID_AUTH_HEADERS = {"Authorization": "Bearer wrong_token"}


def setup_module():
    """Ensure service is initialized with clean database."""
    svc = get_service()
    svc.db.execute("DELETE FROM memories")
    svc.db.execute("DELETE FROM requests")
    svc.db.commit()


def teardown_module():
    try:
        Path(tmp_db.name).unlink(missing_ok=True)
    except Exception:
        pass


def test_01_health_check():
    """GET /health must return 200 OK without requiring authentication."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "service" in data


def test_02_authentication_enforcement():
    """Unauthenticated or invalid key requests to /add and /search must be rejected with 401."""
    # /add without auth
    resp = client.post("/add", json={"request_id": "r1", "user_id": "u1", "session_id": "s1", "messages": []})
    assert resp.status_code == 401

    # /add with wrong token
    resp = client.post("/add", json={"request_id": "r1", "user_id": "u1", "session_id": "s1", "messages": []}, headers=INVALID_AUTH_HEADERS)
    assert resp.status_code == 401

    # /search without auth
    resp = client.post("/search", json={"user_id": "u1", "query": "hello"})
    assert resp.status_code == 401

    # /search with X-Api-Key header (must be accepted)
    resp = client.post("/search", json={"user_id": "u1", "query": "hello"}, headers={"X-Api-Key": "aml_secret_token_12345"})
    assert resp.status_code == 200


def test_03_top_k_boundary_validation():
    """top_k > 100 must be rejected with 422 Unprocessable Entity per AML contract."""
    resp = client.post("/search", json={"user_id": "u1", "query": "test", "top_k": 101}, headers=AUTH_HEADERS)
    assert resp.status_code == 422

    resp = client.post("/search", json={"user_id": "u1", "query": "test", "top_k": 0}, headers=AUTH_HEADERS)
    assert resp.status_code == 422

    resp = client.post("/search", json={"user_id": "u1", "query": "test", "top_k": 50}, headers=AUTH_HEADERS)
    assert resp.status_code == 200


def test_04_add_and_search_contract_compliance():
    """Verify standard Add, Search, and response structure compliance."""
    add_payload = {
        "request_id": "req_smoke_001",
        "user_id": "user_alice",
        "session_id": "session_001",
        "messages": [
            {"role": "user", "content": "I adopted a golden retriever puppy named Luna yesterday.", "timestamp": "2026-10-20"},
            {"role": "assistant", "content": "Congratulations on adopting Luna! Golden retrievers are wonderful dogs.", "timestamp": "2026-10-20"}
        ]
    }
    resp = client.post("/add", json=add_payload, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    assert resp.json().get("success") is True

    # Search for Luna
    search_payload = {
        "user_id": "user_alice",
        "query": "What kind of dog did Alice adopt and what is its name?",
        "top_k": 10
    }
    resp = client.post("/search", json=search_payload, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0

    first_item = data["data"][0]
    assert "id" in first_item
    assert "content" in first_item
    assert "score" in first_item
    assert isinstance(first_item["score"], float)


def test_05_idempotency_and_duplicate_handling():
    """Identical duplicate Add requests return 200 OK; conflicting duplicate returns 409 Conflict."""
    # 1. Exact identical payload with same request_id -> returns 200 OK (idempotent)
    identical_payload = {
        "request_id": "req_smoke_001",
        "user_id": "user_alice",
        "session_id": "session_001",
        "messages": [
            {"role": "user", "content": "I adopted a golden retriever puppy named Luna yesterday.", "timestamp": "2026-10-20"},
            {"role": "assistant", "content": "Congratulations on adopting Luna! Golden retrievers are wonderful dogs.", "timestamp": "2026-10-20"}
        ]
    }
    resp = client.post("/add", json=identical_payload, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    assert resp.json().get("success") is True

    # 2. Conflicting payload with reused request_id -> returns 409 Conflict per AML contract
    conflicting_payload = {
        "request_id": "req_smoke_001",  # Same request_id
        "user_id": "user_alice",
        "session_id": "session_001",
        "messages": [{"role": "user", "content": "Completely different conflicting content"}]
    }
    resp_conflict = client.post("/add", json=conflicting_payload, headers=AUTH_HEADERS)
    assert resp_conflict.status_code == 409
    assert "request_id reused" in resp_conflict.json()["detail"].lower()


def test_06_user_isolation():
    """User A's memories must NEVER be visible or leaked when querying User B."""
    # Add User Bob
    bob_payload = {
        "request_id": "req_smoke_bob_001",
        "user_id": "user_bob",
        "session_id": "session_bob_001",
        "messages": [
            {"role": "user", "content": "I am working as a quantum computing researcher in Zurich."}
        ]
    }
    resp = client.post("/add", json=bob_payload, headers=AUTH_HEADERS)
    assert resp.status_code == 200

    # Query Bob for Alice's dog Luna (should find NOTHING)
    resp = client.post("/search", json={"user_id": "user_bob", "query": "Luna puppy golden retriever", "top_k": 20}, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    bob_results = resp.json()["data"]
    assert not any("Luna" in item["content"] for item in bob_results)

    # Query Alice for Bob's quantum job (should find NOTHING)
    resp = client.post("/search", json={"user_id": "user_alice", "query": "quantum computing Zurich", "top_k": 20}, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    alice_results = resp.json()["data"]
    assert not any("quantum" in item["content"].lower() for item in alice_results)


def test_07_concurrent_stress_load():
    """Verify that 20 concurrent search requests complete cleanly with zero errors."""
    def _search_task(i: int) -> int:
        res = client.post("/search", json={"user_id": "user_alice", "query": f"Luna puppy {i}", "top_k": 10}, headers=AUTH_HEADERS)
        return res.status_code

    with ThreadPoolExecutor(max_workers=8) as executor:
        statuses = list(executor.map(_search_task, range(20)))

    assert all(code == 200 for code in statuses)
