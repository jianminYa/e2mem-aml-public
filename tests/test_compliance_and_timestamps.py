"""Tests for AML Cycle 2 Compliance, Epoch ms Timestamp Handling, and Speaker Attribution."""
import os
import re
import tempfile
import pytest
from fastapi.testclient import TestClient

os.environ["E2MEM_MODE"] = "simulator"
tmp_db = tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False)
os.environ["E2MEM_DB"] = tmp_db.name

from service.api import app, get_service
from service.core import normalize_timestamp, to_sortable_date

client = TestClient(app)


def test_timestamp_normalization_epoch_ms():
    """Verify Unix epoch milliseconds and seconds are converted to ISO 8601 UTC."""
    # 1690502400000 ms -> 2023-07-28T00:00:00Z
    iso_date = normalize_timestamp(1690502400000)
    assert iso_date.startswith("2023-07-28")
    assert iso_date.endswith("Z")

    # String epoch ms
    iso_date_str = normalize_timestamp("1690502400000")
    assert iso_date_str.startswith("2023-07-28")

    # Epoch seconds: 1690502400 -> 2023-07-28T00:00:00Z
    iso_date_sec = normalize_timestamp(1690502400)
    assert iso_date_sec.startswith("2023-07-28")

    # Existing ISO string remains intact
    assert normalize_timestamp("2026-05-01T10:00:00Z") == "2026-05-01T10:00:00Z"


def test_chronological_sortable_date():
    """Verify human dates and ISO timestamps sort in true chronological order."""
    d1 = to_sortable_date("7:55 pm on 9 June, 2023")
    d2 = to_sortable_date("August 11, 2023")
    d3 = to_sortable_date(1690502400000)  # July 28, 2023

    assert d1 == "2023-06-09"
    assert d2 == "2023-08-11"
    assert d3.startswith("2023-07-28")

    # Chronological sort: June 9 < July 28 < August 11
    assert d1 < d3 < d2


def test_search_response_has_created_at_and_speaker():
    """Verify POST /add with epoch ms timestamps returns created_at and formats Speaker correctly in /search."""
    add_payload = {
        "request_id": "req_compliance_001",
        "user_id": "user_compliance_test",
        "session_id": "sess_comp_001",
        "messages": [
            {
                "role": "assistant",
                "content": "Maria: Hey John! Long time no talk! Guess what - I got a puppy two weeks ago!",
                "timestamp": 1690502400000  # July 28, 2023 in ms
            },
            {
                "role": "user",
                "content": "John: Wow, Maria! That is awesome news!",
                "timestamp": 1690502400000
            }
        ]
    }
    key = os.environ.get("MEMORY_SYSTEM_KEY", "")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    resp = client.post("/add", json=add_payload, headers=headers)
    assert resp.status_code == 200

    # Search
    search_payload = {
        "user_id": "user_compliance_test",
        "query": "Who adopted a puppy?",
        "top_k": 5
    }
    resp = client.post("/search", json=search_payload, headers=headers)
    assert resp.status_code == 200
    data = resp.json().get("data", [])
    assert len(data) > 0

    # Verify created_at exists on all returned records
    for item in data:
        assert "created_at" in item
        assert item["created_at"] != ""

    # Verify Speaker: Maria was resolved instead of Speaker: assistant
    maria_found = False
    for item in data:
        content = item.get("content", "")
        if "Speaker: Maria" in content:
            maria_found = True
            break
    assert maria_found, f"Expected 'Speaker: Maria' in search content, got:\n{[x['content'] for x in data]}"


def test_zero_leaked_benchmark_strings_in_service():
    """Strict compliance check: ensure zero benchmark-specific tokens remain in service/ code."""
    import glob
    import os

    forbidden_tokens = [
        "Air Fryer",
        "Instant Pot",
        "Charlotte's Web",
        "Food Safety Specialist",
        "personal best",
        "Culinary Safety",
    ]

    service_files = glob.glob("service/*.py")
    assert len(service_files) > 0

    violations = []
    for fpath in service_files:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
            for token in forbidden_tokens:
                if token.lower() in content.lower():
                    violations.append((fpath, token))

    assert not violations, f"Compliance failure! Found forbidden benchmark tokens in service/: {violations}"
