import io

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture
def client(store, fake_llm):
    return TestClient(app)


@pytest.fixture
def auth(client):
    r = client.post("/auth/token", json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["documents_indexed_chunks"] > 0


def test_bad_login(client):
    assert client.post("/auth/token", json={"username": "admin", "password": "nope"}).status_code == 401


def test_requires_token(client):
    assert client.post("/query", json={"question": "anything here"}).status_code == 401
    bad = {"Authorization": "Bearer not-a-jwt"}
    assert client.post("/query", json={"question": "anything here"}, headers=bad).status_code == 401


def test_query_returns_grounded_answer(client, auth):
    r = client.post("/query", json={"question": "How quickly must Class I recalls be pulled?"}, headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert "2 hours" in body["answer"]
    assert body["citations"][0]["source"] == "product_recall_policy.pdf"


@pytest.mark.parametrize("payload", [
    {"question": "x"},
    {"question": "valid question", "top_k": 0},
    {"question": "valid question", "provider": "not-a-provider"},
    {"question": "a" * 2001},
])
def test_validation_errors(client, auth, payload):
    assert client.post("/query", json=payload, headers=auth).status_code == 422


def test_query_streaming(client, auth):
    with client.stream("POST", "/query", json={"question": "Class I recall timing?", "stream": True},
                       headers=auth) as r:
        body = "".join(r.iter_text())
    assert "event: sources" in body and "event: token" in body and "event: done" in body


def test_chat_streaming(client, auth):
    with client.stream("POST", "/chat", json={"message": "Which products in the recall policy had sales last quarter?",
                                              "thread_id": "api1"}, headers=auth) as r:
        body = "".join(r.iter_text())
    assert "event: step" in body and "sql_query" in body and "event: done" in body


def test_chat_non_streaming(client, auth):
    r = client.post("/chat", json={"message": "What is the PTO carryover?", "thread_id": "api2", "stream": False},
                    headers=auth)
    assert r.status_code == 200 and r.json()["intent"] == "docs"


def test_ingest_upload_and_type_check(client, auth):
    files = {"files": ("new_policy.md", io.BytesIO(b"# Parking Policy\n\n## Spaces\n\nEmployees park in rows F and G."),
                       "text/markdown")}
    r = client.post("/ingest", files=files, headers=auth)
    assert r.status_code == 200 and r.json()[0]["chunks"] >= 1
    bad = {"files": ("evil.exe", io.BytesIO(b"MZ"), "application/octet-stream")}
    assert client.post("/ingest", files=bad, headers=auth).status_code == 415


def test_rate_limit(client, auth, monkeypatch):
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "rate_limit_per_minute", 2)
    codes = [client.get("/metrics", headers=auth).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
