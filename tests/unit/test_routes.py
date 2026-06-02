from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.main import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health_ok(client):
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = True
    mock_collection = MagicMock()
    mock_collection.points_count = 42
    mock_qdrant.get_collection.return_value = mock_collection

    with patch("qdrant_client.QdrantClient", return_value=mock_qdrant):
        response = client.get("/health")

    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "ok"
    assert data["points"] == 42


def test_health_degraded_when_collection_missing(client):
    mock_qdrant = MagicMock()
    mock_qdrant.collection_exists.return_value = False

    with patch("qdrant_client.QdrantClient", return_value=mock_qdrant):
        response = client.get("/health")

    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "degraded"
    assert data["qdrant"] == "collection_missing"


def test_health_degraded_on_qdrant_error(client):
    with patch("qdrant_client.QdrantClient", side_effect=ConnectionError("refused")):
        response = client.get("/health")

    assert response.status_code == 503
    data = response.get_json()
    assert data["status"] == "degraded"
    assert "refused" in data["qdrant"]


def test_ask_empty_query(client):
    response = client.post("/ask", json={"query": ""})
    assert response.status_code == 200
    data = response.get_json()
    assert data["answer"] == "Query is empty"
    assert data["sources"] == []


def test_ask_success(client):
    with patch("backend.app.api.routes.get_answer") as mock_get_answer:
        mock_get_answer.return_value = {
            "answer": "Test answer.",
            "sources": [{"title": "Guidance", "page": 1}],
        }
        response = client.post("/ask", json={"query": "What is required?"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["answer"] == "Test answer."
    assert len(data["sources"]) == 1
    mock_get_answer.assert_called_once_with("What is required?")


def test_ask_agent_empty_query(client):
    response = client.post("/ask_agent", json={"query": ""})
    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is False
    assert data["answer"] == "Query is empty"
    assert data["sources"] == []
    assert data["steps"] == []


def test_ask_agent_success(client):
    with patch("backend.app.api.routes.get_agent_answer") as mock_agent:
        mock_agent.return_value = {
            "answer": "Agent answer.",
            "sources": [{"title": "Guidance", "pdf_id": "1"}],
            "steps": [{"tool": "search_guidance", "args": {"query": "test"}}],
        }
        response = client.post("/ask_agent", json={"query": "List REMS guidances"})

    assert response.status_code == 200
    data = response.get_json()
    assert data["success"] is True
    assert data["answer"] == "Agent answer."
    assert len(data["sources"]) == 1
    assert data["steps"][0]["tool"] == "search_guidance"
    mock_agent.assert_called_once_with("List REMS guidances")
