# serving/tests/test_e5_api.py
# Tests for E5 — FastAPI endpoints.
# Uses TestClient — no running server needed.

import pytest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path):
    """Create a FastAPI test client with isolated dependencies."""
    import json
    (tmp_path / "manifest.json").write_text(json.dumps({"metadata": {"title": "Test"}}))
    (tmp_path / "metadata.json").write_text(json.dumps({"duration": 100, "fps": 30}))
    (tmp_path / "segments.json").write_text("[]")
    (tmp_path / "entities.json").write_text("[]")
    (tmp_path / "relations.json").write_text("[]")
    (tmp_path / "multimodal_chunks.json").write_text("[]")
    (tmp_path / "chunk_segment_map.json").write_text("{}")
    (tmp_path / "triplets.json").write_text("[]")
    from serving.fastapi.app import app
    from serving.fastapi.routes.query import set_workflow
    from local.storage.registry_provider import set_registry, reset_registry
    from local.storage.lecture_registry import LectureRegistry

    mock_registry = MagicMock(spec=LectureRegistry)
    mock_registry.get_lecture.return_value = {"lecture_id": "lec_001", "status": "READY", "package_path": str(tmp_path)}
    mock_registry.list_lectures.return_value = [{"lecture_id": "lec_001", "status": "READY", "package_path": str(tmp_path)}]
    set_registry(mock_registry)

    mock_workflow = MagicMock()
    mock_workflow.run.return_value = {
        "answer": "BFS explores level by level.",
        "sources": [{"chunk_id": "c_001", "timestamp": "00:00:10", "segment_id": "s_1"}],
        "graph_path": ["BFS", "Dijkstra"],
    }
    set_workflow(mock_workflow)

    with patch("serving.fastapi.learning_service._retrieve_context", return_value="Mock context"), \
         patch("serving.fastapi.learning_service._call_llm", return_value='[{"title":"mock","description":"mock"}]'):
        yield TestClient(app)

    reset_registry()
    set_workflow(None)


class TestFastAPI:

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_list_lectures(self, client):
        response = client.get("/lectures")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_lecture(self, client):
        response = client.get("/lecture/lec_001")
        assert response.status_code == 200
        assert response.json()["lecture_id"] == "lec_001"

    def test_get_status(self, client):
        response = client.get("/status/lec_001")
        assert response.status_code == 200
        assert response.json()["lecture_id"] == "lec_001"

    def test_query_without_workflow(self, client):
        """Query without initialized workflow returns 503."""
        from serving.fastapi.routes.query import set_workflow
        set_workflow(None)
        response = client.post("/query", json={"query": "test"})
        assert response.status_code == 503

    def test_query_with_workflow(self, client):
        """Query with injected workflow returns answer."""
        response = client.post("/query", json={"query": "How does BFS work?"})
        assert response.status_code == 200

        data = response.json()
        assert data["answer"] == "BFS explores level by level."
        assert len(data["sources"]) == 1
        assert data["graph_path"] == ["BFS", "Dijkstra"]

    def test_query_empty_rejected(self, client):
        """Empty query is rejected by validation."""
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

    def test_notes_endpoint(self, client):
        response = client.post("/notes", json={"lecture_id": "lec_001"})
        if response.status_code != 200:
            print("Response:", response.text)
        assert response.status_code == 200

    def test_flashcards_endpoint(self, client):
        response = client.post("/flashcards", json={"lecture_id": "lec_001"})
        assert response.status_code == 200

    def test_quiz_endpoint(self, client):
        response = client.post("/quiz", json={"lecture_id": "lec_001"})
        assert response.status_code == 200

    def test_learning_path_endpoint(self, client):
        response = client.post("/learning_path", json={"lecture_id": "lec_001"})
        assert response.status_code == 200

    def test_settings_endpoint(self, client):
        response = client.put("/settings", json={"top_k": 10})
        assert response.status_code == 200
        assert response.json()["status"] == "updated"
