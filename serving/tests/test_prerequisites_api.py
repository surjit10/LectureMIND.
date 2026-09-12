# serving/tests/test_prerequisites_api.py
# Tests for Prerequisite API routes and QueryResponse integration.

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from schemas.query import QueryResponse


@pytest.fixture
def client(tmp_path):
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
    mock_workflow._neo4j_driver = MagicMock()
    mock_workflow.run.return_value = {
        "answer": "Thrashing occurs when virtual memory working sets exceed physical RAM.",
        "sources": [{"chunk_id": "c_005", "timestamp": "00:05:00", "segment_id": "s_3"}],
        "graph_path": ["Page Table", "Page Fault", "Thrashing"],
        "graph_results": [{"start_name": "Thrashing"}],
    }
    set_workflow(mock_workflow)

    with patch("serving.fastapi.learning_service._retrieve_context", return_value="Mock context"), \
         patch("serving.fastapi.learning_service._call_llm", return_value='[{"title":"mock","description":"mock"}]'):
        yield TestClient(app)

    reset_registry()
    set_workflow(None)


def test_query_endpoint_includes_prerequisites(client):
    """Verify POST /query returns prerequisites array and debug trace."""
    mock_prereqs = [
        {
            "concept": "Page Tables",
            "entity_id": "e_pt",
            "type": "Concept",
            "depth": 2,
            "timestamp": 20.44,
            "chunk_id": "c_04",
            "confidence": 0.85,
            "parent_concept": "Page Faults",
            "path": ["Page Tables", "Page Faults", "Thrashing"],
            "edge_provenance": [
                {
                    "source": "Page Tables",
                    "target": "Page Faults",
                    "confidence": 0.85,
                    "chunk_id": "c_04",
                    "timestamp": 20.44,
                }
            ],
        }
    ]

    with patch("retrieval.graph_retriever.neo4j_retriever.get_prerequisites_for_query", return_value=mock_prereqs), \
         patch("serving.fastapi.routes.query.activate_lecture"):
        response = client.post("/query", json={"query": "Explain Thrashing", "lecture_id": "lec_001"})
        assert response.status_code == 200
        data = response.json()

        # Check top-level additive fields
        assert "prerequisites" in data
        assert len(data["prerequisites"]) == 1
        assert data["prerequisites"][0]["concept"] == "Page Tables"
        assert data["prerequisites"][0]["depth"] == 2
        assert data["prerequisites"][0]["timestamp"] == 20.44

        # Check backward compatibility: graph_path remains unchanged!
        assert data["graph_path"] == ["Page Table", "Page Fault", "Thrashing"]

        # Check debug trace
        assert "debug" in data
        assert "prerequisites" in data["debug"]


def test_get_prerequisites_endpoint(client):
    """Verify dedicated GET /lecture/{lecture_id}/prerequisites/{concept_name} endpoint."""
    mock_items = [
        {
            "concept": "Page Faults",
            "entity_id": "e_pf",
            "type": "Concept",
            "depth": 1,
            "timestamp": 25.55,
            "chunk_id": "c_05",
            "confidence": 0.88,
            "parent_concept": "Thrashing",
            "path": ["Page Faults", "Thrashing"],
            "edge_provenance": [],
        }
    ]

    with patch("serving.fastapi.routes.prerequisites.get_prerequisites_for_concept", return_value=mock_items):
        response = client.get("/lecture/lec_001/prerequisites/Thrashing?min_confidence=0.60&max_depth=3")
        assert response.status_code == 200
        data = response.json()

        assert data["lecture_id"] == "lec_001"
        assert data["concept"] == "Thrashing"
        assert data["count"] == 1
        assert len(data["prerequisites"]) == 1
        assert data["prerequisites"][0]["concept"] == "Page Faults"
        assert data["prerequisites"][0]["depth"] == 1


def test_get_prerequisites_empty_concept(client):
    """Empty concept name returns 400 Bad Request."""
    response = client.get("/lecture/lec_001/prerequisites/%20")
    assert response.status_code == 400
