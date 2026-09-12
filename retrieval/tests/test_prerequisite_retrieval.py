# retrieval/tests/test_prerequisite_retrieval.py
# Tests for Neo4j prerequisite backward traversal and path preservation.

import pytest
from unittest.mock import MagicMock

from retrieval.graph_retriever.neo4j_retriever import (
    get_prerequisites_for_concept,
    get_prerequisites_for_query,
)


def test_prerequisite_traversal_path_preservation():
    """
    CRITICAL USER CORRECTION 3:
    The Neo4j traversal should preserve the actual path, not flatten it.
    For: Page Tables -> Page Faults -> Thrashing
    API must represent depth, timestamps, intermediate hops, and edge provenance.
    """
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    # 2 hops: Page Tables -> Page Faults -> Thrashing
    # Hop 1: Page Faults -> Thrashing (depth 1)
    # Hop 2: Page Tables -> Page Faults -> Thrashing (depth 2)
    record_depth1 = {
        "entity_id": "e_pf",
        "concept": "Page Faults",
        "type": "Concept",
        "target_concept": "Thrashing",
        "depth": 1,
        "path_nodes": ["Page Faults", "Thrashing"],
        "edge_provenance": [
            {
                "source": "Page Faults",
                "target": "Thrashing",
                "confidence": 0.88,
                "chunk_id": "c_05",
                "timestamp": 25.55,
            }
        ],
        "direct_confidence": 0.88,
        "chunk_id": "c_05",
        "timestamp": 25.55,
    }
    record_depth2 = {
        "entity_id": "e_pt",
        "concept": "Page Tables",
        "type": "Concept",
        "target_concept": "Thrashing",
        "depth": 2,
        "path_nodes": ["Page Tables", "Page Faults", "Thrashing"],
        "edge_provenance": [
            {
                "source": "Page Tables",
                "target": "Page Faults",
                "confidence": 0.85,
                "chunk_id": "c_04",
                "timestamp": 20.44,
            },
            {
                "source": "Page Faults",
                "target": "Thrashing",
                "confidence": 0.88,
                "chunk_id": "c_05",
                "timestamp": 25.55,
            },
        ],
        "direct_confidence": 0.85,
        "chunk_id": "c_04",
        "timestamp": 20.44,
    }

    # Cypher order: depth DESC, timestamp ASC
    mock_session.run.return_value = [record_depth2, record_depth1]

    items = get_prerequisites_for_concept(
        concept_name="Thrashing",
        lecture_id="lec_os",
        driver=mock_driver,
        min_confidence=0.65,
    )

    assert len(items) == 2
    # Verify depth 2 (Page Tables)
    pt = items[0]
    assert pt["concept"] == "Page Tables"
    assert pt["depth"] == 2
    assert pt["timestamp"] == 20.44
    assert pt["parent_concept"] == "Page Faults"
    assert pt["path"] == ["Page Tables", "Page Faults", "Thrashing"]
    assert len(pt["edge_provenance"]) == 2

    # Verify depth 1 (Page Faults)
    pf = items[1]
    assert pf["concept"] == "Page Faults"
    assert pf["depth"] == 1
    assert pf["timestamp"] == 25.55
    assert pf["parent_concept"] == "Thrashing"
    assert pf["path"] == ["Page Faults", "Thrashing"]
    assert len(pf["edge_provenance"]) == 1


def test_prerequisite_branching_preservation():
    """
    Branching topology:
                 ┌── Page Tables ──┐
    Virtual Memory                  ── Page Fault ── Thrashing
                 └── Address Space ─┘
    Both branches must be retained in the structured return.
    """
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    branch_pt = {
        "entity_id": "e_vm",
        "concept": "Virtual Memory",
        "type": "Concept",
        "target_concept": "Thrashing",
        "depth": 3,
        "path_nodes": ["Virtual Memory", "Page Tables", "Page Fault", "Thrashing"],
        "edge_provenance": [
            {"source": "Virtual Memory", "target": "Page Tables", "confidence": 0.9},
            {"source": "Page Tables", "target": "Page Fault", "confidence": 0.85},
            {"source": "Page Fault", "target": "Thrashing", "confidence": 0.88},
        ],
        "direct_confidence": 0.9,
        "chunk_id": "c_01",
        "timestamp": 5.0,
    }
    branch_as = {
        "entity_id": "e_vm",
        "concept": "Virtual Memory",
        "type": "Concept",
        "target_concept": "Thrashing",
        "depth": 3,
        "path_nodes": ["Virtual Memory", "Address Space", "Page Fault", "Thrashing"],
        "edge_provenance": [
            {"source": "Virtual Memory", "target": "Address Space", "confidence": 0.85},
            {"source": "Address Space", "target": "Page Fault", "confidence": 0.82},
            {"source": "Page Fault", "target": "Thrashing", "confidence": 0.88},
        ],
        "direct_confidence": 0.85,
        "chunk_id": "c_01",
        "timestamp": 5.0,
    }

    mock_session.run.return_value = [branch_pt, branch_as]

    items = get_prerequisites_for_concept(
        concept_name="Thrashing",
        lecture_id="lec_os",
        driver=mock_driver,
    )

    assert len(items) == 2
    paths = [item["path"] for item in items]
    assert ["Virtual Memory", "Page Tables", "Page Fault", "Thrashing"] in paths
    assert ["Virtual Memory", "Address Space", "Page Fault", "Thrashing"] in paths


def test_single_lecture_isolation_enforced():
    """Verify that omitting lecture_id raises ValueError to prevent cross-lecture leakage."""
    with pytest.raises(ValueError, match="lecture_id is required"):
        get_prerequisites_for_concept("Thrashing", lecture_id="")


def test_get_prerequisites_for_query():
    """Verify get_prerequisites_for_query extracts concept and invokes traversal."""
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    record = {
        "entity_id": "e_pt",
        "concept": "Page Tables",
        "type": "Concept",
        "target_concept": "Thrashing",
        "depth": 1,
        "path_nodes": ["Page Tables", "Thrashing"],
        "edge_provenance": [],
        "direct_confidence": 0.85,
        "chunk_id": "c_01",
        "timestamp": 20.0,
    }
    mock_session.run.return_value = [record]

    items = get_prerequisites_for_query(
        query="What causes Thrashing in modern virtual memory?",
        lecture_id="lec_os",
        driver=mock_driver,
    )

    assert len(items) >= 1
    assert items[0]["concept"] == "Page Tables"
