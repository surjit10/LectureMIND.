# local/tests/test_prerequisite_enricher.py
# Tests for local prerequisite enricher and Neo4j loading.

import json
from pathlib import Path
from unittest.mock import MagicMock

from local.loaders.prerequisite_enricher import (
    enrich_package,
    load_prerequisites_into_neo4j,
    enrich_and_load,
)
from local.loaders.neo4j_loader import load_neo4j


def test_enrich_package(tmp_path: Path):
    pkg = tmp_path / "package"
    pkg.mkdir()

    entities = [
        {"entity_id": "e_pt", "name": "Page Table", "type": "Concept"},
        {"entity_id": "e_pf", "name": "Page Fault", "type": "Concept"},
    ]
    chunks = [
        {
            "lecture_id": "lec_01",
            "chunk_id": "c_01",
            "timestamp": 10.0,
            "transcript": "Page tables are used for virtual memory address translation.",
            "visual_context": "",
            "ocr_text": "",
        },
        {
            "lecture_id": "lec_01",
            "chunk_id": "c_02",
            "timestamp": 20.0,
            "transcript": "Before we move on to page faults, let's first understand page tables.",
            "visual_context": "",
            "ocr_text": "",
        },
    ]
    segments = [
        {
            "segment_id": "s_01",
            "title": "Paging",
            "start": 0.0,
            "end": 30.0,
            "chunks": ["c_01", "c_02"],
        }
    ]
    relations = [
        {
            "relation_id": "r_01",
            "source_entity_id": "e_pt",
            "relation": "USED_BY",
            "target_entity_id": "e_pf",
        }
    ]

    (pkg / "entities.json").write_text(json.dumps(entities))
    (pkg / "multimodal_chunks.json").write_text(json.dumps(chunks))
    (pkg / "segments.json").write_text(json.dumps(segments))
    (pkg / "relations.json").write_text(json.dumps(relations))

    result = enrich_package(pkg, min_confidence=0.60)
    assert "prerequisites" in result
    assert len(result["prerequisites"]) == 1

    prereqs_path = pkg / "prerequisites.json"
    assert prereqs_path.exists()
    saved = json.loads(prereqs_path.read_text(encoding="utf-8"))
    assert len(saved["prerequisites"]) == 1
    assert saved["prerequisites"][0]["source_name"] == "Page Table"
    assert saved["prerequisites"][0]["target_name"] == "Page Fault"


def test_load_prerequisites_into_neo4j():
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    prereqs = [
        {
            "source_id": "e_pt",
            "target_id": "e_pf",
            "relation_id": "prereq_000001",
            "confidence": 0.85,
            "evidence_chunk_id": "c_01",
            "evidence_timestamp": 10.0,
            "signals": {"discourse": 0.95},
            "discourse_pattern": "pedagogical_precedence",
            "discourse_snippet": "before page faults understand page tables",
        }
    ]

    count = load_prerequisites_into_neo4j(mock_driver, prereqs, "lec_01")
    assert count == 1
    assert mock_session.run.call_count == 1
    args, kwargs = mock_session.run.call_args
    assert kwargs["source"] == "e_pt"
    assert kwargs["target"] == "e_pf"
    assert kwargs["lecture_id"] == "lec_01"
    assert kwargs["confidence"] == 0.85


def test_load_neo4j_with_prerequisites(tmp_path: Path):
    """Verify load_neo4j automatically loads prerequisites.json if present."""
    pkg = tmp_path / "package"
    pkg.mkdir()

    entities = [
        {"entity_id": "e_01", "name": "Alpha", "type": "Concept"},
        {"entity_id": "e_02", "name": "Beta", "type": "Concept"},
    ]
    relations = [
        {"relation_id": "r_01", "source_entity_id": "e_01", "relation": "EXPLAINS", "target_entity_id": "e_02"},
    ]
    prerequisites = {
        "prerequisites": [
            {
                "source_id": "e_01",
                "target_id": "e_02",
                "relation_id": "prereq_000001",
                "confidence": 0.90,
                "evidence_chunk_id": "c_01",
                "evidence_timestamp": 12.0,
                "signals": {},
            }
        ]
    }

    (pkg / "entities.json").write_text(json.dumps(entities))
    (pkg / "relations.json").write_text(json.dumps(relations))
    (pkg / "prerequisites.json").write_text(json.dumps(prerequisites))

    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    result = load_neo4j(pkg, "lec_01", driver=mock_driver)
    assert result["node_count"] == 2
    assert result["relationship_count"] == 1
    assert result["prerequisite_count"] == 1
    # 2 nodes + 1 standard rel + 1 prereq rel = 4 session.run calls
    assert mock_session.run.call_count == 4


def test_neo4j_prerequisite_merge_prevents_duplicate_parallel_edges():
    """
    REGRESSION TEST: Verify load_prerequisites_into_neo4j does NOT include relation_id
    in the Cypher MERGE pattern, preventing duplicate parallel edges when A9 and
    prerequisite inference both process the same concept pair.
    """
    mock_session = MagicMock()
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    prereqs = [
        {
            "source_id": "e_pt",
            "target_id": "e_pf",
            "relation_id": "prereq_000001",
            "confidence": 0.92,
            "evidence_chunk_id": "c_01",
            "evidence_timestamp": 10.0,
            "signals": {"graph": 1.0},
        }
    ]

    load_prerequisites_into_neo4j(mock_driver, prereqs, "lec_01")
    assert mock_session.run.call_count == 1
    cypher_query = mock_session.run.call_args[0][0]

    # Verify MERGE does NOT contain relation_id in pattern
    assert "MERGE (s)-[r:PREREQUISITE_OF]->(t)" in cypher_query, "MERGE must match on semantic relationship, not relation_id"
    assert "MERGE (s)-[r:PREREQUISITE_OF {relation_id:" not in cypher_query, "Must not require relation_id match"

    # Verify coalesce is used to preserve existing relation_id while storing inferred_relation_id
    assert "r.inferred_relation_id = $relation_id" in cypher_query
    assert "r.relation_id = coalesce(r.relation_id, $relation_id)" in cypher_query


def test_edge_merging_simulation_preserves_single_semantic_edge():
    """
    Simulate stateful Neo4j relationship execution across Stage D1 (A9 loader)
    and Stage D1.5 (prerequisite enricher) to verify that an existing explicit
    A9 PREREQUISITE_OF relationship is updated rather than duplicated.
    """
    # In-memory graph state tracking: (source, target, rel_type) -> relationship properties
    graph_edges = {}

    def fake_run(query: str, **kwargs):
        src = kwargs.get("source")
        tgt = kwargs.get("target")
        lec = kwargs.get("lecture_id")
        rel_id = kwargs.get("relation_id")

        if "MERGE (s)-[r:PREREQUISITE_OF]->(t)" in query or "MERGE (s)-[r:PREREQUISITE_OF" in query:
            key = (src, tgt, "PREREQUISITE_OF", lec)
            if key not in graph_edges:
                graph_edges[key] = {"relation_id": rel_id, "lecture_id": lec}
            # Simulate SET operations
            edge = graph_edges[key]
            if "confidence" in kwargs:
                edge["confidence"] = kwargs["confidence"]
                edge["inferred_relation_id"] = rel_id
                # coalesce: if relation_id was already set (e.g. from A9), preserve it
                edge["relation_id"] = edge.get("relation_id") or rel_id
                edge["is_inferred"] = True

    mock_session = MagicMock()
    mock_session.run.side_effect = fake_run
    mock_driver = MagicMock()
    mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
    mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

    # 1. First: A9 loads explicit relation
    from local.loaders.neo4j_loader import _create_relationships
    from schemas.relation import Relation
    from schemas.enums import RelationType

    a9_rel = Relation(
        relation_id="rel_a9_001",
        source_entity_id="e_01",
        relation=RelationType.PREREQUISITE_OF,
        target_entity_id="e_02",
    )
    _create_relationships(mock_driver, [a9_rel], "lec_01")
    assert len(graph_edges) == 1
    edge = graph_edges[("e_01", "e_02", "PREREQUISITE_OF", "lec_01")]
    assert edge["relation_id"] == "rel_a9_001"

    # 2. Second: Prerequisite enricher runs for the same pair
    inferred_prereqs = [
        {
            "source_id": "e_01",
            "target_id": "e_02",
            "relation_id": "prereq_000001",
            "confidence": 0.95,
            "evidence_chunk_id": "c_01",
            "evidence_timestamp": 5.0,
            "signals": {},
        }
    ]
    load_prerequisites_into_neo4j(mock_driver, inferred_prereqs, "lec_01")

    # MUST still have exactly 1 edge in graph, NOT 2
    assert len(graph_edges) == 1, f"Expected 1 merged edge, found {len(graph_edges)}"
    merged = graph_edges[("e_01", "e_02", "PREREQUISITE_OF", "lec_01")]
    assert merged["relation_id"] == "rel_a9_001", "Original A9 relation_id must be preserved"
    assert merged["inferred_relation_id"] == "prereq_000001", "Inferred relation ID must be recorded"
    assert merged["confidence"] == 0.95, "Confidence must be updated"

