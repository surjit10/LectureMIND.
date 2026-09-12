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
