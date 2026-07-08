# cloud/tests/test_a9_relation_extraction.py
# Tests for Stage A9 — Relation Extraction via Qwen2.5-7B-Instruct.
# vLLM is mocked — no model downloads.

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from config import CloudSettings
from schemas.relation import Relation
from schemas.enums import RelationType


@pytest.fixture
def cloud_settings(tmp_path):
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def setup_a9_inputs(cloud_settings):
    """Create entities.json, segments.json, and multimodal_chunks.json."""
    lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
    lecture_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000001",
         "timestamp": 10.0, "transcript": "BFS uses queue for traversal",
         "visual_context": "BFS traversal diagram", "ocr_text": "Breadth First Search O(V+E)"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000002",
         "timestamp": 20.0, "transcript": "DFS uses stack for traversal",
         "visual_context": "DFS tree structure", "ocr_text": "Depth First Search O(V+E)"},
    ]
    entities = [
        {"entity_id": "lec_001_entity_000001", "name": "BFS", "type": "Algorithm"},
        {"entity_id": "lec_001_entity_000002", "name": "DFS", "type": "Algorithm"},
        {"entity_id": "lec_001_entity_000003", "name": "O(V+E)", "type": "Formula"},
    ]
    segments = [
        {"segment_id": "seg_001", "title": "Graph Traversal",
         "start": 10.0, "end": 25.0,
         "chunks": ["lec_001_chunk_000001", "lec_001_chunk_000002"]},
    ]
    (lecture_dir / "multimodal_chunks.json").write_text(json.dumps(chunks))
    (lecture_dir / "entities.json").write_text(json.dumps(entities))
    (lecture_dir / "segments.json").write_text(json.dumps(segments))
    return lecture_dir


def _make_mock_llm():
    """Mock LLM backend that returns structured relation JSON."""
    mock = MagicMock()

    relation_json = json.dumps([
        {
            "source_entity_id": "lec_001_entity_000001",
            "relation": "PREREQUISITE_OF",
            "target_entity_id": "lec_001_entity_000002",
        },
        {
            "source_entity_id": "lec_001_entity_000003",
            "relation": "DERIVED_FROM",
            "target_entity_id": "lec_001_entity_000001",
        },
    ])

    def generate_fn(prompts, **kwargs):
        return [relation_json for _ in prompts]

    mock.generate.side_effect = generate_fn
    return mock


class TestRelationExtractor:

    def test_extract_relations_produces_valid_relations(self, cloud_settings, setup_a9_inputs):
        """Relation extraction produces validated Relation records."""
        from cloud.extraction.relation_extractor import extract_relations

        relations = extract_relations(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _make_mock_llm(),
        )

        assert len(relations) >= 1
        assert all(isinstance(r, Relation) for r in relations)

    def test_relation_id_format(self, cloud_settings, setup_a9_inputs):
        """relation_id must follow rel_NNNNNN format."""
        from cloud.extraction.relation_extractor import extract_relations

        relations = extract_relations(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _make_mock_llm(),
        )

        for rel in relations:
            assert rel.relation_id.startswith("rel_")

    def test_relations_use_entity_ids_not_names(self, cloud_settings, setup_a9_inputs):
        """Relations MUST use source_entity_id/target_entity_id, never names."""
        from cloud.extraction.relation_extractor import extract_relations

        relations = extract_relations(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _make_mock_llm(),
        )

        for rel in relations:
            assert rel.source_entity_id.startswith("lec_001_entity_")
            assert rel.target_entity_id.startswith("lec_001_entity_")
            assert rel.source_entity_id != "BFS"
            assert rel.source_entity_id != "DFS"
            assert rel.target_entity_id != "BFS"
            assert rel.target_entity_id != "DFS"

    def test_relation_types_are_valid(self, cloud_settings, setup_a9_inputs):
        """All relation types must be from the allowed RelationType enum."""
        from cloud.extraction.relation_extractor import extract_relations

        relations = extract_relations(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _make_mock_llm(),
        )

        valid_types = {t.value for t in RelationType}
        for rel in relations:
            assert rel.relation.value in valid_types

    def test_relations_json_written(self, cloud_settings, setup_a9_inputs):
        """relations.json must exist after extraction."""
        from cloud.extraction.relation_extractor import extract_relations

        extract_relations(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _make_mock_llm(),
        )

        relations_path = setup_a9_inputs / "relations.json"
        assert relations_path.exists()

        data = json.loads(relations_path.read_text())
        assert isinstance(data, list)
        assert len(data) >= 1

        for item in data:
            Relation(**item)

    def test_no_self_relations(self, cloud_settings, setup_a9_inputs):
        """No entity should have a relation to itself."""
        from cloud.extraction.relation_extractor import extract_relations

        relations = extract_relations(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _make_mock_llm(),
        )

        for rel in relations:
            assert rel.source_entity_id != rel.target_entity_id

    def test_invalid_entity_references_filtered(self, cloud_settings, setup_a9_inputs):
        """Relations referencing non-existent entity_ids must be filtered."""
        from cloud.extraction.relation_extractor import _validate_entity_references

        valid_ids = {"lec_001_entity_000001", "lec_001_entity_000002"}
        raw = [
            {"source_entity_id": "lec_001_entity_000001", "relation": "PREREQUISITE_OF",
             "target_entity_id": "lec_001_entity_000002"},
            {"source_entity_id": "lec_001_entity_FAKE", "relation": "EXPLAINS",
             "target_entity_id": "lec_001_entity_000001"},
        ]
        filtered = _validate_entity_references(raw, valid_ids)
        assert len(filtered) == 1
        assert filtered[0]["source_entity_id"] == "lec_001_entity_000001"

    def test_duplicate_relations_removed(self, cloud_settings, setup_a9_inputs):
        """Duplicate (source, relation, target) triples must be removed."""
        from cloud.extraction.relation_extractor import _deduplicate_relations

        raw = [
            {"source_entity_id": "e1", "relation": "EXPLAINS", "target_entity_id": "e2"},
            {"source_entity_id": "e1", "relation": "EXPLAINS", "target_entity_id": "e2"},
            {"source_entity_id": "e1", "relation": "USED_BY", "target_entity_id": "e2"},
        ]
        deduped = _deduplicate_relations(raw)
        assert len(deduped) == 2

    def test_malformed_json_handled(self, cloud_settings, setup_a9_inputs):
        """Malformed JSON from LLM must not crash."""
        from cloud.extraction.relation_extractor import _parse_relation_json

        result = _parse_relation_json("not json at all {}")
        assert result == []

    def test_custom_extractor_fn(self, cloud_settings, setup_a9_inputs):
        """Custom extractor_fn injection must work."""
        from cloud.extraction.relation_extractor import extract_relations

        def mock_extractor(entities, chunks):
            return [{
                "source_entity_id": entities[0]["entity_id"],
                "relation": "EXPLAINS",
                "target_entity_id": entities[1]["entity_id"],
            }]

        relations = extract_relations(
            "lec_001", cloud_settings=cloud_settings, extractor_fn=mock_extractor,
        )

        assert len(relations) == 1
        assert relations[0].relation.value == "EXPLAINS"

    def test_missing_entities_raises(self, cloud_settings):
        """Missing entities.json must raise FileNotFoundError."""
        from cloud.extraction.relation_extractor import extract_relations

        lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
        lecture_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="entities.json"):
            extract_relations("lec_001", cloud_settings=cloud_settings)

    def test_missing_chunks_raises(self, cloud_settings):
        """Missing multimodal_chunks.json must raise FileNotFoundError."""
        from cloud.extraction.relation_extractor import extract_relations

        lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
        lecture_dir.mkdir(parents=True, exist_ok=True)
        entities = [{"entity_id": "e1", "name": "Test", "type": "Concept"}]
        (lecture_dir / "entities.json").write_text(json.dumps(entities))

        with pytest.raises(FileNotFoundError, match="multimodal_chunks.json"):
            extract_relations("lec_001", cloud_settings=cloud_settings)
