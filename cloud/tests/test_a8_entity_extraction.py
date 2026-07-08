# cloud/tests/test_a8_entity_extraction.py
# Tests for Stage A8 — Entity Extraction via Qwen2.5-7B-Instruct.
# vLLM is mocked — no model downloads.

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from config import CloudSettings
from schemas.entity import Entity
from schemas.enums import EntityType


@pytest.fixture
def cloud_settings(tmp_path):
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def setup_a8_inputs(cloud_settings):
    """Create segments.json and multimodal_chunks.json for entity extraction."""
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
    segments = [
        {"segment_id": "seg_001", "title": "Graph Traversal",
         "start": 10.0, "end": 25.0,
         "chunks": ["lec_001_chunk_000001", "lec_001_chunk_000002"]},
    ]
    (lecture_dir / "multimodal_chunks.json").write_text(json.dumps(chunks))
    (lecture_dir / "segments.json").write_text(json.dumps(segments))
    return lecture_dir


def _make_mock_llm():
    """Mock LLM backend that returns structured entity JSON."""
    mock = MagicMock()

    entity_json = json.dumps([
        {"name": "Breadth First Search", "type": "Algorithm"},
        {"name": "Depth First Search", "type": "Algorithm"},
        {"name": "O(V+E)", "type": "Formula"},
        {"name": "BFS traversal diagram", "type": "Diagram"},
    ])

    def generate_fn(prompts, **kwargs):
        return [entity_json for _ in prompts]

    mock.generate.side_effect = generate_fn
    return mock


class TestEntityExtractor:

    def test_extract_entities_produces_valid_entities(self, cloud_settings, setup_a8_inputs):
        """Entity extraction produces validated Entity records."""
        from cloud.extraction.entity_extractor import extract_entities

        entities = extract_entities(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _make_mock_llm(),
        )

        assert len(entities) >= 1
        assert all(isinstance(e, Entity) for e in entities)

    def test_entity_id_format(self, cloud_settings, setup_a8_inputs):
        """entity_id must follow {lecture_id}_entity_{number} format."""
        from cloud.extraction.entity_extractor import extract_entities

        entities = extract_entities(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _make_mock_llm(),
        )

        for entity in entities:
            assert entity.entity_id.startswith("lec_001_entity_")

    def test_entity_types_are_valid(self, cloud_settings, setup_a8_inputs):
        """All entity types must be from the allowed EntityType enum."""
        from cloud.extraction.entity_extractor import extract_entities

        entities = extract_entities(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _make_mock_llm(),
        )

        valid_types = {t.value for t in EntityType}
        for entity in entities:
            assert entity.type.value in valid_types

    def test_entities_json_written(self, cloud_settings, setup_a8_inputs):
        """entities.json must exist after extraction."""
        from cloud.extraction.entity_extractor import extract_entities

        extract_entities(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _make_mock_llm(),
        )

        entities_path = setup_a8_inputs / "entities.json"
        assert entities_path.exists()

        data = json.loads(entities_path.read_text())
        assert isinstance(data, list)
        assert len(data) >= 1

        for item in data:
            Entity(**item)

    def test_no_duplicate_entities(self, cloud_settings, setup_a8_inputs):
        """Entity names must be deduplicated."""
        from cloud.extraction.entity_extractor import extract_entities

        entities = extract_entities(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _make_mock_llm(),
        )

        names_lower = [e.name.lower() for e in entities]
        assert len(names_lower) == len(set(names_lower))

    def test_custom_extractor_fn(self, cloud_settings, setup_a8_inputs):
        """Custom extractor_fn injection must work."""
        from cloud.extraction.entity_extractor import extract_entities

        def mock_extractor(text, title):
            return [("Custom Entity", "Concept")]

        entities = extract_entities(
            "lec_001", cloud_settings=cloud_settings, extractor_fn=mock_extractor,
        )

        assert any(e.name == "Custom Entity" for e in entities)

    def test_segment_titles_become_lecture_segment_entities(self, cloud_settings, setup_a8_inputs):
        """Segment titles should be added as LectureSegment type entities."""
        from cloud.extraction.entity_extractor import extract_entities

        entities = extract_entities(
            "lec_001", cloud_settings=cloud_settings,
            llm_loader=lambda: _make_mock_llm(),
        )

        segment_entities = [e for e in entities if e.type == EntityType.LectureSegment]
        assert len(segment_entities) >= 1
        assert any(e.name == "Graph Traversal" for e in segment_entities)

    def test_invalid_entity_type_filtered(self, cloud_settings, setup_a8_inputs):
        """LLM output with invalid entity types must be filtered out."""
        from cloud.extraction.entity_extractor import _parse_entity_json

        raw = json.dumps([
            {"name": "Valid", "type": "Algorithm"},
            {"name": "Invalid", "type": "FakeType"},
        ])
        entities = _parse_entity_json(raw)
        assert len(entities) == 1
        assert entities[0]["name"] == "Valid"

    def test_malformed_json_handled(self, cloud_settings, setup_a8_inputs):
        """Malformed JSON from LLM must not crash."""
        from cloud.extraction.entity_extractor import _parse_entity_json

        result = _parse_entity_json("not json at all")
        assert result == []

    def test_missing_segments_raises(self, cloud_settings):
        """Missing segments.json must raise FileNotFoundError."""
        from cloud.extraction.entity_extractor import extract_entities

        lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
        lecture_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="segments.json"):
            extract_entities("lec_001", cloud_settings=cloud_settings)

    def test_missing_chunks_raises(self, cloud_settings):
        """Missing multimodal_chunks.json must raise FileNotFoundError."""
        from cloud.extraction.entity_extractor import extract_entities

        lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
        lecture_dir.mkdir(parents=True, exist_ok=True)
        segments = [{"segment_id": "seg_001", "title": "Test", "start": 0.0, "end": 1.0, "chunks": ["c1"]}]
        (lecture_dir / "segments.json").write_text(json.dumps(segments))

        with pytest.raises(FileNotFoundError, match="multimodal_chunks.json"):
            extract_entities("lec_001", cloud_settings=cloud_settings)
