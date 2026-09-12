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

    def test_invalid_entity_type_mapped_to_concept(self, cloud_settings, setup_a8_inputs):
        """LLM output with invalid entity types is preserved and mapped to Concept
        (A8 fix: invalid types are no longer dropped, so no entity information is lost)."""
        from cloud.extraction.entity_extractor import _parse_entity_json

        raw = json.dumps([
            {"name": "Valid", "type": "Algorithm"},
            {"name": "Invalid", "type": "FakeType"},
        ])
        entities = _parse_entity_json(raw)
        assert len(entities) == 2
        assert {e["name"] for e in entities} == {"Valid", "Invalid"}
        invalid = next(e for e in entities if e["name"] == "Invalid")
        assert invalid["type"] == "Concept"

    def test_malformed_json_handled(self, cloud_settings, setup_a8_inputs):
        """Malformed JSON from LLM must not crash."""
        from cloud.extraction.entity_extractor import _parse_entity_json

        result = _parse_entity_json("not json at all")
        assert result == []

    def test_invalid_escapes_repaired(self, cloud_settings, setup_a8_inputs):
        r"""LaTeX-style invalid escapes (W\_q, O\(V\)) must be repaired, not dropped."""
        from cloud.extraction.entity_extractor import _parse_entity_json

        # Single backslashes in the JSON literal ("W\_q") are invalid escapes.
        raw = '[{"name": "W\\_q", "type": "Formula"}, {"name": "O\\(V\\)", "type": "Formula"}]'
        entities = _parse_entity_json(raw)
        assert len(entities) == 2
        assert entities[0]["name"] == "W\\_q"
        assert entities[1]["name"] == "O\\(V\\)"

    def test_fenced_json_parsed(self, cloud_settings, setup_a8_inputs):
        """Fenced ```json ... ``` responses must parse."""
        from cloud.extraction.entity_extractor import _parse_entity_json

        raw = '```json\n[{"name": "BFS", "type": "Algorithm"}]\n```'
        entities = _parse_entity_json(raw)
        assert len(entities) == 1
        assert entities[0]["name"] == "BFS"
        assert entities[0]["type"] == "Algorithm"

    def test_entity_type_normalization(self, cloud_settings, setup_a8_inputs):
        """Variable/TextElement/unknown types normalize deterministically to Concept;
        case variants normalize to the canonical type."""
        from cloud.extraction.entity_extractor import _parse_entity_json, normalize_entity_type

        assert normalize_entity_type("Variable") == "Concept"
        assert normalize_entity_type("TextElement") == "Concept"
        assert normalize_entity_type("concept") == "Concept"
        assert normalize_entity_type("Algorithm") == "Algorithm"

        raw = json.dumps([
            {"name": "x", "type": "Variable"},
            {"name": "y", "type": "TextElement"},
            {"name": "z", "type": "concept"},
        ])
        entities = _parse_entity_json(raw)
        assert len(entities) == 3
        assert all(e["type"] == "Concept" for e in entities)

    def test_extraction_stats_counters(self, cloud_settings, setup_a8_inputs):
        """ExtractionStats records parse/accepted/normalized counters."""
        from cloud.extraction.entity_extractor import _parse_entity_json
        from cloud.utils.diagnostics import ExtractionStats

        stats = ExtractionStats()
        _parse_entity_json('[{"name": "A", "type": "Algorithm"}]', stats=stats)
        _parse_entity_json('[{"name": "B", "type": "Variable"}]', stats=stats)

        assert stats.requests == 2
        assert stats.valid == 2
        assert stats.accepted == 2
        assert stats.normalized == 1
        assert stats.rejected == 0
        assert "A8 EXTRACTION QUALITY REPORT" in stats.report("A8")

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

    def test_extract_entities_covers_all_chunks_across_windows(self, cloud_settings, tmp_path):
        """Entity extraction with multiple windows covers 100% of chunks without truncation."""
        from cloud.extraction.entity_extractor import extract_entities
        from unittest.mock import MagicMock

        lecture_dir = Path(cloud_settings.lecture_dir("lec_002"))
        lecture_dir.mkdir(parents=True, exist_ok=True)

        chunks = [
            {"lecture_id": "lec_002", "chunk_id": f"lec_002_chunk_{i:06d}",
             "timestamp": float(i * 30), "transcript": f"Transcript content for chunk {i}",
             "visual_context": f"Diagram {i}", "ocr_text": f"OCR Text {i}"}
            for i in range(1, 7)
        ]
        segments = [
            {"segment_id": "seg_001", "title": "Full Segment", "start": 0.0, "end": 180.0,
             "chunks": [c["chunk_id"] for c in chunks]}
        ]
        (lecture_dir / "multimodal_chunks.json").write_text(json.dumps(chunks))
        (lecture_dir / "segments.json").write_text(json.dumps(segments))

        prompts_seen = []
        mock_llm = MagicMock()
        def generate_side_effect(prompts, **kwargs):
            prompts_seen.extend(prompts)
            return [json.dumps([{"name": f"Entity_W{idx}", "type": "Concept"}]) for idx, _ in enumerate(prompts)]
        mock_llm.generate.side_effect = generate_side_effect

        # Set token budget small enough to force multiple windows
        cloud_settings.EXTRACTION_WINDOW_TOKEN_BUDGET = 50

        entities = extract_entities("lec_002", cloud_settings=cloud_settings, llm_loader=lambda: mock_llm)

        assert len(prompts_seen) > 1, "Should have created multiple windows for 6 chunks with budget=50"
        # Verify all chunk IDs were included in the prompts
        all_cids = {c["chunk_id"] for c in chunks}
        found_cids = {cid for cid in all_cids if any(cid in p for p in prompts_seen)}
        assert found_cids == all_cids, "All source chunks must be present in prompts"
        assert any(e.name.startswith("Entity_W") for e in entities)

    def test_overlapping_windows_consolidate_duplicate_entities(self, cloud_settings, tmp_path):
        """Overlapping windows extracting the same entity must consolidate to a single entity."""
        from cloud.extraction.entity_extractor import extract_entities
        from unittest.mock import MagicMock

        lecture_dir = Path(cloud_settings.lecture_dir("lec_003"))
        lecture_dir.mkdir(parents=True, exist_ok=True)

        chunks = [
            {"lecture_id": "lec_003", "chunk_id": f"lec_003_chunk_{i:06d}",
             "timestamp": float(i * 30), "transcript": f"Discussion of Transformers and Alternating Current {i}",
             "visual_context": "", "ocr_text": ""}
            for i in range(1, 5)
        ]
        segments = [
            {"segment_id": "seg_001", "title": "Intro", "start": 0.0, "end": 120.0,
             "chunks": [c["chunk_id"] for c in chunks]}
        ]
        (lecture_dir / "multimodal_chunks.json").write_text(json.dumps(chunks))
        (lecture_dir / "segments.json").write_text(json.dumps(segments))

        mock_llm = MagicMock()
        # Returns identical entity name across all windows
        mock_llm.generate.side_effect = lambda prompts, **kwargs: [
            json.dumps([{"name": "Alternating Current", "type": "Concept"}]) for _ in prompts
        ]

        cloud_settings.EXTRACTION_WINDOW_TOKEN_BUDGET = 40
        entities = extract_entities("lec_003", cloud_settings=cloud_settings, llm_loader=lambda: mock_llm)

        ac_entities = [e for e in entities if e.name == "Alternating Current"]
        assert len(ac_entities) == 1, "Identical entity from multiple windows must consolidate to exactly 1 entity"

    def test_similar_but_distinct_names_remain_distinct(self):
        """Distinct terms like Cache vs Cache Controller must NOT be merged."""
        from cloud.extraction.entity_extractor import _deduplicate_entities

        raw = [
            {"name": "Cache", "type": "Concept"},
            {"name": "cache", "type": "Concept"},
            {"name": "Cache Controller", "type": "Concept"},
            {"name": "Cache Coherence", "type": "Concept"},
        ]
        deduped = _deduplicate_entities(raw)
        names = [e["name"] for e in deduped]
        assert len(deduped) == 3
        assert "Cache" in names
        assert "Cache Controller" in names
        assert "Cache Coherence" in names
