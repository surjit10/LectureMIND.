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
    """Mock LLM backend that returns structured relation JSON.

    The prompt now uses compact aliases (E1, E2, E3, ...) so the mock
    must respond with the same alias style.  The relation extractor
    remaps these back to real entity_ids via _remap_relations().
    """
    mock = MagicMock()

    relation_json = json.dumps([
        {
            "source_entity_id": "E1",
            "relation": "PREREQUISITE_OF",
            "target_entity_id": "E2",
        },
        {
            "source_entity_id": "E3",
            "relation": "DERIVED_FROM",
            "target_entity_id": "E1",
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

    def test_invalid_escapes_repaired(self, cloud_settings, setup_a9_inputs):
        r"""A9 must repair invalid escapes (W\_q) instead of discarding the whole
        segment's relations — the measured production failure mode."""
        from cloud.extraction.relation_extractor import _parse_relation_json

        # Single backslash in the JSON literal ("W\_q") is an invalid escape.
        raw = ('[{"source_entity_id": "e1", "relation": "EXPLAINS", '
               '"target_entity_id": "e2", "note": "W\\_q"}]')
        relations = _parse_relation_json(raw)
        assert len(relations) == 1
        assert relations[0]["relation"] == "EXPLAINS"
        assert relations[0]["source_entity_id"] == "e1"
        assert relations[0]["target_entity_id"] == "e2"

    def test_fenced_json_parsed(self, cloud_settings, setup_a9_inputs):
        """Fenced ```json ... ``` responses must parse."""
        from cloud.extraction.relation_extractor import _parse_relation_json

        raw = '```json\n[{"source_entity_id": "e1", "relation": "USED_BY", "target_entity_id": "e2"}]\n```'
        relations = _parse_relation_json(raw)
        assert len(relations) == 1
        assert relations[0]["relation"] == "USED_BY"

    def test_relation_type_normalization(self, cloud_settings, setup_a9_inputs):
        """Case variants normalize to canonical types; unknown types like MODIFIES
        are rejected intentionally (closed RelationType enum)."""
        from cloud.extraction.relation_extractor import _parse_relation_json, normalize_relation_type

        assert normalize_relation_type("explains") == "EXPLAINS"
        assert normalize_relation_type("MODIFIES") is None
        assert normalize_relation_type("PREREQUISITE_OF") == "PREREQUISITE_OF"

        raw = json.dumps([
            {"source_entity_id": "e1", "relation": "explains", "target_entity_id": "e2"},
            {"source_entity_id": "e1", "relation": "MODIFIES", "target_entity_id": "e2"},
        ])
        relations = _parse_relation_json(raw)
        assert len(relations) == 1
        assert relations[0]["relation"] == "EXPLAINS"

    def test_relation_diagnostics_counters(self, cloud_settings, setup_a9_inputs):
        """ExtractionStats records rejected/normalized for relation parsing."""
        from cloud.extraction.relation_extractor import _parse_relation_json
        from cloud.utils.diagnostics import ExtractionStats

        stats = ExtractionStats()
        raw = json.dumps([
            {"source_entity_id": "e1", "relation": "explains", "target_entity_id": "e2"},
            {"source_entity_id": "e1", "relation": "MODIFIES", "target_entity_id": "e2"},
        ])
        relations = _parse_relation_json(raw, stats=stats)
        assert len(relations) == 1
        assert stats.requests == 1
        assert stats.valid == 1
        assert stats.accepted == 1
        assert stats.normalized == 1
        assert stats.rejected == 1

    def test_failed_segment_retried_once_with_larger_budget(self, cloud_settings, setup_a9_inputs):
        """A segment whose JSON parse fails is retried once with a larger token budget."""
        from cloud.extraction.relation_extractor import _extract_relations_batch
        from cloud.utils.diagnostics import ExtractionStats

        good = json.dumps([
            {"source_entity_id": "lec_001_entity_000001", "relation": "EXPLAINS",
             "target_entity_id": "lec_001_entity_000002"},
        ])

        calls = []

        def generate_fn(prompts, **kwargs):
            calls.append((list(prompts), kwargs))
            if len(calls) == 1:
                # First pass: segment 0 is garbage, segment 1 is fine.
                return ["not json at all", good]
            # Retry pass: only the failed segment 0 is regenerated.
            return [good]

        mock = MagicMock()
        mock.generate.side_effect = generate_fn

        stats = ExtractionStats()
        results = _extract_relations_batch(["prompt0", "prompt1"], mock, stats=stats)

        assert len(results) == 2
        assert results[0][0]["relation"] == "EXPLAINS"  # recovered by the retry
        assert results[1][0]["relation"] == "EXPLAINS"  # parsed on the first pass
        assert stats.retries == 1
        # Retry regenerates only the failed prompt with a larger budget.
        retry_prompts, retry_kwargs = calls[1]
        assert len(retry_prompts) == 1
        assert retry_prompts[0] == "prompt0"
        assert retry_kwargs["max_tokens"] > calls[0][1]["max_tokens"]

    def test_truncated_segment_retried_to_recover_tail(self, cloud_settings, setup_a9_inputs):
        """A truncated (partially salvaged) segment is retried to recover the tail."""
        from cloud.extraction.relation_extractor import _extract_relations_batch
        from cloud.utils.diagnostics import ExtractionStats

        rel = ('{"source_entity_id": "lec_001_entity_000001", '
               '"relation": "EXPLAINS", "target_entity_id": "lec_001_entity_000002"}')
        truncated = "[%s, %s, {\"source" % (rel, rel)  # cut off mid-object
        full = "[%s, %s, %s]" % (rel, rel, rel)

        calls = []

        def generate_fn(prompts, **kwargs):
            calls.append(len(prompts))
            if len(calls) == 1:
                return [truncated]
            return [full]

        mock = MagicMock()
        mock.generate.side_effect = generate_fn

        stats = ExtractionStats()
        results = _extract_relations_batch(["prompt0"], mock, stats=stats)

        assert len(results[0]) == 3  # full list recovered via the retry
        assert stats.retries == 1

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


class TestCompactEntityAliases:
    """Tests for the compact entity alias system that fixes the star graph."""

    def test_build_compact_entity_list_format(self):
        """Compact list uses E1, E2, ... aliases and returns a valid mapping."""
        from cloud.extraction.relation_extractor import _build_compact_entity_list

        entities = [
            {"entity_id": "long_lecture_id_entity_000001", "name": "BFS", "type": "Algorithm"},
            {"entity_id": "long_lecture_id_entity_000002", "name": "DFS", "type": "Algorithm"},
        ]
        text, alias_map = _build_compact_entity_list(entities)

        assert "E1: BFS (Algorithm)" in text
        assert "E2: DFS (Algorithm)" in text
        # Full entity_ids must NOT appear in the compact list.
        assert "long_lecture_id_entity_000001" not in text
        assert alias_map["E1"] == "long_lecture_id_entity_000001"
        assert alias_map["E2"] == "long_lecture_id_entity_000002"

    def test_compact_list_dramatically_shorter(self):
        """Compact list must be much shorter than the full-ID version."""
        from cloud.extraction.relation_extractor import (
            _build_entity_list_str,
            _build_compact_entity_list,
        )

        entities = [
            {"entity_id": f"CS162_Lecture_1_What_is_an_Operating_System_720P_entity_{i:06d}",
             "name": f"Entity {i}", "type": "Concept"}
            for i in range(1, 101)
        ]
        full_str = _build_entity_list_str(entities)
        compact_str, _ = _build_compact_entity_list(entities)

        # Compact should be roughly 3-4x shorter.
        assert len(compact_str) < len(full_str) / 2

    def test_filter_entities_for_segment(self):
        """Per-segment filtering only returns entities mentioned in the text."""
        from cloud.extraction.relation_extractor import _filter_entities_for_segment

        entities = [
            {"entity_id": "e1", "name": "BFS", "type": "Algorithm"},
            {"entity_id": "e2", "name": "DFS", "type": "Algorithm"},
            {"entity_id": "e3", "name": "Dijkstra", "type": "Algorithm"},
        ]
        seg_text = "BFS uses a queue for traversal, unlike DFS."

        filtered = _filter_entities_for_segment(entities, seg_text)
        names = {e["name"] for e in filtered}
        assert "BFS" in names
        assert "DFS" in names
        assert "Dijkstra" not in names

    def test_remap_relations(self):
        """Compact aliases are correctly remapped to real entity_ids."""
        from cloud.extraction.relation_extractor import _remap_relations

        alias_map = {"E1": "real_id_001", "E2": "real_id_002"}
        relations = [
            {"source_entity_id": "E1", "relation": "EXPLAINS", "target_entity_id": "E2"},
        ]
        remapped = _remap_relations(relations, alias_map)
        assert len(remapped) == 1
        assert remapped[0]["source_entity_id"] == "real_id_001"
        assert remapped[0]["target_entity_id"] == "real_id_002"

    def test_remap_drops_unmapped_aliases(self):
        """Relations with aliases not in the map are dropped."""
        from cloud.extraction.relation_extractor import _remap_relations

        alias_map = {"E1": "real_id_001"}
        relations = [
            {"source_entity_id": "E1", "relation": "EXPLAINS", "target_entity_id": "E99"},
        ]
        remapped = _remap_relations(relations, alias_map)
        assert len(remapped) == 0
