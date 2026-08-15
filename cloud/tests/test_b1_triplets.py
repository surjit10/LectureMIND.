# cloud/tests/test_b1_triplets.py
# Tests for Stage B1 — Triplet Generation.
# No model downloads — pure logic tests.

import json
import pytest
from pathlib import Path

from schemas.triplet import RerankerTriplet
from config import CloudSettings


@pytest.fixture
def cloud_settings(tmp_path):
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def setup_segments_and_chunks(cloud_settings):
    """Create segments.json and multimodal_chunks.json with 2 segments."""
    lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
    lecture_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        {
            "lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000001",
            "timestamp": 10.0, "transcript": "BFS uses queue",
            "visual_context": "BFS diagram", "ocr_text": "O(V+E)",
        },
        {
            "lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000002",
            "timestamp": 20.0, "transcript": "BFS explores level by level",
            "visual_context": "BFS tree", "ocr_text": "Queue",
        },
        {
            "lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000003",
            "timestamp": 30.0, "transcript": "DFS uses stack",
            "visual_context": "DFS recursion", "ocr_text": "O(V+E)",
        },
    ]
    segments = [
        {"segment_id": "seg_1", "title": "BFS", "start": 10.0, "end": 25.0,
         "chunks": ["lec_001_chunk_000001", "lec_001_chunk_000002"]},
        {"segment_id": "seg_2", "title": "DFS", "start": 25.0, "end": 40.0,
         "chunks": ["lec_001_chunk_000003"]},
    ]

    (lecture_dir / "multimodal_chunks.json").write_text(json.dumps(chunks))
    (lecture_dir / "segments.json").write_text(json.dumps(segments))
    return lecture_dir


class TestTripletGenerator:

    def test_generate_triplets_success(self, cloud_settings, setup_segments_and_chunks):
        """Produces valid triplets with positive from same segment, negative from sibling."""
        from cloud.training.triplet_generator import generate_triplets

        triplets = generate_triplets("lec_001", cloud_settings=cloud_settings)

        assert len(triplets) >= 2  # At least one per chunk with a sibling.
        assert all(isinstance(t, RerankerTriplet) for t in triplets)

        # Verify output file.
        output_path = setup_segments_and_chunks / "triplets.json"
        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert len(data) == len(triplets)

    def test_triplets_have_non_empty_fields(self, cloud_settings, setup_segments_and_chunks):
        """All triplet fields must be non-empty."""
        from cloud.training.triplet_generator import generate_triplets

        triplets = generate_triplets("lec_001", cloud_settings=cloud_settings)

        for t in triplets:
            assert t.query.strip() != ""
            assert t.positive.strip() != ""
            assert t.negative.strip() != ""

    def test_triplet_schema_validation(self):
        """RerankerTriplet rejects empty fields."""
        with pytest.raises(Exception):
            RerankerTriplet(query="", positive="text", negative="text")

    def test_fenced_json_parsed(self):
        """Fenced ```json ... ``` responses must parse to query strings."""
        from cloud.training.triplet_generator import _parse_queries_json

        raw = '```json\n["What is BFS?", "Explain DFS traversal"]\n```'
        queries = _parse_queries_json(raw)
        assert queries == ["What is BFS?", "Explain DFS traversal"]

    def test_invalid_escapes_repaired(self):
        r"""Invalid escapes (W\_q, O\(V\)) must be repaired, not discarded."""
        from cloud.training.triplet_generator import _parse_queries_json

        # Single backslashes in the JSON literal are invalid escapes.
        raw = '["What is W\\_q?", "Explain O\\(V\\)"]'
        queries = _parse_queries_json(raw)
        assert len(queries) == 2
        assert queries[0] == "What is W\\_q?"
        assert queries[1] == "Explain O\\(V\\)"

    def test_malformed_json_falls_back_to_regex(self):
        """Unparseable JSON falls back to regex string extraction, not crash."""
        from cloud.training.triplet_generator import _parse_queries_json

        queries = _parse_queries_json("the questions are: \"What is BFS?\" and \"What is DFS?\"")
        assert queries == ["What is BFS?", "What is DFS?"]

    def test_query_diagnostics_counters(self):
        """ExtractionStats records request/valid/accepted for query parsing."""
        from cloud.training.triplet_generator import _parse_queries_json
        from cloud.utils.diagnostics import ExtractionStats

        stats = ExtractionStats()
        queries = _parse_queries_json('["What is BFS?"]', stats=stats)
        assert queries == ["What is BFS?"]
        assert stats.requests == 1
        assert stats.valid == 1
        assert stats.accepted == 1
        assert "B1 EXTRACTION QUALITY REPORT" in stats.report("B1")

    def test_missing_segments_raises(self, cloud_settings):
        """Missing segments.json raises FileNotFoundError."""
        from cloud.training.triplet_generator import generate_triplets

        lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
        lecture_dir.mkdir(parents=True, exist_ok=True)
        (lecture_dir / "multimodal_chunks.json").write_text("[]")

        with pytest.raises(FileNotFoundError, match="segments.json"):
            generate_triplets("lec_001", cloud_settings=cloud_settings)

    def test_custom_query_generator(self, cloud_settings, setup_segments_and_chunks):
        """Custom query generator is used when provided."""
        from cloud.training.triplet_generator import generate_triplets

        def custom_gen(chunk, title):
            return f"What is {title}?"

        triplets = generate_triplets(
            "lec_001", cloud_settings=cloud_settings,
            query_generator=custom_gen,
        )

        for t in triplets:
            assert t.query.startswith("What is ")
