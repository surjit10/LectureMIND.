# local/tests/test_d2_qdrant_loader.py
# Tests for D2 — Qdrant Vector Store Loader.
# Qdrant is fully mocked — no server needed.

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from config import LocalSettings, SharedSettings
from local.loaders.qdrant_loader import (
    load_qdrant, _build_payloads, _build_chunk_segment_map,
    _chunk_id_to_uuid, QdrantLoadError,
)


@pytest.fixture
def package_dir(tmp_path):
    """Create a minimal package with 3 chunks, 1 segment, and embeddings."""
    pkg = tmp_path / "package"
    pkg.mkdir()

    chunks = [
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000001",
         "timestamp": 10.0, "transcript": "BFS", "visual_context": "diagram", "ocr_text": "O(V+E)"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000002",
         "timestamp": 20.0, "transcript": "DFS", "visual_context": "tree", "ocr_text": "stack"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000003",
         "timestamp": 30.0, "transcript": "Dijkstra", "visual_context": "graph", "ocr_text": "O(E log V)"},
    ]
    segments = [
        {"segment_id": "seg_1", "title": "Graph Algos", "start": 10.0, "end": 35.0,
         "chunks": ["lec_001_chunk_000001", "lec_001_chunk_000002", "lec_001_chunk_000003"]},
    ]
    embedding_ids = [
        {"row_index": 0, "chunk_id": "lec_001_chunk_000001"},
        {"row_index": 1, "chunk_id": "lec_001_chunk_000002"},
        {"row_index": 2, "chunk_id": "lec_001_chunk_000003"},
    ]

    (pkg / "multimodal_chunks.json").write_text(json.dumps(chunks))
    (pkg / "segments.json").write_text(json.dumps(segments))
    (pkg / "embedding_ids.json").write_text(json.dumps(embedding_ids))
    np.save(str(pkg / "embeddings.npy"), np.random.randn(3, 1024).astype(np.float32))

    return pkg


class TestQdrantLoader:

    def test_deterministic_uuid(self):
        """Same chunk_id always produces the same UUID."""
        id1 = _chunk_id_to_uuid("lec_001_chunk_000001")
        id2 = _chunk_id_to_uuid("lec_001_chunk_000001")
        id3 = _chunk_id_to_uuid("lec_001_chunk_000002")
        assert id1 == id2
        assert id1 != id3

    def test_build_chunk_segment_map_from_segments(self, package_dir):
        """Falls back to segments.json when chunk_segment_map.json absent."""
        csm = _build_chunk_segment_map(package_dir)
        assert csm["lec_001_chunk_000001"] == "seg_1"
        assert csm["lec_001_chunk_000002"] == "seg_1"

    def test_build_chunk_segment_map_from_csm_file(self, package_dir):
        """Uses chunk_segment_map.json when present."""
        csm_data = {
            "lec_001_chunk_000001": "seg_custom",
            "lec_001_chunk_000002": "seg_custom",
            "lec_001_chunk_000003": "seg_custom",
        }
        (package_dir / "chunk_segment_map.json").write_text(json.dumps(csm_data))
        csm = _build_chunk_segment_map(package_dir)
        assert csm["lec_001_chunk_000001"] == "seg_custom"

    def test_payload_has_exactly_7_fields(self, package_dir):
        """Each payload must have exactly the 7 specified fields."""
        embedding_ids = json.loads((package_dir / "embedding_ids.json").read_text())
        csm = _build_chunk_segment_map(package_dir)
        payloads = _build_payloads(package_dir, embedding_ids, csm, lecture_id="lecture_test001")

        expected_keys = {"lecture_id", "chunk_id", "segment_id", "timestamp",
                         "transcript", "visual_context", "ocr_text"}
        for p in payloads:
            assert set(p.keys()) == expected_keys

    def test_payload_lecture_id_is_runtime_id(self, package_dir):
        """Payload lecture_id must be the injected runtime ID, not the package-internal one."""
        embedding_ids = json.loads((package_dir / "embedding_ids.json").read_text())
        csm = _build_chunk_segment_map(package_dir)
        payloads = _build_payloads(
            package_dir, embedding_ids, csm, lecture_id="lecture_b3fb8ab8"
        )
        for p in payloads:
            assert p["lecture_id"] == "lecture_b3fb8ab8", (
                f"Expected runtime lecture_id, got {p['lecture_id']!r}"
            )

    def test_load_qdrant_with_mock(self, package_dir):
        """Full load with mock client — lecture_id must be stored in every payload."""
        mock_client = MagicMock()

        result = load_qdrant(
            package_dir,
            lecture_id="lecture_test001",
            qdrant_client=mock_client,
        )

        assert result["point_count"] == 3
        assert result["vector_dimension"] == 1024
        assert mock_client.upsert.called
        # Verify all upserted payloads carry the runtime lecture_id.
        all_points = []
        for call in mock_client.upsert.call_args_list:
            all_points.extend(call.kwargs.get("points", call.args[1] if len(call.args) > 1 else []))
        for pt in all_points:
            payload = pt.payload if hasattr(pt, "payload") else pt.__dict__.get("payload", {})
            assert payload["lecture_id"] == "lecture_test001"

    def test_wrong_dimension_raises(self, package_dir):
        """embeddings.npy with dim != 1024 raises error."""
        np.save(str(package_dir / "embeddings.npy"), np.random.randn(3, 768).astype(np.float32))

        with pytest.raises(QdrantLoadError, match="1024"):
            load_qdrant(package_dir, lecture_id="lecture_test001", qdrant_client=MagicMock())
