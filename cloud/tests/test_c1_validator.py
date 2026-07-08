# cloud/tests/test_c1_validator.py
# Tests for Stage C1 — Knowledge Package Validation.
# Uses small fixture data — no models.

import json
import pytest
import numpy as np
from pathlib import Path

from config import CloudSettings, SharedSettings
from cloud.packaging.validator import validate_package, ValidationError


@pytest.fixture
def cloud_settings(tmp_path):
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def valid_package(cloud_settings):
    """Create a minimal but fully valid knowledge package."""
    lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
    lecture_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000001",
         "timestamp": 10.0, "transcript": "BFS", "visual_context": "diagram", "ocr_text": "O(V+E)"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000002",
         "timestamp": 20.0, "transcript": "DFS", "visual_context": "tree", "ocr_text": "stack"},
    ]
    segments = [
        {"segment_id": "seg_1", "title": "Graph Traversal", "start": 10.0, "end": 25.0,
         "chunks": ["lec_001_chunk_000001", "lec_001_chunk_000002"]},
    ]
    entities = [
        {"entity_id": "lec_001_entity_000001", "name": "BFS", "type": "Algorithm"},
        {"entity_id": "lec_001_entity_000002", "name": "DFS", "type": "Algorithm"},
    ]
    relations = [
        {"relation_id": "rel_001", "source_entity_id": "lec_001_entity_000001",
         "relation": "PREREQUISITE_OF", "target_entity_id": "lec_001_entity_000002"},
    ]
    embedding_ids = [
        {"row_index": 0, "chunk_id": "lec_001_chunk_000001"},
        {"row_index": 1, "chunk_id": "lec_001_chunk_000002"},
    ]
    manifest = {
        "lecture_id": "lec_001", "created_at": "2026-06-17T12:00:00Z",
        "embedding_model": "BAAI/bge-large-en-v1.5", "embedding_dimension": 1024,
        "chunk_count": 2, "segment_count": 1, "entity_count": 2,
        "relation_count": 1, "reranker_version": "v1.0",
    }

    (lecture_dir / "multimodal_chunks.json").write_text(json.dumps(chunks))
    (lecture_dir / "segments.json").write_text(json.dumps(segments))
    (lecture_dir / "entities.json").write_text(json.dumps(entities))
    (lecture_dir / "relations.json").write_text(json.dumps(relations))
    (lecture_dir / "embedding_ids.json").write_text(json.dumps(embedding_ids))
    (lecture_dir / "manifest.json").write_text(json.dumps(manifest))

    embeddings = np.random.randn(2, 1024).astype(np.float32)
    np.save(str(lecture_dir / "embeddings.npy"), embeddings)

    return lecture_dir


class TestValidator:

    def test_valid_package_passes(self, cloud_settings, valid_package):
        """Fully valid package passes all checks."""
        result = validate_package("lec_001", cloud_settings=cloud_settings)
        assert result["status"] == "VALID"
        assert result["chunk_count"] == 2
        assert result["entity_count"] == 2

    def test_missing_file_fails(self, cloud_settings, valid_package):
        """Missing required file raises ValidationError."""
        (valid_package / "entities.json").unlink()
        with pytest.raises(ValidationError, match="entities.json"):
            validate_package("lec_001", cloud_settings=cloud_settings)

    def test_wrong_embedding_dimension_fails(self, cloud_settings, valid_package):
        """embeddings.npy with dim != 1024 raises ValidationError."""
        wrong_emb = np.random.randn(2, 768).astype(np.float32)
        np.save(str(valid_package / "embeddings.npy"), wrong_emb)
        with pytest.raises(ValidationError, match="1024"):
            validate_package("lec_001", cloud_settings=cloud_settings)

    def test_referential_integrity_bad_entity(self, cloud_settings, valid_package):
        """Relation referencing non-existent entity raises ValidationError."""
        relations = [
            {"relation_id": "rel_001", "source_entity_id": "lec_001_entity_FAKE",
             "relation": "PREREQUISITE_OF", "target_entity_id": "lec_001_entity_000002"},
        ]
        (valid_package / "relations.json").write_text(json.dumps(relations))
        with pytest.raises(ValidationError, match="source_entity_id"):
            validate_package("lec_001", cloud_settings=cloud_settings)

    def test_referential_integrity_bad_chunk(self, cloud_settings, valid_package):
        """embedding_ids referencing non-existent chunk raises ValidationError."""
        ids = [
            {"row_index": 0, "chunk_id": "lec_001_chunk_FAKE"},
            {"row_index": 1, "chunk_id": "lec_001_chunk_000002"},
        ]
        (valid_package / "embedding_ids.json").write_text(json.dumps(ids))
        with pytest.raises(ValidationError, match="chunk_id"):
            validate_package("lec_001", cloud_settings=cloud_settings)

    def test_manifest_count_mismatch_fails(self, cloud_settings, valid_package):
        """Manifest with wrong chunk_count raises ValidationError."""
        manifest = json.loads((valid_package / "manifest.json").read_text())
        manifest["chunk_count"] = 999
        (valid_package / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValidationError, match="chunk_count"):
            validate_package("lec_001", cloud_settings=cloud_settings)
