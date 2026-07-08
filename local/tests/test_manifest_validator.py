# local/tests/test_manifest_validator.py
# Tests for local manifest validation.

import json
import pytest
import numpy as np
from pathlib import Path

from config import SharedSettings
from local.loaders.manifest_validator import validate_manifest, ManifestValidationError


@pytest.fixture
def valid_package(tmp_path):
    """Create a minimal valid package for manifest testing."""
    pkg = tmp_path / "package"
    pkg.mkdir()

    chunks = [
        {"lecture_id": "lec_001", "chunk_id": "c1", "timestamp": 10.0,
         "transcript": "BFS", "visual_context": "d", "ocr_text": "t"},
        {"lecture_id": "lec_001", "chunk_id": "c2", "timestamp": 20.0,
         "transcript": "DFS", "visual_context": "d", "ocr_text": "t"},
    ]
    segments = [{"segment_id": "s1", "title": "Algos", "start": 10.0, "end": 25.0, "chunks": ["c1", "c2"]}]
    entities = [{"entity_id": "e1", "name": "BFS", "type": "Algorithm"}]
    relations = []
    manifest = {
        "lecture_id": "lec_001", "created_at": "2026-06-17T12:00:00Z",
        "embedding_model": "BAAI/bge-large-en-v1.5", "embedding_dimension": 1024,
        "chunk_count": 2, "segment_count": 1, "entity_count": 1,
        "relation_count": 0, "reranker_version": "v1.0",
    }
    embedding_ids = [{"row_index": 0, "chunk_id": "c1"}, {"row_index": 1, "chunk_id": "c2"}]

    (pkg / "multimodal_chunks.json").write_text(json.dumps(chunks))
    (pkg / "segments.json").write_text(json.dumps(segments))
    (pkg / "entities.json").write_text(json.dumps(entities))
    (pkg / "relations.json").write_text(json.dumps(relations))
    (pkg / "manifest.json").write_text(json.dumps(manifest))
    (pkg / "embedding_ids.json").write_text(json.dumps(embedding_ids))
    np.save(str(pkg / "embeddings.npy"), np.random.randn(2, 1024).astype(np.float32))
    return pkg


class TestManifestValidator:

    def test_valid_manifest_passes(self, valid_package):
        result = validate_manifest(valid_package)
        assert result["status"] == "VALID"
        assert result["chunk_count"] == 2

    def test_wrong_chunk_count_fails(self, valid_package):
        manifest = json.loads((valid_package / "manifest.json").read_text())
        manifest["chunk_count"] = 999
        (valid_package / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ManifestValidationError, match="chunk_count"):
            validate_manifest(valid_package)

    def test_wrong_embedding_dimension_fails(self, valid_package):
        np.save(str(valid_package / "embeddings.npy"), np.random.randn(2, 768).astype(np.float32))
        with pytest.raises(ManifestValidationError, match="dimension"):
            validate_manifest(valid_package)

    def test_missing_manifest_fails(self, tmp_path):
        pkg = tmp_path / "empty"
        pkg.mkdir()
        with pytest.raises(ManifestValidationError, match="manifest.json"):
            validate_manifest(pkg)
