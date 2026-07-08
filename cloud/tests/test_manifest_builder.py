# cloud/tests/test_manifest_builder.py
# Tests for Manifest Generation.
# Uses small fixture data — no models.

import json
import pytest
from pathlib import Path

from config import CloudSettings, SharedSettings
from schemas.manifest import Manifest


@pytest.fixture
def cloud_settings(tmp_path):
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def setup_manifest_inputs(cloud_settings):
    """Create all prerequisite files for manifest building."""
    lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
    lecture_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000001",
         "timestamp": 10.0, "transcript": "BFS", "visual_context": "diagram", "ocr_text": "O(V+E)"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000002",
         "timestamp": 20.0, "transcript": "DFS", "visual_context": "tree", "ocr_text": "stack"},
    ]
    segments = [
        {"segment_id": "seg_001", "title": "Graph Traversal", "start": 10.0, "end": 25.0,
         "chunks": ["lec_001_chunk_000001", "lec_001_chunk_000002"]},
    ]
    entities = [
        {"entity_id": "lec_001_entity_000001", "name": "BFS", "type": "Algorithm"},
        {"entity_id": "lec_001_entity_000002", "name": "DFS", "type": "Algorithm"},
    ]
    relations = [
        {"relation_id": "rel_000001", "source_entity_id": "lec_001_entity_000001",
         "relation": "PREREQUISITE_OF", "target_entity_id": "lec_001_entity_000002"},
    ]

    (lecture_dir / "multimodal_chunks.json").write_text(json.dumps(chunks))
    (lecture_dir / "segments.json").write_text(json.dumps(segments))
    (lecture_dir / "entities.json").write_text(json.dumps(entities))
    (lecture_dir / "relations.json").write_text(json.dumps(relations))
    return lecture_dir


class TestManifestBuilder:

    def test_build_manifest_produces_valid_manifest(self, cloud_settings, setup_manifest_inputs):
        """Manifest builder produces a validated Manifest."""
        from cloud.packaging.manifest_builder import build_manifest

        manifest = build_manifest("lec_001", cloud_settings=cloud_settings)

        assert isinstance(manifest, Manifest)
        assert manifest.lecture_id == "lec_001"

    def test_manifest_counts_are_correct(self, cloud_settings, setup_manifest_inputs):
        """Manifest counts must match actual artifact counts."""
        from cloud.packaging.manifest_builder import build_manifest

        manifest = build_manifest("lec_001", cloud_settings=cloud_settings)

        assert manifest.chunk_count == 2
        assert manifest.segment_count == 1
        assert manifest.entity_count == 2
        assert manifest.relation_count == 1

    def test_manifest_embedding_dimension_is_1024(self, cloud_settings, setup_manifest_inputs):
        """embedding_dimension must always be 1024."""
        from cloud.packaging.manifest_builder import build_manifest

        manifest = build_manifest("lec_001", cloud_settings=cloud_settings)

        assert manifest.embedding_dimension == 1024

    def test_manifest_json_written(self, cloud_settings, setup_manifest_inputs):
        """manifest.json must exist after building."""
        from cloud.packaging.manifest_builder import build_manifest

        build_manifest("lec_001", cloud_settings=cloud_settings)

        manifest_path = setup_manifest_inputs / "manifest.json"
        assert manifest_path.exists()

        data = json.loads(manifest_path.read_text())
        Manifest(**data)

    def test_manifest_has_required_fields(self, cloud_settings, setup_manifest_inputs):
        """manifest.json must contain all required schema fields."""
        from cloud.packaging.manifest_builder import build_manifest

        build_manifest("lec_001", cloud_settings=cloud_settings)

        manifest_path = setup_manifest_inputs / "manifest.json"
        data = json.loads(manifest_path.read_text())

        required_keys = {
            "lecture_id", "created_at", "embedding_model",
            "embedding_dimension", "chunk_count", "segment_count",
            "entity_count", "relation_count", "reranker_version",
        }
        assert set(data.keys()) == required_keys

    def test_manifest_embedding_model(self, cloud_settings, setup_manifest_inputs):
        """embedding_model must be set correctly."""
        from cloud.packaging.manifest_builder import build_manifest

        manifest = build_manifest("lec_001", cloud_settings=cloud_settings)

        assert manifest.embedding_model == cloud_settings.BGE_MODEL_PATH

    def test_manifest_reranker_version(self, cloud_settings, setup_manifest_inputs):
        """reranker_version must be set correctly."""
        from cloud.packaging.manifest_builder import build_manifest

        manifest = build_manifest("lec_001", cloud_settings=cloud_settings)

        assert manifest.reranker_version == "BAAI/bge-reranker-base"

    def test_missing_chunks_raises(self, cloud_settings):
        """Missing multimodal_chunks.json must raise FileNotFoundError."""
        from cloud.packaging.manifest_builder import build_manifest

        lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
        lecture_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="multimodal_chunks.json"):
            build_manifest("lec_001", cloud_settings=cloud_settings)

    def test_missing_segments_raises(self, cloud_settings):
        """Missing segments.json must raise FileNotFoundError."""
        from cloud.packaging.manifest_builder import build_manifest

        lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
        lecture_dir.mkdir(parents=True, exist_ok=True)
        (lecture_dir / "multimodal_chunks.json").write_text("[]")

        with pytest.raises(FileNotFoundError, match="segments.json"):
            build_manifest("lec_001", cloud_settings=cloud_settings)
