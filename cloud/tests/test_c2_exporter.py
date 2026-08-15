# cloud/tests/test_c2_exporter.py
# Tests for Stage C2 — Knowledge Package Export.
# Uses the same fixture as C1 + verifies zip contents.

import json
import zipfile
import pytest
import numpy as np
from pathlib import Path

from config import CloudSettings
from cloud.packaging.exporter import export_package


@pytest.fixture
def cloud_settings(tmp_path):
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def valid_package(cloud_settings):
    """Create a minimal valid knowledge package (reuses C1 fixture pattern)."""
    lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
    lecture_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000001",
         "timestamp": 10.0, "transcript": "BFS", "visual_context": "diagram", "ocr_text": "text"},
    ]
    segments = [
        {"segment_id": "seg_1", "title": "BFS", "start": 10.0, "end": 20.0,
         "chunks": ["lec_001_chunk_000001"]},
    ]
    entities = [
        {"entity_id": "lec_001_entity_000001", "name": "BFS", "type": "Algorithm"},
    ]
    relations = []
    embedding_ids = [{"row_index": 0, "chunk_id": "lec_001_chunk_000001"}]
    manifest = {
        "lecture_id": "lec_001", "created_at": "2026-06-17T12:00:00Z",
        "embedding_model": "BAAI/bge-large-en-v1.5", "embedding_dimension": 1024,
        "chunk_count": 1, "segment_count": 1, "entity_count": 1,
        "relation_count": 0, "reranker_version": "v1.0",
    }

    (lecture_dir / "multimodal_chunks.json").write_text(json.dumps(chunks))
    (lecture_dir / "segments.json").write_text(json.dumps(segments))
    (lecture_dir / "entities.json").write_text(json.dumps(entities))
    (lecture_dir / "relations.json").write_text(json.dumps(relations))
    (lecture_dir / "embedding_ids.json").write_text(json.dumps(embedding_ids))
    (lecture_dir / "manifest.json").write_text(json.dumps(manifest))
    (lecture_dir / "chunk_segment_map.json").write_text(json.dumps({"lec_001_chunk_000001": "seg_1"}))
    np.save(str(lecture_dir / "embeddings.npy"), np.random.randn(1, 1024).astype(np.float32))
    # Triplets carry the reranker training data — now part of the package.
    (lecture_dir / "triplets.json").write_text(json.dumps(
        [{"query": "What is BFS?", "positive": "BFS explores level by level",
          "negative": "DFS explores depth first"}]
    ))

    # Also create excluded files that must NOT end up in the zip.
    (lecture_dir / "vlm_output.jsonl").write_text('{"test": true}\n')
    (lecture_dir / "ocr_output.jsonl").write_text('{"test": true}\n')
    (lecture_dir / "transcript.json").write_text("[]")
    (lecture_dir / "metadata.json").write_text("{}")
    frames_dir = lecture_dir / "frames"
    frames_dir.mkdir()
    (frames_dir / "frame_000001.jpg").write_bytes(b"fake")
    logs_dir = lecture_dir / "logs"
    logs_dir.mkdir()
    (logs_dir / "pipeline.log").write_text("log entry")

    return lecture_dir


class TestExporter:

    def test_export_creates_zip(self, cloud_settings, valid_package, tmp_path):
        """Export produces knowledge_package.zip in transfer/."""
        transfer_dir = tmp_path / "transfer"
        zip_path = export_package(
            "lec_001", cloud_settings=cloud_settings,
            transfer_dir=str(transfer_dir),
        )
        assert zip_path.exists()
        assert zip_path.name == "lec_001_knowledge_package.zip"

    def test_zip_contains_required_files(self, cloud_settings, valid_package, tmp_path):
        """Zip must contain all required package files, at the ZIP ROOT."""
        transfer_dir = tmp_path / "transfer"
        zip_path = export_package(
            "lec_001", cloud_settings=cloud_settings,
            transfer_dir=str(transfer_dir),
        )

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()

        # Exact top-level names — the local importer validates/extracts
        # top-level files, so a subfolder prefix would break local import.
        required = {
            "manifest.json", "multimodal_chunks.json", "segments.json",
            "entities.json", "embeddings.npy", "embedding_ids.json",
            "chunk_segment_map.json", "triplets.json", "metadata.json",
        }
        assert required.issubset(set(names)), f"Missing files: {required - set(names)}"
        # No subfolder prefix anywhere.
        assert all("/" not in n for n in names), f"Expected flat layout, got: {names}"
        # reranker_model/ must NOT be present in the package.
        assert not any("reranker_model" in n for n in names), "reranker_model/ must not be exported"

    def test_zip_excludes_forbidden_files(self, cloud_settings, valid_package, tmp_path):
        """Zip must NOT contain vlm_output, ocr_output, frames, logs, etc."""
        transfer_dir = tmp_path / "transfer"
        zip_path = export_package(
            "lec_001", cloud_settings=cloud_settings,
            transfer_dir=str(transfer_dir),
        )

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()

        for excluded in ["vlm_output.jsonl", "ocr_output.jsonl", "transcript.json",
                         "frames.json", "training_metrics.json"]:
            assert not any(excluded in n for n in names), f"{excluded} should not be in zip"

        assert not any("/frames/" in n for n in names), "frames/ dir should not be in zip"
        assert not any("/logs/" in n for n in names), "logs/ dir should not be in zip"
        assert not any("reranker_model" in n for n in names), "reranker_model/ dir should not be in zip"

    def test_export_with_validation_failure(self, cloud_settings, tmp_path):
        """Export must fail if validation fails (no package dir)."""
        transfer_dir = tmp_path / "transfer"
        lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
        lecture_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(Exception):
            export_package("lec_001", cloud_settings=cloud_settings, transfer_dir=str(transfer_dir))
