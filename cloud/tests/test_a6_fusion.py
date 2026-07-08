# cloud/tests/test_a6_fusion.py
# Tests for Stage A6 — Multimodal Fusion (MASTER CONTRACT).
# Tests the timestamp alignment logic and schema validation.

import json
import pytest
from pathlib import Path

from schemas.chunk import MultimodalChunk
from config import CloudSettings


@pytest.fixture
def cloud_settings(tmp_path):
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def setup_fusion_inputs(cloud_settings):
    """Create all prerequisite files for fusion."""
    lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
    lecture_dir.mkdir(parents=True, exist_ok=True)

    # transcript.json
    transcript = [
        {"segment_id": 1, "start": 10.0, "end": 15.0, "text": "BFS uses queue"},
        {"segment_id": 2, "start": 20.0, "end": 25.0, "text": "DFS uses stack"},
        {"segment_id": 3, "start": 50.0, "end": 55.0, "text": "No matching frame nearby"},
    ]
    (lecture_dir / "transcript.json").write_text(json.dumps(transcript))

    # frames.json — maps frame_id to timestamp.
    frames = [
        {"frame_id": 1, "timestamp": 10.5, "image_path": "frames/frame_000001.jpg"},
        {"frame_id": 2, "timestamp": 20.3, "image_path": "frames/frame_000002.jpg"},
    ]
    (lecture_dir / "frames.json").write_text(json.dumps(frames))

    # vlm_output.jsonl
    vlm_lines = [
        json.dumps({"frame_id": 1, "caption": "BFS traversal diagram", "objects": ["queue", "graph"]}),
        json.dumps({"frame_id": 2, "caption": "DFS tree structure", "objects": ["stack", "tree"]}),
    ]
    (lecture_dir / "vlm_output.jsonl").write_text("\n".join(vlm_lines) + "\n")

    # ocr_output.jsonl
    ocr_lines = [
        json.dumps({"frame_id": 1, "ocr_text": ["Breadth First Search", "O(V+E)"]}),
        json.dumps({"frame_id": 2, "ocr_text": ["Depth First Search"]}),
    ]
    (lecture_dir / "ocr_output.jsonl").write_text("\n".join(ocr_lines) + "\n")

    return lecture_dir


class TestMultimodalFusion:

    def test_fusion_produces_valid_chunks(self, cloud_settings, setup_fusion_inputs):
        """Fusion produces validated MultimodalChunk records."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        chunks = fuse("lec_001", cloud_settings=cloud_settings)

        assert len(chunks) == 3
        assert all(isinstance(c, MultimodalChunk) for c in chunks)

    def test_chunk_id_format(self, cloud_settings, setup_fusion_inputs):
        """chunk_id must follow {lecture_id}_chunk_{number} format."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        chunks = fuse("lec_001", cloud_settings=cloud_settings)

        assert chunks[0].chunk_id == "lec_001_chunk_000001"
        assert chunks[1].chunk_id == "lec_001_chunk_000002"
        assert chunks[2].chunk_id == "lec_001_chunk_000003"

    def test_timestamp_alignment(self, cloud_settings, setup_fusion_inputs):
        """First chunk aligns with frame at 10.5s (within ±2s of transcript start 10.0s)."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        chunks = fuse("lec_001", cloud_settings=cloud_settings)

        # Chunk 1: transcript start=10.0, nearest frame at 10.5 → within ±2s.
        assert chunks[0].visual_context == "BFS traversal diagram"
        assert "Breadth First Search" in chunks[0].ocr_text

        # Chunk 2: transcript start=20.0, nearest frame at 20.3 → within ±2s.
        assert chunks[1].visual_context == "DFS tree structure"
        assert "Depth First Search" in chunks[1].ocr_text

    def test_no_match_outside_tolerance(self, cloud_settings, setup_fusion_inputs):
        """Chunk 3 at timestamp 50.0 has no frame within ±2s → empty visual/ocr."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        chunks = fuse("lec_001", cloud_settings=cloud_settings)

        # Chunk 3: transcript start=50.0, no frame nearby → empty.
        assert chunks[2].visual_context == ""
        assert chunks[2].ocr_text == ""

    def test_lecture_id_propagated(self, cloud_settings, setup_fusion_inputs):
        """Every chunk must carry the correct lecture_id."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        chunks = fuse("lec_001", cloud_settings=cloud_settings)

        for chunk in chunks:
            assert chunk.lecture_id == "lec_001"

    def test_output_file_written(self, cloud_settings, setup_fusion_inputs):
        """multimodal_chunks.json must exist after fusion."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        fuse("lec_001", cloud_settings=cloud_settings)

        output_path = Path(cloud_settings.lecture_dir("lec_001")) / "multimodal_chunks.json"
        assert output_path.exists()

        data = json.loads(output_path.read_text())
        assert isinstance(data, list)
        assert len(data) == 3

        # Validate every chunk against the schema.
        for item in data:
            MultimodalChunk(**item)

    def test_no_extra_fields_in_output(self, cloud_settings, setup_fusion_inputs):
        """Output chunks must NOT contain segment_id or any field outside the schema."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        fuse("lec_001", cloud_settings=cloud_settings)

        output_path = Path(cloud_settings.lecture_dir("lec_001")) / "multimodal_chunks.json"
        data = json.loads(output_path.read_text())

        allowed_keys = {"lecture_id", "chunk_id", "timestamp", "transcript", "visual_context", "ocr_text"}
        for item in data:
            assert set(item.keys()) == allowed_keys

    def test_fusion_missing_transcript_raises(self, cloud_settings):
        """Missing transcript.json must raise FileNotFoundError."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
        lecture_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="transcript.json"):
            fuse("lec_001", cloud_settings=cloud_settings)

    def test_ocr_text_is_flattened_string(self, cloud_settings, setup_fusion_inputs):
        """ocr_text in output must be a string, not a list."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        chunks = fuse("lec_001", cloud_settings=cloud_settings)

        for chunk in chunks:
            assert isinstance(chunk.ocr_text, str)

    def test_stream_read_jsonl(self):
        """_stream_jsonl must parse line-by-line, not as JSON array."""
        from cloud.ingestion.fusion.multimodal_fusion import _stream_jsonl
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"frame_id": 1, "caption": "test", "objects": ["a"]}\n')
            f.write('{"frame_id": 2, "caption": "test2", "objects": ["b"]}\n')
            f.flush()
            path = Path(f.name)

        records = _stream_jsonl(path)
        assert len(records) == 2
        assert records[0]["frame_id"] == 1
        path.unlink()
