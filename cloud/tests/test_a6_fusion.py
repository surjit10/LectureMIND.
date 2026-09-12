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

    def test_output_has_all_schema_fields(self, cloud_settings, setup_fusion_inputs):
        """Output chunks must contain all MultimodalChunk schema fields (including V2 provenance)."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        fuse("lec_001", cloud_settings=cloud_settings)

        output_path = Path(cloud_settings.lecture_dir("lec_001")) / "multimodal_chunks.json"
        data = json.loads(output_path.read_text())

        expected_keys = {
            "lecture_id", "chunk_id", "timestamp", "transcript",
            "visual_context", "ocr_text",
            "start_time", "end_time", "segment_ids",
        }
        for item in data:
            assert set(item.keys()) == expected_keys, f"Got keys: {set(item.keys())}"

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

    def test_merge_segments_merges_short_atoms(self, cloud_settings, setup_fusion_inputs):
        """With merge_segments=True, short transcript atoms are merged into semantic chunks."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        chunks = fuse("lec_001", cloud_settings=cloud_settings, merge_segments=True)

        # 3 short atoms (~55 chars total) → 1 merged chunk.
        assert len(chunks) == 1
        assert chunks[0].segment_ids == [1, 2, 3]
        assert chunks[0].start_time == 10.0
        assert chunks[0].end_time == 55.0
        assert "BFS" in chunks[0].transcript
        assert "DFS" in chunks[0].transcript

    def test_merge_preserves_visual_from_first_atom(self, cloud_settings, setup_fusion_inputs):
        """Merged chunk takes visual_context and ocr_text from the first atom's aligned frame."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        chunks = fuse("lec_001", cloud_settings=cloud_settings, merge_segments=True)

        assert chunks[0].visual_context == "BFS traversal diagram"
        assert "Breadth First Search" in chunks[0].ocr_text

    def test_merge_keeps_lecture_id(self, cloud_settings, setup_fusion_inputs):
        """Merged chunks must carry the correct lecture_id."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        chunks = fuse("lec_001", cloud_settings=cloud_settings, merge_segments=True)
        assert chunks[0].lecture_id == "lec_001"
        assert "lec_001_chunk" in chunks[0].chunk_id

    def test_merge_no_extra_fields(self, cloud_settings, setup_fusion_inputs):
        """Merged chunks must NOT contain fields outside the schema."""
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        chunks = fuse("lec_001", cloud_settings=cloud_settings, merge_segments=True)

        expected_keys = {
            "lecture_id", "chunk_id", "timestamp", "transcript",
            "visual_context", "ocr_text",
            "start_time", "end_time", "segment_ids",
        }
        for c in chunks:
            assert set(c.model_dump().keys()) == expected_keys


class TestMergeAtoms:

    def test_empty_atoms(self):
        """Empty atom list returns empty groups."""
        from cloud.ingestion.fusion.multimodal_fusion import _merge_atoms
        assert _merge_atoms([]) == []

    def test_single_atom(self):
        """Single atom returns one group of one."""
        from cloud.ingestion.fusion.multimodal_fusion import _merge_atoms
        atoms = [{"segment_id": 1, "start": 0.0, "end": 2.0, "text": "Hello world."}]
        groups = _merge_atoms(atoms)
        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_merge_below_min_absorbed(self):
        """Atoms below MIN_CHUNK_CHARS merge into one group."""
        from cloud.ingestion.fusion.multimodal_fusion import _merge_atoms
        atoms = [
            {"segment_id": 1, "start": 0.0, "end": 1.0, "text": "Short A."},
            {"segment_id": 2, "start": 1.5, "end": 2.5, "text": "Short B."},
        ]
        groups = _merge_atoms(atoms)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_sentence_boundary_at_target_triggers_break(self):
        """A sentence-ending atom at or above TARGET chars breaks into a new group."""
        from cloud.ingestion.fusion.multimodal_fusion import _merge_atoms, TARGET_CHUNK_CHARS

        # Build text long enough to exceed TARGET_CHUNK_CHARS.
        long_sentence = "This is a very long sentence that exceeds the target chunk characters threshold. " * 15
        assert len(long_sentence) >= TARGET_CHUNK_CHARS, f"Need {TARGET_CHUNK_CHARS}, got {len(long_sentence)}"

        atoms = [
            {"segment_id": 1, "start": 0.0, "end": 3.0, "text": long_sentence},
            {"segment_id": 2, "start": 3.5, "end": 6.0, "text": "Next topic sentence here."},
        ]
        groups = _merge_atoms(atoms)
        assert len(groups) == 2

    def test_silence_gap_triggers_break(self):
        """A long silence gap with enough content triggers a boundary."""
        from cloud.ingestion.fusion.multimodal_fusion import _merge_atoms, SILENCE_GAP_THRESHOLD

        atoms = [
            {"segment_id": 1, "start": 0.0, "end": 2.0,
             "text": "This is a segment with enough content to satisfy the minimum. " * 5},
            {"segment_id": 2, "start": 2.0 + SILENCE_GAP_THRESHOLD + 0.5,
             "end": 7.0, "text": "After a long pause."},
        ]
        groups = _merge_atoms(atoms)
        assert len(groups) == 2

    def test_no_boundary_keeps_growing(self):
        """Absent any boundary signal, atoms keep accumulating."""
        from cloud.ingestion.fusion.multimodal_fusion import _merge_atoms

        atoms = [
            {"segment_id": 1, "start": 0.0, "end": 1.0, "text": "Word."},
            {"segment_id": 2, "start": 1.5, "end": 2.5, "text": "More."},
            {"segment_id": 3, "start": 3.0, "end": 4.0, "text": "Still."},
            {"segment_id": 4, "start": 4.5, "end": 5.5, "text": "Going."},
        ]
        groups = _merge_atoms(atoms)
        assert len(groups) == 1
        assert len(groups[0]) == 4

    def test_deterministic(self):
        """Same input always produces the same output."""
        from cloud.ingestion.fusion.multimodal_fusion import _merge_atoms

        base = [
            {"segment_id": 1, "start": 0.0, "end": 1.0, "text": "A. "},
            {"segment_id": 2, "start": 1.5, "end": 2.5, "text": "B. "},
            {"segment_id": 3, "start": 3.0, "end": 4.0, "text": "C. "},
        ]
        r1 = _merge_atoms(base)
        r2 = _merge_atoms(base)
        assert r1 == r2


class TestTranscriptCleaning:
    """Tests for deterministic transcript cleaning (filler word removal)."""

    def test_removes_verbal_fillers(self):
        from cloud.ingestion.fusion.multimodal_fusion import _clean_transcript

        raw = "Um, uh, the operating system manages memory."
        cleaned = _clean_transcript(raw)
        assert "um" not in cleaned.lower()
        assert "uh" not in cleaned.lower()
        assert "operating system manages memory" in cleaned

    def test_removes_hedge_phrases(self):
        from cloud.ingestion.fusion.multimodal_fusion import _clean_transcript

        raw = "This is, you know, sort of a graph traversal algorithm."
        cleaned = _clean_transcript(raw)
        assert "you know" not in cleaned
        assert "sort of" not in cleaned
        assert "graph traversal algorithm" in cleaned

    def test_removes_sentence_initial_fillers(self):
        from cloud.ingestion.fusion.multimodal_fusion import _clean_transcript

        raw = "So, the kernel handles interrupts. Okay, next topic."
        cleaned = _clean_transcript(raw)
        assert cleaned.startswith("the kernel handles")
        assert "Okay" not in cleaned

    def test_preserves_technical_content(self):
        from cloud.ingestion.fusion.multimodal_fusion import _clean_transcript

        text = "Virtual memory uses page tables and TLB caches for address translation."
        assert _clean_transcript(text) == text

    def test_cleans_dangling_commas(self):
        from cloud.ingestion.fusion.multimodal_fusion import _clean_transcript

        raw = "Well, um, so, you know, CPU scheduling occurs here."
        cleaned = _clean_transcript(raw)
        assert not cleaned.startswith(",")
        assert ", ," not in cleaned
        assert "CPU scheduling occurs here" in cleaned

    def test_empty_string(self):
        from cloud.ingestion.fusion.multimodal_fusion import _clean_transcript

        assert _clean_transcript("") == ""
        assert _clean_transcript(None) is None


class TestVisualPropagation:
    """Tests for visual context carry-forward."""

    def test_visual_propagation_carries_forward(self, cloud_settings, setup_fusion_inputs):
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        chunks = fuse("lec_001", cloud_settings=cloud_settings, propagate_visual=True)
        assert len(chunks) == 3
        # Chunk 1 and 2 have their own frames
        assert chunks[0].visual_context == "BFS traversal diagram"
        assert chunks[1].visual_context == "DFS tree structure"
        # Chunk 3 (timestamp 50.0s, no frame within ±2s) inherits from chunk 2
        assert chunks[2].visual_context == "DFS tree structure"
        assert "Depth First Search" in chunks[2].ocr_text

    def test_visual_propagation_disabled_by_default(self, cloud_settings, setup_fusion_inputs):
        from cloud.ingestion.fusion.multimodal_fusion import fuse

        chunks = fuse("lec_001", cloud_settings=cloud_settings)
        assert len(chunks) == 3
        assert chunks[2].visual_context == ""
        assert chunks[2].ocr_text == ""

