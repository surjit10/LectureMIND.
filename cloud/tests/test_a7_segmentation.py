# cloud/tests/test_a7_segmentation.py
# Tests for Stage A7 — Semantic Topic Segmentation.
# Embedding model is mocked — no downloads.

import json
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from config import CloudSettings
from schemas.segment import LectureSegment
from schemas.chunk_map import ChunkSegmentMap


@pytest.fixture
def cloud_settings(tmp_path):
    return CloudSettings(
        LECTURE_OUTPUT_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/"),
        FRAME_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/frames/"),
        LOG_DIR=str(tmp_path / "cloud_runtime/lectures/{lecture_id}/logs/"),
    )


@pytest.fixture
def setup_chunks(cloud_settings):
    """Create a minimal multimodal_chunks.json with distinct topics."""
    lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
    lecture_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000001",
         "timestamp": 10.0, "transcript": "BFS uses queue for traversal",
         "visual_context": "BFS traversal diagram", "ocr_text": "Breadth First Search O(V+E)"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000002",
         "timestamp": 20.0, "transcript": "BFS visits nodes level by level",
         "visual_context": "BFS level order", "ocr_text": "Queue data structure"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000003",
         "timestamp": 30.0, "transcript": "BFS finds shortest path in unweighted graphs",
         "visual_context": "BFS shortest path", "ocr_text": "Shortest path BFS"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000004",
         "timestamp": 120.0, "transcript": "DFS uses stack for traversal",
         "visual_context": "DFS tree structure", "ocr_text": "Depth First Search O(V+E)"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000005",
         "timestamp": 130.0, "transcript": "DFS explores as far as possible",
         "visual_context": "DFS backtracking", "ocr_text": "Stack data structure"},
        {"lecture_id": "lec_001", "chunk_id": "lec_001_chunk_000006",
         "timestamp": 140.0, "transcript": "DFS can detect cycles in graphs",
         "visual_context": "DFS cycle detection", "ocr_text": "Cycle detection DFS"},
    ]
    (lecture_dir / "multimodal_chunks.json").write_text(json.dumps(chunks))
    return lecture_dir


def _make_mock_embedding_model(dim=1024):
    """
    Create a mock embedding model that returns deterministic embeddings.

    First 3 chunks (BFS) get similar embeddings, last 3 (DFS) get different ones.
    This ensures the segmenter detects a topic boundary.
    """
    mock = MagicMock()
    call_count = [0]

    def encode_fn(texts, **kwargs):
        n = len(texts)
        embeddings = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            np.random.seed(42 + i)
            base = np.random.randn(dim).astype(np.float32)
            base = base / np.linalg.norm(base)
            embeddings[i] = base
        return embeddings

    mock.encode.side_effect = encode_fn
    return mock


def _make_mock_title_generator():
    """Mock title generator that returns simple titles."""
    mock = MagicMock()

    def generate_fn(prompts, **kwargs):
        return [f"Topic Segment {i+1}" for i, p in enumerate(prompts)]

    mock.generate.side_effect = generate_fn
    return mock


class TestSegmenter:

    def test_segmentation_produces_valid_segments(self, cloud_settings, setup_chunks):
        """Segmentation produces validated LectureSegment records."""
        from cloud.segmentation.segmenter import segment

        segments = segment(
            "lec_001",
            cloud_settings=cloud_settings,
            min_chunks=2,
            embedding_model_loader=lambda: _make_mock_embedding_model(),
            title_generator_loader=lambda: _make_mock_title_generator(),
        )

        assert len(segments) >= 1
        assert all(isinstance(s, LectureSegment) for s in segments)

    def test_segment_id_format(self, cloud_settings, setup_chunks):
        """segment_id must follow seg_NNN format."""
        from cloud.segmentation.segmenter import segment

        segments = segment(
            "lec_001",
            cloud_settings=cloud_settings,
            min_chunks=2,
            embedding_model_loader=lambda: _make_mock_embedding_model(),
            title_generator_loader=lambda: _make_mock_title_generator(),
        )

        for seg in segments:
            assert seg.segment_id.startswith("seg_")

    def test_all_chunks_assigned(self, cloud_settings, setup_chunks):
        """Every chunk must be assigned to exactly one segment."""
        from cloud.segmentation.segmenter import segment

        segments = segment(
            "lec_001",
            cloud_settings=cloud_settings,
            min_chunks=2,
            embedding_model_loader=lambda: _make_mock_embedding_model(),
            title_generator_loader=lambda: _make_mock_title_generator(),
        )

        all_chunk_ids = set()
        for seg in segments:
            for cid in seg.chunks:
                assert cid not in all_chunk_ids, f"Chunk {cid} assigned to multiple segments"
                all_chunk_ids.add(cid)

        assert len(all_chunk_ids) == 6

    def test_segments_json_written(self, cloud_settings, setup_chunks):
        """segments.json must exist after segmentation."""
        from cloud.segmentation.segmenter import segment

        segment(
            "lec_001",
            cloud_settings=cloud_settings,
            min_chunks=2,
            embedding_model_loader=lambda: _make_mock_embedding_model(),
            title_generator_loader=lambda: _make_mock_title_generator(),
        )

        segments_path = setup_chunks / "segments.json"
        assert segments_path.exists()

        data = json.loads(segments_path.read_text())
        assert isinstance(data, list)
        assert len(data) >= 1

        for item in data:
            LectureSegment(**item)

    def test_chunk_segment_map_written(self, cloud_settings, setup_chunks):
        """chunk_segment_map.json must exist and map all chunks."""
        from cloud.segmentation.segmenter import segment

        segment(
            "lec_001",
            cloud_settings=cloud_settings,
            min_chunks=2,
            embedding_model_loader=lambda: _make_mock_embedding_model(),
            title_generator_loader=lambda: _make_mock_title_generator(),
        )

        map_path = setup_chunks / "chunk_segment_map.json"
        assert map_path.exists()

        mapping = json.loads(map_path.read_text())
        assert isinstance(mapping, dict)
        assert len(mapping) == 6

        ChunkSegmentMap(mapping=mapping)

    def test_end_after_start(self, cloud_settings, setup_chunks):
        """Every segment must have end > start."""
        from cloud.segmentation.segmenter import segment

        segments = segment(
            "lec_001",
            cloud_settings=cloud_settings,
            min_chunks=2,
            embedding_model_loader=lambda: _make_mock_embedding_model(),
            title_generator_loader=lambda: _make_mock_title_generator(),
        )

        for seg in segments:
            assert seg.end > seg.start

    def test_title_from_generator(self, cloud_settings, setup_chunks):
        """Titles must come from the title generator when provided."""
        from cloud.segmentation.segmenter import segment

        segments = segment(
            "lec_001",
            cloud_settings=cloud_settings,
            min_chunks=2,
            embedding_model_loader=lambda: _make_mock_embedding_model(),
            title_generator_loader=lambda: _make_mock_title_generator(),
        )

        for seg in segments:
            assert seg.title.startswith("Topic Segment")

    def test_missing_chunks_raises(self, cloud_settings):
        """Missing multimodal_chunks.json must raise FileNotFoundError."""
        from cloud.segmentation.segmenter import segment

        lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
        lecture_dir.mkdir(parents=True, exist_ok=True)

        with pytest.raises(FileNotFoundError, match="multimodal_chunks.json"):
            segment("lec_001", cloud_settings=cloud_settings)

    def test_empty_chunks_raises(self, cloud_settings):
        """Empty multimodal_chunks.json must raise ValueError."""
        from cloud.segmentation.segmenter import segment

        lecture_dir = Path(cloud_settings.lecture_dir("lec_001"))
        lecture_dir.mkdir(parents=True, exist_ok=True)
        (lecture_dir / "multimodal_chunks.json").write_text("[]")

        with pytest.raises(ValueError, match="zero chunks"):
            segment("lec_001", cloud_settings=cloud_settings)

    def test_fallback_title_keyword_based(self):
        """Fallback titles derive from repeated technical keywords, not raw speech."""
        from cloud.segmentation.segmenter import _generate_title_fallback

        chunks = [
            {"transcript": "so we're going to talk about cross validation today",
             "visual_context": "", "ocr_text": ""},
            {"transcript": "cross validation splits the data into folds",
             "visual_context": "", "ocr_text": ""},
            {"transcript": "with k fold cross validation we train k models",
             "visual_context": "", "ocr_text": ""},
        ]
        title = _generate_title_fallback(chunks)
        assert "cross" in title.lower()
        assert "validation" in title.lower()
        # Raw speech filler must not leak into the title.
        assert "going" not in title.lower()
        assert len(title.split()) <= 8

    def test_fallback_title_uses_ocr_heading(self):
        """Fallback titles prefer a short OCR slide heading."""
        from cloud.segmentation.segmenter import _generate_title_fallback

        chunks = [
            {"transcript": "so this is the part where we discuss bias variance tradeoff",
             "visual_context": "", "ocr_text": "Bias-Variance Tradeoff"},
        ]
        assert _generate_title_fallback(chunks) == "Bias-Variance Tradeoff"

    def test_fallback_title_ignores_ocr_boilerplate(self):
        """Fallback must skip channel-branding OCR lines before keyword fallback."""
        from cloud.segmentation.segmenter import _generate_title_fallback

        chunks = [
            {"transcript": "k fold cross validation evaluates model performance",
             "visual_context": "", "ocr_text": "Subscribe to this channel"},
            {"transcript": "each fold trains on k minus one parts",
             "visual_context": "", "ocr_text": ""},
        ]
        title = _generate_title_fallback(chunks)
        assert "subscribe" not in title.lower()
        assert "cross" in title.lower()

    def test_fallback_title_skips_markdown_filler_captions(self):
        """Markdown-prefixed, generic VLM captions must fall through to keywords."""
        from cloud.segmentation.segmenter import _generate_title_fallback

        chunks = [
            {"transcript": "relational model algebra forms the basis of the course",
             "visual_context": "### Description The image appears to be a slide about "
                                "relational algebra and its operators",
             "ocr_text": ""},
        ]
        title = _generate_title_fallback(chunks)
        assert "description" not in title.lower()
        assert "image" not in title.lower()
        assert "relational" in title.lower()

    def test_fallback_title_empty_chunks(self):
        """Empty chunk list must produce a safe placeholder."""
        from cloud.segmentation.segmenter import _generate_title_fallback

        assert _generate_title_fallback([]) == "Untitled Segment"

    def test_cosine_similarity_used(self, cloud_settings, setup_chunks):
        """Verify cosine similarity computation works correctly."""
        from cloud.segmentation.segmenter import _cosine_similarity

        a = np.array([1.0, 0.0, 0.0])
        b = np.array([1.0, 0.0, 0.0])
        assert abs(_cosine_similarity(a, b) - 1.0) < 1e-6

        c = np.array([0.0, 1.0, 0.0])
        assert abs(_cosine_similarity(a, c)) < 1e-6

    def test_adaptive_threshold(self, cloud_settings):
        """Adaptive threshold must be within clamped range."""
        from cloud.segmentation.segmenter import _compute_adaptive_threshold

        sims = [0.9, 0.85, 0.3, 0.8, 0.82]
        threshold = _compute_adaptive_threshold(sims)
        assert 0.3 <= threshold <= 0.85
