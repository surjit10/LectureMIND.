# cloud/tests/test_window_builder.py
# Tests for Chunk-Aware Multimodal Extraction Window Builder.

import pytest
from cloud.extraction.window_builder import (
    build_extraction_windows,
    clean_visual_context,
    format_chunk_multimodal,
)


def mock_token_counter(text: str) -> int:
    """Deterministic token counter for testing (1 word = 1 token)."""
    return max(1, len(text.split()))


@pytest.fixture
def sample_chunks():
    return [
        {
            "lecture_id": "lec_001",
            "chunk_id": f"lec_001_chunk_{i:06d}",
            "timestamp": float(i * 30),
            "transcript": f"Spoken explanation for concept {i}. Detailed theoretical analysis.",
            "visual_context": f"### Educational Slide Description\n#### Background - Blue grid background\nDiagram showing circuit component {i} with voltage V=IR.",
            "ocr_text": f"Component {i}: Formula V = I * R and Frequency f = 50Hz",
        }
        for i in range(1, 13)
    ]


class TestWindowBuilder:

    def test_100_percent_coverage(self, sample_chunks):
        """Every single source chunk must appear in at least one window."""
        windows, metrics = build_extraction_windows(
            sample_chunks,
            token_budget=60,
            overlap_chunks=1,
            token_counter=mock_token_counter,
        )

        all_source_ids = {c["chunk_id"] for c in sample_chunks}
        covered_ids = {cid for w in windows for cid in w.chunk_ids}

        assert metrics["uncovered_chunks"] == 0
        assert metrics["coverage_percentage"] == 100.0
        assert all_source_ids <= covered_ids
        assert len(windows) > 1

    def test_adjacent_window_overlap(self, sample_chunks):
        """Adjacent windows must overlap by the configured number of chunks."""
        windows, metrics = build_extraction_windows(
            sample_chunks,
            token_budget=120,
            overlap_chunks=1,
            token_counter=mock_token_counter,
        )

        assert metrics["overlap_count"] > 0
        for i in range(len(windows) - 1):
            curr_window = windows[i]
            next_window = windows[i + 1]
            shared_chunks = set(curr_window.chunk_ids) & set(next_window.chunk_ids)
            assert len(shared_chunks) >= 1

    def test_token_budget_compliance(self, sample_chunks):
        """No window should exceed the configured token budget (unless single chunk > budget)."""
        budget = 70
        windows, _ = build_extraction_windows(
            sample_chunks,
            token_budget=budget,
            overlap_chunks=1,
            token_counter=mock_token_counter,
        )

        for w in windows:
            # If multiple chunks, token_count must not exceed budget
            if len(w.chunks) > 1:
                assert w.token_count <= budget

    def test_short_lecture_creates_single_window(self, sample_chunks):
        """A short lecture that fits within the budget must create exactly 1 window."""
        short_chunks = sample_chunks[:3]
        windows, metrics = build_extraction_windows(
            short_chunks,
            token_budget=5000,
            overlap_chunks=1,
            token_counter=mock_token_counter,
        )

        assert len(windows) == 1
        assert metrics["windows_created"] == 1
        assert metrics["coverage_percentage"] == 100.0
        assert metrics["total_source_chunks"] == 3
        assert len(windows[0].chunk_ids) == 3

    def test_multimodal_preservation(self, sample_chunks):
        """Windows must preserve transcript, OCR, and visual context with chunk provenance."""
        windows, _ = build_extraction_windows(
            sample_chunks[:2],
            token_budget=5000,
            overlap_chunks=1,
            token_counter=mock_token_counter,
        )

        text = windows[0].text
        assert "SPOKEN TRANSCRIPT:" in text
        assert "SLIDE / OCR TEXT:" in text
        assert "VISUAL CONTEXT:" in text
        assert "lec_001_chunk_000001" in text
        assert "lec_001_chunk_000002" in text
        assert "V = I * R" in text

    def test_conservative_visual_cleaning(self):
        """Only structural boilerplate is removed; formulas and technical words are kept."""
        raw = (
            "### Educational Slide Description\n"
            "#### Background - Blue grid background\n"
            "Transformer primary coil connected to AC source with EMF E = -N(dPhi/dt)."
        )
        cleaned = clean_visual_context(raw)
        assert "Educational Slide Description" not in cleaned
        assert "Blue grid background" not in cleaned
        assert "Transformer primary coil connected to AC source" in cleaned
        assert "EMF E = -N(dPhi/dt)" in cleaned

    def test_missing_token_counter_raises_error(self, sample_chunks):
        """Fails loudly if token_counter is None (enforces no heuristic guessing)."""
        with pytest.raises(ValueError, match="token_counter callable must be provided"):
            build_extraction_windows(sample_chunks, token_counter=None)

    def test_empty_chunks_raises_error(self):
        """Fails loudly if empty chunks list is passed."""
        with pytest.raises(ValueError, match="empty chunks list"):
            build_extraction_windows([], token_counter=mock_token_counter)
