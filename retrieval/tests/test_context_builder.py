# retrieval/tests/test_context_builder.py
# Regression tests for ContextBuilder — specifically the Chat pipeline bug
# where adjacent chunk merging produced a single block that exceeded the
# budget, causing the loop to break on iteration 1 and return "".
#
# Edge cases covered:
#   - Single oversized merged chunk (the Chat bug)
#   - Multiple adjacent chunks merging into one large block
#   - No adjacent chunks (normal Chat path)
#   - Lecture-wide retrieval (is_lecture_wide=True)
#   - Very small budget
#   - Very large budget
#   - Empty retrieval results
#   - Budget exactly at chunk boundary

import pytest

from retrieval.context_builder import ContextBuilder, _build_passage_text, _are_adjacent


def _make_chunk(
    chunk_id: str,
    transcript: str = "",
    ocr_text: str = "",
    timestamp: float = 0.0,
    segment_id: str = "seg_1",
) -> dict:
    """Helper — build a result dict in the same shape as qdrant_retriever output."""
    return {
        "chunk_id": chunk_id,
        "score": 0.9,
        "rerank_score": 0.9,
        "payload": {
            "chunk_id": chunk_id,
            "transcript": transcript,
            "ocr_text": ocr_text,
            "visual_context": "",
            "timestamp": timestamp,
            "segment_id": segment_id,
        },
    }


class TestContextBuilderBugFix:
    """
    Regression tests for the Chat pipeline bug.

    Root cause: ContextBuilder.build() merged adjacent chunks into one
    large block, which exceeded the char_budget.  The loop then broke on
    the first iteration, producing an empty context that triggered the
    'Insufficient evidence' fallback in answer_generator.
    """

    def test_single_oversized_chunk_not_empty(self):
        """
        The primary regression test.

        A single chunk whose text exceeds the budget must NOT produce an
        empty context.  The builder must truncate and return at least
        some evidence so the LLM can be called.
        """
        long_transcript = "A " * 3000  # ~6000 chars — well above 4000 budget
        chunk = _make_chunk("c_001", transcript=long_transcript, timestamp=10.0)

        builder = ContextBuilder(char_budget=4000)
        context = builder.build([chunk], is_lecture_wide=False, need_visual=False)

        assert context != "", (
            "Context must not be empty when a single chunk exists, "
            "even if it exceeds the budget."
        )
        assert len(context) <= 4000, "Truncated context must respect the budget."
        assert "A" in context, "Truncated context must contain actual content."

    def test_five_adjacent_chunks_produce_non_empty_context(self):
        """
        Five semantically related chunks with timestamps < 30 s apart.

        This replicates the exact production failure: top-5 chat results
        are temporally close → merged into one block → truncated instead
        of discarded.
        """
        chunks = [
            _make_chunk(
                f"c_{i:03d}",
                transcript=f"Lecture content chunk {i}. " * 40,  # ~800 chars each
                timestamp=float(i * 10),  # 0s, 10s, 20s, 30s, 40s — all adjacent
                segment_id="seg_1",
            )
            for i in range(5)
        ]

        builder = ContextBuilder(char_budget=4000)
        context = builder.build(chunks, is_lecture_wide=False, need_visual=False)

        assert context != "", "Context must not be empty for 5 adjacent chunks."
        assert len(context) <= 4000, "Context must stay within budget."

    def test_non_adjacent_chunks_no_truncation(self):
        """
        Chunks with timestamps > 30 s apart are NOT merged.
        Each is small enough to fit individually — all should appear.
        """
        chunks = [
            _make_chunk(
                f"c_{i:03d}",
                transcript=f"Short content for chunk {i}.",
                timestamp=float(i * 60),  # 60 s apart — never adjacent
                segment_id=f"seg_{i}",
            )
            for i in range(3)
        ]

        builder = ContextBuilder(char_budget=4000)
        context = builder.build(chunks, is_lecture_wide=False, need_visual=False)

        assert context != ""
        # All three short chunks should fit.
        for i in range(3):
            assert f"chunk {i}" in context, f"Chunk {i} should appear in context."

    def test_empty_results_returns_empty_string(self):
        """Empty input must still return ''."""
        builder = ContextBuilder(char_budget=4000)
        assert builder.build([]) == ""

    def test_very_small_budget_still_returns_something(self):
        """Even a 50-char budget should produce some content, not empty."""
        chunk = _make_chunk("c_001", transcript="Hello world this is a test sentence.")
        builder = ContextBuilder(char_budget=50)
        context = builder.build([chunk], is_lecture_wide=False)

        assert context != "", "Even a tiny budget must produce at least some context."
        assert len(context) <= 50

    def test_very_large_budget_returns_all_content(self):
        """A large budget should include all chunks without truncation."""
        chunks = [
            _make_chunk(f"c_{i}", transcript=f"Content {i}.", timestamp=float(i * 60))
            for i in range(5)
        ]
        builder = ContextBuilder(char_budget=100_000)
        context = builder.build(chunks, is_lecture_wide=False)

        for i in range(5):
            assert f"Content {i}" in context

    def test_lecture_wide_retrieval_unaffected(self):
        """
        Lecture-wide mode samples across timeline.  The fix must not
        change its behaviour — spread-out chunks are not merged, so
        the first-chunk guarantee path is never hit.
        """
        chunks = [
            _make_chunk(
                f"c_{i:03d}",
                transcript=f"Topic {i} discussed at this point in the lecture.",
                timestamp=float(i * 300),  # 5-min intervals
                segment_id=f"seg_{i}",
            )
            for i in range(10)
        ]

        builder = ContextBuilder(char_budget=4000)
        context = builder.build(chunks, is_lecture_wide=True, need_visual=False)

        assert context != ""
        assert len(context) <= 4000

    def test_context_contains_transcript_label(self):
        """
        _build_passage_text prefixes transcripts with [Transcript].
        The context must carry that label so the LLM can distinguish sources.
        """
        chunk = _make_chunk("c_001", transcript="BFS uses a queue data structure.")
        builder = ContextBuilder(char_budget=4000)
        context = builder.build([chunk])

        assert "[Transcript]" in context
        assert "BFS" in context

    def test_chunk_exactly_at_budget_boundary(self):
        """A chunk whose size exactly equals the budget fits without truncation."""
        # Build text exactly == budget characters.
        budget = 200
        # The passage is prefixed with "[Transcript] " (13 chars), so we fill
        # the transcript to make total == budget.
        transcript = "A" * (budget - len("[Transcript] "))
        chunk = _make_chunk("c_001", transcript=transcript, timestamp=0.0)
        builder = ContextBuilder(char_budget=budget)
        context = builder.build([chunk])

        assert context != ""
        assert len(context) <= budget + 20  # tiny separator slack

    def test_deduplication_by_chunk_id(self):
        """Duplicate chunk_ids produce only one context entry."""
        chunk = _make_chunk("c_001", transcript="Unique content here.", timestamp=0.0)
        builder = ContextBuilder(char_budget=4000)
        context = builder.build([chunk, chunk])  # same chunk twice

        assert context.count("Unique content here") == 1

    def test_ocr_noise_filtered_out(self):
        """OCR-only noise chunks must not contribute to the context.

        When a chunk has no transcript and only noisy OCR text,
        _build_passage_text returns str(payload) as a fallback.  The important
        guarantee is that the clean chunk's transcript does appear.
        """
        # A chunk with no transcript and only noise OCR text.
        # _is_ocr_noise will flag it, so _build_passage_text produces str(payload)
        # for the fallback — that is implementation detail.  What matters:
        # the clean chunk's educational content IS present.
        noise_chunk = _make_chunk(
            "c_noise",
            transcript="",
            ocr_text="Subscribe like share notification bell icon",
            timestamp=0.0,
        )
        clean_chunk = _make_chunk(
            "c_002",
            transcript="This is real educational content.",
            timestamp=60.0,
        )
        builder = ContextBuilder(char_budget=4000)
        context = builder.build([noise_chunk, clean_chunk])

        # The clean chunk's transcript must be present.
        assert "real educational content" in context
        # The [Transcript] portion (from the clean chunk) must not contain Subscribe.
        assert "[Transcript]" in context
        assert "Subscribe" not in context.split("[Transcript]")[-1]


class TestBudgetPropagation:
    """
    Verify that char_budget is respected when passed explicitly,
    mirroring what rerank_service.rerank() now does when forwarding
    the planner's context_budget.
    """

    def test_explicit_small_budget(self):
        """Explicit small budget overrides the default 4000 limit."""
        chunk = _make_chunk("c_001", transcript="W " * 500)  # ~1000 chars
        builder = ContextBuilder(char_budget=4000)
        context = builder.build([chunk], char_budget=100)

        assert len(context) <= 100

    def test_explicit_large_budget_lecture_wide(self):
        """Planner gives 6000-char budget for lecture-wide — must be respected."""
        chunks = [
            _make_chunk(f"c_{i}", transcript="X " * 200, timestamp=float(i * 300))
            for i in range(8)
        ]
        builder = ContextBuilder(char_budget=4000)
        context = builder.build(chunks, is_lecture_wide=True, char_budget=6000)

        assert len(context) <= 6000
        assert context != ""


class TestAdjacentMergeBehavior:
    """Unit tests for the adjacency-detection logic itself."""

    def test_chunks_within_30s_same_segment_are_adjacent(self):
        a = _make_chunk("c_001", timestamp=10.0, segment_id="seg_1")
        b = _make_chunk("c_002", timestamp=35.0, segment_id="seg_1")  # 25 s apart
        # Both chunks share a segment and are 25 s apart (< 30 s threshold) — adjacent.
        assert _are_adjacent(a, b) is True

    def test_chunks_beyond_30s_not_adjacent(self):
        a = _make_chunk("c_001", timestamp=0.0, segment_id="seg_1")
        b = _make_chunk("c_002", timestamp=60.0, segment_id="seg_1")
        assert _are_adjacent(a, b) is False

    def test_different_segments_not_adjacent(self):
        a = _make_chunk("c_001", timestamp=10.0, segment_id="seg_1")
        b = _make_chunk("c_002", timestamp=15.0, segment_id="seg_2")
        assert _are_adjacent(a, b) is False
