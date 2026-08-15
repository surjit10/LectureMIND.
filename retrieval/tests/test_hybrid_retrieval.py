"""Tests for hybrid retrieval: BM25 index, RRF fusion, and rerank integration."""

import json
import pytest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from retrieval.hybrid.bm25_retriever import (
    _BM25Index,
    _tokenize,
    bm25_search,
    clear_lecture_index,
    rrf_fuse,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_qdrant_points():
    """Simulate Qdrant scroll results for a small lecture."""
    class MockPoint:
        def __init__(self, cid, transcript, ocr_text, timestamp=0.0):
            self.id = cid
            self.payload = {
                "chunk_id": cid,
                "transcript": transcript,
                "ocr_text": ocr_text,
                "timestamp": timestamp,
                "lecture_id": "lec_test",
            }
    return [
        MockPoint("c1", "BFS uses a queue data structure for traversal.", "BFS Queue O(V+E)"),
        MockPoint("c2", "DFS uses a stack instead.", "DFS Stack"),
        MockPoint("c3", "The enrollment limit for this course is 428 students.", ""),
        MockPoint("c4", "Okay.", ""),
        MockPoint("c5", "Now let us talk about time complexity.", ""),
        MockPoint("c6", "The honor code requires all students to submit their own work.", "Honor Code Policy"),
    ]


@pytest.fixture
def mock_qdrant_client(mock_qdrant_points):
    """Mock a QdrantClient that returns the fixture points."""
    client = MagicMock()
    # Use return_value so every scroll call returns the full list.
    # The _BM25Index loop checks `if not page` to break.
    client.scroll.return_value = (mock_qdrant_points, None)
    return client


@pytest.fixture
def bm25_index(mock_qdrant_client):
    """Build a BM25 index from the mock points."""
    return _BM25Index("lec_test", mock_qdrant_client)


# ── Tokenization ──────────────────────────────────────────────────────

class TestTokenize:
    def test_simple_sentence(self):
        assert _tokenize("BFS uses queue") == ["bfs", "uses", "queue"]

    def test_lowercase(self):
        assert _tokenize("DFS Stack") == ["dfs", "stack"]

    def test_punctuation_stripped(self):
        assert _tokenize("Okay.") == ["okay"]

    def test_short_tokens_filtered(self):
        # "a" (len 1) filtered; "an" (len 2), "the" (len 3) kept
        assert _tokenize("a the") == ["the"]

    def test_numbers_preserved(self):
        assert _tokenize("428 students") == ["428", "students"]

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_mixed_case(self):
        result = _tokenize("BFS Queue O(V+E)")
        assert "bfs" in result
        assert "queue" in result
        assert "o" not in result  # Single char filtered
        assert "v" not in result
        assert "e" not in result


# ── BM25 Index ────────────────────────────────────────────────────────

class TestBM25Index:
    def test_build_index(self, bm25_index):
        """Index correctly builds from Qdrant points."""
        assert bm25_index.doc_count == 6
        assert bm25_index.avgdl > 0
        assert len(bm25_index.idf) > 0

    def test_doc_texts(self, bm25_index):
        """doc_texts contains (chunk_id, combined_text, payload)."""
        ids = [dt[0] for dt in bm25_index.doc_texts]
        assert "c1" in ids
        assert "c3" in ids
        # c3 has transcript + empty ocr
        c3_entry = [dt for dt in bm25_index.doc_texts if dt[0] == "c3"][0]
        assert "enrollment" in c3_entry[1]

    def test_score_empty_index(self):
        """Empty index returns empty scores."""
        # Build an index that finds no points.
        client = MagicMock()
        client.scroll.return_value = ([], None)
        idx = _BM25Index("lec_empty", client)
        assert idx.score(["bfs"]) == []

    def test_score_terms_present(self, bm25_index):
        """Query terms present in the index return non-zero scores."""
        scored = bm25_index.score(["bfs", "queue"])
        assert len(scored) > 0
        assert scored[0][1] > 0  # Top score > 0

    def test_score_terms_absent(self, bm25_index):
        """Query terms not in the index return empty."""
        scored = bm25_index.score(["zzzzzzzz", "aaaaaaa"])
        assert scored == []

    def test_clear(self, bm25_index):
        """Clear releases memory."""
        bm25_index.clear()
        assert bm25_index.doc_terms == []
        assert bm25_index.idf == {}
        assert bm25_index.doc_texts == []


# ── BM25 Search ───────────────────────────────────────────────────────

class TestBM25Search:
    def test_build_and_search(self, bm25_index):
        """Search returns results with correct structure."""
        results = []
        scored = bm25_index.score(["bfs", "queue"])
        for doc_idx, score in scored:
            cid, text, payload = bm25_index.doc_texts[doc_idx]
            results.append({"chunk_id": cid, "score": score, "payload": payload})
            if len(results) >= 5:
                break

        assert len(results) > 0
        assert results[0]["chunk_id"] == "c1"  # BFS queue doc
        assert results[0]["score"] > 0


# ── RRF Fusion ────────────────────────────────────────────────────────

def _make_result(chunk_id: str, score: float) -> Dict[str, Any]:
    """Helper to create a result dict like vector_results / BM25 results."""
    return {
        "chunk_id": chunk_id,
        "score": score,
        "payload": {"chunk_id": chunk_id, "transcript": f"text {chunk_id}"},
    }


class TestRRFFuse:
    def test_empty_vector(self):
        """Empty vector results + BM25 results = BM25-only fusion."""
        bm25 = [_make_result("c1", 10.0), _make_result("c2", 5.0)]
        fused = rrf_fuse([], bm25, top_k=5)
        assert len(fused) == 2
        assert fused[0]["chunk_id"] == "c1"

    def test_empty_bm25(self):
        """Empty BM25 results = vector-only fusion."""
        vector = [_make_result("c1", 0.9), _make_result("c2", 0.8)]
        fused = rrf_fuse(vector, [], top_k=5)
        assert len(fused) == 2

    def test_both_empty(self):
        """Both empty returns empty."""
        assert rrf_fuse([], [], top_k=5) == []

    def test_dedup(self):
        """Same chunk in both lists appears once."""
        vector = [_make_result("c1", 0.9), _make_result("c2", 0.8)]
        bm25 = [_make_result("c2", 10.0), _make_result("c3", 5.0)]
        fused = rrf_fuse(vector, bm25, top_k=10)
        assert len(fused) == 3
        c2_entries = [f for f in fused if f["chunk_id"] == "c2"]
        assert len(c2_entries) == 1

    def test_top_k_truncation(self):
        """top_k limits the number of fused results."""
        vector = [_make_result(f"c{i}", 0.5) for i in range(10)]
        bm25 = [_make_result(f"c{i}", 10.0) for i in range(10, 20)]
        fused = rrf_fuse(vector, bm25, top_k=5)
        assert len(fused) == 5

    def test_rrf_ordering(self):
        """BM25-only results get higher RRF score when vector doesn't have them."""
        vector = [_make_result("c1", 0.9)]
        bm25 = [_make_result("c2", 10.0)]
        fused = rrf_fuse(vector, bm25, top_k=5)
        # c2 has RRF = 1/61, c1 has RRF = 1/61 too (same rank 1) but c2
        # has higher vector score as tiebreaker. Actually wait:
        # c1: rank 1 in vector → 1/(60+1) = 0.01639
        # c2: rank 1 in BM25 → 1/(60+1) = 0.01639
        # Tiebreaker: score (c2=10.0 vs c1=0.9) → c2 first
        # Wait, the tiebreaker is -best_score, so higher score = first.
        assert fused[0]["chunk_id"] == "c2"  # BM25 score 10.0 as tiebreaker


# ── Rerank integration via mock ───────────────────────────────────────

class TestHybridRerank:
    @pytest.fixture(autouse=True)
    def _setup_reranker(self):
        """Prime the global reranker singleton so rerank() doesn't raise."""
        import retrieval.reranker.rerank_service as rs
        from unittest.mock import MagicMock
        mock_service = MagicMock()
        mock_service.score_pairs.return_value = [
            ("[Transcript] BFS uses queue", 0.95),
        ]
        rs._GLOBAL_RERANKER_SERVICE = mock_service
        yield
        rs._GLOBAL_RERANKER_SERVICE = None

    def test_rerank_calls_bm25_when_hybrid_enabled(self):
        """rerank() calls bm25_search when enable_hybrid=True."""
        from retrieval.reranker.rerank_service import rerank

        query = "BFS uses queue"
        vector_results = [
            {"chunk_id": "c1", "score": 0.9, "payload": {"transcript": "BFS uses queue", "chunk_id": "c1"}},
        ]
        graph_results = []

        with patch(
            "retrieval.reranker.rerank_service.bm25_search",
            return_value=[],
        ) as mock_bm25:
            reranked, context = rerank(
                query, graph_results, vector_results,
                lecture_id="lec_test",
                enable_hybrid=True,
            )
            mock_bm25.assert_called_once()

    def test_rerank_skips_hybrid_when_disabled(self):
        """rerank() does NOT call bm25_search when enable_hybrid=False."""
        from retrieval.reranker.rerank_service import rerank

        query = "BFS uses queue"
        vector_results = [
            {"chunk_id": "c1", "score": 0.9, "payload": {"transcript": "BFS uses queue", "chunk_id": "c1"}},
        ]
        graph_results = []

        with patch(
            "retrieval.reranker.rerank_service.bm25_search",
        ) as mock_bm25:
            reranked, context = rerank(
                query, graph_results, vector_results,
                lecture_id="lec_test",
                enable_hybrid=False,
            )
            mock_bm25.assert_not_called()

    def test_rerank_scores_same_visibility_as_context(self):
        """
        Rerank passages must match what the LLM context will show: visual
        context is excluded from scoring when need_visual=False.

        Regression (2026-08-11): the reranker scored passages with
        visual_context always included, so visual-boosted slides outranked
        the transcript chunk that actually named the course materials — and
        the LLM never saw those visuals.
        """
        import retrieval.reranker.rerank_service as rs
        from retrieval.reranker.rerank_service import rerank

        query = "What are the main course materials for CS162?"
        vector_results = [
            {
                "chunk_id": "c_textbook",
                "score": 0.8,
                "payload": {
                    "chunk_id": "c_textbook",
                    "transcript": "The textbook is Principles and Practices of Operating Systems.",
                    "ocr_text": "",
                    "visual_context": "",
                    "timestamp": 100.0,
                },
            },
            {
                "chunk_id": "c_visual_slide",
                "score": 0.7,
                "payload": {
                    "chunk_id": "c_visual_slide",
                    "transcript": "So what is an operating system?",
                    "ocr_text": "",
                    "visual_context": "### Title: What is an Operating System? Key concepts...",
                    "timestamp": 200.0,
                },
            },
        ]
        graph_results = []

        with patch("retrieval.reranker.rerank_service.bm25_search", return_value=[]):
            rerank(
                query, graph_results, vector_results,
                lecture_id="lec_test",
                enable_hybrid=False,
                need_visual=False,
            )

        # Passages handed to the cross-encoder must NOT contain visual text.
        passages = rs._GLOBAL_RERANKER_SERVICE.score_pairs.call_args[0][1]
        visual_passages = [p for p in passages if "What is an Operating System? Key concepts" in p]
        assert visual_passages == [], "visual_context must not be scored when need_visual=False"
        assert any("textbook" in p for p in passages)

    def test_rerank_hybrid_graceful_fallback(self):
        """rerank() falls back gracefully when BM25 raises."""
        from retrieval.reranker.rerank_service import rerank

        query = "BFS uses queue"
        vector_results = [
            {"chunk_id": "c1", "score": 0.9, "payload": {"transcript": "BFS uses queue", "chunk_id": "c1"}},
        ]
        graph_results = []

        with patch(
            "retrieval.reranker.rerank_service.bm25_search",
            side_effect=RuntimeError("Qdrant connection failed"),
        ):
            # Should not raise; logs warning and falls back to dense.
            reranked, context = rerank(
                query, graph_results, vector_results,
                lecture_id="lec_test",
                enable_hybrid=True,
            )
            # Should still produce reranked results (dense-only fallback)
            assert len(reranked) > 0