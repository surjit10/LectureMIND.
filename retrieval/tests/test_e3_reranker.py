# retrieval/tests/test_e3_reranker.py
# Tests for E3 — Reranker fusion and reranking.

import pytest
import numpy as np
from unittest.mock import MagicMock

from retrieval.reranker.rerank_service import rerank, _extract_passages, MAX_CONTEXT_CHARS
from local.services.reranker_service import RerankerService


class TestReranker:

    def _make_vector_results(self):
        return [
            {"chunk_id": "c_001", "score": 0.9, "payload": {
                "chunk_id": "c_001", "transcript": "BFS uses queue",
                "visual_context": "BFS diagram", "ocr_text": "O(V+E)"}},
            {"chunk_id": "c_002", "score": 0.8, "payload": {
                "chunk_id": "c_002", "transcript": "DFS uses stack",
                "visual_context": "DFS tree", "ocr_text": "O(V+E)"}},
        ]

    def test_deduplication_by_chunk_id(self):
        """Duplicate chunk_ids are removed."""
        vr = [
            {"chunk_id": "c_001", "score": 0.9, "payload": {"transcript": "a"}},
            {"chunk_id": "c_001", "score": 0.8, "payload": {"transcript": "b"}},
        ]
        combined = _extract_passages([], vr)
        assert len(combined) == 1

    def test_rerank_with_mock_service(self):
        """Reranking produces reranked_results and final_context."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.95, 0.60])
        service = RerankerService(mock_model)

        reranked, context = rerank(
            "How does BFS work?",
            graph_results=[],
            vector_results=self._make_vector_results(),
            reranker_service=service,
        )

        assert len(reranked) == 2
        assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]
        assert len(context) > 0

    def test_context_max_length(self):
        """final_context is capped at MAX_CONTEXT_CHARS."""
        # Create results with very long text.
        long_text = "x" * 3000
        vr = [
            {"chunk_id": f"c_{i}", "score": 0.5, "payload": {
                "chunk_id": f"c_{i}", "transcript": long_text}}
            for i in range(5)
        ]

        reranked, context = rerank("test", [], vr)
        assert len(context) <= MAX_CONTEXT_CHARS + 100  # Allow for join separators.

    def test_empty_results(self):
        """Empty input returns empty output."""
        reranked, context = rerank("test", [], [])
        assert reranked == []
        assert context == ""
