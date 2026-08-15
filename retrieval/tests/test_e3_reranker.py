# retrieval/tests/test_e3_reranker.py
# Tests for E3 — Reranker fusion and reranking.

import pytest
import numpy as np
from unittest.mock import MagicMock

from retrieval.reranker.rerank_service import rerank, _extract_passages, MAX_CONTEXT_CHARS
from retrieval.context_builder import _build_graph_context
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

        # Inject a mock reranker so this test is independent of the
        # application-level global singleton (set by app.py at startup).
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5] * len(vr))
        service = RerankerService(mock_model)

        reranked, context = rerank("test", [], vr, reranker_service=service)
        assert len(context) <= MAX_CONTEXT_CHARS + 100  # Allow for join separators.

    def test_empty_results(self):
        """Empty input returns empty output."""
        reranked, context = rerank("test", [], [])
        assert reranked == []
        assert context == ""

    def _make_graph_results(self):
        return [
            {
                "start_id": "e_001", "start_name": "Operating System Diagram",
                "start_type": "Concept", "related_id": "e_002",
                "related_name": "Network Diagram", "related_type": "Concept",
                "rel_types": ["DERIVED_FROM"], "hops": 1,
            },
            {
                "start_id": "e_003", "start_name": "BFS", "start_type": "Algorithm",
                "related_id": "e_004", "related_name": "Queue", "related_type": "Concept",
                "rel_types": ["USES"], "hops": 1,
            },
            {
                "start_id": "e_005", "start_name": "Isolated Node",
                "start_type": "Concept", "related_name": None,
                "rel_types": [], "hops": 0,
            },
        ]

    def test_graph_context_rendered_into_final_context(self):
        """Graph paths must reach final_context even when the reranker sees
        only vector passages (relationship grounding)."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9])
        service = RerankerService(mock_model)

        reranked, context = rerank(
            "How is the OS diagram related to the network diagram?",
            graph_results=self._make_graph_results(),
            vector_results=self._make_vector_results()[:1],
            reranker_service=service,
        )

        assert "[Graph] Operating System Diagram -DERIVED_FROM-> Network Diagram" in context
        assert "[Graph] BFS -USES-> Queue" in context
        assert "[Graph] Isolated Node" in context
        # Vector transcript must still be present.
        assert "[Transcript]" in context

    def test_graph_only_no_vectors_returns_graph_context(self):
        """Graph-only queries (no vector passages) must still get graph context,
        not an empty context that triggers the Insufficient-evidence fallback."""
        reranked, context = rerank(
            "What depends on the network diagram?",
            graph_results=self._make_graph_results(),
            vector_results=[],
        )
        assert reranked == []
        assert "[Graph] Operating System Diagram -DERIVED_FROM-> Network Diagram" in context

    def test_graph_context_dedupes_paths(self):
        """Duplicate graph paths appear only once."""
        gr = self._make_graph_results()
        context = _build_graph_context(gr + gr)
        assert context.count("DERIVED_FROM") == 1

    def test_graph_context_empty_input(self):
        """No graph results -> no graph context."""
        assert _build_graph_context([]) == ""
        assert _build_graph_context([{"start_name": ""}]) == ""
