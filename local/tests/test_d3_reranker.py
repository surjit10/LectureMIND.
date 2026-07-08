# local/tests/test_d3_reranker.py
# Tests for D3 — Global Reranker Loader + Singleton Behavior.
# No model downloads — CrossEncoder loading is mocked.
#
# Architecture:
#   - _load_cross_encoder() loads from a fixed path (global_reranker/).
#   - The singleton is set once by app.py startup via _rs._GLOBAL_RERANKER_SERVICE.
#   - rerank() raises RuntimeError if called before singleton is initialized.

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

from local.services.reranker_service import RerankerService


# ---------------------------------------------------------------------------
# _load_cross_encoder
# ---------------------------------------------------------------------------

class TestGlobalRerankerLoader:

    def test_load_cross_encoder_calls_sentence_transformers(self, tmp_path):
        """_load_cross_encoder() invokes CrossEncoder with the given path."""
        model_dir = tmp_path / "global_reranker"
        model_dir.mkdir()

        mock_model = MagicMock()
        with patch(
            "sentence_transformers.CrossEncoder",
            return_value=mock_model,
        ) as mock_ce:
            from local.loaders.reranker_loader import _load_cross_encoder
            result = _load_cross_encoder(model_dir)

        mock_ce.assert_called_once_with(str(model_dir), device="cpu")
        assert result is mock_model

    def test_load_cross_encoder_accepts_path_object(self, tmp_path):
        """_load_cross_encoder() accepts a Path without converting to str first."""
        model_dir = tmp_path / "global_reranker"
        model_dir.mkdir()

        with patch("sentence_transformers.CrossEncoder", return_value=MagicMock()):
            from local.loaders.reranker_loader import _load_cross_encoder
            # Should not raise
            _load_cross_encoder(model_dir)


# ---------------------------------------------------------------------------
# Singleton behavior
# ---------------------------------------------------------------------------

class TestRerankerSingleton:

    def setup_method(self):
        """Reset singleton before each test."""
        from retrieval.reranker import rerank_service as _rs
        _rs._GLOBAL_RERANKER_SERVICE = None

    def teardown_method(self):
        """Reset singleton after each test."""
        from retrieval.reranker import rerank_service as _rs
        _rs._GLOBAL_RERANKER_SERVICE = None

    def test_rerank_raises_when_singleton_not_set(self):
        """rerank() must raise RuntimeError if singleton is None and no injection."""
        from retrieval.reranker.rerank_service import rerank
        with pytest.raises(RuntimeError, match="Global reranker singleton is not initialized"):
            rerank("test query", graph_results=[], vector_results=[
                {"chunk_id": "c1", "transcript": "BFS", "visual_context": "", "ocr_text": ""},
            ])

    def test_rerank_works_with_injected_service(self):
        """rerank() works when reranker_service is injected (test path)."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9])
        service = RerankerService(mock_model)

        from retrieval.reranker.rerank_service import rerank
        results, context = rerank(
            "BFS query",
            graph_results=[],
            vector_results=[
                {"chunk_id": "c1", "transcript": "BFS explores level by level",
                 "visual_context": "", "ocr_text": ""},
            ],
            reranker_service=service,
        )
        assert len(results) == 1
        assert results[0]["rerank_score"] == pytest.approx(0.9)

    def test_singleton_set_directly_is_used(self):
        """rerank() uses _GLOBAL_RERANKER_SERVICE when set externally (as app.py does)."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.75])

        from retrieval.reranker import rerank_service as _rs
        _rs._GLOBAL_RERANKER_SERVICE = RerankerService(mock_model)

        from retrieval.reranker.rerank_service import rerank
        results, _ = rerank(
            "some query",
            graph_results=[],
            vector_results=[
                {"chunk_id": "c1", "transcript": "hello", "visual_context": "", "ocr_text": ""},
            ],
        )
        assert len(results) == 1
        assert results[0]["rerank_score"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# RerankerService scoring (preserved verbatim)
# ---------------------------------------------------------------------------

class TestRerankerService:

    def test_score_pairs_sorted_descending(self):
        """Scores are returned sorted descending."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.3, 0.9, 0.1])

        service = RerankerService(mock_model)
        results = service.score_pairs("query", ["p1", "p2", "p3"])

        assert len(results) == 3
        assert results[0] == ("p2", 0.9)
        assert results[1] == ("p1", 0.3)
        assert results[2] == ("p3", 0.1)

    def test_empty_passages_returns_empty(self):
        """Empty passage list returns empty results."""
        service = RerankerService(MagicMock())
        assert service.score_pairs("query", []) == []

    def test_single_passage(self):
        """Single passage returns single result."""
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.85])

        service = RerankerService(mock_model)
        results = service.score_pairs("query", ["only passage"])

        assert len(results) == 1
        assert results[0] == ("only passage", 0.85)

