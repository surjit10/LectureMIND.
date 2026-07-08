# retrieval/tests/test_e2b_vector_retriever.py
# Tests for E2b — Qdrant Vector Retriever.

import pytest
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock

from retrieval.vector_retriever.qdrant_retriever import retrieve_vectors


class TestVectorRetriever:

    def _make_mock_model(self, dim=1024):
        """Create a mock embedding model."""
        mock = MagicMock()
        mock.encode.return_value = np.random.randn(1, dim).astype(np.float32)
        return mock

    def test_retrieve_top_k(self):
        """Returns top-K results from mock Qdrant."""
        mock_model = self._make_mock_model()
        mock_hit = SimpleNamespace(
            payload={"chunk_id": "c_001", "transcript": "BFS uses queue",
                     "visual_context": "diagram", "ocr_text": "O(V+E)",
                     "segment_id": "s_001", "timestamp": 10.0, "lecture_id": "lec_001"},
            score=0.95,
        )
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        mock_client.query_points.return_value = mock_response

        results = retrieve_vectors(
            "BFS explanation",
            qdrant_client=mock_client,
            embedding_model=mock_model,
            top_k=5,
        )

        assert len(results) == 1
        assert results[0]["chunk_id"] == "c_001"
        assert results[0]["score"] == 0.95
        assert "payload" in results[0]

    def test_no_raw_vectors_in_results(self):
        """Results must not contain raw vectors."""
        mock_model = self._make_mock_model()
        mock_hit = SimpleNamespace(
            payload={"chunk_id": "c_001"}, score=0.8,
        )
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = [mock_hit]
        mock_client.query_points.return_value = mock_response

        results = retrieve_vectors("test", qdrant_client=mock_client, embedding_model=mock_model)

        for r in results:
            assert "vector" not in r

    def test_wrong_dimension_raises(self):
        """Query vector != 1024 raises ValueError."""
        mock_model = self._make_mock_model(dim=768)
        mock_client = MagicMock()

        with pytest.raises(ValueError, match="1024"):
            retrieve_vectors("test", qdrant_client=mock_client, embedding_model=mock_model)
