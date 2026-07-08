# evaluation/tests/test_evaluation.py
# Tests for evaluation suite — uses mocks, no real services.

import json
import pytest
from pathlib import Path

from evaluation.retrieval.eval_retrieval import (
    split_dataset, evaluate_retrieval, _fallback_metrics,
)
from evaluation.ragas.eval_ragas import _placeholder_scores


class TestRetrievalEval:

    def test_split_70_15_15(self):
        """Dataset splits correctly."""
        data = [{"q": i} for i in range(100)]
        train, val, test = split_dataset(data)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_fallback_metrics(self):
        """Fallback metrics compute without ranx."""
        queries = [
            {"query": "test", "relevant_chunk_ids": ["c_001", "c_002"]},
        ]

        def mock_retrieve(q):
            return ["c_001", "c_003", "c_002"]

        metrics = _fallback_metrics(queries, mock_retrieve, [5, 10])
        assert "MRR" in metrics
        assert "Recall@5" in metrics
        assert "Recall@10" in metrics
        assert metrics["MRR"] == 1.0  # c_001 is first relevant at rank 1.
        assert metrics["Recall@5"] == 1.0  # Both c_001, c_002 in top 5.

    def test_evaluate_retrieval_fallback(self):
        """evaluate_retrieval uses fallback when ranx not available."""
        queries = [
            {"query": "q1", "relevant_chunk_ids": ["c_001"]},
        ]
        metrics = evaluate_retrieval(
            queries, lambda q: ["c_001", "c_002"], ranx_module=None,
        )
        assert metrics["MRR"] > 0


class TestRagasEval:

    def test_placeholder_scores(self):
        """Placeholder returns all three metrics."""
        scores = _placeholder_scores([{"query": "test"}])
        assert "Faithfulness" in scores
        assert "Answer Relevancy" in scores
        assert "Context Precision" in scores
