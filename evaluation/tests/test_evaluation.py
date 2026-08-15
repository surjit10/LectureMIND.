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


class TestRerankerMetrics:

    def test_avg_score_reads_rerank_score(self):
        """Reranked results carry rerank_score, not score."""
        from evaluation.metrics.reranker_metrics import average_cross_encoder_score
        results = [
            {"rerank_score": 0.8},
            {"rerank_score": 0.6},
        ]
        assert average_cross_encoder_score(results) == 0.7

    def test_avg_score_legacy_score_fallback(self):
        from evaluation.metrics.reranker_metrics import average_cross_encoder_score
        results = [{"score": 0.5}, {"score": 0.7}]
        assert average_cross_encoder_score(results) == 0.6

    def test_avg_score_empty(self):
        from evaluation.metrics.reranker_metrics import average_cross_encoder_score
        assert average_cross_encoder_score([]) == 0.0
        assert average_cross_encoder_score([{"chunk_id": "c1"}]) == 0.0


class TestAnswerMetrics:

    def test_answer_f1_perfect(self):
        from evaluation.metrics.answer_metrics import calculate_answer_f1
        assert calculate_answer_f1("the cat sat", "the cat sat") == 1.0

    def test_answer_f1_partial(self):
        from evaluation.metrics.answer_metrics import calculate_answer_f1
        f1 = calculate_answer_f1("the cat sat on the mat", "the dog sat")
        assert 0.0 < f1 < 1.0

    def test_answer_f1_no_overlap(self):
        from evaluation.metrics.answer_metrics import calculate_answer_f1
        assert calculate_answer_f1("aaa bbb", "ccc ddd") == 0.0

    def test_keyword_recall_empty_returns_zero(self):
        """Empty keyword list must NOT report perfect recall."""
        from evaluation.metrics.answer_metrics import calculate_keyword_recall
        assert calculate_keyword_recall([], "some answer") == 0.0


class TestCitationMetrics:

    def test_completeness_all_sources_backed(self):
        from evaluation.metrics.citation_metrics import calculate_citation_completeness
        sources = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        reranked = [{"payload": {"chunk_id": "c1"}}, {"chunk_id": "c2"}]
        assert calculate_citation_completeness("ctx", sources, reranked) == 1.0

    def test_completeness_hallucinated_source(self):
        from evaluation.metrics.citation_metrics import calculate_citation_completeness
        sources = [{"chunk_id": "c1"}, {"chunk_id": "FAKE"}]
        reranked = [{"payload": {"chunk_id": "c1"}}]
        assert calculate_citation_completeness("ctx", sources, reranked) == 0.5

    def test_completeness_no_sources_no_context_ok(self):
        from evaluation.metrics.citation_metrics import calculate_citation_completeness
        assert calculate_citation_completeness("", [], []) == 1.0


class TestDatasetLoaderVisualFlag:

    def test_need_visual_loaded(self):
        """Optional need_visual flag loads and defaults to False."""
        from evaluation.dataset_loader import DatasetLoader
        import json, tempfile, os
        data = [{
            "lecture_id": "l", "query": "q", "ground_truth_answer": "a",
            "expected_chunk_ids": ["c1"], "expected_route": "vector_only",
            "question_type": "visual", "difficulty": "easy", "topic": "t",
            "need_visual": True,
        }]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            samples = DatasetLoader(path).load()
            assert samples[0].need_visual is True
        finally:
            os.unlink(path)

    def test_need_visual_invalid_type_rejected(self):
        from evaluation.dataset_loader import DatasetLoader
        import json, tempfile, os
        data = [{
            "lecture_id": "l", "query": "q", "ground_truth_answer": "a",
            "expected_chunk_ids": ["c1"], "expected_route": "vector_only",
            "question_type": "visual", "difficulty": "easy", "topic": "t",
            "need_visual": "yes",
        }]
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = f.name
        try:
            with pytest.raises(ValueError):
                DatasetLoader(path).load()
        finally:
            os.unlink(path)

    def test_visual_routing_accuracy(self):
        """Visual routing accuracy: 1 when flags match, 0 when they don't."""
        from evaluation.metrics.planner_metrics import calculate_routing_accuracy
        assert calculate_routing_accuracy("need_visual", "need_visual") == 1.0
        assert calculate_routing_accuracy("need_visual", "no_visual") == 0.0

    def test_route_enum_name_matches_dataset(self):
        """Dataset expected_route values match RetrievalRoute enum names."""
        from schemas.enums import RetrievalRoute
        assert RetrievalRoute.graph_and_vector.name == "graph_and_vector"
        assert RetrievalRoute.vector_only.name == "vector_only"
        assert RetrievalRoute.graph_only.name == "graph_only"

    def test_dataset_ground_truth_anchors_exist(self):
        """Every expected chunk ID in the QA set exists in the package."""
        import json
        from evaluation.dataset_loader import DatasetLoader
        pkg = Path("data/packages/lecture_cs162_v17")
        if not pkg.exists():
            pytest.skip("lecture package not present")
        mm = json.loads((pkg / "multimodal_chunks.json").read_text(encoding="utf-8"))
        ids = {c["chunk_id"] for c in mm}
        samples = DatasetLoader("evaluation/datasets/cs162_lecture1_qa_50.json").load()
        assert len(samples) >= 50
        for s in samples:
            for cid in s.expected_chunk_ids:
                assert cid in ids, f"anchor not in package: {cid} ({s.query})"
