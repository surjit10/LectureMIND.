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

    def test_completeness_candidate_not_in_context_is_invalid(self):
        """A citation to a candidate dropped from prompt context (in_context=False) is invalid."""
        from evaluation.metrics.citation_metrics import calculate_citation_completeness
        sources = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        reranked = [
            {"payload": {"chunk_id": "c1"}, "in_context": True},
            {"payload": {"chunk_id": "c2"}, "in_context": False},
        ]
        assert calculate_citation_completeness("ctx", sources, reranked) == 0.5

    def test_completeness_all_in_context_valid(self):
        from evaluation.metrics.citation_metrics import calculate_citation_completeness
        sources = [{"chunk_id": "c1"}]
        reranked = [
            {"payload": {"chunk_id": "c1"}, "in_context": True},
            {"payload": {"chunk_id": "c2"}, "in_context": False},
        ]
        assert calculate_citation_completeness("ctx", sources, reranked) == 1.0

    def test_completeness_explicit_context_ids(self):
        from evaluation.metrics.citation_metrics import calculate_citation_completeness
        sources = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        reranked = [{"chunk_id": "c1"}, {"chunk_id": "c2"}]
        assert calculate_citation_completeness("ctx", sources, reranked, context_chunk_ids=["c1"]) == 0.5

    def test_completeness_no_sources_no_context_ok(self):
        from evaluation.metrics.citation_metrics import calculate_citation_completeness
        assert calculate_citation_completeness("", [], []) == 1.0


class TestMRRTruncation:

    def test_mrr_unbounded_counts_beyond_k(self):
        """k=None (default): a hit at rank 8 contributes 1/8."""
        from evaluation.metrics.retrieval_metrics import calculate_mrr
        assert calculate_mrr(["x"], ["a", "b", "c", "d", "e", "f", "g", "x"]) == pytest.approx(1 / 8)

    def test_mrr_at_5_zeroes_hits_beyond_rank_5(self):
        """Strict MRR@5: a relevant chunk ranked 7th contributes 0, matching Hit@5."""
        from evaluation.metrics.retrieval_metrics import calculate_mrr
        assert calculate_mrr(["x"], ["a", "b", "c", "d", "e", "f", "g", "x"], k=5) == 0.0

    def test_mrr_at_5_counts_within_rank_5(self):
        from evaluation.metrics.retrieval_metrics import calculate_mrr
        assert calculate_mrr(["x"], ["a", "b", "c", "x"], k=5) == pytest.approx(0.25)

    def test_mrr_ignores_duplicate_retrievals(self):
        """Duplicates must not shift reciprocal-rank positions."""
        from evaluation.metrics.retrieval_metrics import calculate_mrr
        # 'x' is genuinely the 3rd unique document.
        assert calculate_mrr(["x"], ["a", "x", "x", "x"], k=None) == pytest.approx(0.5)


class TestDeduplicatedMetrics:

    def test_precision_at_5_constant_k_denominator(self):
        """Standard P@k divides by k, not by the number of retrieved docs."""
        from evaluation.metrics.retrieval_metrics import calculate_precision_at_k
        assert calculate_precision_at_k(["x"], ["x", "y"], 5) == pytest.approx(1 / 5)

    def test_precision_at_5_duplicates_counted_once(self):
        from evaluation.metrics.retrieval_metrics import calculate_precision_at_k
        assert calculate_precision_at_k(["x"], ["x", "x", "x", "x", "x"], 5) == pytest.approx(1 / 5)

    def test_ndcg_duplicates_never_exceed_one(self):
        from evaluation.metrics.retrieval_metrics import calculate_ndcg
        assert calculate_ndcg(["x"], ["x", "x", "x", "x", "x"], 5) == pytest.approx(1.0)

    def test_recall_dedup(self):
        from evaluation.metrics.retrieval_metrics import calculate_recall_at_k
        assert calculate_recall_at_k(["x", "y"], ["x", "x", "y"], 5) == 1.0

    def test_hit_dedup(self):
        from evaluation.metrics.retrieval_metrics import calculate_hit_at_k
        assert calculate_hit_at_k(["x"], ["y", "y", "x"], 5) == 1.0


class TestRPrecision:

    def test_r_precision_single_chunk_hit(self):
        from evaluation.metrics.retrieval_metrics import calculate_r_precision
        assert calculate_r_precision(["c1"], ["c1", "c2", "c3"]) == 1.0

    def test_r_precision_single_chunk_miss(self):
        from evaluation.metrics.retrieval_metrics import calculate_r_precision
        assert calculate_r_precision(["c1"], ["c2", "c1", "c3"]) == 0.0

    def test_r_precision_two_chunks_partial(self):
        from evaluation.metrics.retrieval_metrics import calculate_r_precision
        assert calculate_r_precision(["c1", "c2"], ["c1", "x", "c2"]) == 0.5

    def test_r_precision_two_chunks_full(self):
        from evaluation.metrics.retrieval_metrics import calculate_r_precision
        assert calculate_r_precision(["c1", "c2"], ["c2", "c1", "x"]) == 1.0

    def test_r_precision_multiple_chunks(self):
        from evaluation.metrics.retrieval_metrics import calculate_r_precision
        assert pytest.approx(calculate_r_precision(["A", "B", "C"], ["A", "X", "B", "Y", "C"])) == 2 / 3

    def test_r_precision_zero_expected(self):
        from evaluation.metrics.retrieval_metrics import calculate_r_precision
        assert calculate_r_precision([], ["c1", "c2"]) == 0.0

    def test_r_precision_fewer_retrieved_than_r(self):
        from evaluation.metrics.retrieval_metrics import calculate_r_precision
        assert pytest.approx(calculate_r_precision(["A", "B", "C"], ["A"])) == 1 / 3


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
        """Every expected chunk ID in the QA set exists in the preserved CS162 package.

        Uses the read-only package snapshot in 0-output/ (the benchmark's
        indexed package data/packages/lecture_cs162_v17 is gitignored and not
        shipped, so the preserved ZIP is the source of truth for chunk IDs).
        """
        import json
        import zipfile
        from evaluation.dataset_loader import DatasetLoader
        pkg_zip = Path("0-output/CS162_Lecture_1_What_is_an_Operating_System_720P_knowledge_package.zip")
        if not pkg_zip.exists():
            pytest.skip("preserved CS162 package ZIP not present")
        with zipfile.ZipFile(pkg_zip) as zf:
            mm = json.loads(zf.read("multimodal_chunks.json").decode("utf-8"))
        ids = {c["chunk_id"] for c in mm}
        samples = DatasetLoader("evaluation/datasets/cs162_lecture1_qa_50.json").load()
        assert len(samples) >= 50
        for s in samples:
            for cid in s.expected_chunk_ids:
                assert cid in ids, f"anchor not in package: {cid} ({s.query})"
