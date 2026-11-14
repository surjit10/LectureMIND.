# evaluation/tests/test_benchmark_post_rerank.py
# Post-rerank retrieval metrics and context recall — mocked workflow, no services.

from unittest.mock import MagicMock, patch

from agent.planner.planner import QueryPlanner
from evaluation.benchmark_runner import BenchmarkRunner
from evaluation.dataset_loader import BenchmarkSample
from evaluation.metrics import retrieval_metrics


def _r(cid, **extra):
    return {"chunk_id": cid, "payload": {"chunk_id": cid}, **extra}


class TestRetrievalMetricHelpers:

    def test_ranked_chunk_ids_dedupes_and_reads_payload(self):
        results = [_r("a"), {"payload": {"chunk_id": "b"}}, _r("a"), {"chunk_id": "c"}]
        assert retrieval_metrics.ranked_chunk_ids(results) == ["a", "b", "c"]

    def test_context_recall(self):
        assert retrieval_metrics.calculate_context_recall(["a", "b"], ["b", "x"]) == 0.5
        assert retrieval_metrics.calculate_context_recall(["a"], []) == 0.0
        assert retrieval_metrics.calculate_context_recall(["a", "a"], ["a"]) == 1.0


def _runner_with_state(state):
    runner = BenchmarkRunner.__new__(BenchmarkRunner)  # skip live service wiring
    runner.workflow = MagicMock()
    runner.workflow.run.return_value = state
    runner._planner = QueryPlanner()
    return runner


def _sample(**kw):
    base = dict(lecture_id="lec", query="What is a page table?", ground_truth_answer="maps pages",
                expected_chunk_ids=["gold"], expected_route="vector_only",
                question_type="factual", difficulty="easy", topic="vm")
    base.update(kw)
    return BenchmarkSample(**base)


class TestBenchmarkPostRerankMetrics:

    def test_post_rerank_metrics_follow_reranked_order(self):
        """Pre-rerank pool has gold at rank 6; the reranker lifts it to rank 1."""
        vector = [_r(f"v{i}") for i in range(5)] + [_r("gold")]
        reranked = [_r("gold", in_context=True)] + [_r(f"v{i}", in_context=(i == 0)) for i in range(5)]
        state = {"retrieval_route": None, "graph_results": [], "vector_results": vector,
                 "reranked_results": reranked, "final_context": "ctx", "answer": "maps pages",
                 "sources": [{"chunk_id": "gold"}, {"chunk_id": "v0"}], "telemetry": {}}
        with patch("retrieval.reranker.rerank_service.should_apply_hybrid", return_value=False):
            m = _runner_with_state(state)._evaluate_sample(_sample())["metrics"]
        assert m["hit_at_5"] == 0.0                 # legacy key: pre-rerank pool
        assert m["post_rerank_hit_at_5"] == 1.0
        assert m["post_rerank_mrr_at_5"] == 1.0
        assert m["context_recall"] == 1.0

    def test_unanswerable_samples_have_no_post_rerank_denominator(self):
        state = {"retrieval_route": None, "graph_results": [], "vector_results": [_r("x")],
                 "reranked_results": [_r("x", in_context=True)], "final_context": "ctx",
                 "answer": "Insufficient evidence found in lecture.", "sources": [], "telemetry": {}}
        with patch("retrieval.reranker.rerank_service.should_apply_hybrid", return_value=False):
            m = _runner_with_state(state)._evaluate_sample(
                _sample(expected_chunk_ids=[], answerable=False))["metrics"]
        for key in ("post_rerank_hit_at_5", "post_rerank_mrr_at_5", "context_recall"):
            assert m[key] is None
        assert m["refusal_correct"] == 1.0

    def test_benchmark_skips_hybrid_for_lecture_wide_queries(self):
        """The benchmark must use the same hybrid gate as runtime (no RRF for summaries)."""
        state = {"retrieval_route": None, "graph_results": [], "vector_results": [_r("a")],
                 "reranked_results": [_r("a", in_context=True)], "final_context": "ctx",
                 "answer": "summary", "sources": [{"chunk_id": "a"}], "telemetry": {},
                 "query_plan": QueryPlanner().plan_full("Summarize the lecture")}
        with patch("retrieval.hybrid.bm25_retriever.bm25_search") as mock_bm25:
            _runner_with_state(state)._evaluate_sample(_sample(query="Summarize the lecture",
                                                               expected_chunk_ids=["a"]))
        mock_bm25.assert_not_called()
