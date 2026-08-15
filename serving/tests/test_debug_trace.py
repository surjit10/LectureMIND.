# serving/tests/test_debug_trace.py
# Feature 3 — Retrieval Explainability: the /query debug trace.
# Verifies the trace is a pure pass-through of workflow state (no recomputation)
# and that QueryResponse.debug is additive + backward compatible.

import asyncio

import pytest

from schemas.query import QueryRequest, QueryResponse
from schemas.enums import RetrievalRoute
from serving.fastapi.routes import query as query_routes
from serving.fastapi.routes.query import _build_debug_trace


def _fake_state():
    return {
        "query": "How is a relation connected to a table?",
        "retrieval_route": RetrievalRoute.graph_and_vector,
        "graph_results": [
            {
                "start_name": "Relational Model",
                "related_name": "Relational Algebra",
                "rel_types": ["EXPLAINS"],
                "hops": 1,
            }
        ],
        "vector_results": [
            {"chunk_id": "lec_a_chunk_1", "score": 0.82},
            {"chunk_id": "lec_a_chunk_2", "score": 0.71},
        ],
        "reranked_results": [
            {
                "chunk_id": "lec_a_chunk_1",
                "payload": {"chunk_id": "lec_a_chunk_1"},
                "rerank_score": 0.91,
            }
        ],
        "final_context": "Some grounded context text.",
        "graph_path": ["Relational Model", "Relational Algebra"],
        "telemetry": {
            "planner_latency": 0.001,
            "retrieval_latency": 0.5,
            "reranker_latency": 0.01,
            "generation_latency": 2.0,
            "total_latency": 2.5,
        },
    }


class TestBuildDebugTrace:
    def test_passthrough_of_state(self):
        trace = _build_debug_trace(_fake_state())
        assert trace["retrieval_route"] == "graph+vector"
        assert trace["stage_counts"] == {"vector": 2, "graph": 1, "reranked": 1}
        assert trace["vector_results"][0]["score"] == pytest.approx(0.82)
        assert trace["reranked_results"][0]["rerank_score"] == pytest.approx(0.91)
        assert trace["graph_results"][0]["start"] == "Relational Model"
        assert trace["telemetry"]["generation_latency"] == pytest.approx(2.0)
        assert trace["final_context_chars"] == len("Some grounded context text.")
        # Planner intent re-derived by the lightweight heuristic (no I/O).
        # "connected to" triggers graph routing → factual_qa intent.
        assert trace["plan"]["intent"] == "factual_qa"

    def test_enum_route_stringified(self):
        trace = _build_debug_trace({"query": "q", "retrieval_route": RetrievalRoute.vector_only})
        assert trace["retrieval_route"] == "vector_only"

    def test_empty_state_is_safe(self):
        trace = _build_debug_trace({"query": "", "telemetry": {}})
        assert trace["retrieval_route"] == ""
        assert trace["stage_counts"] == {"vector": 0, "graph": 0, "reranked": 0}


class TestQueryResponseDebugField:
    def test_default_empty_backward_compatible(self):
        resp = QueryResponse(answer="a", sources=[], graph_path=[])
        assert resp.debug == {}

    def test_debug_populated(self):
        resp = QueryResponse(answer="a", sources=[], graph_path=[], debug={"x": 1})
        assert resp.debug == {"x": 1}


class TestQueryEndpointDebugContract:
    """End-to-end: POST /query must surface the trace in response.debug."""

    class _FakeWorkflow:
        def __init__(self, state):
            self._state = state

        def run(self, query, lecture_id=""):
            return dict(self._state)

    def test_query_endpoint_populates_debug(self, monkeypatch):
        state = _fake_state()
        monkeypatch.setattr(query_routes, "_workflow", self._FakeWorkflow(state))
        monkeypatch.setattr(query_routes, "_active_package", {"lecture_id": "lecture_a"})
        try:
            resp = asyncio.run(
                query_routes.query_endpoint(QueryRequest(query="How is a relation connected to a table?"))
            )
        finally:
            monkeypatch.setattr(query_routes, "_workflow", None)
            monkeypatch.setattr(query_routes, "_active_package", None)

        assert resp.answer == ""
        assert resp.debug["retrieval_route"] == "graph+vector"
        assert resp.debug["stage_counts"] == {"vector": 2, "graph": 1, "reranked": 1}
        assert resp.debug["telemetry"]["total_latency"] == pytest.approx(2.5)
