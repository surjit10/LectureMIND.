"""Tests for QueryRequest and QueryPipelineState schemas."""
import pytest
from pydantic import ValidationError
from schemas.query import QueryRequest, QueryResponse
from schemas.pipeline_state import QueryPipelineState
from schemas.enums import RetrievalRoute


def test_query_request_valid():
    qr = QueryRequest(query="Why BFS before Dijkstra?")
    assert qr.query == "Why BFS before Dijkstra?"


def test_query_request_empty_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(query="")


def test_query_request_whitespace_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(query="   ")


def test_query_request_extra_field_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(query="test", retrieval_route="graph_only")


def test_pipeline_state_defaults():
    state = QueryPipelineState()
    assert state.query == ""
    assert state.retrieval_route is None
    assert state.graph_results == []
    assert state.vector_results == []
    assert state.reranked_results == []
    assert state.final_context == ""
    assert state.answer == ""
    assert state.sources == []
    assert state.graph_path == []


def test_pipeline_state_with_route():
    state = QueryPipelineState(
        query="test",
        retrieval_route=RetrievalRoute.graph_and_vector,
    )
    assert state.retrieval_route == RetrievalRoute.graph_and_vector


def test_pipeline_state_invalid_route_rejected():
    with pytest.raises(ValidationError):
        QueryPipelineState(query="test", retrieval_route="invalid_route")


def test_pipeline_state_extra_field_rejected():
    with pytest.raises(ValidationError):
        QueryPipelineState(query="test", invented_field="value")


def test_retrieval_route_enum_is_closed():
    expected = {"graph_only", "vector_only", "graph+vector"}
    actual = {r.value for r in RetrievalRoute}
    assert actual == expected


def test_query_response_valid():
    resp = QueryResponse(
        answer="BFS was taught first because it introduces queues.",
        sources=[{"chunk_id": "lec_001_chunk_000001", "timestamp": "00:20:00"}],
        graph_path=["Queue", "BFS", "Dijkstra"],
    )
    assert resp.answer != ""
    assert len(resp.sources) == 1
    assert len(resp.graph_path) == 3
