# agent/tests/test_e4_answer_generator.py
# Tests for E4 — Answer Generator.

import pytest
from unittest.mock import MagicMock

from agent.langgraph.nodes.answer_generator import (
    answer_generator_node, _build_sources, _build_graph_path, _format_timestamp,
)


class TestAnswerGenerator:

    def test_empty_context_returns_disclaimer(self):
        """No context = 'Insufficient evidence' answer."""
        state = {
            "query": "test", "final_context": "",
            "reranked_results": [], "graph_results": [],
        }
        result = answer_generator_node(state, ollama_client=MagicMock())
        assert result["answer"] == "Insufficient evidence found in lecture."

    def test_answer_with_mock_ollama(self):
        """Ollama mock returns a grounded answer."""
        mock_client = MagicMock()
        mock_client.chat.return_value = {
            "message": {"content": "BFS explores level by level."}
        }
        state = {
            "query": "How does BFS work?",
            "final_context": "BFS uses a queue to explore nodes level by level.",
            "reranked_results": [
                {"chunk_id": "c_001", "payload": {
                    "chunk_id": "c_001", "timestamp": 1280.0, "segment_id": "seg_5"}}
            ],
            "graph_results": [],
        }

        result = answer_generator_node(state, ollama_client=mock_client)
        assert result["answer"] == "BFS explores level by level."
        assert len(result["sources"]) == 1
        assert result["sources"][0]["chunk_id"] == "c_001"

    def test_sources_from_reranked_only(self):
        """Sources must come from reranked_results only."""
        reranked = [
            {"chunk_id": "c_001", "payload": {"chunk_id": "c_001", "timestamp": 60.0, "segment_id": "s_1"}},
            {"chunk_id": "c_002", "payload": {"chunk_id": "c_002", "timestamp": 120.0, "segment_id": "s_2"}},
        ]
        sources = _build_sources(reranked)
        assert len(sources) == 2
        assert sources[0]["chunk_id"] == "c_001"
        assert sources[1]["timestamp"] == "00:02:00"

    def test_graph_path_from_graph_results(self):
        """graph_path populated only from graph results."""
        graph = [
            {"start_name": "Queue", "related_name": "BFS"},
            {"start_name": "BFS", "related_name": "Dijkstra"},
        ]
        path = _build_graph_path(graph)
        assert "Queue" in path
        assert "BFS" in path
        assert "Dijkstra" in path

    def test_graph_path_empty_without_graph(self):
        """No graph results = empty graph_path."""
        assert _build_graph_path([]) == []

    def test_timestamp_formatting(self):
        assert _format_timestamp(0) == "00:00:00"
        assert _format_timestamp(61) == "00:01:01"
        assert _format_timestamp(3661) == "01:01:01"
