# retrieval/tests/test_e2a_graph_retriever.py
# Tests for E2a — Neo4j Graph Retriever.

import pytest
from unittest.mock import MagicMock

from retrieval.graph_retriever.neo4j_retriever import extract_entities, retrieve_graph


class TestEntityExtraction:

    def test_extracts_capitalized_terms(self):
        entities = extract_entities("Why was BFS taught before Dijkstra?")
        assert "BFS" in entities
        assert "Dijkstra" in entities

    def test_extracts_acronyms(self):
        entities = extract_entities("DFS and BFS comparison")
        assert "DFS" in entities
        assert "BFS" in entities

    def test_empty_query(self):
        assert extract_entities("") == []

    def test_no_entities(self):
        assert extract_entities("hello world foo bar") == []


class TestGraphRetriever:

    def test_retrieve_graph_with_mock_driver(self):
        """Mock Neo4j driver returns graph results."""
        mock_record = {
            "start_id": "e_001", "start_name": "BFS", "start_type": "Algorithm",
            "related_id": "e_002", "related_name": "Dijkstra", "related_type": "Algorithm",
            "rel_types": ["PREREQUISITE_OF"], "hops": 1,
        }
        mock_session = MagicMock()
        mock_session.run.return_value = [mock_record]
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        results = retrieve_graph("Why BFS before Dijkstra?", driver=mock_driver)

        assert len(results) == 1
        assert results[0]["start_name"] == "BFS"
        assert results[0]["related_name"] == "Dijkstra"

    def test_no_entities_returns_empty(self):
        """Query with no extractable entities returns empty."""
        results = retrieve_graph("hello world", driver=MagicMock())
        assert results == []
