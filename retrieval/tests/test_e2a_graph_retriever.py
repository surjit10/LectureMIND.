# retrieval/tests/test_e2a_graph_retriever.py
# Tests for E2a — Neo4j Graph Retriever.

import pytest
from unittest.mock import MagicMock

from retrieval.graph_retriever.neo4j_retriever import (
    _find_fuzzy_candidates,
    _score_fuzzy_entity_match,
    _stem_word,
    extract_entities,
    retrieve_graph,
)


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

    def test_extracts_lowercase_terms(self):
        """Natural-language queries contribute lowercase candidate terms."""
        entities = extract_entities("How is a relation connected to a table?")
        assert "relation" in entities
        assert "table" in entities

    def test_stopword_only_query_returns_empty(self):
        """A query consisting only of stopwords yields no entities."""
        assert extract_entities("what is the and of to in") == []


class TestFuzzyEntityFallback:

    def test_stem_word_inflections(self):
        """Plural/inflection normalization handles common English forms."""
        assert _stem_word("processes") == "process"
        assert _stem_word("classes") == "class"
        assert _stem_word("primitives") == "primitive"
        assert _stem_word("abstractions") == "abstraction"
        assert _stem_word("algorithms") == "algorithm"
        assert _stem_word("variables") == "variable"
        assert _stem_word("dijkstra") == "dijkstra"

    def test_fuzzy_candidates_inflection_match(self):
        """Plural query term matches singular multi-word entity."""
        entities = ["Process Abstraction", "Virtual Memory", "Page Table"]
        candidates = _find_fuzzy_candidates(
            "Why are Unix processes lightweight?",
            entities,
            threshold=0.70,
            limit=3,
        )
        assert "Process Abstraction" in candidates

    def test_fuzzy_candidates_typo_match(self):
        """Query with minor typo matches correct graph entity."""
        entities = ["Dijkstra", "Bellman-Ford", "Breadth-First Search"]
        candidates = _find_fuzzy_candidates(
            "Explain Dijsktra shortest path",
            entities,
            threshold=0.70,
            limit=3,
        )
        assert "Dijkstra" in candidates

    def test_fuzzy_candidates_unrelated_query_returns_empty(self):
        """Completely unrelated query never returns false positive graph entities."""
        entities = ["Process Abstraction", "Virtual Memory", "Page Table"]
        candidates = _find_fuzzy_candidates(
            "What is the capital of France?",
            entities,
            threshold=0.72,
            limit=3,
        )
        assert candidates == []

    def test_fuzzy_candidates_empty_input(self):
        """Empty query or empty entity list returns empty."""
        assert _find_fuzzy_candidates("", ["Process Abstraction"]) == []
        assert _find_fuzzy_candidates("Explain processes", []) == []


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

        results = retrieve_graph(
            "Why BFS before Dijkstra?", driver=mock_driver, lecture_id="lec_001",
        )

        assert len(results) == 1
        assert results[0]["start_name"] == "BFS"
        assert results[0]["related_name"] == "Dijkstra"

    def test_retrieve_graph_fallback_resolution_mock(self):
        """When exact and substring return empty, Stage 3 fallback resolves candidate."""
        mock_node_record = {"name": "Process Abstraction"}
        mock_traversal_record = {
            "start_id": "e_003", "start_name": "Process Abstraction", "start_type": "Concept",
            "related_id": "e_004", "related_name": "Address Space", "related_type": "Concept",
            "rel_types": ["EXPLAINS"], "hops": 1,
        }

        mock_session = MagicMock()
        # Call 1 (exact): []
        # Call 2 (substring): []
        # Call 3 (fetch names): [mock_node_record]
        # Call 4 (traversal on candidate): [mock_traversal_record]
        mock_session.run.side_effect = [
            [],
            [],
            [mock_node_record],
            [mock_traversal_record],
        ]
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        results = retrieve_graph(
            "Why are processes isolated?", driver=mock_driver, lecture_id="lec_001",
        )

        assert len(results) == 1
        assert results[0]["start_name"] == "Process Abstraction"
        assert results[0]["related_name"] == "Address Space"

    def test_stopword_query_returns_empty(self):
        """Query with no extractable entities returns empty."""
        results = retrieve_graph(
            "what is the of to", driver=MagicMock(), lecture_id="lec_001",
        )
        assert results == []

    def test_missing_lecture_id_raises(self):
        """Missing lecture_id must fail loudly, never traverse all lectures."""
        with pytest.raises(ValueError, match="lecture_id is required"):
            retrieve_graph("BFS", driver=MagicMock())
