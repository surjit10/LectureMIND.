# agent/tests/test_e2_workflow.py
# Tests for E2 — LangGraph Workflow (end-to-end with mocks).

import pytest
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock

from schemas.enums import RetrievalRoute
from agent.dspy.planner import QueryPlanner
from agent.langgraph.workflow import QueryWorkflow
from local.services.reranker_service import RerankerService


class TestWorkflow:

    def _make_workflow(self, route=RetrievalRoute.vector_only):
        """Create a workflow with all services mocked."""
        # Mock planner.
        planner = QueryPlanner()

        # Mock Neo4j driver.
        mock_session = MagicMock()
        mock_session.run.return_value = [
            {"start_id": "e_001", "start_name": "BFS", "start_type": "Algorithm",
             "related_id": "e_002", "related_name": "Dijkstra", "related_type": "Algorithm",
             "rel_types": ["PREREQUISITE_OF"], "hops": 1},
        ]
        mock_driver = MagicMock()
        mock_driver.session.return_value.__enter__ = MagicMock(return_value=mock_session)
        mock_driver.session.return_value.__exit__ = MagicMock(return_value=False)

        # Mock Qdrant.
        mock_hit = SimpleNamespace(
            payload={"chunk_id": "c_001", "transcript": "BFS uses queue",
                     "visual_context": "diagram", "ocr_text": "O(V+E)",
                     "segment_id": "s_001", "timestamp": 10.0, "lecture_id": "lec_001"},
            score=0.92,
        )
        mock_qdrant = MagicMock()
        mock_qdrant.search.return_value = [mock_hit]

        # Mock embedding model.
        mock_embed = MagicMock()
        mock_embed.encode.return_value = np.random.randn(1, 1024).astype(np.float32)

        # Mock reranker.
        mock_reranker_model = MagicMock()
        mock_reranker_model.predict.return_value = np.array([0.95])
        reranker_service = RerankerService(mock_reranker_model)

        # Mock Ollama.
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {
            "message": {"content": "BFS explores level by level using a queue."}
        }

        return QueryWorkflow(
            planner=planner,
            neo4j_driver=mock_driver,
            qdrant_client=mock_qdrant,
            embedding_model=mock_embed,
            reranker_service=reranker_service,
            ollama_client=mock_ollama,
        )

    def test_vector_only_workflow(self):
        """Vector-only query produces answer with sources."""
        workflow = self._make_workflow()
        state = workflow.run("Summarize lecture section on BFS")

        assert state["retrieval_route"] == RetrievalRoute.vector_only
        assert state["answer"] != ""
        assert len(state["sources"]) >= 0
        assert state["graph_results"] == []  # No graph retrieval.

    def test_graph_only_workflow(self):
        """Graph-only query triggers graph retriever."""
        workflow = self._make_workflow()
        state = workflow.run("What is prerequisite of Dijkstra?")

        assert state["retrieval_route"] == RetrievalRoute.graph_only
        assert len(state["graph_results"]) > 0
        assert state["vector_results"] == []  # No vector retrieval.

    def test_state_has_all_fields(self):
        """Final state contains all QueryPipelineState fields."""
        workflow = self._make_workflow()
        state = workflow.run("Tell me about BFS")

        required_keys = {
            "query", "retrieval_route", "graph_results", "vector_results",
            "reranked_results", "final_context", "answer", "sources", "graph_path",
        }
        assert required_keys.issubset(set(state.keys()))

    def test_lecture_wide_retrieval(self):
        """Lecture-wide query triggers vector_only and returns answer."""
        workflow = self._make_workflow()
        state = workflow.run("Explain the lecture")

        assert state["retrieval_route"] == RetrievalRoute.vector_only
        assert state["answer"] != ""

