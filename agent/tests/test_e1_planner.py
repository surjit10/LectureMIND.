# agent/tests/test_e1_planner.py
# Tests for E1 — DSPy Route Planner.

import pytest
from schemas.enums import RetrievalRoute
from agent.dspy.planner import QueryPlanner


class TestQueryPlanner:

    def test_graph_query_prerequisite(self):
        """'prerequisite' keyword routes to graph_only."""
        planner = QueryPlanner()
        assert planner.plan("What is prerequisite of Dynamic Programming?") == RetrievalRoute.graph_only

    def test_graph_query_before(self):
        """'taught before' routes to graph_only."""
        planner = QueryPlanner()
        assert planner.plan("Why was BFS taught before Dijkstra?") == RetrievalRoute.graph_only

    def test_graph_query_related(self):
        """'related' routes to graph_only."""
        planner = QueryPlanner()
        assert planner.plan("How is Queue related to BFS?") == RetrievalRoute.graph_only

    def test_vector_query_summarize(self):
        """'summarize' routes to vector_only."""
        planner = QueryPlanner()
        assert planner.plan("Summarize lecture section on trees.") == RetrievalRoute.vector_only

    def test_vector_query_what_was_said(self):
        """'what was said' routes to vector_only."""
        planner = QueryPlanner()
        assert planner.plan("What was said about memory allocation?") == RetrievalRoute.vector_only

    def test_mixed_query(self):
        """Graph + vector keywords route to graph+vector."""
        planner = QueryPlanner()
        result = planner.plan("Why BFS before Dijkstra and summarize where it was explained?")
        assert result == RetrievalRoute.graph_and_vector

    def test_empty_query_defaults_to_vector(self):
        """Empty query defaults to vector_only."""
        planner = QueryPlanner()
        assert planner.plan("") == RetrievalRoute.vector_only

    def test_unknown_query_defaults_to_vector(self):
        """Unknown query without keywords defaults to vector_only."""
        planner = QueryPlanner()
        assert planner.plan("hello world") == RetrievalRoute.vector_only

    def test_dspy_module_injection(self):
        """Custom DSPy module is used when injected."""
        from unittest.mock import MagicMock
        from types import SimpleNamespace

        mock_module = MagicMock(return_value=SimpleNamespace(route="graph_only"))
        planner = QueryPlanner(dspy_module=mock_module)
        result = planner.plan("test query")

        assert result == RetrievalRoute.graph_only
        mock_module.assert_called_once_with(query="test query")

    def test_dspy_module_fallback(self):
        """Broken DSPy module falls back to heuristic."""
        from unittest.mock import MagicMock

        mock_module = MagicMock(side_effect=RuntimeError("fail"))
        planner = QueryPlanner(dspy_module=mock_module)
        result = planner.plan("What is prerequisite of X?")

        assert result == RetrievalRoute.graph_only

    def test_lecture_summary_queries(self):
        """Lecture-wide queries trigger is_lecture_wide=True."""
        planner = QueryPlanner()
        queries = [
            "Explain the lecture",
            "Explain this lecture",
            "What is this lecture about?",
            "What is this lecture on",
            "What is explained in this lecture?",
            "What is explain in this lecture",
            "What explained in this lecture",
            "What's explained in this lecture",
            "What is this lecture explaining",
            "What does this lecture explain",
            "Explain what is in this lecture",
            "What does this lecture cover",
            "What is covered in the lecture",
            "What concepts are taught",
            "What topics are covered",
            "Summarize the lecture",
            "Give summary",
            "Lecture overview",
            "Lecture summary",
            "Key concepts",
            "Key takeaways",
            "Main ideas",
            "Topics covered",
            "What did we learn?",
            "Explain today's lecture",
            "Overview of today's lecture"
        ]
        for q in queries:
            plan = planner.plan_full(q)
            assert plan.is_lecture_wide is True, f"Failed for query: {q}"
            assert plan.top_k == 15, f"Failed for query: {q}"
            assert plan.context_budget == 6000, f"Failed for query: {q}"

    def test_factual_qa_queries(self):
        """Factual queries do not trigger is_lecture_wide."""
        planner = QueryPlanner()
        queries = [
            "What is normalization?",
            "Explain relational algebra.",
            "Define BCNF.",
            "What is a primary key?",
            "Explain Softmax.",
            "What is PageRank?"
        ]
        for q in queries:
            plan = planner.plan_full(q)
            assert plan.is_lecture_wide is False, f"Failed for query: {q}"
            assert plan.top_k == 15, f"Failed for query: {q}"
            assert plan.context_budget == 4000, f"Failed for query: {q}"

    def test_specific_course_facts_are_not_lecture_wide(self):
        """Grading/syllabus questions are point facts: semantic retrieval, not timeline sampling."""
        planner = QueryPlanner()
        for q in [
            "What is the grading breakdown for CS162?",
            "What is the syllabus of CS162 according to the lecture?",
            "What did the professor say about the course grading policy?",
        ]:
            assert planner.plan_full(q).is_lecture_wide is False, q

    def test_topic_scoped_summaries_use_semantic_retrieval(self):
        """A summary/notes/quiz request naming a topic keeps its intent but is not lecture-wide."""
        planner = QueryPlanner()
        for q, intent in [
            ("Give an overview of virtual memory", "summary"),
            ("Summarize the section on paging", "summary"),
            ("Quiz me on paging", "quiz"),
            ("Generate notes on deadlock", "notes"),
        ]:
            plan = planner.plan_full(q)
            assert plan.intent == intent, q
            assert plan.is_lecture_wide is False, q
            assert plan.context_budget == 4000, q

    def test_scope_only_requests_stay_lecture_wide(self):
        planner = QueryPlanner()
        for q in ["Summarize lecture 3", "Quiz me", "Generate study notes",
                  "Give me a quick recap of everything we covered so far"]:
            assert planner.plan_full(q).is_lecture_wide is True, q

    def test_ordinary_words_do_not_trigger_quiz_or_notes(self):
        planner = QueryPlanner()
        for q in ["I have a question about deadlock", "What is the best practice for locking?"]:
            plan = planner.plan_full(q)
            assert plan.intent not in ("quiz", "notes"), q
            assert plan.is_lecture_wide is False, q

    def test_relate_and_connects_route_to_graph(self):
        planner = QueryPlanner()
        for q in ["How does paging relate to virtual memory?",
                  "Explain how the kernel connects to hardware",
                  "What are the prerequisites of Process Abstraction?",
                  "What relations exist between threads and processes?",
                  "What are the relationships between kernel and memory?"]:
            assert planner.plan(q) in (RetrievalRoute.graph_only, RetrievalRoute.graph_and_vector), q

    def test_fully_connected_layer_is_not_a_graph_query(self):
        planner = QueryPlanner()
        for q in ["Why are fully connected layers stacked with nonlinearities?",
                  "How do PyTorch fully-connected layers work?",
                  "What is the relational model?"]:
            assert planner.plan(q) == RetrievalRoute.vector_only, q
