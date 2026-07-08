# agent/langgraph/workflow.py
# E2 — LangGraph Single-Pass Workflow.
#
# Orchestrates: Planner → Conditional Routing → Retrieval → Reranker → Answer
#
# No memory. No tool calling. No autonomous loops. Single-pass only.
# State matches the frozen QueryPipelineState exactly.

import logging
import time
from typing import Any, Dict, Optional

from schemas.enums import RetrievalRoute
from agent.dspy.planner import QueryPlanner

logger = logging.getLogger(__name__)


class QueryWorkflow:
    """
    Single-pass query workflow.

    Flow:
        1. Planner classifies route
        2. Conditional routing to Graph / Vector / Both
        3. Reranker fuses and reranks
        4. Answer Generator produces grounded answer

    In production, this would be a LangGraph StateGraph.
    The node functions are structured to be drop-in compatible
    with langgraph.graph.StateGraph when the dependency is available.
    """

    def __init__(
        self,
        planner: Optional[QueryPlanner] = None,
        neo4j_driver: Optional[Any] = None,
        qdrant_client: Optional[Any] = None,
        embedding_model: Optional[Any] = None,
        reranker_service: Optional[Any] = None,
        ollama_client: Optional[Any] = None,
    ):
        self._planner = planner or QueryPlanner()
        self._neo4j_driver = neo4j_driver
        self._qdrant_client = qdrant_client
        self._embedding_model = embedding_model
        self._reranker_service = reranker_service
        self._ollama_client = ollama_client

    def run(self, query: str, lecture_id: str = "") -> Dict[str, Any]:
        """
        Execute the full single-pass query pipeline.

        Args:
            query: User query string.
            lecture_id: The ID of the active lecture.

        Returns:
            Final state dict matching QueryPipelineState.
        """
        # Initialize state.
        state = {
            "query": query,
            "lecture_id": lecture_id,
            "retrieval_route": None,
            "graph_results": [],
            "vector_results": [],
            "reranked_results": [],
            "final_context": "",
            "answer": "",
            "sources": [],
            "graph_path": [],
            "telemetry": {},
            "llm_metadata": {},
        }

        t_total_start = time.perf_counter()

        # Step 1: Plan route.
        t_plan_start = time.perf_counter()
        route = self._planner.plan(query)
        state["retrieval_route"] = route
        state["telemetry"]["planner_latency"] = time.perf_counter() - t_plan_start
        logger.info("Workflow: Route = %s", route.value)

        # Step 2: Conditional retrieval.
        t_ret_start = time.perf_counter()
        if route in (RetrievalRoute.graph_only, RetrievalRoute.graph_and_vector):
            from agent.langgraph.nodes.graph_retriever import graph_retriever_node
            graph_update = graph_retriever_node(state, driver=self._neo4j_driver)
            state.update(graph_update)

        if route in (RetrievalRoute.vector_only, RetrievalRoute.graph_and_vector):
            from agent.langgraph.nodes.vector_retriever import vector_retriever_node
            vector_update = vector_retriever_node(
                state,
                qdrant_client=self._qdrant_client,
                embedding_model=self._embedding_model,
            )
            state.update(vector_update)
            
        state["telemetry"]["retrieval_latency"] = time.perf_counter() - t_ret_start

        # Step 3: Rerank.
        t_rerank_start = time.perf_counter()
        from agent.langgraph.nodes.reranker import reranker_node
        rerank_update = reranker_node(
            state, reranker_service=self._reranker_service,
        )
        state.update(rerank_update)
        state["telemetry"]["reranker_latency"] = time.perf_counter() - t_rerank_start

        # Step 4: Generate answer.
        t_gen_start = time.perf_counter()
        from agent.langgraph.nodes.answer_generator import answer_generator_node
        answer_update = answer_generator_node(
            state, ollama_client=self._ollama_client,
        )
        state.update(answer_update)
        state["telemetry"]["generation_latency"] = time.perf_counter() - t_gen_start
        
        state["telemetry"]["total_latency"] = time.perf_counter() - t_total_start

        logger.info("Workflow: Complete. Answer length = %d", len(state["answer"]))
        return state
