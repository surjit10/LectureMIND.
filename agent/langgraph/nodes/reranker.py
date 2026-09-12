# agent/langgraph/nodes/reranker.py
# LangGraph node — wraps the rerank service.

import logging
from typing import Any, Dict, Optional

from retrieval.reranker.rerank_service import rerank
from local.services.reranker_service import RerankerService

logger = logging.getLogger(__name__)

# Module-level planner instance (shared, stateless, lightweight).
_planner = None


def _get_planner():
    """Lazy-load planner singleton (avoids circular imports at module load)."""
    global _planner
    if _planner is None:
        from agent.dspy.planner import QueryPlanner
        _planner = QueryPlanner()
    return _planner


def reranker_node(
    state: Dict[str, Any],
    reranker_service: Optional[RerankerService] = None,
    reranker_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    LangGraph node: rerank combined results.

    Populates state.reranked_results and state.final_context only.

    V7: Calls plan_full() on the query to resolve is_lecture_wide and
    need_visual, then forwards them to ContextBuilder via rerank().
    No state schema changes required.
    """
    query = state.get("query", "")
    graph_results = state.get("graph_results", [])
    vector_results = state.get("vector_results", [])

    # Derive is_lecture_wide / need_visual from the query plan.
    plan = state.get("query_plan")
    if plan is None:
        try:
            plan = _get_planner().plan_full(query)
        except Exception as exc:
            logger.warning("Reranker node: plan_full failed, using defaults: %s", exc)
            plan = None

    if plan is not None:
        is_lecture_wide = getattr(plan, "is_lecture_wide", False)
        need_visual = getattr(plan, "need_visual", False)
        context_budget = getattr(plan, "context_budget", None)
    else:
        is_lecture_wide = False
        need_visual = False
        context_budget = None  # rerank() will fall back to MAX_CONTEXT_CHARS.

    lecture_id = state.get("lecture_id", None)

    reranked, final_context = rerank(
        query, graph_results, vector_results,
        reranker_service=reranker_service,
        reranker_model=reranker_model,
        is_lecture_wide=is_lecture_wide,
        need_visual=need_visual,
        char_budget=context_budget,
        lecture_id=lecture_id,
    )

    return {
        "reranked_results": reranked,
        "final_context": final_context,
    }
