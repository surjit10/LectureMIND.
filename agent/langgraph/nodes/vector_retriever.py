# agent/langgraph/nodes/vector_retriever.py
# LangGraph node — wraps the Qdrant retriever.

import logging
from typing import Any, Dict, Optional

from retrieval.vector_retriever.qdrant_retriever import retrieve_vectors

logger = logging.getLogger(__name__)

_planner = None


def _get_planner():
    """Lazy-load planner singleton."""
    global _planner
    if _planner is None:
        from agent.dspy.planner import QueryPlanner
        _planner = QueryPlanner()
    return _planner


def vector_retriever_node(
    state: Dict[str, Any],
    qdrant_client: Optional[Any] = None,
    embedding_model: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    LangGraph node: populate state.vector_results from Qdrant.

    Only modifies vector_results. Does not touch other fields.

    V7: Uses plan_full() to determine top_k — lecture-wide queries
    retrieve more chunks (top_k=15) to enable timeline sampling; normal
    queries now retrieve a candidate pool of NORMAL_TOP_K (15) so the
    reranker can recover chunks cosine ranks 6-15.
    """
    query = state.get("query", "")
    lecture_id = state.get("lecture_id", "")

    # Determine top_k from the query plan (lecture-wide = more chunks).
    plan = state.get("query_plan")
    if plan is None:
        try:
            plan = _get_planner().plan_full(query)
        except Exception as exc:
            logger.warning("Vector retriever: plan_full failed, using default top_k: %s", exc)
            plan = None

    if plan is not None:
        top_k = getattr(plan, "top_k", 15)
        is_lecture_wide = getattr(plan, "is_lecture_wide", False)
    else:
        from agent.dspy.planner import NORMAL_TOP_K
        top_k = NORMAL_TOP_K
        is_lecture_wide = False

    if is_lecture_wide and lecture_id:
        from retrieval.vector_retriever.qdrant_retriever import retrieve_lecture_wide
        results = retrieve_lecture_wide(
            qdrant_client=qdrant_client,
            top_k=top_k,
            lecture_id=lecture_id,
        )
    else:
        results = retrieve_vectors(
            query,
            qdrant_client=qdrant_client,
            embedding_model=embedding_model,
            top_k=top_k,
            lecture_id=lecture_id if lecture_id else None,
        )
        
    return {"vector_results": results}
