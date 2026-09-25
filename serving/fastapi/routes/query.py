# serving/fastapi/routes/query.py
# POST /query endpoint — the core query pipeline route.

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from schemas.query import QueryRequest, QueryResponse
from agent.langgraph.workflow import QueryWorkflow

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level workflow reference — set by app.py on startup.
_workflow: Optional[QueryWorkflow] = None
_active_package: Optional[Dict[str, Any]] = None

# Module-level planner singleton (matches the pattern used by the nodes).
_planner = None


def _get_planner():
    """Lazy-load the stateless planner singleton."""
    global _planner
    if _planner is None:
        from agent.dspy.planner import QueryPlanner
        _planner = QueryPlanner()
    return _planner


def get_workflow() -> Optional[QueryWorkflow]:
    """Return the currently active workflow, or None."""
    return _workflow


def clear_workflow() -> None:
    """
    Safely destroy the current workflow and release owned resources.

    Closes the Neo4j driver and Qdrant client that were created by the
    upload route and passed to the workflow.  Shared infrastructure
    (Qdrant collection data, Neo4j data on disk) is NOT touched.
    """
    global _workflow, _active_package
    if _workflow is None:
        return
    # Close Neo4j driver if owned by this workflow.
    driver = getattr(_workflow, "_neo4j_driver", None)
    if driver is not None:
        try:
            driver.close()
            logger.info("Lifecycle: Neo4j driver closed.")
        except Exception as exc:
            logger.warning("Lifecycle: Error closing Neo4j driver: %s", exc)
        _workflow._neo4j_driver = None
    # Close Qdrant client if owned by this workflow.
    qdrant = getattr(_workflow, "_qdrant_client", None)
    if qdrant is not None:
        try:
            qdrant.close()
            logger.info("Lifecycle: Qdrant client closed.")
        except Exception as exc:
            logger.warning("Lifecycle: Error closing Qdrant client: %s", exc)
        _workflow._qdrant_client = None
    _workflow = None
    _active_package = None
    logger.info("Lifecycle: Previous workflow cleared.")


def set_workflow(workflow: QueryWorkflow) -> None:
    """Inject the workflow instance (called by app.py on startup)."""
    global _workflow
    _workflow = workflow


def activate_lecture(lecture_id: str) -> None:
    """
    Switch the active lecture context.

    Destroys the previous workflow's DB connections, creates fresh Neo4j and
    Qdrant connections for the new lecture, and updates the registry.

    The global reranker singleton is NOT touched. It was loaded once at
    application startup and remains in memory for the entire process lifetime.
    """
    global _workflow, _active_package
    if _active_package and _active_package.get("lecture_id") == lecture_id:
        return  # Already active

    logger.info("Lifecycle: Activating fresh context for lecture: %s", lecture_id)
    
    from local.storage.registry_provider import get_registry
    lecture = get_registry().get_lecture(lecture_id)
    if not lecture or lecture.get("status") != "READY":
        raise ValueError(f"Lecture {lecture_id} not found or not READY. Cannot activate.")

    # 1. Release previous lecture resources completely
    clear_workflow()
    
    # 2. Establish fresh DB connections
    from neo4j import GraphDatabase
    from qdrant_client import QdrantClient
    from config import local_settings
    from local.loaders.package_loader import load_package

    driver = GraphDatabase.driver(local_settings.NEO4J_URI)
    qdrant_client = QdrantClient(url=local_settings.QDRANT_URL)
    
    workflow = QueryWorkflow(neo4j_driver=driver, qdrant_client=qdrant_client)
    package = load_package(lecture_id)
    
    # 3. Set the new active context
    set_active_package(lecture_id, package)
    set_workflow(workflow)

    # 4. Persist to registry so it survives restarts
    get_registry().set_active_lecture(lecture_id)
    logger.info("Lifecycle: Successfully activated lecture: %s", lecture_id)


def set_active_package(lecture_id: str, package: Dict[str, Any]) -> None:
    """Register the loaded knowledge package with the query system."""
    global _active_package
    _active_package = package


def _build_debug_trace(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the Developer-Mode pipeline trace from existing workflow state.

    Pure pass-through of data the workflow already computed — no additional
    retrieval, no recomputation of embeddings/reranking. The planner intent
    is re-derived once via the lightweight heuristic plan_full() because
    state intentionally does not carry it (frozen schema).
    """
    route = state.get("retrieval_route")
    route_str = ""
    if route is not None:
        route_str = route.value if hasattr(route, "value") else str(route)

    plan_info: Dict[str, Any] = {}
    try:
        plan = _get_planner().plan_full(state.get("query", ""))
        plan_info = {
            "intent": plan.intent,
            "is_lecture_wide": plan.is_lecture_wide,
            "answer_style": plan.answer_style,
            "top_k": plan.top_k,
            "context_budget": plan.context_budget,
            "need_visual": plan.need_visual,
        }
    except Exception as exc:
        logger.warning("Debug trace: plan_full failed: %s", exc)

    def _round(value) -> float:
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return 0.0

    vector_results = []
    for r in state.get("vector_results", [])[:10]:
        vector_results.append({
            "chunk_id": r.get("chunk_id", ""),
            "score": _round(r.get("score", 0.0)),
        })

    graph_results = []
    for r in state.get("graph_results", [])[:10]:
        graph_results.append({
            "start": r.get("start_name", ""),
            "related": r.get("related_name", ""),
            "rel_types": r.get("rel_types", []),
            "hops": r.get("hops", 0),
        })

    reranked_results = []
    for r in state.get("reranked_results", [])[:10]:
        payload = r.get("payload", r)
        reranked_results.append({
            "chunk_id": payload.get("chunk_id", r.get("chunk_id", "")),
            "rerank_score": _round(r.get("rerank_score", 0.0)),
        })

    return {
        "retrieval_route": route_str or "",
        "plan": plan_info,
        "telemetry": {k: _round(v) for k, v in state.get("telemetry", {}).items()},
        "stage_counts": {
            "vector": len(state.get("vector_results", [])),
            "graph": len(state.get("graph_results", [])),
            "reranked": len(state.get("reranked_results", [])),
        },
        "vector_results": vector_results,
        "graph_results": graph_results,
        "reranked_results": reranked_results,
        "final_context_chars": len(state.get("final_context", "")),
        "graph_path": state.get("graph_path", []),
        "prerequisites": state.get("prerequisites", []),
    }


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    QueryPlanner → Single-pass Orchestrator → Neo4j/Qdrant → Reranker → Answer Generator

    Request: QueryRequest (query field only)
    Response: QueryResponse (answer, sources, graph_path, prerequisites)
    """
    try:
        if request.lecture_id:
            activate_lecture(request.lecture_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if _workflow is None:
        raise HTTPException(
            status_code=503,
            detail="Query engine not initialized. Load a knowledge package first.",
        )

    try:
        # Fallback to active package if the request doesn't provide it
        lecture_id = request.lecture_id or (_active_package.get("lecture_id") if _active_package else "")
        state = _workflow.run(request.query, lecture_id=lecture_id)

        # Socratic Prerequisite Back-Tracker: trace conceptual dependencies
        prerequisites = []
        driver = getattr(_workflow, "_neo4j_driver", None)
        if lecture_id and driver is not None:
            try:
                from retrieval.graph_retriever.neo4j_retriever import get_prerequisites_for_query
                prerequisites = get_prerequisites_for_query(
                    query=request.query,
                    lecture_id=lecture_id,
                    driver=driver,
                    graph_results=state.get("graph_results", []),
                )
            except Exception as exc:
                logger.warning("Prerequisite back-tracker lookup skipped: %s", exc)

        state["prerequisites"] = prerequisites

        return QueryResponse(
            answer=state.get("answer", ""),
            sources=state.get("sources", []),
            graph_path=state.get("graph_path", []),
            prerequisites=prerequisites,
            debug=_build_debug_trace(state),
        )
    except Exception as exc:
        logger.error("Query pipeline failed: %s", exc)
        return QueryResponse(
            answer=f"Query execution incomplete due to missing dependencies or models: {str(exc)}",
            sources=[],
            graph_path=[],
            prerequisites=[],
        )
