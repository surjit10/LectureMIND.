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


@router.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Execute the full query pipeline.

    DSPy Planner → LangGraph → Neo4j/Qdrant → Reranker → Answer Generator

    Request: QueryRequest (query field only)
    Response: QueryResponse (answer, sources, graph_path)
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
        return QueryResponse(
            answer=state.get("answer", ""),
            sources=state.get("sources", []),
            graph_path=state.get("graph_path", []),
        )
    except Exception as exc:
        logger.error("Query pipeline failed: %s", exc)
        return QueryResponse(
            answer=f"Query execution incomplete due to missing dependencies or models: {str(exc)}",
            sources=[],
            graph_path=[]
        )
