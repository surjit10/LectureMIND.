# agent/langgraph/state.py
# LangGraph state definition — uses the FROZEN QueryPipelineState.
#
# This module re-exports the Chunk 1 schema as a TypedDict for LangGraph compatibility.
# No fields added. No fields removed.

from typing import Any, Dict, List, Optional, TypedDict

from schemas.enums import RetrievalRoute


class GraphState(TypedDict, total=False):
    """
    LangGraph-compatible state matching QueryPipelineState exactly.

    Fields are identical to schemas.pipeline_state.QueryPipelineState.
    TypedDict is used because LangGraph operates on dicts, not Pydantic models.
    """
    query: str
    retrieval_route: Optional[RetrievalRoute]
    graph_results: List[Dict[str, Any]]
    vector_results: List[Dict[str, Any]]
    reranked_results: List[Dict[str, Any]]
    final_context: str
    answer: str
    sources: List[Dict[str, Any]]
    graph_path: List[str]
    llm_metadata: Dict[str, Any]
    telemetry: Dict[str, float]
