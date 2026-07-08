# schemas/pipeline_state.py
# QueryPipelineState — INTERNAL LangGraph state schema.
#
# This is the internal routing state used inside the LangGraph agent.
# It is NOT the API request schema. See query.py for QueryRequest.
#
# No stage may add fields to this schema without updating the specification.

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict
from .enums import RetrievalRoute


class QueryPipelineState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    retrieval_route: Optional[RetrievalRoute] = None
    graph_results: List[Dict[str, Any]] = []
    vector_results: List[Dict[str, Any]] = []
    reranked_results: List[Dict[str, Any]] = []
    final_context: str = ""
    answer: str = ""
    sources: List[Dict[str, Any]] = []
    graph_path: List[str] = []
    llm_metadata: Dict[str, Any] = {}
    telemetry: Dict[str, float] = {}
