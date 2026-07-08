# schemas/query.py
# QueryRequest — external API schema consumed by POST /query.
# QueryResponse — API response schema from POST /query.
#
# QueryRequest is the inbound payload (only `query` field).
# QueryPipelineState is the internal LangGraph routing schema — see pipeline_state.py.
# These two are intentionally separate per the specification.

from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, field_validator


class QueryRequest(BaseModel):
    """External API request schema for POST /query."""
    model_config = ConfigDict(extra="forbid")

    query: str
    lecture_id: str = ""

    @field_validator("query")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty.")
        return v


class QueryResponse(BaseModel):
    """External API response schema from POST /query."""
    model_config = ConfigDict(extra="forbid")

    answer: str
    sources: List[Dict[str, Any]] = []
    graph_path: List[str] = []
