# schemas/triplet.py
# RerankerTriplet produced by Stage B1 (Llama-3.1-8B-Instruct hard-negative mining).
# Used to fine-tune the cross-encoder reranker in Stage B2.
#
# {
#   "query": "How BFS works",
#   "positive": "BFS segment text",
#   "negative": "DFS segment text"
# }

from pydantic import BaseModel, ConfigDict, field_validator


class RerankerTriplet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    positive: str
    negative: str

    @field_validator("query", "positive", "negative")
    @classmethod
    def must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query, positive, and negative must not be empty.")
        return v
