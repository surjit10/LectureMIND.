# evaluation/metrics/citation_metrics.py
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def calculate_citation_coverage(expected_ids: List[str], sources: List[Dict[str, Any]]) -> float:
    """Measures if the generated sources cover all expected chunk IDs."""
    if not expected_ids:
        return 1.0
    if not sources:
        return 0.0
        
    cited_ids = {s.get("chunk_id") for s in sources if s.get("chunk_id")}
    hits = sum(1 for cid in expected_ids if cid in cited_ids)
    return hits / len(expected_ids)

def calculate_citation_completeness(
    final_context: str,
    sources: List[Dict[str, Any]],
    reranked_results: List[Dict[str, Any]],
    context_chunk_ids: Optional[List[str]] = None,
) -> float:
    """
    Fraction of cited sources that are actually backed by a chunk included in the LLM context.

    A citation is complete and grounded only if its chunk_id corresponds to a chunk
    that was actually included in the context prompt provided to the LLM.

    Resolution order for valid context chunk IDs:
      1. Explicit `context_chunk_ids` if provided.
      2. Chunks tagged with `in_context=True` in `reranked_results` (populated by rerank_service).
      3. Fallback: all candidate IDs in `reranked_results` if no `in_context` tags exist
         (for backward compatibility with legacy test fixtures).
    """
    if not sources:
        # No sources with non-empty context is a grounding failure;
        # no sources and no context is a refusal, which is acceptable.
        return 1.0 if not final_context else 0.0

    cited_ids = [s.get("chunk_id") for s in sources if s.get("chunk_id")]
    if not cited_ids:
        return 1.0 if not final_context else 0.0

    if context_chunk_ids is not None:
        valid_context_ids = set(context_chunk_ids)
    elif any("in_context" in r for r in reranked_results):
        valid_context_ids = set()
        for r in reranked_results:
            if r.get("in_context") is True:
                payload = r.get("payload", r)
                cid = payload.get("chunk_id", r.get("chunk_id", ""))
                if cid:
                    valid_context_ids.add(cid)
    else:
        valid_context_ids = set()
        for r in reranked_results:
            payload = r.get("payload", r)
            cid = payload.get("chunk_id", r.get("chunk_id", ""))
            if cid:
                valid_context_ids.add(cid)

    hits = sum(1 for cid in cited_ids if cid in valid_context_ids)
    return hits / len(cited_ids)

def get_citation_count(sources: List[Dict[str, Any]]) -> int:
    """Returns the total number of unique citations used."""
    return len({s.get("chunk_id") for s in sources if s.get("chunk_id")})

def get_chunk_coverage(sources: List[Dict[str, Any]], reranked_results: List[Dict[str, Any]]) -> float:
    """Percentage of the reranked chunks that actually ended up as cited sources."""
    if not reranked_results:
        return 1.0 if not sources else 0.0
    
    cited_ids = {s.get("chunk_id") for s in sources if s.get("chunk_id")}
    
    total_reranked = 0
    for r in reranked_results:
        payload = r.get("payload", r)
        cid = payload.get("chunk_id", r.get("chunk_id", ""))
        if cid:
            total_reranked += 1
            
    if total_reranked == 0:
        return 0.0
        
    return len(cited_ids) / total_reranked
