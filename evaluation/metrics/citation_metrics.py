# evaluation/metrics/citation_metrics.py
import logging
from typing import List, Dict, Any

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
) -> float:
    """
    Fraction of cited sources that are actually backed by a retrieved chunk.

    A citation is "complete" only if its chunk_id also appears in the
    reranked results that produced the context.  This catches hallucinated
    or stale citations that cite IDs the retriever never returned.
    """
    if not sources:
        # No sources with non-empty context is a grounding failure;
        # no sources and no context is a refusal, which is acceptable.
        return 1.0 if not final_context else 0.0

    reranked_ids = set()
    for r in reranked_results:
        payload = r.get("payload", r)
        cid = payload.get("chunk_id", r.get("chunk_id", ""))
        if cid:
            reranked_ids.add(cid)

    cited_ids = [s.get("chunk_id") for s in sources if s.get("chunk_id")]
    if not cited_ids:
        return 1.0 if not final_context else 0.0

    hits = sum(1 for cid in cited_ids if cid in reranked_ids)
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
