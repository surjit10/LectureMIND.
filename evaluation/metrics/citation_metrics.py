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

def calculate_citation_completeness(final_context: str, sources: List[Dict[str, Any]]) -> float:
    """
    Measures if sources are well-represented.
    Basic implementation checks if sources list is non-empty when context is used.
    """
    if final_context and sources:
        return 1.0
    if not final_context and not sources:
        return 1.0
    return 0.0

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
