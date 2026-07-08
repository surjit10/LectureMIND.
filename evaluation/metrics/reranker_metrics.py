# evaluation/metrics/reranker_metrics.py
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def evaluate_ranking_quality(expected_ids: List[str], reranked_results: List[Dict[str, Any]]) -> float:
    """Evaluates if the expected chunk IDs were successfully pushed to the top by the reranker."""
    if not expected_ids or not reranked_results:
        return 0.0
    
    reranked_ids = []
    for r in reranked_results:
        payload = r.get("payload", r)
        cid = payload.get("chunk_id", r.get("chunk_id", ""))
        if cid:
            reranked_ids.append(cid)
            
    for idx, cid in enumerate(reranked_ids):
        if cid in expected_ids:
            return 1.0 / (idx + 1)
    return 0.0

def average_cross_encoder_score(reranked_results: List[Dict[str, Any]]) -> float:
    """Calculates the average cross-encoder score of the returned results."""
    if not reranked_results:
        return 0.0
    scores = [float(r.get("score", 0.0)) for r in reranked_results if "score" in r]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def score_distribution(reranked_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Returns basic stats about the score distribution."""
    if not reranked_results:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    scores = [float(r.get("score", 0.0)) for r in reranked_results if "score" in r]
    if not scores:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores)
    }
