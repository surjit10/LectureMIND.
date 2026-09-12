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
    """Calculates the average cross-encoder score of the returned results.

    The reranker stores its score under ``rerank_score`` (see
    retrieval/reranker/rerank_service.py).  ``score`` is accepted as a
    fallback for legacy result dicts.
    """
    if not reranked_results:
        return 0.0
    scores = []
    for r in reranked_results:
        if "rerank_score" in r:
            try:
                scores.append(float(r["rerank_score"]))
            except (TypeError, ValueError):
                continue
        elif "score" in r:
            try:
                scores.append(float(r["score"]))
            except (TypeError, ValueError):
                continue
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


def top1_cross_encoder_score(reranked_results: List[Dict[str, Any]]) -> float:
    """Returns the cross-encoder score of the top-ranked candidate."""
    if not reranked_results:
        return 0.0
    first = reranked_results[0]
    raw = first.get("rerank_score", first.get("score"))
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def top_k_cross_encoder_score(reranked_results: List[Dict[str, Any]], k: int = 3) -> float:
    """Returns the average cross-encoder score of the top-k ranked candidates."""
    if not reranked_results:
        return 0.0
    scores = []
    for r in reranked_results[:k]:
        raw = r.get("rerank_score", r.get("score"))
        if raw is not None:
            try:
                scores.append(float(raw))
            except (TypeError, ValueError):
                continue
    return sum(scores) / len(scores) if scores else 0.0

def score_distribution(reranked_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Returns basic stats about the score distribution (uses rerank_score)."""
    if not reranked_results:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    scores = []
    for r in reranked_results:
        raw = r.get("rerank_score", r.get("score"))
        if raw is None:
            continue
        try:
            scores.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not scores:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": sum(scores) / len(scores)
    }
