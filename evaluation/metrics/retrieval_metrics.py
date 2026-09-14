# evaluation/metrics/retrieval_metrics.py
import logging
from typing import List, Optional
import math

logger = logging.getLogger(__name__)


def _dedupe_preserve_order(ids: List[str]) -> List[str]:
    """Remove duplicate chunk IDs while preserving ranking order.

    Retrieval metrics assume a ranked list of *unique* documents. Duplicate
    entries (e.g. the same chunk returned by both dense and lexical legs)
    would otherwise be scored multiple times — inflating Precision/NDCG and
    corrupting reciprocal-rank positions.
    """
    seen = set()
    out = []
    for cid in ids:
        if cid and cid not in seen:
            seen.add(cid)
            out.append(cid)
    return out


def calculate_precision_at_k(expected_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
    """Standard Precision@k: relevant items among the first k, divided by k.

    The denominator is the constant ``k`` (standard IR definition), not the
    number of actually retrieved documents. Deduplicates the retrieved list
    first so a duplicated chunk cannot be counted as multiple hits.
    """
    if not retrieved_ids or not expected_ids or k <= 0:
        return 0.0
    top_k = _dedupe_preserve_order(retrieved_ids)[:k]
    hits = sum(1 for cid in top_k if cid in expected_ids)
    return hits / k

def calculate_r_precision(expected_ids: List[str], retrieved_ids: List[str]) -> float:
    """Calculates R-Precision: precision at rank R, where R = len(expected_ids).

    Standard definition:
        R = len(expected_ids)
        top_R = retrieved_ids[:R]
        R-Precision = len(set(top_R) & set(expected_ids)) / R

    Returns 0.0 if expected_ids or retrieved_ids is empty, or R <= 0.
    """
    if not expected_ids or not retrieved_ids:
        return 0.0
    R = len(expected_ids)
    if R <= 0:
        return 0.0
    top_r = _dedupe_preserve_order(retrieved_ids)[:R]
    hits = sum(1 for cid in top_r if cid in expected_ids)
    return hits / R


def calculate_normalized_precision_at_k(expected_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
    """[DEPRECATED] Precision normalized by min(k, len(expected_ids)).

    Deprecated: When len(expected_ids) <= k, this formula reduces identically to
    Recall@k. Use calculate_r_precision or standard calculate_precision_at_k instead.
    """
    if not retrieved_ids or not expected_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for cid in top_k if cid in expected_ids)
    max_possible_hits = min(len(top_k), len(expected_ids))
    return hits / max_possible_hits if max_possible_hits > 0 else 0.0

def calculate_recall_at_k(expected_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
    if not expected_ids:
        return 1.0  # vacuous truth; dataset loader rejects empty expected sets
    if not retrieved_ids:
        return 0.0
    top_k = _dedupe_preserve_order(retrieved_ids)[:k]
    hits = sum(1 for cid in top_k if cid in expected_ids)
    return hits / len(expected_ids)

def calculate_hit_at_k(expected_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
    if not expected_ids:
        return 1.0  # vacuous truth; dataset loader rejects empty expected sets
    if not retrieved_ids:
        return 0.0
    top_k = _dedupe_preserve_order(retrieved_ids)[:k]
    return 1.0 if any(cid in expected_ids for cid in top_k) else 0.0

def calculate_mrr(expected_ids: List[str], retrieved_ids: List[str], k: Optional[int] = None) -> float:
    """Reciprocal rank of the first relevant retrieved chunk.

    Args:
        k: When given, ranks beyond ``k`` contribute 0 (strict MRR@k — a
           relevant chunk at rank 7 scores 0.0 for MRR@5, matching Hit@5
           semantics). When ``None`` (default), the full ranked list is
           considered (unbounded MRR).
    """
    if not expected_ids or not retrieved_ids:
        return 0.0
    ranked = _dedupe_preserve_order(retrieved_ids)
    if k is not None:
        if k <= 0:
            return 0.0
        ranked = ranked[:k]
    for idx, cid in enumerate(ranked):
        if cid in expected_ids:
            return 1.0 / (idx + 1)
    return 0.0

def calculate_ndcg(expected_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
    if not expected_ids:
        return 1.0  # vacuous truth; dataset loader rejects empty expected sets
    if not retrieved_ids:
        return 0.0
    top_k = _dedupe_preserve_order(retrieved_ids)[:k]
    dcg = 0.0
    for i, cid in enumerate(top_k):
        if cid in expected_ids:
            dcg += 1.0 / math.log2(i + 2)
            
    idcg = 0.0
    ideal_hits = min(len(set(expected_ids)), k)
    for i in range(ideal_hits):
        idcg += 1.0 / math.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0.0
