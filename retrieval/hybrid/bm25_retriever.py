# retrieval/hybrid/bm25_retriever.py
# V2 Hybrid — Okapi BM25 Lexical Retriever + RRF Fusion.
#
# Adds a pure-Python BM25 index (no rank_bm25 dependency) built lazily from
# Qdrant payloads. The index is cached per lecture_id and built on first query.
#
# RRF (Reciprocal Rank Fusion) combines vector and BM25 results into a unified
# candidate pool before cross-encoder reranking.
#
# Feature-flagged — off by default (ENABLE_HYBRID_RETRIEVAL in LocalSettings).

import logging
import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from config import LocalSettings

logger = logging.getLogger(__name__)

# ── BM25 tuning constants ────────────────────────────────────────────
K1 = 1.5          # Term-frequency saturation.
B = 0.75          # Length normalization.

# RRF constant (standard value from the original paper).
RRF_K = 60

# Collection to search in Qdrant.
_COLLECTION_NAME = "lecturemind_chunks"

# Module-level BM25 index cache: {lecture_id: _BM25Index}.
_BM25_INDEXES: Dict[str, "_BM25Index"] = {}


# ── Tokeniser ─────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase, keep only alphanumeric tokens of length >= 2."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if len(t) >= 2]


# ── BM25 Index ────────────────────────────────────────────────────────

class _BM25Index:
    """Okapi BM25 index for a single lecture, built from Qdrant payloads."""

    def __init__(self, lecture_id: str, qdrant_client: Any):
        """
        Build the index by scrolling all Qdrant points for this lecture.

        Args:
            lecture_id: The runtime lecture ID.
            qdrant_client: An open QdrantClient instance (NOT closed here).
        """
        self.lecture_id = lecture_id
        self.doc_texts: List[Tuple[str, str, Dict[str, Any]]] = []
        # (chunk_id, full_text, payload_dict)

        # Scroll all chunks from Qdrant.
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        points = []
        offset = None
        while True:
            page, next_offset = qdrant_client.scroll(
                collection_name=_COLLECTION_NAME,
                scroll_filter=Filter(must=[
                    FieldCondition(key="lecture_id", match=MatchValue(value=lecture_id)),
                ]),
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            points.extend(page)
            if not page or next_offset is None:
                break
            offset = next_offset

        self.doc_terms: List[Counter] = []

        if not points:
            logger.warning("BM25: No Qdrant points found for lecture '%s'", lecture_id)
            self.doc_count = 0
            self.avgdl = 0.0
            self.idf: Dict[str, float] = {}
            return

        # Build text for each chunk: transcript + ocr_text (same as what
        # gets embedded for dense retrieval and what _build_passage_text uses).
        for point in points:
            payload = point.payload if hasattr(point, "payload") else point.get("payload", {})
            chunk_id = payload.get("chunk_id", "") or str(point.id)
            text = " ".join([
                payload.get("transcript") or "",
                payload.get("ocr_text") or "",
            ]).strip()
            tokens = _tokenize(text)
            self.doc_texts.append((chunk_id, text, dict(payload)))
            self.doc_terms.append(Counter(tokens))

        self.doc_count = len(self.doc_terms)

        # Average document length (in tokens).
        total_terms = sum(sum(t.values()) for t in self.doc_terms)
        self.avgdl = total_terms / self.doc_count if self.doc_count > 0 else 0.0

        # Compute IDF: log((N - df + 0.5) / (df + 0.5) + 1.0)
        doc_freq: Counter = Counter()
        for terms in self.doc_terms:
            for term in terms:
                doc_freq[term] += 1

        self.idf = {}
        for term, df in doc_freq.items():
            self.idf[term] = math.log(
                (self.doc_count - df + 0.5) / (df + 0.5) + 1.0,
            )

        logger.info(
            "BM25: Built index for '%s' — %d docs, avgdl=%.1f, vocab=%d",
            lecture_id, self.doc_count, self.avgdl, len(self.idf),
        )

    def score(self, query_tokens: List[str]) -> List[Tuple[int, float]]:
        """
        Score all documents for a tokenized query.

        Returns:
            List of (doc_idx, BM25_score) sorted descending by score.
        """
        if self.doc_count == 0:
            return []

        scores: List[Tuple[int, float]] = []

        for doc_idx in range(self.doc_count):
            doc_len = sum(self.doc_terms[doc_idx].values())
            if doc_len == 0:
                scores.append((doc_idx, 0.0))
                continue

            total = 0.0
            for term in query_tokens:
                idf = self.idf.get(term)
                if idf is None:
                    continue
                tf = self.doc_terms[doc_idx].get(term, 0)
                if tf == 0:
                    continue
                numerator = tf * (K1 + 1.0)
                denominator = tf + K1 * (1.0 - B + B * doc_len / self.avgdl)
                total += idf * numerator / denominator

            if total > 0.0:
                scores.append((doc_idx, total))

        scores.sort(key=lambda x: -x[1])
        return scores

    def clear(self) -> None:
        """Release memory for this index."""
        self.doc_terms.clear()
        self.idf.clear()
        self.doc_texts.clear()


# ── Public API ────────────────────────────────────────────────────────

def build_or_get_index(lecture_id: str) -> "_BM25Index":
    """
    Return the cached BM25 index for a lecture, building it on first call.

    The index is built by scrolling all Qdrant points for the lecture via
    a temporary client (opened and closed within this function).

    Args:
        lecture_id: The runtime lecture ID to query.

    Returns:
        _BM25Index instance.
    """
    if lecture_id in _BM25_INDEXES:
        return _BM25_INDEXES[lecture_id]

    from qdrant_client import QdrantClient
    settings = LocalSettings()
    client = QdrantClient(url=settings.QDRANT_URL)
    try:
        index = _BM25Index(lecture_id, client)
        _BM25_INDEXES[lecture_id] = index
    finally:
        client.close()

    return index


def clear_lecture_index(lecture_id: str) -> None:
    """Remove a lecture's BM25 index from the cache (e.g. on lecture unload)."""
    index = _BM25_INDEXES.pop(lecture_id, None)
    if index is not None:
        index.clear()
        logger.info("BM25: Cleared index for '%s'.", lecture_id)


def bm25_search(
    query: str,
    lecture_id: str,
    top_k: int = 15,
    qdrant_client: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Search a lecture's BM25 index and return top-k candidates.

    The index is built lazily on first call. Subsequent calls are
    instant (pre-built). Callers that already hold an open QdrantClient
    may pass it to avoid a redundant connect/close.

    Args:
        query: Raw query string (tokenized internally).
        lecture_id: The runtime lecture ID.
        top_k: Number of results to return.
        qdrant_client: Optional open QdrantClient (reused if provided;
                       otherwise a temporary client is created).

    Returns:
        List of dicts with chunk_id, score, bm25_score, payload.
        Empty list if the index has no documents.
    """
    if lecture_id not in _BM25_INDEXES:
        if qdrant_client is not None:
            index = _BM25Index(lecture_id, qdrant_client)
            _BM25_INDEXES[lecture_id] = index
        else:
            index = build_or_get_index(lecture_id)
    else:
        index = _BM25_INDEXES[lecture_id]

    if index.doc_count == 0:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        logger.warning("BM25: Query '%s' produced zero tokens.", query)
        return []

    scored = index.score(query_tokens)
    results: List[Dict[str, Any]] = []
    seen_ids: set = set()

    for doc_idx, bm25_score in scored:
        chunk_id, text, payload = index.doc_texts[doc_idx]
        if chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)

        results.append({
            "chunk_id": chunk_id,
            "score": bm25_score,
            "payload": payload,
            "bm25_score": bm25_score,
        })

        if len(results) >= top_k:
            break

    logger.info(
        "BM25: Searched '%s' — %d results from %d docs (top_k=%d).",
        lecture_id, len(results), index.doc_count, top_k,
    )
    return results


# ── RRF Fusion ────────────────────────────────────────────────────────

def rrf_fuse(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    top_k: int = 15,
    k: float = RRF_K,
) -> List[Dict[str, Any]]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank(leg))) for each unique chunk.
    Results are sorted by RRF score (desc), then by the original
    vector score as a tiebreaker.

    Args:
        vector_results: Output from Qdrant dense retrieval.
        bm25_results: Output from bm25_search().
        top_k: Maximum number of fused results to return.
        k: RRF constant (default 60 — standard from the original paper).

    Returns:
        Fused list, deduplicated by chunk_id, sorted by RRF score desc.
        Each entry has chunk_id, score (vector score for duplicates),
        payload, and rrf_score.
    """
    rrf_scores: Dict[str, float] = {}      # chunk_id → RRF score
    best_scores: Dict[str, float] = {}      # chunk_id → vector score
    best_payload: Dict[str, Dict] = {}      # chunk_id → payload

    # Process vector results.
    for rank, vr in enumerate(vector_results, start=1):
        cid = vr.get("chunk_id", "")
        if not cid:
            continue
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in best_scores:
            best_scores[cid] = float(vr.get("score", 0.0))
            best_payload[cid] = vr.get("payload", {})

    # Process BM25 results.
    for rank, br in enumerate(bm25_results, start=1):
        cid = br.get("chunk_id", "")
        if not cid:
            continue
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in best_scores:
            best_scores[cid] = float(br.get("score", 0.0))
            best_payload[cid] = br.get("payload", {})

    # Sort by RRF score desc, then by vector score desc.
    ranked = sorted(
        rrf_scores.items(),
        key=lambda x: (-x[1], -best_scores.get(x[0], 0.0)),
    )

    fused: List[Dict[str, Any]] = []
    for cid, rrf_score in ranked[:top_k]:
        fused.append({
            "chunk_id": cid,
            "score": best_scores.get(cid, 0.0),
            "payload": best_payload.get(cid, {}),
            "rrf_score": rrf_score,
        })

    logger.info(
        "RRF: Fused %d vector + %d BM25 → %d results (top_k=%d).",
        len(vector_results), len(bm25_results), len(fused), top_k,
    )
    return fused