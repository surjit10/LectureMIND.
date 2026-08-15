# retrieval/vector_retriever/qdrant_retriever.py
# E2b — Qdrant Vector Retriever.
#
# Embeds the query with bge-large-en-v1.5 (1024-dim),
# searches Qdrant for top-K similar chunks.
#
# Returns chunk_id, score, and payload. Never returns raw vectors.
# Populates state.vector_results only.

import logging
from typing import Any, Dict, List, Optional

from config import LocalSettings, SharedSettings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "lecturemind_chunks"
DEFAULT_TOP_K = 5
_GLOBAL_EMBEDDING_MODEL = None

# Candidate-quality filter (applied at query time only — stored data and
# payloads are never modified). Whisper-segment chunks like "OK?" / "Okay."
# are pure transcript filler that steal retrieval slots (measured: 6.8% of
# top-5 slots across the 76-question set, ~10% of the corpus). A candidate
# is filtered ONLY when its transcript is shorter than MIN_TRANSCRIPT_CHARS
# AND it carries no OCR or visual content — multimodal evidence and
# legitimate short facts (e.g. "The class has a limit of 428." = 29 chars)
# are never removed.
MIN_TRANSCRIPT_CHARS = 15
# Fetch this many extra candidates so filtering filler still yields the
# requested top_k valid results without shrinking the returned pool.
RETRIEVAL_FETCH_SLACK = 15


def _is_transcript_filler(payload: Dict[str, Any]) -> bool:
    """True when a candidate is transcript-only filler ("OK?", "Okay.").

    Only short transcript chunks with no OCR and no visual content qualify.
    Chunks whose evidence comes from ocr_text / visual_context (short or
    empty transcript) are valid multimodal candidates and are never filtered.
    """
    transcript = (payload.get("transcript") or "").strip()
    if len(transcript) >= MIN_TRANSCRIPT_CHARS:
        return False
    if (payload.get("ocr_text") or "").strip():
        return False
    if (payload.get("visual_context") or "").strip():
        return False
    return True


def _embed_query(query: str, model: Any) -> List[float]:
    """Embed a single query string using the embedding model."""
    vector = model.encode([query], normalize_embeddings=True)
    return vector[0].tolist()


def preload_embedding_model(device: str = "cpu") -> Any:
    """
    Preload the global embedding model (bge-large-en-v1.5) at startup.

    Loads the model once and registers it as the module-level singleton so
    the first query never pays the model-loading latency. Idempotent — safe
    to call from app startup even if a workflow later injects its own model.

    Returns:
        The loaded SentenceTransformer instance.
    """
    global _GLOBAL_EMBEDDING_MODEL
    if _GLOBAL_EMBEDDING_MODEL is not None:
        return _GLOBAL_EMBEDDING_MODEL
    logger.info("E2b: Preloading embedding model BAAI/bge-large-en-v1.5 (device=%s)", device)
    from sentence_transformers import SentenceTransformer
    _GLOBAL_EMBEDDING_MODEL = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    return _GLOBAL_EMBEDDING_MODEL


def retrieve_vectors(
    query: str,
    qdrant_client: Optional[Any] = None,
    embedding_model: Optional[Any] = None,
    local_settings: LocalSettings | None = None,
    shared_settings: SharedSettings | None = None,
    top_k: int = DEFAULT_TOP_K,
    collection_name: str = COLLECTION_NAME,
    lecture_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve top-K similar chunks from Qdrant.

    Args:
        query: User query string.
        qdrant_client: Optional pre-created Qdrant client.
        embedding_model: Optional pre-loaded embedding model.
        local_settings: Injected LocalSettings.
        shared_settings: SharedSettings for EMBEDDING_DIMENSION.
        top_k: Number of results to return.
        collection_name: Qdrant collection name.
        lecture_id: Optional lecture ID to filter results.

    Returns:
        List of dicts with chunk_id, score, and full payload.
    """
    settings = local_settings or LocalSettings()
    ss = shared_settings or SharedSettings()

    global _GLOBAL_EMBEDDING_MODEL
    # Load embedding model if not provided.
    close_client = False
    
    if embedding_model is not None:
        _GLOBAL_EMBEDDING_MODEL = embedding_model
    elif _GLOBAL_EMBEDDING_MODEL is None:
        logger.info("E2b: Lazy loading embedding model BAAI/bge-large-en-v1.5")
        from sentence_transformers import SentenceTransformer
        _GLOBAL_EMBEDDING_MODEL = SentenceTransformer(
            "BAAI/bge-large-en-v1.5",
            device="cpu"
        )

    # Embed the query.
    query_vector = _embed_query(query, _GLOBAL_EMBEDDING_MODEL)

    if len(query_vector) != ss.EMBEDDING_DIMENSION:
        raise ValueError(
            f"E2b: Query vector dimension {len(query_vector)} != {ss.EMBEDDING_DIMENSION}"
        )

    # Connect to Qdrant.
    if qdrant_client is None:
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(url=settings.QDRANT_URL)
        close_client = True

    try:
        query_filter = None
        if lecture_id:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(must=[FieldCondition(key="lecture_id", match=MatchValue(value=lecture_id))])
        else:
            # Data isolation: a query without a lecture_id must never search
            # across every lecture in the shared collection. Reject loudly.
            raise ValueError(
                "E2b: lecture_id is required for vector retrieval — refusing to "
                "search across all lectures (cross-lecture data leak)."
            )

        # Fetch top_k + slack so the filler filter below never shrinks the
        # externally requested pool below top_k valid candidates.
        search_results = qdrant_client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k + RETRIEVAL_FETCH_SLACK,
            query_filter=query_filter,
        ).points

        results = []
        for hit in search_results:
            payload = hit.payload if hasattr(hit, "payload") else hit.get("payload", {})
            score = hit.score if hasattr(hit, "score") else hit.get("score", 0.0)

            if _is_transcript_filler(payload):
                continue

            results.append({
                "chunk_id": payload.get("chunk_id", ""),
                "score": float(score),
                "payload": payload,
            })
            if len(results) >= top_k:
                break

        logger.info("E2b: Retrieved %d vector results (top_k=%d).", len(results), top_k)
        return results

    finally:
        if close_client:
            qdrant_client.close()


def retrieve_lecture_wide(
    qdrant_client: Optional[Any] = None,
    local_settings: LocalSettings | None = None,
    top_k: int = 15,
    collection_name: str = COLLECTION_NAME,
    lecture_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve representative chunks across the lecture timeline.
    Bypasses semantic search completely to avoid bias towards meta-commentary.
    """
    settings = local_settings or LocalSettings()
    
    if not lecture_id:
        raise ValueError(
            "E2b: lecture_id is required for lecture-wide retrieval — refusing to "
            "scroll across all lectures (cross-lecture data leak)."
        )

    close_client = False
    if qdrant_client is None:
        from qdrant_client import QdrantClient
        qdrant_client = QdrantClient(url=settings.QDRANT_URL)
        close_client = True
        
    try:
        query_filter = None
        if lecture_id:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            query_filter = Filter(must=[FieldCondition(key="lecture_id", match=MatchValue(value=lecture_id))])
            
        # Scroll to get all chunks (up to a reasonable limit).
        # Paginate with offset so lectures with more than one page of
        # points (page size 1000) are fully covered.
        results = []
        offset = None
        MAX_SCROLL_POINTS = 10000
        while len(results) < MAX_SCROLL_POINTS:
            page, next_offset = qdrant_client.scroll(
                collection_name=collection_name,
                scroll_filter=query_filter,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            results.extend(page)
            if not page or next_offset is None:
                break
            offset = next_offset

        if not results:
            return []
            
        # Sort chronologically
        def get_ts(point):
            try:
                return float(point.payload.get("timestamp", 0))
            except (TypeError, ValueError):
                return 0.0
                
        results.sort(key=get_ts)
        
        # Select representative chunks by fetching clustered windows
        window_size = 8
        num_windows = max(1, top_k // 2)  # Typically 8 windows (top_k is usually 15 or 20)
        
        if len(results) <= window_size * num_windows:
            sampled = results
        else:
            step = len(results) / num_windows
            sampled = []
            for i in range(num_windows):
                start_idx = int(i * step)
                start_idx = min(start_idx, len(results) - window_size)
                sampled.extend(results[start_idx : start_idx + window_size])
                
        # Deduplicate in case windows overlap
        unique_sampled = []
        seen = set()
        for hit in sampled:
            cid = hit.id
            if cid not in seen:
                seen.add(cid)
                unique_sampled.append(hit)
        sampled = unique_sampled
            
        ret_results = []
        for hit in sampled:
            payload = hit.payload if hasattr(hit, "payload") else hit.get("payload", {})
            ret_results.append({
                "chunk_id": payload.get("chunk_id", ""),
                "score": 1.0,  # Constant score since this is unbiased sampling
                "payload": payload,
            })
            
        logger.info("E2b: Retrieved %d lecture-wide chunks (sampled from %d).", len(ret_results), len(results))
        return ret_results
        
    finally:
        if close_client:
            qdrant_client.close()

