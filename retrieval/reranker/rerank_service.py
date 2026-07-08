# retrieval/reranker/rerank_service.py
# E3 — Reranker Node Service.
#
# Combines graph_results + vector_results, deduplicates by chunk_id,
# reranks using the global CrossEncoder singleton.
#
# Populates state.reranked_results and state.final_context.
#
# The global reranker singleton (_GLOBAL_RERANKER_SERVICE) is set exactly
# once by serving/fastapi/app.py at application startup and lives for the
# entire process lifetime. Lecture switching has no effect on it.

import logging
import threading
from typing import Any, Dict, List, Optional

from local.services.reranker_service import RerankerService
from retrieval.context_builder import ContextBuilder, _build_passage_text

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 4000
_GLOBAL_RERANKER_SERVICE = None
_RERANKER_LOCK = threading.Lock()
_CONTEXT_BUILDER = ContextBuilder(char_budget=MAX_CONTEXT_CHARS)


# clear_reranker() has been removed.
# The global singleton is set once at startup and never invalidated.
# Lecture switching does not affect the reranker.

def reload_global_reranker(reranker_dir: str) -> None:
    """
    Thread-safe hot reload of the global reranker model.
    """
    global _GLOBAL_RERANKER_SERVICE
    from local.loaders.reranker_loader import _load_cross_encoder
    
    with _RERANKER_LOCK:
        model = _load_cross_encoder(reranker_dir)
        _GLOBAL_RERANKER_SERVICE = RerankerService(model)
        logger.info("E3: Global reranker hot-reloaded successfully.")

def _extract_passages(
    graph_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Combine graph and vector results, deduplicate by chunk_id.

    Graph results don't carry chunk payloads directly, so we only
    use vector results that have full payload. Graph results
    contribute entity context.
    """
    seen_chunks = set()
    combined = []

    # Vector results have full chunk payloads.
    for vr in vector_results:
        chunk_id = vr.get("chunk_id", "")
        if chunk_id and chunk_id not in seen_chunks:
            seen_chunks.add(chunk_id)
            combined.append(vr)

    # Graph results may contain entity context without chunk payloads.
    # Include them as supplementary context if they have chunk_id.
    for gr in graph_results:
        chunk_id = gr.get("chunk_id", "")
        if chunk_id and chunk_id not in seen_chunks:
            seen_chunks.add(chunk_id)
            combined.append(gr)

    return combined


# _build_passage_text is imported from context_builder for backward compat.
# External callers that imported it from here continue to work.



def rerank(
    query: str,
    graph_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    reranker_service: Optional[RerankerService] = None,
    reranker_model: Optional[Any] = None,
    is_lecture_wide: bool = False,
    need_visual: bool = False,
    char_budget: Optional[int] = None,
) -> tuple:
    """
    Fuse, deduplicate, and rerank results.

    Args:
        query: User query string.
        graph_results: Results from graph retriever.
        vector_results: Results from vector retriever.
        reranker_service: Pre-initialized RerankerService from Chunk 4.
        reranker_model: Raw CrossEncoder model (creates service if needed).
        is_lecture_wide: When True, ContextBuilder samples across timeline.
        need_visual: When True, visual context is included.

    Returns:
        (reranked_results, final_context) tuple.
    """
    effective_budget = char_budget if char_budget is not None else MAX_CONTEXT_CHARS

    # Combine and deduplicate.
    combined = _extract_passages(graph_results, vector_results)

    if not combined:
        logger.info("E3: No passages to rerank.")
        return [], ""

    # Build passage texts.
    passages = [_build_passage_text(r) for r in combined]

    global _GLOBAL_RERANKER_SERVICE
    # Allow test injection via function arguments.
    # In production the singleton is always set by startup before the first query.
    with _RERANKER_LOCK:
        active_service = _GLOBAL_RERANKER_SERVICE
        if active_service is None:
            if reranker_service is not None:
                active_service = reranker_service
            elif reranker_model is not None:
                active_service = RerankerService(reranker_model)
            else:
                raise RuntimeError(
                    "E3: Global reranker singleton is not initialized. "
                    "serving/fastapi/app.py must call _run_model_recovery() before "
                    "any query is processed. Do not load the reranker from disk here."
                )

    # Rerank.
    if active_service is not None:
        scored = active_service.score_pairs(query, passages)

        # Map scores back to combined results.
        passage_to_idx = {p: i for i, p in enumerate(passages)}
        reranked = []
        for passage_text, score in scored:
            idx = passage_to_idx.get(passage_text)
            if idx is not None:
                entry = {**combined[idx], "rerank_score": score}
                reranked.append(entry)
    else:
        # No reranker — use original order.
        logger.warning("E3: No reranker available, using original order.")
        reranked = [{**r, "rerank_score": r.get("score", 0.0)} for r in combined]

    # Build final_context via ContextBuilder.
    # ContextBuilder handles dedup, merge, chronological order,
    # OCR noise filtering, and lecture-wide temporal sampling.
    global _CONTEXT_BUILDER
    final_context = _CONTEXT_BUILDER.build(
        reranked,
        is_lecture_wide=is_lecture_wide,
        need_visual=need_visual,
        char_budget=effective_budget,
    )

    logger.info("E3: Reranked %d passages, context=%d chars.", len(reranked), len(final_context))
    return reranked, final_context
