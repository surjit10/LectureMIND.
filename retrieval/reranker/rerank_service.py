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

from config import LocalSettings
from local.services.reranker_service import RerankerService
from retrieval.context_builder import (
    ContextBuilder,
    _build_graph_context,
    _build_passage_text,
    _CHUNK_SEPARATOR,
)

# Hybrid retrieval (BM25 + RRF) — opt-in, off by default.
from retrieval.hybrid.bm25_retriever import bm25_search, rrf_fuse

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
    lecture_id: Optional[str] = None,
    enable_hybrid: Optional[bool] = None,
) -> tuple:
    """
    Fuse, deduplicate, and rerank results.

    Hybrid retrieval: when enable_hybrid=True and lecture_id is given, BM25
    lexical retrieval runs alongside the dense vector search and results are
    fused via Reciprocal Rank Fusion (RRF) before cross-encoder reranking.

    Args:
        query: User query string.
        graph_results: Results from graph retriever.
        vector_results: Results from vector retriever.
        reranker_service: Pre-initialized RerankerService from Chunk 4.
        reranker_model: Raw CrossEncoder model (creates service if needed).
        is_lecture_wide: When True, ContextBuilder samples across timeline.
        need_visual: When True, visual context is included.
        char_budget: Character budget for context (overrides default).
        lecture_id: Runtime lecture ID for BM25 index scope.
        enable_hybrid: When True, run BM25 + RRF fusion. If None, falls
            back to LocalSettings.ENABLE_HYBRID_RETRIEVAL (default False).

    Returns:
        (reranked_results, final_context) tuple.
    """
    effective_budget = char_budget if char_budget is not None else MAX_CONTEXT_CHARS

    # ── Hybrid retrieval: BM25 + RRF fusion on vector candidates ──
    # When enabled, augment the dense vector candidates with lexical BM25
    # results fused via Reciprocal Rank Fusion. This catches entity names,
    # numeric facts, and exact-phrase matches that dense embeddings may miss.
    fused_vectors = vector_results
    if enable_hybrid is None:
        enable_hybrid = LocalSettings().ENABLE_HYBRID_RETRIEVAL
    if enable_hybrid and lecture_id:
        try:
            bm25_retrieved = bm25_search(query, lecture_id, top_k=15)
            if bm25_retrieved:
                fused_vectors = rrf_fuse(vector_results, bm25_retrieved, top_k=15)
                logger.info(
                    "E3: Hybrid retrieval — BM25 contributed %d candidates, "
                    "RRF fused to %d total.",
                    len(bm25_retrieved), len(fused_vectors),
                )
        except Exception as exc:
            logger.warning(
                "E3: Hybrid retrieval failed (fallback to dense only): %s", exc,
            )

    # Combine graph and vector results, deduplicate by chunk_id.
    combined = _extract_passages(graph_results, fused_vectors)

    # Render graph paths as labelled context. Graph results carry entity
    # names + relation types (not chunk payloads), so they never enter the
    # rerank pool — there is no passage text to score. They are, however,
    # the core evidence for relationship questions, so they are appended to
    # final_context directly (budget reserved below).
    # Cap the graph section at half the budget so a broad graph hit can
    # never starve the transcript evidence.
    graph_budget = min(2000, max(0, effective_budget // 2))
    graph_context = _build_graph_context(graph_results, char_budget=graph_budget)

    if not combined:
        if graph_context:
            logger.info(
                "E3: No vector passages; graph context only (%d chars).",
                len(graph_context),
            )
            return [], graph_context
        logger.info("E3: No passages to rerank.")
        return [], ""

    # Build passage texts with the SAME visibility the LLM context will use:
    # visual context is only included when need_visual=True.  Previously the
    # reranker scored passages with visual_context always included, so chunks
    # whose relevance came from slide descriptions were ranked high but the
    # LLM never saw those visuals (need_visual=False) — starving genuinely
    # relevant transcript evidence (diagnosed: "course materials" ranked the
    # visual-boosted "What is an OS?" / "Societal Scale" slides above the
    # textbook chunk that names the actual course materials).
    passages = [_build_passage_text(r, need_visual=need_visual) for r in combined]

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
            elif LocalSettings().ENABLE_AUTO_MODEL_RECOVERY:
                raise RuntimeError(
                    "E3: Global reranker singleton is not initialized. "
                    "serving/fastapi/app.py must call _run_model_recovery() before "
                    "any query is processed. Do not load the reranker from disk here."
                )
            else:
                logger.warning(
                    "E3: Global reranker singleton is not initialized and auto-recovery is disabled; using original order."
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
    # Reserve budget for the graph section first so relationship evidence
    # is never squeezed out by transcript chunks.
    global _CONTEXT_BUILDER
    vector_budget = effective_budget
    if graph_context:
        vector_budget = max(
            0, effective_budget - len(graph_context) - len(_CHUNK_SEPARATOR)
        )

    final_context, used_ids = _CONTEXT_BUILDER.build_with_metadata(
        reranked,
        is_lecture_wide=is_lecture_wide,
        need_visual=need_visual,
        char_budget=vector_budget,
    )

    # Tag entries that actually made it into the LLM context prompt
    used_id_set = set(used_ids)
    for entry in reranked:
        payload = entry.get("payload", entry)
        cid = payload.get("chunk_id", entry.get("chunk_id", ""))
        entry["in_context"] = bool(cid in used_id_set)

    # Prepend the graph section: relationship paths are the primary
    # evidence for relationship questions, so they lead the context.
    if graph_context:
        if final_context:
            final_context = graph_context + _CHUNK_SEPARATOR + final_context
        else:
            final_context = graph_context

    logger.info(
        "E3: Reranked %d passages, graph=%d chars, context=%d chars.",
        len(reranked), len(graph_context), len(final_context),
    )
    return reranked, final_context
