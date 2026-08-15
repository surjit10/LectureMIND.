# retrieval/course_retriever.py
# Feature 1 — Course fan-out retriever.
#
# A course is metadata only (local/storage/course_registry.py). This module
# turns a course query into N lecture-scoped retrievals, one per member
# lecture, then merges the results in the application layer.
#
# ISOLATION: every underlying call passes an explicit lecture_id, so the
# existing hard guards in qdrant_retriever / neo4j_retriever apply verbatim.
# No lecture graph is ever merged; results are deduplicated on
# (lecture_id, chunk_id) because chunk_id strings may repeat across lectures.

import logging
from typing import Any, Dict, List, Optional

from local.storage.course_registry import get_course_registry
from retrieval.vector_retriever.qdrant_retriever import (
    preload_embedding_model,
    retrieve_lecture_wide,
    retrieve_vectors,
)
from retrieval.graph_retriever.neo4j_retriever import retrieve_graph

# NOTE: module-level imports are safe — the heavy dependencies (torch,
# sentence_transformers, qdrant_client, neo4j) are all lazy-loaded inside
# the retriever functions; these modules only import config/logging at load.

logger = logging.getLogger(__name__)


def _dedup_key(result: Dict[str, Any]) -> str:
    """Dedupe across lectures: chunk_ids repeat across lectures, so key on (lecture, chunk)."""
    payload = result.get("payload", result)
    lecture_id = payload.get("lecture_id", "") or result.get("lecture_id", "")
    chunk_id = payload.get("chunk_id", "") or result.get("chunk_id", "")
    return f"{lecture_id}::{chunk_id}"


def retrieve_course_context(
    query: str,
    course_id: str,
    *,
    qdrant_client: Optional[Any] = None,
    embedding_model: Optional[Any] = None,
    driver: Optional[Any] = None,
    is_lecture_wide: bool = False,
    top_k_per_lecture: int = 5,
    include_graph: bool = True,
) -> Dict[str, Any]:
    """
    Fan out a query to every READY lecture in a course and merge results.

    Args:
        query: The user's course-scoped question.
        course_id: Course whose member lectures are queried.
        qdrant_client / embedding_model / driver: Optional shared handles.
            When omitted, retrievers open/close their own (existing behavior).
        is_lecture_wide: When True, each lecture is covered via timeline
            sampling (retrieve_lecture_wide) instead of top-K similarity.
        top_k_per_lecture: Chunks retrieved per lecture.
        include_graph: When True, also run graph traversal per lecture.

    Returns:
        {
          "vector_results": [...],   # each tagged with payload.lecture_id
          "graph_results": [...],
          "lectures_used": [...],    # lecture_ids actually queried
          "skipped_lectures": [...], # ids that failed / were missing
        }
    """
    course = get_course_registry().get_course(course_id)
    if course is None:
        raise KeyError(f"Course not found: {course_id}")
    lecture_ids = course.get("lecture_ids", [])

    vector_results: List[Dict[str, Any]] = []
    graph_results: List[Dict[str, Any]] = []
    lectures_used: List[str] = []
    skipped: List[str] = []

    # Reuse the process-wide embedding model singleton instead of reloading.
    if embedding_model is None:
        try:
            embedding_model = preload_embedding_model()
        except Exception as exc:
            logger.warning("Course retriever: embedding model unavailable: %s", exc)

    for lecture_id in lecture_ids:
        # --- Vector fan-out (each call passes its own lecture_id guard) ---
        try:
            if is_lecture_wide:
                results = retrieve_lecture_wide(
                    qdrant_client=qdrant_client,
                    top_k=top_k_per_lecture,
                    lecture_id=lecture_id,
                )
            else:
                results = retrieve_vectors(
                    query,
                    qdrant_client=qdrant_client,
                    embedding_model=embedding_model,
                    top_k=top_k_per_lecture,
                    lecture_id=lecture_id,
                )
            vector_results.extend(results)
        except Exception as exc:
            logger.warning("Course retriever: vector retrieval failed for %s: %s", lecture_id, exc)
            skipped.append(lecture_id)
            continue

        # --- Graph fan-out ---
        if include_graph:
            try:
                graph_results.extend(
                    retrieve_graph(query, driver=driver, lecture_id=lecture_id)
                )
            except Exception as exc:
                logger.warning("Course retriever: graph retrieval failed for %s: %s", lecture_id, exc)

        lectures_used.append(lecture_id)

    # --- Dedupe on (lecture_id, chunk_id); keep the highest-scoring copy ---
    seen: Dict[str, Dict[str, Any]] = {}
    for result in vector_results:
        key = _dedup_key(result)
        score = float(result.get("score", 0.0) or 0.0)
        if key not in seen or score > float(seen[key].get("score", 0.0) or 0.0):
            seen[key] = result
    vector_results = sorted(seen.values(), key=lambda r: float(r.get("score", 0.0) or 0.0), reverse=True)

    logger.info(
        "Course retriever: %d lecture(s) → %d vector, %d graph (skipped %d).",
        len(lectures_used), len(vector_results), len(graph_results), len(skipped),
    )
    return {
        "vector_results": vector_results,
        "graph_results": graph_results,
        "lectures_used": lectures_used,
        "skipped_lectures": skipped,
    }
