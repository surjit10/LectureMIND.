# retrieval/context_builder.py
# V7 — Context Builder.
#
# Receives reranked chunks and produces the final LLM context string.
# Responsibilities:
#   - Remove duplicate chunks (by chunk_id and text)
#   - Merge adjacent chunks from the same segment/lecture region
#   - Prioritize Transcript > OCR > Visuals
#   - Preserve chronological order
#   - Enforce token budget without breaking passage continuity
#   - Filter OCR noise (subscribe slides, channel branding, watermarks)
#   - Lecture-wide mode: sample representative chunks across timeline
#
# Nothing in retrieval, reranking, embeddings, or orchestration is changed.

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OCR noise patterns — channel branding, subscribe slides, social handles.
# These appear in YouTube-sourced lectures and pollute educational context.
# ---------------------------------------------------------------------------

_NOISE_PATTERNS = [
    re.compile(r"\bsubscribe\b", re.IGNORECASE),
    re.compile(r"\blike\b.{0,20}\bshare\b", re.IGNORECASE),
    re.compile(r"\bshare\b.{0,20}\blike\b", re.IGNORECASE),
    re.compile(r"\bnotification\b", re.IGNORECASE),
    re.compile(r"\bbell\s+icon\b", re.IGNORECASE),
    re.compile(r"\b@[A-Za-z0-9_]+\b"),        # social handles
    re.compile(r"https?://\S+"),               # URLs (rarely educational in OCR)
    re.compile(r"\bwatermark\b", re.IGNORECASE),
    re.compile(r"\blogo\b.{0,20}\bchannel\b", re.IGNORECASE),
    re.compile(r"\boutro\b", re.IGNORECASE),
    re.compile(r"\bthank\s+you\s+for\s+watching\b", re.IGNORECASE),
    re.compile(r"\bdon[\'']t\s+forget\s+to\b", re.IGNORECASE),
]

# Minimum word count for OCR text to be considered educational.
_MIN_OCR_WORDS = 5

# Noise patterns whose matches are REMOVED from OCR text rather than
# discarding the whole slide. A single URL or social handle embedded in an
# otherwise educational slide must not destroy the content (diagnosed: the
# "Increasing Software Complexity" slide's OCR — Linux 2.2 / Firefox / Windows
# ranking — was dropped entirely because it contained one informationisbeautiful.net
# URL, so the LLM could never name the systems).
_LOCALIZED_NOISE_PATTERNS = [
    re.compile(r"(?:https?://|www\.)\S+"),
    re.compile(r"\b@[A-Za-z0-9_]+\b"),
]


def _sanitize_ocr(text: str) -> str:
    """Remove localized noise (URLs, handles) from OCR text, keep the rest."""
    if not text:
        return ""
    cleaned = text
    for pattern in _LOCALIZED_NOISE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()

# Separator used between context chunks.
_CHUNK_SEPARATOR = "\n\n---\n\n"


def _is_ocr_noise(text: str) -> bool:
    """
    Return True if OCR text is likely non-educational noise.

    Checks for channel branding patterns AND word count threshold.
    Legitimate educational slide content is never discarded.
    """
    if not text.strip():
        return True
    # Too short to be educational.
    word_count = len(text.split())
    if word_count < _MIN_OCR_WORDS:
        # Still check if it has substantial educational content.
        if not any(char.isalpha() for char in text):
            return True
    # Check for known noise patterns.
    for pattern in _NOISE_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _build_passage_text(result: Dict[str, Any], need_visual: bool = True) -> str:
    """
    Extract passage text from a result dict, labelled by source type.

    Priority: Transcript > OCR > Visual descriptions.
    OCR noise is filtered. Visuals included only when need_visual=True.
    """
    payload = result.get("payload", result)

    transcript = (payload.get("transcript") or "").strip()
    ocr_text = (payload.get("ocr_text") or "").strip()
    visual_context = (payload.get("visual_context") or "").strip()

    parts = []
    if transcript:
        parts.append(f"[Transcript] {transcript}")

    if ocr_text:
        # Strip localized noise (URLs, handles) first — an educational slide
        # containing one URL must keep its content.
        clean_ocr = _sanitize_ocr(ocr_text)
        if clean_ocr and not _is_ocr_noise(clean_ocr):
            parts.append(f"[OCR] {clean_ocr}")

    if visual_context and need_visual:
        parts.append(f"[Visual] {visual_context}")

    return " ".join(parts) if parts else str(payload)


def _build_graph_context(
    graph_results: List[Dict[str, Any]],
    char_budget: int = 2000,
) -> str:
    """
    Render graph retrieval results as labelled entity-relation paths.

    Graph results carry entity names and relation types (no transcript):
        {"start_name": "Operating System Diagram", "related_name": "Network Diagram",
         "rel_types": ["DERIVED_FROM"], "hops": 1}

    Renders as:
        [Graph] Operating System Diagram -DERIVED_FROM-> Network Diagram

    Isolated start nodes (no related_name) render as a plain entity line.
    Duplicate paths are dropped; short (fewer-hop) paths are preferred;
    output is capped at char_budget so a broad graph hit (e.g. entity
    substring match) can never blow the final context budget.
    Empty results return "".
    """
    if not graph_results:
        return ""

    # Prefer direct (1-hop) relations first, then longer paths.
    ordered = sorted(
        graph_results,
        key=lambda gr: (
            (gr.get("related_name") or "").strip() == "",  # isolated last
            int(gr.get("hops") or 0),
        ),
    )

    lines: List[str] = []
    seen: set = set()
    total_chars = 0
    for gr in ordered:
        start = (gr.get("start_name") or "").strip()
        related = (gr.get("related_name") or "").strip()
        if not start:
            continue
        if related:
            rels = gr.get("rel_types") or []
            rel_str = ",".join(rels) if rels else "RELATED"
            line = f"[Graph] {start} -{rel_str}-> {related}"
        else:
            line = f"[Graph] {start}"
        if line in seen:
            continue
        if total_chars + len(line) + 1 > char_budget:
            break
        seen.add(line)
        lines.append(line)
        total_chars += len(line) + 1

    return "\n".join(lines)


def _block_score(result: Dict[str, Any]) -> float:
    """Best available rerank score for a (possibly merged) block."""
    try:
        return float(result.get("rerank_score") or result.get("score") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _get_timestamp(result: Dict[str, Any]) -> float:
    """Extract numeric timestamp for chronological ordering."""
    payload = result.get("payload", result)
    ts = payload.get("timestamp", 0)
    try:
        return float(ts)
    except (TypeError, ValueError):
        return 0.0


def _are_adjacent(a: Dict[str, Any], b: Dict[str, Any], gap_seconds: float = 30.0) -> bool:
    """
    Return True if two chunks are from the same region (same segment,
    timestamps within gap_seconds of each other).
    """
    a_payload = a.get("payload", a)
    b_payload = b.get("payload", b)
    # Must be from same lecture
    if a_payload.get("lecture_id") and b_payload.get("lecture_id"):
        if a_payload["lecture_id"] != b_payload["lecture_id"]:
            return False
    
    # Must be from same segment
    if a_payload.get("segment_id") and b_payload.get("segment_id"):
        if a_payload["segment_id"] != b_payload["segment_id"]:
            return False

    ts_a = _get_timestamp(a)
    ts_b = _get_timestamp(b)
    return abs(ts_a - ts_b) <= gap_seconds


def _merge_adjacent(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge adjacent chunks from the same lecture region into one entry.

    Merges transcript, ocr_text, and visual_context by concatenation.
    The merged chunk uses the timestamp of the first chunk in the group.
    Tracks constituent_chunk_ids across merged chunks.
    """
    if not chunks:
        return []

    merged = []
    current = dict(chunks[0])
    current_payload = dict(current.get("payload", current))
    init_cid = current_payload.get("chunk_id") or current.get("chunk_id", "")
    init_snippet = (current_payload.get("transcript") or current_payload.get("ocr_text") or "")[:40].strip()
    current["constituent_chunks"] = [(init_cid, init_snippet)] if init_cid else []

    for nxt in chunks[1:]:
        nxt_payload = nxt.get("payload", nxt)
        nxt_cid = nxt_payload.get("chunk_id") or nxt.get("chunk_id", "")
        nxt_snippet = (nxt_payload.get("transcript") or nxt_payload.get("ocr_text") or "")[:40].strip()

        if _are_adjacent(current, nxt):
            # Merge text fields.
            for field in ("transcript", "ocr_text", "visual_context"):
                a_text = (current_payload.get(field) or "").strip()
                b_text = (nxt_payload.get(field) or "").strip()
                if a_text and b_text:
                    current_payload[field] = f"{a_text} {b_text}"
                elif b_text:
                    current_payload[field] = b_text
            # Keep the highest rerank score from the group, so score-based
            # selection treats the merged block as its best constituent.
            cur_score = float(current.get("rerank_score") or 0.0)
            nxt_score = float(nxt.get("rerank_score") or 0.0)
            if nxt_score > cur_score:
                current["rerank_score"] = nxt.get("rerank_score")
            # Keep the earliest timestamp.
            if _get_timestamp(nxt) < _get_timestamp(current):
                current_payload["timestamp"] = nxt_payload.get("timestamp", 0)
            # Record constituent chunk id and snippet.
            if nxt_cid and not any(nxt_cid == c[0] for c in current["constituent_chunks"]):
                current["constituent_chunks"].append((nxt_cid, nxt_snippet))
        else:
            # Flush current.
            if "payload" in current:
                current["payload"] = current_payload
            else:
                current.update(current_payload)
            merged.append(current)
            current = dict(nxt)
            current_payload = dict(current.get("payload", current))
            current["constituent_chunks"] = [(nxt_cid, nxt_snippet)] if nxt_cid else []

    # Flush last.
    if "payload" in current:
        current["payload"] = current_payload
    else:
        current.update(current_payload)
    merged.append(current)

    return merged


def _sample_lecture_wide(chunks: List[Dict[str, Any]], n_buckets: int = 8) -> List[Dict[str, Any]]:
    """
    Sample representative chunks across the lecture timeline.

    Divides the timeline into n_buckets equal-width time windows and
    takes one chunk from each bucket.  This ensures:
        Beginning → Early concepts → Middle → Late concepts → Conclusion
    rather than just the top semantically similar chunks.
    """
    if not chunks:
        return []
    if len(chunks) <= n_buckets:
        return chunks

    timestamps = [_get_timestamp(c) for c in chunks]
    t_min = min(timestamps)
    t_max = max(timestamps)
    if t_max == t_min:
        # No temporal spread — return evenly spaced by index.
        step = max(1, len(chunks) // n_buckets)
        return [chunks[i] for i in range(0, len(chunks), step)][:n_buckets]

    bucket_width = (t_max - t_min) / n_buckets
    selected: Dict[int, Dict[str, Any]] = {}

    for chunk in chunks:
        ts = _get_timestamp(chunk)
        bucket = min(int((ts - t_min) / bucket_width), n_buckets - 1)
        if bucket not in selected:
            selected[bucket] = chunk

    # Preserve chronological order.
    return [selected[b] for b in sorted(selected.keys())]


class ContextBuilder:
    """
    Assembles the final LLM context from reranked retrieval results.

    Usage:
        builder = ContextBuilder()
        context = builder.build(reranked_results, query_plan=plan)

    Does NOT perform retrieval or reranking.  Input must already be
    reranked by the existing E3 reranker pipeline.
    """

    def __init__(self, char_budget: int = 4000):
        self._default_budget = char_budget

    def build(
        self,
        reranked_results: List[Dict[str, Any]],
        *,
        is_lecture_wide: bool = False,
        need_visual: bool = False,
        char_budget: Optional[int] = None,
    ) -> str:
        """
        Build the final LLM context string.
        (Backward compatible — delegates to build_with_metadata).

        Args:
            reranked_results: Output of the E3 reranker (list of dicts).
            is_lecture_wide: When True, sample across the full timeline
                instead of taking only top-ranked chunks.
            need_visual: When True, include [Visual] labels in context.
            char_budget: Maximum character budget (overrides default).

        Returns:
            Assembled context string ready for the LLM prompt.
        """
        final_context, _ = self.build_with_metadata(
            reranked_results,
            is_lecture_wide=is_lecture_wide,
            need_visual=need_visual,
            char_budget=char_budget,
        )
        return final_context

    def build_with_metadata(
        self,
        reranked_results: List[Dict[str, Any]],
        *,
        is_lecture_wide: bool = False,
        need_visual: bool = False,
        char_budget: Optional[int] = None,
    ) -> Tuple[str, List[str]]:
        """
        Build the final LLM context string and return the list of used chunk IDs.

        Args:
            reranked_results: Output of the E3 reranker (list of dicts).
            is_lecture_wide: When True, sample across the full timeline
                instead of taking only top-ranked chunks.
            need_visual: When True, include [Visual] labels in context.
            char_budget: Maximum character budget (overrides default).

        Returns:
            Tuple of (assembled_context_string, list_of_used_chunk_ids).
        """
        budget = char_budget if char_budget is not None else self._default_budget

        if not reranked_results:
            return "", []

        # 1. Deduplicate by chunk_id.
        seen_ids: set = set()
        unique: List[Dict[str, Any]] = []
        for r in reranked_results:
            payload = r.get("payload", r)
            cid = payload.get("chunk_id") or r.get("chunk_id", "")
            if cid and cid in seen_ids:
                continue
            if cid:
                seen_ids.add(cid)
            unique.append(r)

        # 2. Sort chronologically (preserves lecture narrative flow).
        unique.sort(key=_get_timestamp)

        # 3. Merge adjacent chunks from the same region.
        merged = _merge_adjacent(unique)

        # 4. Lecture-wide: sample representative blocks across timeline.
        if is_lecture_wide:
            merged = _sample_lecture_wide(merged, n_buckets=8)

        # 5. Selection order.
        #    Normal queries: iterate by rerank score so the budget favours the
        #    strongest evidence.  Previously the pool was cut chronologically,
        #    so a top-ranked chunk late in the lecture timeline was silently
        #    dropped and the LLM answered "Insufficient evidence" despite the
        #    evidence being retrieved (measured on the benchmark: 25/50
        #    refusals before score-ordered selection, 0 after).
        #    Lecture-wide queries keep chronological selection — timeline
        #    coverage is the point, and sampling already spread the blocks.
        if is_lecture_wide:
            order = merged
        else:
            order = sorted(merged, key=_block_score, reverse=True)

        # 6. Build text for each chunk, filter by OCR noise.
        selected: List[Any] = []  # (entry, text) pairs in selection order
        seen_texts: set = set()
        total_chars = 0

        for entry in order:
            text = _build_passage_text(entry, need_visual=need_visual)
            if not text.strip():
                continue
            # Skip exact duplicate text blocks.
            if text in seen_texts:
                continue
            seen_texts.add(text)
            # Enforce budget without breaking a passage mid-sentence.
            if total_chars + len(text) + len(_CHUNK_SEPARATOR) > budget:
                # First-chunk guarantee: if nothing has been added yet, this
                # merged block is the only evidence available.  Truncating it
                # is always better than returning an empty context, which
                # causes answer_generator to return the fallback disclaimer
                # without ever calling the LLM.
                if total_chars == 0:
                    # Try to cut at a sentence boundary within the budget.
                    truncated = text[:budget]
                    boundary = truncated.rfind(". ")
                    if boundary > budget // 2:
                        # Enough content before the boundary — cut cleanly.
                        truncated = truncated[: boundary + 1]
                    selected.append((entry, truncated))
                    logger.debug(
                        "ContextBuilder: First chunk truncated from %d to %d chars "
                        "(budget=%d).",
                        len(text), len(truncated), budget,
                    )
                break
            selected.append((entry, text))
            total_chars += len(text) + len(_CHUNK_SEPARATOR)

        # 7. Render the selected evidence chronologically so the narrative
        #    flow is preserved even though selection prioritised rerank score.
        if not is_lecture_wide:
            selected.sort(key=lambda pair: _get_timestamp(pair[0]))
        context_parts = [text for _, text in selected]

        # Extract constituent chunk IDs that made it into selected context.
        used_ids: List[str] = []
        seen_used: set = set()
        for entry, sel_text in selected:
            constituents = entry.get("constituent_chunks")
            if constituents:
                for cid, snippet in constituents:
                    if cid and cid not in seen_used:
                        if not snippet or snippet in sel_text:
                            seen_used.add(cid)
                            used_ids.append(cid)
            else:
                payload = entry.get("payload", entry)
                cid = payload.get("chunk_id") or entry.get("chunk_id", "")
                if cid and cid not in seen_used:
                    seen_used.add(cid)
                    used_ids.append(cid)

        final = _CHUNK_SEPARATOR.join(context_parts)
        logger.info(
            "ContextBuilder: %d input → %d merged → %d used, %d chars (lecture_wide=%s)",
            len(reranked_results), len(merged), len(context_parts), len(final), is_lecture_wide,
        )
        return final, used_ids
