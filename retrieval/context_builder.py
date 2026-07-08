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
from typing import Any, Dict, List, Optional

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

    if ocr_text and not _is_ocr_noise(ocr_text):
        parts.append(f"[OCR] {ocr_text}")

    if visual_context and need_visual:
        parts.append(f"[Visual] {visual_context}")

    return " ".join(parts) if parts else str(payload)


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
    """
    if not chunks:
        return []

    merged = []
    current = dict(chunks[0])
    current_payload = dict(current.get("payload", current))

    for nxt in chunks[1:]:
        if _are_adjacent(current, nxt):
            nxt_payload = nxt.get("payload", nxt)
            # Merge text fields.
            for field in ("transcript", "ocr_text", "visual_context"):
                a_text = (current_payload.get(field) or "").strip()
                b_text = (nxt_payload.get(field) or "").strip()
                if a_text and b_text:
                    current_payload[field] = f"{a_text} {b_text}"
                elif b_text:
                    current_payload[field] = b_text
            # Keep the earliest timestamp.
            if _get_timestamp(nxt) < _get_timestamp(current):
                current_payload["timestamp"] = nxt_payload.get("timestamp", 0)
        else:
            # Flush current.
            if "payload" in current:
                current["payload"] = current_payload
            else:
                current.update(current_payload)
            merged.append(current)
            current = dict(nxt)
            current_payload = dict(current.get("payload", current))

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

        Args:
            reranked_results: Output of the E3 reranker (list of dicts).
            is_lecture_wide: When True, sample across the full timeline
                instead of taking only top-ranked chunks.
            need_visual: When True, include [Visual] labels in context.
            char_budget: Maximum character budget (overrides default).

        Returns:
            Assembled context string ready for the LLM prompt.
        """
        budget = char_budget if char_budget is not None else self._default_budget

        if not reranked_results:
            return ""

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

        # 5. Build text for each chunk, filter by OCR noise.
        context_parts: List[str] = []
        seen_texts: set = set()
        total_chars = 0

        for entry in merged:
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
                    context_parts.append(truncated)
                    logger.debug(
                        "ContextBuilder: First chunk truncated from %d to %d chars "
                        "(budget=%d).",
                        len(text), len(truncated), budget,
                    )
                break
            context_parts.append(text)
            total_chars += len(text) + len(_CHUNK_SEPARATOR)

        final = _CHUNK_SEPARATOR.join(context_parts)
        logger.info(
            "ContextBuilder: %d input → %d merged → %d used, %d chars (lecture_wide=%s)",
            len(reranked_results), len(merged), len(context_parts), len(final), is_lecture_wide,
        )
        return final
