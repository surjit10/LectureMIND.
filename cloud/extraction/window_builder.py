# cloud/extraction/window_builder.py
# Chunk-Aware Multimodal Extraction Window Builder.
#
# Constructs token-budgeted, overlapping extraction windows from MultimodalChunks
# while strictly guaranteeing 100% source-chunk coverage.
#
# Preserves all three modalities:
# 1. Whisper Spoken Audio Transcript
# 2. PaddleOCR Slide Text and Equations
# 3. Qwen2-VL Visual Context
#
# Environment: Kaggle GPU / Local testing compatible.

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Default window configuration:
# 2000 content tokens leaves ~1072 tokens for system prompt and generation within
# TransformersBackend's 3072 input max_length ceiling.
DEFAULT_WINDOW_TOKEN_BUDGET = 2000
DEFAULT_CHUNK_OVERLAP = 1


@dataclass
class ExtractionWindow:
    """Represents a coherent temporal window of multimodal chunks for LLM extraction."""
    window_id: str
    chunk_ids: List[str]
    start_time: float
    end_time: float
    text: str
    token_count: int
    chunks: List[Dict[str, Any]] = field(default_factory=list)


def clean_visual_context(visual_text: str) -> str:
    """
    Conservatively clean visual context to remove generic presentation boilerplate
    while strictly preserving all technical terminology, formulas, equations,
    diagram labels, component names, and numbers.
    """
    if not visual_text:
        return ""

    text = visual_text.strip()

    # Remove generic structural slide headers only
    text = re.sub(
        r"###\s*Educational\s+Slide\s+Description\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Remove boilerplate background descriptions (e.g. '#### Background - Blue grid background')
    text = re.sub(
        r"####\s*Background\s*-\s*[^\n]+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Normalize excessive empty lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def format_chunk_multimodal(chunk: Dict[str, Any]) -> str:
    """
    Format a single multimodal chunk with distinct, labeled modalities.
    Preserves exact chunk_id and timestamp for provenance.
    """
    chunk_id = chunk.get("chunk_id", "unknown_chunk")
    timestamp = float(chunk.get("timestamp", 0.0))
    transcript = (chunk.get("transcript") or "").strip()
    ocr_text = (chunk.get("ocr_text") or "").strip()
    visual_raw = chunk.get("visual_context") or ""
    visual_clean = clean_visual_context(visual_raw)

    sections = [f"--- [Chunk: {chunk_id} | Timestamp: {timestamp:.2f}s] ---"]

    if transcript:
        sections.append(f"SPOKEN TRANSCRIPT:\n{transcript}")
    else:
        sections.append("SPOKEN TRANSCRIPT: (None)")

    if ocr_text:
        sections.append(f"SLIDE / OCR TEXT:\n{ocr_text}")

    if visual_clean:
        sections.append(f"VISUAL CONTEXT:\n{visual_clean}")

    return "\n\n".join(sections)


def build_extraction_windows(
    chunks: List[Dict[str, Any]],
    token_budget: int = DEFAULT_WINDOW_TOKEN_BUDGET,
    overlap_chunks: int = DEFAULT_CHUNK_OVERLAP,
    token_counter: Optional[Callable[[str], int]] = None,
) -> Tuple[List[ExtractionWindow], Dict[str, Any]]:
    """
    Construct chunk-aware, token-budgeted, overlapping extraction windows.

    Guarantees:
    1. Complete source-chunk coverage: Every chunk is present in >= 1 window (source_chunk_ids <= covered_chunk_ids).
    2. Modality preservation: Transcript, OCR, and Visual context are preserved.
    3. Token safety: Window size is managed within token_budget using the provided token_counter.
    4. Adaptive scaling: 1 window for small lectures, N windows for long lectures.

    Args:
        chunks: List of validated MultimodalChunk dictionaries.
        token_budget: Maximum tokens of content per window (default 2000).
        overlap_chunks: Number of chunks to overlap between adjacent windows (default 1).
        token_counter: Callable(text) -> int for exact tokenizer counts.
                       Must be provided; production uses model tokenizer, tests use injected mock.

    Returns:
        Tuple of (list of ExtractionWindows, coverage metrics dict).

    Raises:
        ValueError: If chunks list is empty or token_counter is missing.
        RuntimeError: If source chunk coverage is less than 100%.
    """
    if not chunks:
        raise ValueError("Cannot build extraction windows from empty chunks list.")

    if token_counter is None:
        raise ValueError("token_counter callable must be provided for production token budgeting.")

    # Pre-format each chunk and compute exact token sizes
    formatted_chunks: List[Tuple[Dict[str, Any], str, int]] = []
    for c in chunks:
        fmt = format_chunk_multimodal(c)
        tok_cnt = token_counter(fmt)
        formatted_chunks.append((c, fmt, tok_cnt))

    total_chunks = len(formatted_chunks)
    windows: List[ExtractionWindow] = []
    start_idx = 0
    window_num = 1

    while start_idx < total_chunks:
        curr_chunks: List[Dict[str, Any]] = []
        curr_texts: List[str] = []
        curr_tokens = 0
        end_idx = start_idx

        while end_idx < total_chunks:
            c, fmt, tok_cnt = formatted_chunks[end_idx]

            # If adding this chunk exceeds token_budget and we already have at least one chunk:
            if curr_tokens + tok_cnt > token_budget and len(curr_chunks) > 0:
                break

            curr_chunks.append(c)
            curr_texts.append(fmt)
            curr_tokens += tok_cnt
            end_idx += 1

        window_id = f"window_{window_num:03d}"
        chunk_ids = [c["chunk_id"] for c in curr_chunks]
        start_time = float(curr_chunks[0].get("timestamp", 0.0))
        end_time = float(curr_chunks[-1].get("timestamp", 0.0))
        combined_text = "\n\n".join(curr_texts)

        window = ExtractionWindow(
            window_id=window_id,
            chunk_ids=chunk_ids,
            start_time=start_time,
            end_time=end_time,
            text=combined_text,
            token_count=curr_tokens,
            chunks=curr_chunks,
        )
        windows.append(window)
        window_num += 1

        # If we reached the end of all chunks, we are done
        if end_idx >= total_chunks:
            break

        # Slide forward with overlap:
        # Step forward by at least 1 chunk, preserving overlap_chunks
        step = max(1, (end_idx - start_idx) - overlap_chunks)
        next_start_idx = start_idx + step

        # Guard against zero progress
        if next_start_idx <= start_idx:
            next_start_idx = start_idx + 1

        start_idx = next_start_idx

    # --- Validation: 100% Source-Chunk Coverage Guarantee ---
    all_source_ids: Set[str] = {c["chunk_id"] for c in chunks}
    covered_ids: Set[str] = {cid for w in windows for cid in w.chunk_ids}
    uncovered_ids = all_source_ids - covered_ids
    overlap_count = sum(len(w.chunk_ids) for w in windows) - len(covered_ids)

    coverage_pct = (len(covered_ids) / len(all_source_ids)) * 100.0 if all_source_ids else 0.0

    metrics = {
        "total_source_chunks": len(all_source_ids),
        "unique_covered_chunks": len(covered_ids),
        "uncovered_chunks": len(uncovered_ids),
        "coverage_percentage": round(coverage_pct, 2),
        "overlap_count": overlap_count,
        "windows_created": len(windows),
        "uncovered_chunk_ids": list(uncovered_ids),
    }

    # Coverage assertion: all source chunk ids must be covered
    if not (all_source_ids <= covered_ids):
        raise RuntimeError(
            f"Extraction window coverage violation: {len(uncovered_ids)} chunks uncovered: {uncovered_ids}"
        )

    logger.info(
        "WindowBuilder: Created %d windows covering %d/%d chunks (%.1f%% coverage, %d overlaps).",
        len(windows),
        len(covered_ids),
        len(all_source_ids),
        coverage_pct,
        overlap_count,
    )

    return windows, metrics
