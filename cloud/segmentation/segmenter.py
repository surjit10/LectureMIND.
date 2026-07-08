# cloud/segmentation/segmenter.py
# Stage A7 — Topic Segmentation.
#
# Groups multimodal chunks into coherent lecture segments using
# semantic embeddings (BAAI/bge-large-en-v1.5) for topic boundary detection
# and Qwen2.5-7B-Instruct for semantic title generation via Transformers.
#
# Input:  cloud_runtime/lectures/{lecture_id}/multimodal_chunks.json
# Output: cloud_runtime/lectures/{lecture_id}/segments.json
#         cloud_runtime/lectures/{lecture_id}/chunk_segment_map.json
#         cloud_runtime/lectures/{lecture_id}/embeddings.npy (cached for B0)
#         cloud_runtime/lectures/{lecture_id}/embedding_ids.json (cached for B0)
#
# Environment: Kaggle GPU only.
#
# CRITICAL: Output must validate against LectureSegment and ChunkSegmentMap schemas.
# No extra fields. No renamed fields. No schema modifications.

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from config import CloudSettings, SharedSettings
from schemas.chunk import MultimodalChunk
from schemas.segment import LectureSegment
from schemas.chunk_map import ChunkSegmentMap

logger = logging.getLogger(__name__)

# Segmentation parameters.
MIN_SEGMENT_CHUNKS = 3
MAX_SEGMENT_CHUNKS = 30

# Import model paths from config (GPU memory optimization).
from config import CloudSettings
_cloud_settings_temp = CloudSettings()
EMBEDDING_MODEL = _cloud_settings_temp.BGE_MODEL_PATH

# Title generation prompt.
TITLE_PROMPT = (
    "Generate a concise educational topic title.\n"
    "Maximum 8 words.\n"
    "Return ONLY the title.\n"
    "Do not return markdown.\n"
    "Do not return headings.\n"
    "Do not return explanations.\n"
    "Do not repeat slide labels.\n"
    "Do not return prefixes such as Title, Caption, Analysis, Objects and Concepts.\n\n"
    "Segment content:\n{content}"
)


def _combine_text(chunk: Dict[str, Any]) -> str:
    """Combine transcript, visual_context, and ocr_text into a single string."""
    return (
        chunk.get("transcript", "")
        + " "
        + chunk.get("visual_context", "")
        + " "
        + chunk.get("ocr_text", "")
    ).strip()


def _title_generation_text(chunk: Dict[str, Any]) -> str:
    return (
        chunk.get("transcript", "")
        + " "
        + chunk.get("ocr_text", "")
    ).strip()


def _load_embedding_model(model_name: str = EMBEDDING_MODEL) -> Any:
    """
    Load sentence-transformers embedding model for segmentation on CPU.

    Loads on CPU to avoid GPU memory conflicts with title generation.
    Separated for test mockability.
    """
    from sentence_transformers import SentenceTransformer
    from pathlib import Path

    if not Path(model_name).exists():
        raise FileNotFoundError(f"Model path does not exist: {model_name}")

    model = SentenceTransformer(model_name, device="cpu")
    logger.info("A7: Loaded embedding model '%s' on CPU.", model_name)
    return model


def _compute_embeddings(
    texts: List[str],
    model: Any,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute embeddings for a list of texts using the embedding model.

    Args:
        texts: List of combined chunk texts.
        model: SentenceTransformer model instance.
        batch_size: Batch size for encoding (T4/P100 optimized).

    Returns:
        np.ndarray of shape (N, 1024).
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    return np.array(embeddings, dtype=np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))


def _compute_adaptive_threshold(similarities: List[float]) -> float:
    """
    Compute an adaptive threshold for topic boundary detection.

    Uses mean - 1.0 * std of adjacent cosine similarities.
    This adapts to the lecture's natural topic coherence level.
    """
    if not similarities:
        return 0.5

    arr = np.array(similarities)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    threshold = mean - 1.0 * std

    # Clamp to reasonable range.
    threshold = max(0.3, min(0.85, threshold))
    logger.info(
        "A7: Adaptive threshold: mean=%.4f, std=%.4f, threshold=%.4f",
        mean, std, threshold,
    )
    return threshold


def _find_segment_boundaries(
    embeddings: np.ndarray,
    min_chunks: int = MIN_SEGMENT_CHUNKS,
    max_chunks: int = MAX_SEGMENT_CHUNKS,
) -> List[int]:
    """
    Find segment boundary indices using cosine similarity drops between
    adjacent chunk embeddings with an adaptive threshold.

    Returns a list of indices where new segments begin.
    Index 0 is always included (first segment starts at first chunk).
    """
    n = embeddings.shape[0]
    if n <= min_chunks:
        return [0]

    # Compute adjacent cosine similarities.
    adjacent_sims: List[float] = []
    for i in range(n - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        adjacent_sims.append(sim)

    # Compute adaptive threshold.
    threshold = _compute_adaptive_threshold(adjacent_sims)

    # Detect boundaries.
    boundaries = [0]
    segment_length = 1

    for i, sim in enumerate(adjacent_sims):
        chunk_idx = i + 1
        segment_length += 1

        # Boundary: similarity drop below threshold AND minimum size met,
        # OR maximum segment size reached.
        if (sim < threshold and segment_length >= min_chunks) or \
           segment_length >= max_chunks:
            boundaries.append(chunk_idx)
            segment_length = 1

    return boundaries


def _merge_small_segments(
    boundaries: List[int],
    total_chunks: int,
    min_chunks: int = MIN_SEGMENT_CHUNKS,
) -> List[int]:
    """
    Merge very small trailing segments back into their predecessor.

    If the last segment has fewer than min_chunks, remove that boundary.
    """
    if len(boundaries) <= 1:
        return boundaries

    merged = list(boundaries)

    # Check last segment size.
    while len(merged) > 1:
        last_start = merged[-1]
        last_size = total_chunks - last_start
        if last_size < min_chunks:
            merged.pop()
        else:
            break

    return merged


def _load_title_generator() -> Any:
    """
    Load Qwen2.5-7B-Instruct via Transformers for title generation.

    Uses local Kaggle model path (GPU memory optimization).
    Separated for test mockability.
    """
    from cloud.orchestration.llm_backend import TransformersBackend
    settings = CloudSettings()

    backend = TransformersBackend(
        model_path=settings.QWEN_TEXT_MODEL_PATH,
        max_new_tokens=32,
    )
    logger.info("A7: Loaded Qwen2.5-7B-Instruct for title generation from %s.", settings.QWEN_TEXT_MODEL_PATH)
    return backend


def _generate_titles_batch(
    segment_texts: List[str],
    title_generator: Any,
) -> List[str]:
    """
    Generate titles for multiple segments in a single batch using the LLM backend.

    Args:
        segment_texts: List of combined segment texts (one per segment).
        title_generator: LLM backend instance (TransformersBackend or compatible).

    Returns:
        List of title strings.
    """
    prompts = [
        TITLE_PROMPT.format(content=text[:1500])
        for text in segment_texts
    ]

    outputs = title_generator.generate(
        prompts,
        temperature=0.3,
        max_tokens=32,
        top_p=0.9,
    )

    titles = []
    for raw_title in outputs:
        raw_title = raw_title.strip()
        # Clean up: remove quotes, trailing periods, limit length.
        raw_title = raw_title.strip("\"'").rstrip(".")
        # Enforce max 8 words.
        words = raw_title.split()
        if len(words) > 8:
            raw_title = " ".join(words[:8])
        if not raw_title:
            raw_title = "Untitled Segment"
        titles.append(raw_title)

    return titles


def _generate_title_fallback(chunks: List[Dict[str, Any]]) -> str:
    """
    Fallback title generation when no title_generator is available.

    Uses the first chunk's transcript, truncated to a meaningful phrase.
    """
    if not chunks:
        return "Untitled Segment"

    first_transcript = chunks[0].get("transcript", "").strip()
    if first_transcript:
        title = first_transcript[:60].strip()
        if len(first_transcript) > 60:
            title = title.rsplit(" ", 1)[0]
        return title

    first_visual = chunks[0].get("visual_context", "").strip()
    if first_visual:
        return first_visual[:60].strip()

    return "Untitled Segment"


def segment(
    lecture_id: str,
    cloud_settings: CloudSettings | None = None,
    min_chunks: int = MIN_SEGMENT_CHUNKS,
    max_chunks: int = MAX_SEGMENT_CHUNKS,
    embedding_model_loader: Optional[Callable] = None,
    title_generator_loader: Optional[Callable] = None,
) -> List[LectureSegment]:
    """
    Execute semantic topic segmentation on multimodal chunks.

    Method:
    1. Combine transcript + visual_context + ocr_text per chunk.
    2. Generate embedding per chunk using BAAI/bge-large-en-v1.5.
    3. Compute cosine similarity between adjacent chunks.
    4. Detect topic boundary when similarity < adaptive threshold.
    5. Merge very small segments.
    6. Generate semantic title via Qwen2.5-7B-Instruct.

    Args:
        lecture_id: Unique lecture identifier.
        cloud_settings: Injected CloudSettings.
        min_chunks: Minimum chunks per segment.
        max_chunks: Maximum chunks per segment.
        embedding_model_loader: Optional callable returning embedding model (for tests).
        title_generator_loader: Optional callable returning title generator (for tests).

    Returns:
        List of validated LectureSegment instances.

    Raises:
        FileNotFoundError: If multimodal_chunks.json is missing.
        ValueError: If zero chunks are found.
    """
    settings = cloud_settings or CloudSettings()
    lecture_dir = Path(settings.lecture_dir(lecture_id))

    # Read multimodal_chunks.json (A6 output).
    chunks_path = lecture_dir / "multimodal_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"A7: multimodal_chunks.json not found: {chunks_path}")

    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))

    # Validate each chunk against the master contract.
    for chunk in chunks_data:
        MultimodalChunk(**chunk)

    if not chunks_data:
        raise ValueError("A7: multimodal_chunks.json contains zero chunks.")

    # Step 1: Combine texts.
    texts = [_combine_text(c) for c in chunks_data]

    # Step 2: Generate embeddings.
    if embedding_model_loader is not None:
        emb_model = embedding_model_loader()
    else:
        emb_model = _load_embedding_model()

    embeddings = _compute_embeddings(texts, emb_model)
    logger.info("A7: Computed embeddings for %d chunks, shape=%s", len(texts), embeddings.shape)

    # GPU Memory Cleanup: Release BGE embedding model (running on CPU, but good practice).
    try:
        del emb_model
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass
    except Exception:
        pass

    # Save embeddings for B0 reuse — eliminates duplicate BGE computation.
    try:
        ss = SharedSettings()
        if embeddings.shape[1] == ss.EMBEDDING_DIMENSION:
            chunk_ids = [c["chunk_id"] for c in chunks_data]
            np.save(str(lecture_dir / "embeddings.npy"), embeddings)
            embedding_ids = [
                {"row_index": idx, "chunk_id": cid}
                for idx, cid in enumerate(chunk_ids)
            ]
            (lecture_dir / "embedding_ids.json").write_text(
                json.dumps(embedding_ids, indent=2), encoding="utf-8",
            )
            logger.info("A7: Saved embeddings.npy + embedding_ids.json for B0 reuse.")
    except Exception as exc:
        logger.warning("A7: Could not cache embeddings for B0: %s", exc)

    # Step 3-4: Find boundaries using cosine similarity.
    boundaries = _find_segment_boundaries(
        embeddings,
        min_chunks=min_chunks,
        max_chunks=max_chunks,
    )

    # Step 5: Merge small segments.
    boundaries = _merge_small_segments(boundaries, len(chunks_data), min_chunks)
    logger.info("A7: Found %d segment boundaries: %s", len(boundaries), boundaries)

    # Build segment chunk groups.
    segment_chunk_groups: List[List[Dict[str, Any]]] = []
    for seg_idx, start_idx in enumerate(boundaries):
        end_idx = boundaries[seg_idx + 1] if seg_idx + 1 < len(boundaries) else len(chunks_data)
        segment_chunk_groups.append(chunks_data[start_idx:end_idx])

    # Step 6: Generate titles.
    if title_generator_loader is not None:
        title_gen = title_generator_loader()
        # Batch title generation.
        segment_texts = [
            " ".join(_title_generation_text(c) for c in group)[:1500]
            for group in segment_chunk_groups
        ]
        titles = _generate_titles_batch(segment_texts, title_gen)
        # GPU Memory Cleanup: Release Qwen2.5-7B-Instruct after title generation.
        try:
            del title_gen
            import gc
            gc.collect()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass
    else:
        # Try to load Qwen2.5-7B-Instruct; fallback to heuristic on import failure.
        try:
            title_gen = _load_title_generator()
            segment_texts = [
                " ".join(_title_generation_text(c) for c in group)[:1500]
                for group in segment_chunk_groups
            ]
            titles = _generate_titles_batch(segment_texts, title_gen)
        except (ImportError, Exception) as exc:
            logger.warning("A7: Title generator unavailable (%s), using fallback.", exc)
            titles = [_generate_title_fallback(group) for group in segment_chunk_groups]
        else:
            # GPU Memory Cleanup: Release Qwen2.5-7B-Instruct after title generation (loaded in try block).
            try:
                del title_gen
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
            except Exception:
                pass

    # Build segments.
    segments: List[LectureSegment] = []
    chunk_segment_mapping: Dict[str, str] = {}

    for seg_idx, (group, title) in enumerate(zip(segment_chunk_groups, titles)):
        seg_num = seg_idx + 1
        segment_id = f"seg_{seg_num:03d}"

        chunk_ids = [c["chunk_id"] for c in group]

        # Timestamps.
        start_ts = group[0]["timestamp"]
        end_ts = group[-1]["timestamp"]
        if end_ts <= start_ts:
            end_ts = start_ts + 0.1

        # Validate against schema.
        seg = LectureSegment(
            segment_id=segment_id,
            title=title,
            start=round(start_ts, 3),
            end=round(end_ts, 3),
            chunks=chunk_ids,
        )
        segments.append(seg)

        for cid in chunk_ids:
            chunk_segment_mapping[cid] = segment_id

    if not segments:
        raise ValueError("A7: Segmentation produced zero segments.")

    # Write segments.json.
    segments_path = lecture_dir / "segments.json"
    segments_path.write_text(
        json.dumps([s.model_dump() for s in segments], indent=2),
        encoding="utf-8",
    )
    logger.info(
        "A7: segments.json written (%d segments) to %s",
        len(segments), segments_path,
    )

    # Write chunk_segment_map.json.
    csm = ChunkSegmentMap(mapping=chunk_segment_mapping)
    csm_path = lecture_dir / "chunk_segment_map.json"
    csm_path.write_text(
        json.dumps(csm.mapping, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "A7: chunk_segment_map.json written (%d mappings) to %s",
        len(csm.mapping), csm_path,
    )

    return segments
