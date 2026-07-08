# cloud/training/triplet_generator.py
# Stage B1 — Triplet Generation for reranker fine-tuning.
#
# Generates (query, positive, negative) triplets from segments + chunks.
# Uses Qwen2.5-7B-Instruct via Transformers to generate 3-5 realistic student
# questions per segment, then pairs with positive/negative chunks.
#
# Input: segments.json + multimodal_chunks.json
# Output: cloud_runtime/lectures/{lecture_id}/triplets.json
# Environment: Kaggle GPU only.
#
# Does NOT use entities.json or relations.json.

import json
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import CloudSettings
from schemas.triplet import RerankerTriplet

logger = logging.getLogger(__name__)

# Query synthesis prompt for Qwen2.5-7B-Instruct.
QUERY_SYNTHESIS_PROMPT = (
    "You are a university student studying this lecture topic.\n"
    "Generate exactly {num_questions} realistic questions a student would ask about this content.\n\n"
    "Topic: {title}\n\n"
    "Content:\n{content}\n\n"
    "Rules:\n"
    "* Questions should be specific and educational\n"
    "* Include a mix of conceptual and application questions\n"
    "* Return ONLY valid JSON array of strings\n"
    "* Each question should be 5-15 words\n\n"
    "IMPORTANT:\n"
    "* Return ONLY a JSON array.\n"
    "* Do NOT include explanations.\n"
    "* Do NOT include markdown.\n"
    "* Do NOT include code fences.\n"
    "* Do NOT include commentary before or after the JSON.\n"
    "* If you cannot generate questions, return exactly [].\n\n"
    "JSON output:"
)


def _build_segment_chunk_map(
    segments: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build a mapping from segment_id → list of chunk dicts.

    Uses the chunks list from segments.json to group multimodal chunks.
    """
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    seg_chunks: Dict[str, List[Dict[str, Any]]] = {}

    for seg in segments:
        seg_id = seg["segment_id"]
        seg_chunks[seg_id] = []
        for cid in seg.get("chunks", []):
            if cid in chunk_lookup:
                seg_chunks[seg_id].append(chunk_lookup[cid])

    return seg_chunks


def _chunk_to_text(chunk: Dict[str, Any]) -> str:
    """Combine chunk fields into a single text for triplet purposes."""
    return (
        chunk.get("transcript", "")
        + "\n"
        + chunk.get("visual_context", "")
        + "\n"
        + chunk.get("ocr_text", "")
    ).strip()


def _load_llm() -> Any:
    """
    Load Qwen2.5-7B-Instruct via Transformers from local Kaggle path for query synthesis.

    Separated for test mockability.
    """
    from cloud.orchestration.llm_backend import TransformersBackend
    from config import CloudSettings

    settings = CloudSettings()
    backend = TransformersBackend(
        model_path=settings.QWEN_TEXT_MODEL_PATH,
        max_new_tokens=512,
    )
    logger.info("B1: Loaded Qwen2.5-7B-Instruct for query synthesis from %s.", settings.QWEN_TEXT_MODEL_PATH)
    return backend


def _synthesize_queries_batch(
    segment_infos: List[Dict[str, str]],
    llm: Any,
    num_questions: int = 4,
) -> List[List[str]]:
    """
    Batch synthesize student questions for multiple segments using the LLM backend.

    Args:
        segment_infos: List of dicts with 'title' and 'content' keys.
        llm: LLM backend instance (TransformersBackend or compatible).
        num_questions: Number of questions to generate per segment (3-5).

    Returns:
        List of question lists (one per segment).
    """
    prompts = [
        QUERY_SYNTHESIS_PROMPT.format(
            num_questions=num_questions,
            title=info["title"],
            content=info["content"][:1500],
        )
        for info in segment_infos
    ]

    outputs = llm.generate(
        prompts,
        temperature=0.7,
        max_tokens=512,
        top_p=0.9,
    )

    all_queries: List[List[str]] = []
    for raw_text in outputs:
        queries = _parse_queries_json(raw_text)
        all_queries.append(queries)

    return all_queries


def _parse_queries_json(raw_text: str) -> List[str]:
    """
    Parse LLM output into a list of query strings.

    Handles common JSON formatting issues from LLM output.
    """
    text = (
        raw_text.strip()
        .replace("```json", "")
        .replace("```", "")
    )

    start = text.find("[")
    if start == -1:
        logger.warning("B1: Could not find JSON array in LLM output: %s", text[:200])
        logger.warning("B1 RAW OUTPUT (first 5000 chars):\n%s", raw_text[:5000])
        return []

    depth = 0
    in_string = False
    escape = False
    end = -1

    for i in range(start, len(text)):
        char = text[i]
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    end = i
                    break

    if end == -1:
        logger.warning("B1: Unclosed JSON array in LLM output: %s", text[:200])
        logger.warning("B1 RAW OUTPUT (first 5000 chars):\n%s", raw_text[:5000])
        return []

    json_str = text[start:end + 1]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("B1: JSON parse error: %s — text: %s", exc, json_str[:200])
        logger.warning("B1 RAW OUTPUT (first 5000 chars):\n%s", raw_text[:5000])
        return []

    if not isinstance(parsed, list):
        return []

    # Filter to valid non-empty strings.
    queries = [str(q).strip() for q in parsed if isinstance(q, str) and str(q).strip()]
    return queries


def generate_triplets(
    lecture_id: str,
    cloud_settings: CloudSettings | None = None,
    query_generator: Any = None,
    llm_loader: Optional[Callable] = None,
    seed: int = 42,
    questions_per_segment: int = 4,
) -> List[RerankerTriplet]:
    """
    Generate reranker training triplets from segments and chunks.

    For each segment:
    1. Synthesize 3-5 realistic student questions using Qwen2.5-7B-Instruct.
    2. Pair each question with a positive chunk (same segment) and
       negative chunk (different segment).

    Args:
        lecture_id: Unique lecture identifier.
        cloud_settings: Injected CloudSettings.
        query_generator: Optional callable(chunk, title) -> str. Legacy interface.
        llm_loader: Optional callable returning an LLM backend instance (for tests).
        seed: Random seed for reproducibility.
        questions_per_segment: Number of questions per segment (3-5).

    Returns:
        List of validated RerankerTriplet instances.
    """
    settings = cloud_settings or CloudSettings()
    lecture_dir = Path(settings.lecture_dir(lecture_id))
    rng = random.Random(seed)

    # Load inputs.
    segments_path = lecture_dir / "segments.json"
    chunks_path = lecture_dir / "multimodal_chunks.json"

    if not segments_path.exists():
        raise FileNotFoundError(f"B1: segments.json not found: {segments_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"B1: multimodal_chunks.json not found: {chunks_path}")

    segments = json.loads(segments_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

    if not segments:
        raise ValueError("B1: segments.json contains zero segments.")

    # Build segment → chunks map.
    seg_chunks = _build_segment_chunk_map(segments, chunks)
    seg_ids = [s["segment_id"] for s in segments]
    seg_title_map = {s["segment_id"]: s.get("title", "") for s in segments}
    all_chunks_by_seg = {sid: chunks_list for sid, chunks_list in seg_chunks.items()}

    # Generate queries.
    if query_generator is not None:
        # Legacy per-chunk query generator.
        segment_queries: Dict[str, List[str]] = {}
        for seg_id in seg_ids:
            pos_chunks = all_chunks_by_seg.get(seg_id, [])
            title = seg_title_map.get(seg_id, "")
            queries = []
            for pos_chunk in pos_chunks:
                q = query_generator(pos_chunk, title)
                if q and q.strip():
                    queries.append(q)
            segment_queries[seg_id] = queries
    else:
        # Use LLM batch query synthesis.
        if llm_loader is not None:
            llm = llm_loader()
        else:
            try:
                llm = _load_llm()
            except (ImportError, Exception) as exc:
                logger.warning("B1: LLM backend unavailable (%s), using fallback query generation.", exc)
                llm = None

        if llm is not None:
            # Build segment info for batch synthesis.
            segment_infos: List[Dict[str, str]] = []
            ordered_seg_ids: List[str] = []
            for seg_id in seg_ids:
                pos_chunks = all_chunks_by_seg.get(seg_id, [])
                if not pos_chunks:
                    continue
                title = seg_title_map.get(seg_id, "")
                content = " ".join(_chunk_to_text(c) for c in pos_chunks)
                segment_infos.append({"title": title, "content": content})
                ordered_seg_ids.append(seg_id)

            batch_queries = _synthesize_queries_batch(
                segment_infos, llm, num_questions=questions_per_segment,
            )

            segment_queries = {}
            for seg_id, queries in zip(ordered_seg_ids, batch_queries):
                segment_queries[seg_id] = queries if queries else [
                    f"What is {seg_title_map.get(seg_id, 'this topic')}?"
                ]
        else:
            # Fallback: heuristic query generation.
            segment_queries = {}
            for seg_id in seg_ids:
                pos_chunks = all_chunks_by_seg.get(seg_id, [])
                title = seg_title_map.get(seg_id, "")
                queries = []
                for pos_chunk in pos_chunks:
                    transcript = pos_chunk.get("transcript", "").strip()
                    if transcript:
                        queries.append(f"Explain: {transcript[:100]}")
                    else:
                        queries.append(f"What is {title}?")
                segment_queries[seg_id] = queries

    # Build triplets.
    triplets: List[RerankerTriplet] = []

    for seg_id in seg_ids:
        pos_chunks = all_chunks_by_seg.get(seg_id, [])
        if not pos_chunks:
            continue

        # Sibling segments for hard negatives.
        sibling_ids = [sid for sid in seg_ids if sid != seg_id]
        neg_pool = []
        for sid in sibling_ids:
            neg_pool.extend(all_chunks_by_seg.get(sid, []))

        if not neg_pool:
            continue

        queries = segment_queries.get(seg_id, [])
        if not queries:
            continue

        for query_text in queries:
            if not query_text.strip():
                continue

            # Pick a positive chunk from this segment.
            pos_chunk = rng.choice(pos_chunks)
            pos_text = _chunk_to_text(pos_chunk)

            # Pick a hard negative from sibling segments.
            neg_chunk = rng.choice(neg_pool)
            neg_text = _chunk_to_text(neg_chunk)

            if not pos_text.strip() or not neg_text.strip():
                logger.warning("B1: Skipping triplet with empty field.")
                continue

            # Validate against schema.
            triplet = RerankerTriplet(
                query=query_text,
                positive=pos_text,
                negative=neg_text,
            )
            triplets.append(triplet)

    if not triplets:
        raise ValueError("B1: Generated zero triplets.")

    # Write triplets.json
    output_path = lecture_dir / "triplets.json"
    output_path.write_text(
        json.dumps([t.model_dump() for t in triplets], indent=2),
        encoding="utf-8",
    )
    logger.info("B1: triplets.json written (%d triplets) to %s", len(triplets), output_path)

    return triplets
