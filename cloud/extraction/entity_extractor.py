# cloud/extraction/entity_extractor.py
# Stage A8 — Entity Extraction.
#
# Extracts named entities from segments and their constituent chunks
# using Qwen2.5-7B-Instruct via Transformers for structured extraction.
#
# Input:  segments.json + multimodal_chunks.json
# Output: entities.json
# Environment: Kaggle GPU only.
#
# CRITICAL: Output must validate against Entity schema.
# Only allowed types: Concept, Algorithm, Formula, Code, Diagram, LectureSegment.
# entity_id format: {lecture_id}_entity_{number:06d}

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from config import CloudSettings
from schemas.chunk import MultimodalChunk
from schemas.segment import LectureSegment
from schemas.entity import Entity
from schemas.enums import EntityType
from cloud.utils.diagnostics import ExtractionStats
from cloud.utils.json_repair import parse_json_array
from cloud.extraction.window_builder import build_extraction_windows

logger = logging.getLogger(__name__)

# Allowed entity types for validation.
ALLOWED_ENTITY_TYPES = {t.value for t in EntityType}

# Known LLM-emitted type aliases -> canonical EntityType value.
# Variable: generic CS term ("the variable x") — a real Concept.
# TextElement: slide/OCR artifact name — has no dedicated type, preserved as
# a Concept rather than dropped (the A8 contract never discards information).
_ENTITY_TYPE_ALIASES = {
    "variable": "Concept",
    "textelement": "Concept",
}

# Entity extraction prompt for Qwen2.5-7B-Instruct.
ENTITY_EXTRACTION_PROMPT = (
    "Extract educational entities from the following lecture content.\n\n"
    "Capture meaningful technical entities across categories:\n"
    "* Physical components & hardware (e.g. CPU, Cache, Coil, Core, Sensor, Register)\n"
    "* Scientific & technical principles (e.g. Alternating Current, Locality, Mutual Induction, Virtual Memory)\n"
    "* Formulas, laws, quantities & variables (e.g. Faraday's Law, EMF, Voltage, Big-O, Flux)\n"
    "* Devices, systems & tools (e.g. Transformer, Operating System, Oscilloscope, Database)\n"
    "* Algorithms, methods & architectures (e.g. BFS, Dijkstra, Gradient Descent, Paging)\n"
    "* Key technical concepts & state terms (e.g. Page Fault, Deadlock, Sinusoidal Waveform)\n\n"
    "Rules:\n"
    "* Return ONLY valid JSON array: [{{\"name\":\"entity name\",\"type\":\"EntityType\"}}]\n"
    "* Use only allowed entity types: Concept, Algorithm, Formula, Code, Diagram, LectureSegment\n"
    "* Map physical components, principles, devices, systems, and technical terms to 'Concept'\n"
    "* Map mathematical laws, equations, and expressions to 'Formula'\n"
    "* Map algorithms, procedures, and methods to 'Algorithm'\n"
    "* Map code blocks, functions, and data structures to 'Code'\n"
    "* Map visual figures, charts, and illustrations to 'Diagram'\n"
    "* Do NOT invent entities or extrapolate beyond the provided text\n"
    "* Ignore generic non-educational conversational filler\n\n"
    "Lecture Window: {title}\n\n"
    "Content:\n{content}\n\n"
    "Output format:\n"
    '[{{"name":"entity name","type":"EntityType"}}]\n\n'
    "IMPORTANT:\n"
    "* Return ONLY a JSON array.\n"
    "* Do NOT include explanations.\n"
    "* Do NOT include markdown.\n"
    "* Do NOT include code fences.\n"
    "* Do NOT include commentary before or after the JSON.\n"
    "* If no entities exist, return exactly [].\n\n"
    "JSON output:"
)


def _load_llm() -> Any:
    """
    Load Qwen2.5-7B-Instruct via Transformers from local Kaggle path for entity extraction.

    Separated for test mockability.
    """
    from cloud.orchestration.llm_backend import TransformersBackend
    from config import CloudSettings

    settings = CloudSettings()
    backend = TransformersBackend(
        model_path=settings.QWEN_TEXT_MODEL_PATH,
        max_new_tokens=1024,
    )
    logger.info("A8: Loaded Qwen2.5-7B-Instruct for entity extraction from %s.", settings.QWEN_TEXT_MODEL_PATH)
    return backend


def normalize_entity_type(raw_type: str) -> str:
    """Map an LLM-emitted type string onto the closed EntityType ontology.

    Deterministic normalization order:
    1. Exact canonical match -> unchanged.
    2. Case/whitespace variant of a canonical type -> canonical form.
    3. Known semantic alias (Variable, TextElement) -> Concept.
    4. Anything else -> Concept (A8 contract: invalid types are mapped, never
       dropped, so no entity information is lost).
    """
    t = raw_type.strip()
    if t in ALLOWED_ENTITY_TYPES:
        return t
    lower = t.lower()
    for allowed in ALLOWED_ENTITY_TYPES:
        if allowed.lower() == lower:
            return allowed
    return _ENTITY_TYPE_ALIASES.get(lower, "Concept")


def _extract_entities_batch(
    segment_prompts: List[str],
    llm: Any,
    stats: Optional[ExtractionStats] = None,
) -> List[List[Dict[str, str]]]:
    """
    Batch extract entities from multiple segments using the LLM backend.

    Args:
        segment_prompts: List of formatted extraction prompts.
        llm: LLM backend instance (TransformersBackend or compatible).
        stats: Optional ExtractionStats to record parse/accept counters.

    Returns:
        List of entity lists (one list per segment).
    """
    outputs = llm.generate(
        segment_prompts,
        temperature=0.1,
        max_tokens=1024,
        top_p=0.95,
    )

    all_entities: List[List[Dict[str, str]]] = []
    for raw_text in outputs:
        entities = _parse_entity_json(raw_text, stats=stats)
        all_entities.append(entities)

    return all_entities


def _parse_entity_json(
    raw_text: str,
    stats: Optional[ExtractionStats] = None,
) -> List[Dict[str, str]]:
    """
    Parse LLM output into a list of entity dicts.

    Uses the shared json_repair pipeline (fences -> balanced array -> strict
    load -> safe escape repair, so e.g. W\\_q no longer discards a segment's
    entities). When JSON cannot be parsed even after repair, falls back to
    regex {name, type} extraction so recoverable content is never dropped.
    """
    parsed, status = parse_json_array(raw_text)
    if stats is not None:
        stats.requests += 1
        stats.record_parse(status)

    if parsed is None:
        logger.warning("A8: JSON parse %s; falling back to regex extraction: %s", status, raw_text[:200])
        # Preserve the pre-centralization regex fallback: pull {name, type} pairs.
        entities: List[Dict[str, str]] = []
        for match in re.finditer(
            r'\{[^{}]*"name"\s*:\s*"([^"]+)"\s*,\s*"type"\s*:\s*"([^"]+)"[^{}]*\}',
            raw_text,
        ):
            entities.append({"name": match.group(1), "type": normalize_entity_type(match.group(2))})
        if not entities:
            logger.warning("A8 RAW OUTPUT (first 5000 chars):\n%s", raw_text[:5000])
            return []
        if stats is not None:
            stats.accepted += len(entities)
        return entities

    if not isinstance(parsed, list):
        logger.warning("A8: Expected list, got %s", type(parsed).__name__)
        return []

    # Validate and filter entities.
    valid_entities: List[Dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = item.get("name", "").strip()
        entity_type = item.get("type", "").strip()

        if not name or not entity_type:
            continue

        # Deterministic type normalization (unknown -> Concept, never dropped).
        normalized = normalize_entity_type(entity_type)
        if normalized != entity_type:
            if stats is not None:
                stats.normalized += 1
            logger.warning("A8: Normalized entity type %r -> %r (%s)", entity_type, normalized, name)

        valid_entities.append({"name": name, "type": normalized})

    if stats is not None:
        stats.accepted += len(valid_entities)
    return valid_entities


def _deduplicate_entities(
    raw_entities: List[Dict[str, str]],
    stats: Optional[ExtractionStats] = None,
) -> List[Dict[str, str]]:
    """Deduplicate entities by normalized name, keeping first occurrence."""
    seen: Set[str] = set()
    unique: List[Dict[str, str]] = []
    for ent in raw_entities:
        normalized = ent["name"].strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(ent)
        elif stats is not None:
            stats.duplicates += 1
    return unique


def extract_entities(
    lecture_id: str,
    cloud_settings: CloudSettings | None = None,
    extractor_fn: Optional[Callable] = None,
    llm_loader: Optional[Callable] = None,
) -> List[Entity]:
    """
    Extract entities from segments and their constituent chunks using
    Qwen2.5-7B-Instruct via Transformers.

    Args:
        lecture_id: Unique lecture identifier.
        cloud_settings: Injected CloudSettings.
        extractor_fn: Optional callable(chunks_text, segment_title) -> List[Tuple[str, str]].
                      Legacy interface for backward compatibility.
        llm_loader: Optional callable returning an LLM backend instance (for tests).

    Returns:
        List of validated Entity instances.

    Raises:
        FileNotFoundError: If required input files are missing.
        ValueError: If zero entities are extracted.
    """
    settings = cloud_settings or CloudSettings()
    lecture_dir = Path(settings.lecture_dir(lecture_id))

    segments_path = lecture_dir / "segments.json"
    chunks_path = lecture_dir / "multimodal_chunks.json"

    if not segments_path.exists():
        raise FileNotFoundError(f"A8: segments.json not found: {segments_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"A8: multimodal_chunks.json not found: {chunks_path}")

    segments_data = json.loads(segments_path.read_text(encoding="utf-8"))
    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))

    # Validate inputs.
    for seg in segments_data:
        LectureSegment(**seg)
    for chunk in chunks_data:
        MultimodalChunk(**chunk)

    chunk_lookup = {c["chunk_id"]: c for c in chunks_data}

    # Build combined text per segment.
    segment_texts: List[Tuple[str, str]] = []  # (title, combined_text)
    for seg in segments_data:
        segment_title = seg["title"]
        combined = ""
        for cid in seg["chunks"]:
            if cid in chunk_lookup:
                c = chunk_lookup[cid]
                combined += c["transcript"] + " " + c["visual_context"] + " " + c["ocr_text"] + " "
        segment_texts.append((segment_title, combined.strip()))

    # Extraction diagnostics (pure logging — never affects pipeline output).
    stats = ExtractionStats()

    # Extract entities.
    raw_entities: List[Dict[str, str]] = []

    if extractor_fn is not None:
        # Legacy callable interface.
        for title, text in segment_texts:
            extracted = extractor_fn(text, title)
            for name, etype in extracted:
                raw_entities.append({"name": name, "type": normalize_entity_type(etype)})
    else:
        # Use LLM batch extraction.
        if llm_loader is not None:
            llm = llm_loader()
        else:
            try:
                llm = _load_llm()
            except (ImportError, Exception) as exc:
                logger.error("A8: Failed to load LLM backend: %s", exc)
                raise

        # Determine token counter for budgeting.
        token_counter = None
        if hasattr(llm, "count_tokens") and callable(getattr(llm, "count_tokens")):
            try:
                test_cnt = llm.count_tokens("test")
                if isinstance(test_cnt, int):
                    token_counter = llm.count_tokens
            except Exception:
                token_counter = None

        if token_counter is None:
            # Deterministic mock counter for tests / mocks
            token_counter = lambda t: max(1, len(t.split()))

        token_budget = getattr(settings, "EXTRACTION_WINDOW_TOKEN_BUDGET", 2000)
        overlap_chunks = getattr(settings, "EXTRACTION_WINDOW_OVERLAP_CHUNKS", 1)

        # Build token-budgeted overlapping extraction windows.
        windows, coverage_metrics = build_extraction_windows(
            chunks_data,
            token_budget=token_budget,
            overlap_chunks=overlap_chunks,
            token_counter=token_counter,
        )
        logger.info(
            "A8: Built %d extraction windows covering %d/%d chunks (%.1f%% coverage, %d overlaps).",
            len(windows),
            coverage_metrics["unique_covered_chunks"],
            coverage_metrics["total_source_chunks"],
            coverage_metrics["coverage_percentage"],
            coverage_metrics["overlap_count"],
        )

        # Map each window to relevant segment title(s) if available.
        chunk_to_seg = {}
        for seg in segments_data:
            for cid in seg.get("chunks", []):
                chunk_to_seg[cid] = seg.get("title", "")

        prompts = []
        for w in windows:
            titles = [chunk_to_seg[cid] for cid in w.chunk_ids if cid in chunk_to_seg and chunk_to_seg[cid]]
            unique_titles = list(dict.fromkeys(titles))
            w_title = " | ".join(unique_titles) if unique_titles else f"{w.start_time:.1f}s - {w.end_time:.1f}s"
            prompts.append(ENTITY_EXTRACTION_PROMPT.format(title=w_title, content=w.text))

        # Batch extract.
        batch_results = _extract_entities_batch(prompts, llm, stats=stats)
        for entities_list in batch_results:
            raw_entities.extend(entities_list)

    # Add segment titles as LectureSegment entities.
    for seg in segments_data:
        raw_entities.append({"name": seg["title"], "type": EntityType.LectureSegment.value})

    # Deduplicate.
    unique_entities = _deduplicate_entities(raw_entities, stats=stats)

    logger.info(stats.report("A8"))

    if not unique_entities:
        raise ValueError("A8: Entity extraction produced zero entities.")

    # Build validated Entity objects with generated entity_ids.
    entities: List[Entity] = []
    for idx, ent in enumerate(unique_entities, start=1):
        entity_id = f"{lecture_id}_entity_{idx:06d}"
        entity = Entity(
            entity_id=entity_id,
            name=ent["name"],
            type=EntityType(ent["type"]),
        )
        entities.append(entity)

    # Write entities.json.
    entities_path = lecture_dir / "entities.json"
    entities_path.write_text(
        json.dumps([e.model_dump() for e in entities], indent=2),
        encoding="utf-8",
    )
    logger.info("A8: entities.json written (%d entities) to %s", len(entities), entities_path)
    return entities
