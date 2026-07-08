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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from config import CloudSettings
from schemas.chunk import MultimodalChunk
from schemas.segment import LectureSegment
from schemas.entity import Entity
from schemas.enums import EntityType

logger = logging.getLogger(__name__)

# Allowed entity types for validation.
ALLOWED_ENTITY_TYPES = {t.value for t in EntityType}

# Entity extraction prompt for Qwen2.5-7B-Instruct.
ENTITY_EXTRACTION_PROMPT = (
    "Extract educational entities from the following lecture content.\n\n"
    "Rules:\n"
    "* Return ONLY valid JSON\n"
    "* Use only allowed entity types: Concept, Algorithm, Formula, Code, Diagram, LectureSegment\n"
    "* Ignore generic words\n"
    "* Extract algorithms, formulas, concepts, diagrams and code constructs\n\n"
    "Segment title: {title}\n\n"
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


def _extract_entities_batch(
    segment_prompts: List[str],
    llm: Any,
) -> List[List[Dict[str, str]]]:
    """
    Batch extract entities from multiple segments using the LLM backend.

    Args:
        segment_prompts: List of formatted extraction prompts.
        llm: LLM backend instance (TransformersBackend or compatible).

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
        entities = _parse_entity_json(raw_text)
        all_entities.append(entities)

    return all_entities


def _parse_entity_json(raw_text: str) -> List[Dict[str, str]]:
    """
    Parse LLM output into a list of entity dicts.

    Handles common LLM JSON formatting issues:
    - Extracts JSON array from surrounding text.
    - Validates entity types against allowed enum values.
    - Filters out malformed entries.
    """
    # Try to extract JSON array from the response.
    text = raw_text.strip()
    text = text.replace("```json", "").replace("```", "")

    start = text.find("[")
    if start == -1:
        logger.warning("A8: Could not find JSON array in LLM output: %s", text[:200])
        logger.warning("A8 RAW OUTPUT (first 5000 chars):\n%s", raw_text[:5000])
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
        logger.warning("A8: Unclosed JSON array in LLM output: %s", text[:200])
        logger.warning("A8 RAW OUTPUT (first 5000 chars):\n%s", raw_text[:5000])
        return []

    json_str = text[start:end + 1]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("A8: JSON parse error: %s — text: %s", exc, json_str[:200])
        logger.warning("A8 RAW OUTPUT (first 5000 chars):\n%s", raw_text[:5000])
        return []

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

        # Validate entity type.
        if entity_type not in ALLOWED_ENTITY_TYPES:
            logger.warning("A8: Skipping entity with invalid type: %s (%s)", name, entity_type)
            continue

        valid_entities.append({"name": name, "type": entity_type})

    return valid_entities


def _deduplicate_entities(
    raw_entities: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Deduplicate entities by normalized name, keeping first occurrence."""
    seen: Set[str] = set()
    unique: List[Dict[str, str]] = []
    for ent in raw_entities:
        normalized = ent["name"].strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(ent)
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

    # Extract entities.
    raw_entities: List[Dict[str, str]] = []

    if extractor_fn is not None:
        # Legacy callable interface.
        for title, text in segment_texts:
            extracted = extractor_fn(text, title)
            for name, etype in extracted:
                if etype in ALLOWED_ENTITY_TYPES:
                    raw_entities.append({"name": name, "type": etype})
                else:
                    raw_entities.append({"name": name, "type": "Concept"})
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

        # Build prompts.
        prompts = [
            ENTITY_EXTRACTION_PROMPT.format(title=title, content=text[:3000])
            for title, text in segment_texts
        ]

        # Batch extract.
        batch_results = _extract_entities_batch(prompts, llm)
        for entities_list in batch_results:
            raw_entities.extend(entities_list)

    # Add segment titles as LectureSegment entities.
    for seg in segments_data:
        raw_entities.append({"name": seg["title"], "type": EntityType.LectureSegment.value})

    # Deduplicate.
    unique_entities = _deduplicate_entities(raw_entities)

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
