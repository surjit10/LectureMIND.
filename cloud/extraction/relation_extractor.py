# cloud/extraction/relation_extractor.py
# Stage A9 — Relation Extraction.
#
# Extracts relations between entities using Qwen2.5-7B-Instruct via Transformers.
# Uses entities.json, segments.json, and multimodal_chunks.json as context.
#
# Input:  entities.json + segments.json + multimodal_chunks.json
# Output: relations.json
# Environment: Kaggle GPU only.
#
# CRITICAL: Relations MUST use source_entity_id / target_entity_id.
# NEVER entity text strings.

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from config import CloudSettings
from schemas.chunk import MultimodalChunk
from schemas.segment import LectureSegment
from schemas.entity import Entity
from schemas.relation import Relation
from schemas.enums import RelationType

logger = logging.getLogger(__name__)

# Allowed relation types for validation.
ALLOWED_RELATION_TYPES = {t.value for t in RelationType}

# Relation extraction prompt for Qwen2.5-7B-Instruct.
RELATION_EXTRACTION_PROMPT = (
    "Given the entities and lecture context, "
    "extract valid educational relationships.\n\n"
    "Rules:\n"
    "* Use entity_id references only\n"
    "* Never use entity names as identifiers\n"
    "* Return only valid JSON\n"
    "* Do not invent entities\n"
    "* Do not create duplicate edges\n\n"
    "Allowed relations: PREREQUISITE_OF, INTRODUCED_BEFORE, USED_BY, "
    "DERIVED_FROM, VISUALIZED_BY, EXPLAINS\n\n"
    "Available entities:\n{entity_list}\n\n"
    "Lecture context:\n{context}\n\n"
    "Output format:\n"
    '[{{"source_entity_id":"...","relation":"RELATION_TYPE","target_entity_id":"..."}}]\n\n'
    "IMPORTANT:\n"
    "* Return ONLY a JSON array.\n"
    "* Do NOT include explanations.\n"
    "* Do NOT include markdown.\n"
    "* Do NOT include code fences.\n"
    "* Do NOT include commentary before or after the JSON.\n"
    "* If no relations exist, return exactly [].\n\n"
    "JSON output:"
)


def _load_llm() -> Any:
    """
    Load Qwen2.5-7B-Instruct via Transformers from local Kaggle path for relation extraction.

    Separated for test mockability.
    """
    from cloud.orchestration.llm_backend import TransformersBackend
    from config import CloudSettings

    settings = CloudSettings()
    backend = TransformersBackend(
        model_path=settings.QWEN_TEXT_MODEL_PATH,
        max_new_tokens=2048,
    )
    logger.info("A9: Loaded Qwen2.5-7B-Instruct for relation extraction from %s.", settings.QWEN_TEXT_MODEL_PATH)
    return backend


def _build_entity_list_str(entities: List[Dict[str, Any]]) -> str:
    """Build a formatted entity list string for the prompt."""
    lines = []
    for e in entities:
        lines.append(f'- {e["entity_id"]}: {e["name"]} ({e["type"]})')
    return "\n".join(lines)


def _build_segment_context(
    segments: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    max_chars: int = 3000,
) -> str:
    """Build combined context text from segments and chunks."""
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    context_parts: List[str] = []

    for seg in segments:
        seg_text = f"[{seg['title']}]: "
        for cid in seg["chunks"]:
            if cid in chunk_lookup:
                c = chunk_lookup[cid]
                seg_text += c["transcript"] + " " + c["visual_context"] + " " + c["ocr_text"] + " "
        context_parts.append(seg_text.strip())

    full_context = "\n".join(context_parts)
    return full_context[:max_chars]


def _extract_relations_batch(
    prompts: List[str],
    llm: Any,
) -> List[List[Dict[str, str]]]:
    """
    Batch extract relations using the LLM backend.

    Args:
        prompts: List of formatted relation extraction prompts.
        llm: LLM backend instance (TransformersBackend or compatible).

    Returns:
        List of relation lists (one per prompt).
    """
    outputs = llm.generate(
        prompts,
        temperature=0.1,
        max_tokens=2048,
        top_p=0.95,
    )

    all_relations: List[List[Dict[str, str]]] = []
    for raw_text in outputs:
        relations = _parse_relation_json(raw_text)
        all_relations.append(relations)

    return all_relations


def _parse_relation_json(
    raw_text: str,
) -> List[Dict[str, str]]:
    """
    Parse LLM output into a list of relation dicts.

    Validates relation types and ensures required fields exist.
    """
    text = raw_text.strip()
    text = text.replace("```json", "").replace("```", "")

    start = text.find("[")
    if start == -1:
        logger.warning("A9: Could not find JSON array in LLM output: %s", text[:200])
        logger.warning("A9 RAW OUTPUT (first 5000 chars):\n%s", raw_text[:5000])
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
        logger.warning("A9: Unclosed JSON array in LLM output: %s", text[:200])
        logger.warning("A9 RAW OUTPUT (first 5000 chars):\n%s", raw_text[:5000])
        return []

    json_str = text[start:end + 1]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as exc:
        logger.warning("A9: JSON parse error: %s — text: %s", exc, json_str[:200])
        logger.warning("A9 RAW OUTPUT (first 5000 chars):\n%s", raw_text[:5000])
        return []

    if not isinstance(parsed, list):
        logger.warning("A9: Expected list, got %s", type(parsed).__name__)
        return []

    # Validate and filter relations.
    valid_relations: List[Dict[str, str]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue

        src = item.get("source_entity_id", "").strip()
        rel = item.get("relation", "").strip()
        tgt = item.get("target_entity_id", "").strip()

        if not src or not rel or not tgt:
            continue

        if rel not in ALLOWED_RELATION_TYPES:
            logger.warning("A9: Skipping relation with invalid type: %s", rel)
            continue

        # No self-relations.
        if src == tgt:
            continue

        valid_relations.append({
            "source_entity_id": src,
            "relation": rel,
            "target_entity_id": tgt,
        })

    return valid_relations


def _validate_entity_references(
    raw_relations: List[Dict[str, str]],
    entity_ids: Set[str],
) -> List[Dict[str, str]]:
    """
    Filter relations to only those referencing valid entity_ids.

    Removes any relation where source or target is not in entities.json.
    """
    valid: List[Dict[str, str]] = []
    for rel in raw_relations:
        if rel["source_entity_id"] in entity_ids and rel["target_entity_id"] in entity_ids:
            valid.append(rel)
        else:
            logger.warning(
                "A9: Dropping relation with unknown entity: %s -> %s",
                rel["source_entity_id"], rel["target_entity_id"],
            )
    return valid


def _deduplicate_relations(
    raw_relations: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    """Remove duplicate (source, relation, target) triples."""
    seen: Set[Tuple[str, str, str]] = set()
    unique: List[Dict[str, str]] = []
    for rel in raw_relations:
        key = (rel["source_entity_id"], rel["relation"], rel["target_entity_id"])
        if key not in seen:
            seen.add(key)
            unique.append(rel)
    return unique


def extract_relations(
    lecture_id: str,
    cloud_settings: CloudSettings | None = None,
    extractor_fn: Optional[Callable] = None,
    llm_loader: Optional[Callable] = None,
) -> List[Relation]:
    """
    Extract relations between entities using Qwen2.5-7B-Instruct via Transformers.

    Args:
        lecture_id: Unique lecture identifier.
        cloud_settings: Injected CloudSettings.
        extractor_fn: Optional callable(entities, chunks) -> List[dict].
                      Legacy interface for backward compatibility.
        llm_loader: Optional callable returning an LLM backend instance (for tests).

    Returns:
        List of validated Relation instances.

    Raises:
        FileNotFoundError: If required input files are missing.
        ValueError: If zero relations are extracted.
    """
    settings = cloud_settings or CloudSettings()
    lecture_dir = Path(settings.lecture_dir(lecture_id))

    entities_path = lecture_dir / "entities.json"
    chunks_path = lecture_dir / "multimodal_chunks.json"
    segments_path = lecture_dir / "segments.json"

    if not entities_path.exists():
        raise FileNotFoundError(f"A9: entities.json not found: {entities_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"A9: multimodal_chunks.json not found: {chunks_path}")

    entities_data = json.loads(entities_path.read_text(encoding="utf-8"))
    chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))

    # Load segments if available (used for context).
    segments_data: List[Dict[str, Any]] = []
    if segments_path.exists():
        segments_data = json.loads(segments_path.read_text(encoding="utf-8"))
        for seg in segments_data:
            LectureSegment(**seg)

    # Validate inputs.
    for ent in entities_data:
        Entity(**ent)
    for chunk in chunks_data:
        MultimodalChunk(**chunk)

    entity_ids = {e["entity_id"] for e in entities_data}

    # Extract relations.
    raw_relations: List[Dict[str, str]] = []

    if extractor_fn is not None:
        # Legacy callable interface.
        extracted = extractor_fn(entities_data, chunks_data)
        for rel in extracted:
            raw_relations.append({
                "source_entity_id": rel["source_entity_id"],
                "relation": rel["relation"],
                "target_entity_id": rel["target_entity_id"],
            })
    else:
        # Use LLM batch extraction.
        if llm_loader is not None:
            llm = llm_loader()
        else:
            try:
                llm = _load_llm()
            except (ImportError, Exception) as exc:
                logger.error("A9: Failed to load LLM backend: %s", exc)
                raise

        # Build entity list and context.
        entity_list_str = _build_entity_list_str(entities_data)
        context_str = _build_segment_context(segments_data, chunks_data)

        # For large entity sets, split into batches per segment.
        if segments_data:
            chunk_lookup = {c["chunk_id"]: c for c in chunks_data}
            prompts: List[str] = []
            for seg in segments_data:
                seg_context = f"[{seg['title']}]: "
                for cid in seg["chunks"]:
                    if cid in chunk_lookup:
                        c = chunk_lookup[cid]
                        seg_context += c["transcript"] + " " + c["visual_context"] + " "
                prompt = RELATION_EXTRACTION_PROMPT.format(
                    entity_list=entity_list_str[:2000],
                    context=seg_context[:2000],
                )
                prompts.append(prompt)
        else:
            # Single prompt with all context.
            prompt = RELATION_EXTRACTION_PROMPT.format(
                entity_list=entity_list_str[:2000],
                context=context_str,
            )
            prompts = [prompt]

        # Batch extract.
        batch_results = _extract_relations_batch(prompts, llm)
        for rel_list in batch_results:
            raw_relations.extend(rel_list)

    # Validate entity references.
    raw_relations = _validate_entity_references(raw_relations, entity_ids)

    # Deduplicate.
    raw_relations = _deduplicate_relations(raw_relations)

    if not raw_relations:
        raise ValueError("A9: Relation extraction produced zero relations.")

    # Build validated Relation objects.
    relations: List[Relation] = []
    for idx, rel in enumerate(raw_relations, start=1):
        relation = Relation(
            relation_id=f"rel_{idx:06d}",
            source_entity_id=rel["source_entity_id"],
            relation=RelationType(rel["relation"]),
            target_entity_id=rel["target_entity_id"],
        )
        relations.append(relation)

    # Write relations.json.
    relations_path = lecture_dir / "relations.json"
    relations_path.write_text(
        json.dumps([r.model_dump() for r in relations], indent=2),
        encoding="utf-8",
    )
    logger.info("A9: relations.json written (%d relations) to %s", len(relations), relations_path)
    return relations
