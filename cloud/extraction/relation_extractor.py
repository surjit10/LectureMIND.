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
from cloud.utils.diagnostics import ExtractionStats
from cloud.utils.json_repair import (
    parse_json_array,
    STATUS_FAILED,
    STATUS_PARTIAL,
)
from cloud.extraction.window_builder import build_extraction_windows

logger = logging.getLogger(__name__)

# Allowed relation types for validation.
ALLOWED_RELATION_TYPES = {t.value for t in RelationType}

# Output token budgets for relation extraction. The measured production
# failure mode was max_tokens=2048 truncating the response mid-array on
# lectures with long entity IDs (e.g. CS162 Lecture 1), which dropped a
# segment's relations entirely. _load_llm()'s backend cap must stay >= the
# retry budget — TransformersBackend clamps to min(max_tokens, max_new_tokens).
A9_MAX_TOKENS = 4096
A9_RETRY_MAX_TOKENS = 8192

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
    '[{{"source_entity_id":"E1","relation":"RELATION_TYPE","target_entity_id":"E2"}}]\n\n'
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
        max_new_tokens=A9_RETRY_MAX_TOKENS,
    )
    logger.info("A9: Loaded Qwen2.5-7B-Instruct for relation extraction from %s.", settings.QWEN_TEXT_MODEL_PATH)
    return backend


def _build_entity_list_str(entities: List[Dict[str, Any]]) -> str:
    """Build a formatted entity list string for the prompt."""
    lines = []
    for e in entities:
        lines.append(f'- {e["entity_id"]}: {e["name"]} ({e["type"]})')
    return "\n".join(lines)


def _filter_entities_for_segment(
    entities: List[Dict[str, Any]],
    seg_text: str,
) -> List[Dict[str, Any]]:
    """Return entities whose name appears in the segment text.

    Ensures the LLM only sees entities relevant to the current segment,
    which keeps the prompt compact and focused.
    """
    seg_lower = seg_text.lower()
    return [e for e in entities if e["name"].lower() in seg_lower]


def _build_compact_entity_list(
    entities: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, str]]:
    """Build a compact entity list using short index-based IDs (E1, E2, ...).

    Full entity_ids like ``lecture_cs162_v17_entity_000042`` are 60+ chars
    each.  With 128 entities the formatted list exceeds 12 000 chars — far
    beyond any practical prompt budget.  Short aliases (``E1``, ``E2``, …)
    reduce each line from ~98 chars to ~25 chars, fitting 100+ entities in
    under 3 000 chars.

    Returns:
        (entity_list_str, alias_to_real_id) — the formatted list and a
        mapping from the compact alias back to the real entity_id so
        parsed relations can be remapped.
    """
    lines: List[str] = []
    alias_to_real: Dict[str, str] = {}
    for idx, e in enumerate(entities, start=1):
        alias = f"E{idx}"
        alias_to_real[alias] = e["entity_id"]
        lines.append(f"- {alias}: {e['name']} ({e['type']})")
    return "\n".join(lines), alias_to_real


def _remap_relations(
    relations: List[Dict[str, str]],
    alias_to_real: Dict[str, str],
    valid_entity_ids: Optional[set] = None,
    stats: Optional[ExtractionStats] = None,
) -> List[Dict[str, str]]:
    """Remap compact aliases (E1, E2, …) back to real entity_ids.

    Handles:
      1. Compact aliases: 'E1' -> 'lec_001_entity_000001'
      2. Already-valid entity_ids (from mocks, tests, or models that output real IDs directly)

    Relations whose source or target cannot be remapped or resolved to valid
    entity_ids are dropped and recorded in diagnostics.
    """
    known_ids = set(alias_to_real.values())
    if valid_entity_ids:
        known_ids |= valid_entity_ids

    remapped: List[Dict[str, str]] = []
    for rel in relations:
        src_raw = rel.get("source_entity_id", "")
        tgt_raw = rel.get("target_entity_id", "")
        src = alias_to_real.get(src_raw) or (src_raw if src_raw in known_ids else None)
        tgt = alias_to_real.get(tgt_raw) or (tgt_raw if tgt_raw in known_ids else None)
        if src and tgt:
            remapped.append({
                "source_entity_id": src,
                "relation": rel["relation"],
                "target_entity_id": tgt,
            })
        else:
            if stats is not None:
                stats.rejected += 1
            logger.warning(
                "A9: Dropping relation with unresolvable alias: src=%r -> tgt=%r (known aliases=%s)",
                src_raw, tgt_raw, list(alias_to_real.keys())[:10],
            )
    return remapped


def _prune_entities_for_window(
    seg_entities: List[Dict[str, Any]],
    current_window_text: str,
    neighbor_window_texts: List[str],
    context: str,
    seg_alias_map: Dict[str, str],
    token_counter: Callable[[str], int],
    max_tokens: int,
    seg_id: str,
    window_id: str,
    stats: Optional[ExtractionStats] = None,
) -> Tuple[str, Dict[str, str]]:
    """Deterministically prune segment entities if the prompt exceeds the token ceiling.

    Relevance tiers:
      Tier 3: Mentioned directly in the current window.
      Tier 2: Mentioned in immediate neighboring window(s) (overlap context).
      Tier 1: Mentioned elsewhere in the segment.

    Preserves stable segment aliases (E1, E2, …) even when pruned.
    """
    real_to_alias = {real_id: alias for alias, real_id in seg_alias_map.items()}

    curr_lower = current_window_text.lower()
    neighbor_lower = " ".join(t.lower() for t in neighbor_window_texts)

    scored_entities: List[Tuple[int, int, Dict[str, Any]]] = []
    for idx, e in enumerate(seg_entities):
        name_lower = e["name"].lower()
        if name_lower in curr_lower:
            score = 3
        elif name_lower in neighbor_lower:
            score = 2
        else:
            score = 1
        scored_entities.append((-score, idx, e))

    scored_entities.sort(key=lambda x: (x[0], x[1]))

    kept: List[Dict[str, Any]] = []
    for _, _, e in scored_entities:
        test_kept = kept + [e]
        lines = [f"- {real_to_alias[x['entity_id']]}: {x['name']} ({x['type']})" for x in test_kept]
        test_list_str = "\n".join(lines)
        test_prompt = RELATION_EXTRACTION_PROMPT.format(
            entity_list=test_list_str,
            context=context,
        )
        if token_counter(test_prompt) <= max_tokens or len(kept) < 2:
            kept = test_kept
        else:
            break

    kept_ids = {x["entity_id"] for x in kept}
    final_entities = [e for e in seg_entities if e["entity_id"] in kept_ids]

    final_lines = [f"- {real_to_alias[e['entity_id']]}: {e['name']} ({e['type']})" for e in final_entities]
    final_list_str = "\n".join(final_lines)
    final_alias_map = {real_to_alias[e["entity_id"]]: e["entity_id"] for e in final_entities}

    pruned_count = len(seg_entities) - len(final_entities)
    if pruned_count > 0:
        logger.warning(
            "A9: Pruned %d/%d entities for segment %s window %s to fit token budget (prompt_tokens <= %d)",
            pruned_count, len(seg_entities), seg_id, window_id, max_tokens,
        )
        if stats is not None:
            stats.rejected += pruned_count

    return final_list_str, final_alias_map


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


def normalize_relation_type(raw_type: str) -> Optional[str]:
    """Map an LLM-emitted relation string onto the closed RelationType ontology.

    Deterministic normalization order:
    1. Exact canonical match -> unchanged.
    2. Case variant of a canonical type -> canonical form ("explains" ->
       "EXPLAINS").
    3. Unknown types (e.g. MODIFIES) -> None, i.e. rejected intentionally and
       counted in diagnostics.

    MODIFIES is deliberately NOT added to the closed enum: RelationType is
    spec-locked (schemas/enums.py), and introducing a type requires updating
    the graph retriever's ALL_RELATION_TYPES traversal regex
    (retrieval/graph_retriever/neo4j_retriever.py) and the extraction prompt
    in the SAME change to keep the A9 -> relations.json -> Neo4j import ->
    GraphRetriever chain consistent. Until that decision is made, unknown
    types are rejected but never silently.
    """
    t = raw_type.strip()
    if t in ALLOWED_RELATION_TYPES:
        return t
    lower = t.lower()
    for allowed in ALLOWED_RELATION_TYPES:
        if allowed.lower() == lower:
            return allowed
    return None


def _extract_relations_batch(
    prompts: List[str],
    llm: Any,
    stats: Optional[ExtractionStats] = None,
) -> List[List[Dict[str, str]]]:
    """
    Batch extract relations using the LLM backend.

    Args:
        prompts: List of formatted relation extraction prompts.
        llm: LLM backend instance (TransformersBackend or compatible).
        stats: Optional ExtractionStats to record parse/accept counters.

    Returns:
        List of relation lists (one per prompt).
    """
    outputs = llm.generate(
        prompts,
        temperature=0.1,
        max_tokens=A9_MAX_TOKENS,
        top_p=0.95,
    )

    all_relations: List[List[Dict[str, str]]] = []
    retry_indices: List[int] = []
    for idx, raw_text in enumerate(outputs):
        relations, status = _parse_relation_json_with_status(raw_text, stats=stats)
        all_relations.append(relations)
        if status in (STATUS_FAILED, STATUS_PARTIAL):
            retry_indices.append(idx)

    # Retry truncated/malformed responses once with a larger budget. A failed
    # or partial parse means the response was cut off (or otherwise unusable);
    # one extra call with a doubled budget recovers the tail without unbounded
    # extra GPU time (only the affected segments are regenerated).
    for idx in retry_indices:
        if stats is not None:
            stats.retries += 1
        logger.warning(
            "A9: Segment prompt %d parse failed — retrying once with a larger token budget.",
            idx,
        )
        (retry_text,) = llm.generate(
            [prompts[idx]],
            temperature=0.1,
            max_tokens=A9_RETRY_MAX_TOKENS,
            top_p=0.95,
        )
        retry_relations, _retry_status = _parse_relation_json_with_status(
            retry_text, stats=stats,
        )
        if retry_relations:
            all_relations[idx] = retry_relations

    return all_relations


def _parse_relation_json(
    raw_text: str,
    stats: Optional[ExtractionStats] = None,
) -> List[Dict[str, str]]:
    """
    Parse LLM output into a list of relation dicts.

    Uses the shared json_repair pipeline (fences -> balanced array -> strict
    load -> safe escape repair, and object salvage when the response was
    truncated mid-array). A9 previously had NO escape repair, so a response
    containing e.g. "W\\_q" raised "JSON parse error: Invalid \\escape" and
    the whole segment's relations were discarded. Validates relation types
    via normalize_relation_type() and requires required fields to exist.

    Returns the validated relation list; see
    _parse_relation_json_with_status() for the parse status consumed by the
    retry policy.
    """
    relations, _status = _parse_relation_json_with_status(raw_text, stats=stats)
    return relations


def _parse_relation_json_with_status(
    raw_text: str,
    stats: Optional[ExtractionStats] = None,
) -> Tuple[List[Dict[str, str]], str]:
    """
    Parse LLM output into relations, also returning the json_repair status.

    Status is one of STATUS_OK / STATUS_REPAIRED / STATUS_PARTIAL /
    STATUS_FAILED. STATUS_PARTIAL means the response was truncated mid-array
    and only the complete leading objects were salvaged — the retry policy in
    _extract_relations_batch() re-requests such segments with a larger token
    budget to recover the tail.
    """
    parsed, status = parse_json_array(raw_text)
    if stats is not None:
        stats.requests += 1
        stats.record_parse(status)

    if parsed is None:
        logger.warning("A9: JSON parse %s: %s", status, raw_text[:200])
        logger.warning("A9 RAW OUTPUT (first 5000 chars):\n%s", raw_text[:5000])
        return [], status

    if not isinstance(parsed, list):
        logger.warning("A9: Expected list, got %s", type(parsed).__name__)
        return [], status

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

        normalized = normalize_relation_type(rel)
        if normalized is None:
            if stats is not None:
                stats.rejected += 1
            logger.warning("A9: Skipping relation with invalid type: %s", rel)
            continue
        if normalized != rel:
            if stats is not None:
                stats.normalized += 1
            logger.warning("A9: Normalized relation type %r -> %r", rel, normalized)

        # No self-relations.
        if src == tgt:
            continue

        valid_relations.append({
            "source_entity_id": src,
            "relation": normalized,
            "target_entity_id": tgt,
        })

    if stats is not None:
        stats.accepted += len(valid_relations)
    return valid_relations, status


def _validate_entity_references(
    raw_relations: List[Dict[str, str]],
    entity_ids: Set[str],
    stats: Optional[ExtractionStats] = None,
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
            if stats is not None:
                stats.rejected += 1
            logger.warning(
                "A9: Dropping relation with unknown entity: %s -> %s",
                rel["source_entity_id"], rel["target_entity_id"],
            )
    return valid


def _deduplicate_relations(
    raw_relations: List[Dict[str, str]],
    stats: Optional[ExtractionStats] = None,
) -> List[Dict[str, str]]:
    """Remove duplicate (source, relation, target) triples."""
    seen: Set[Tuple[str, str, str]] = set()
    unique: List[Dict[str, str]] = []
    for rel in raw_relations:
        key = (rel["source_entity_id"], rel["relation"], rel["target_entity_id"])
        if key not in seen:
            seen.add(key)
            unique.append(rel)
        elif stats is not None:
            stats.duplicates += 1
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

    # Extraction diagnostics (pure logging — never affects pipeline output).
    stats = ExtractionStats()

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
            token_counter = lambda t: max(1, len(t.split()))

        token_budget = getattr(settings, "EXTRACTION_WINDOW_TOKEN_BUDGET", 2000)
        overlap_chunks = getattr(settings, "EXTRACTION_WINDOW_OVERLAP_CHUNKS", 1)

        # Segment-level entity visibility + compact aliases.
        # Entities associated with each semantic segment are discovered across
        # all chunks of that segment. Consistent aliases (E1, E2, …) are reused
        # across all windows of that segment so entities in different windows
        # remain jointly addressable, while bounded by token safety.
        segment_alias_maps: List[Dict[str, str]] = []
        prompts: List[str] = []

        max_input_length = getattr(settings, "TRANSFORMERS_MAX_INPUT_LENGTH", 3072)
        safety_margin = getattr(settings, "EXTRACTION_TOKEN_SAFETY_MARGIN", 250)
        max_prompt_budget = max_input_length - safety_margin

        if segments_data:
            chunk_lookup = {c["chunk_id"]: c for c in chunks_data}
            for s_idx, seg in enumerate(segments_data):
                seg_chunks = [chunk_lookup[cid] for cid in seg.get("chunks", []) if cid in chunk_lookup]
                if not seg_chunks:
                    continue

                # 1. Discover all entities mentioned across this semantic segment.
                seg_full_text = f"[{seg['title']}]:\n" + " ".join(
                    (c.get("transcript", "") + " " + c.get("visual_context", "") + " " + c.get("ocr_text", ""))
                    for c in seg_chunks
                )
                seg_entities = _filter_entities_for_segment(entities_data, seg_full_text)
                if len(seg_entities) < 2:
                    seg_entities = entities_data

                # 2. Build consistent compact alias mapping ONCE for this segment.
                seg_entity_list_str, seg_alias_map = _build_compact_entity_list(seg_entities)

                # 3. Build token-budgeted extraction windows for this segment's chunks.
                seg_windows, _ = build_extraction_windows(
                    seg_chunks,
                    token_budget=token_budget,
                    overlap_chunks=overlap_chunks,
                    token_counter=token_counter,
                )

                for w_idx, w in enumerate(seg_windows):
                    seg_context = f"[{seg['title']}]:\n" + w.text

                    # Check if the complete segment entity list fits within prompt budget.
                    candidate_prompt = RELATION_EXTRACTION_PROMPT.format(
                        entity_list=seg_entity_list_str,
                        context=seg_context,
                    )
                    prompt_tokens = token_counter(candidate_prompt)

                    if prompt_tokens <= max_prompt_budget:
                        window_entity_list_str = seg_entity_list_str
                        window_alias_map = seg_alias_map
                        prompt = candidate_prompt
                    else:
                        neighbor_texts = []
                        if w_idx > 0:
                            neighbor_texts.append(seg_windows[w_idx - 1].text)
                        if w_idx < len(seg_windows) - 1:
                            neighbor_texts.append(seg_windows[w_idx + 1].text)

                        window_entity_list_str, window_alias_map = _prune_entities_for_window(
                            seg_entities=seg_entities,
                            current_window_text=w.text,
                            neighbor_window_texts=neighbor_texts,
                            context=seg_context,
                            seg_alias_map=seg_alias_map,
                            token_counter=token_counter,
                            max_tokens=max_prompt_budget,
                            seg_id=seg.get("segment_id", f"seg_{s_idx}"),
                            window_id=w.window_id,
                            stats=stats,
                        )
                        prompt = RELATION_EXTRACTION_PROMPT.format(
                            entity_list=window_entity_list_str,
                            context=seg_context,
                        )

                    segment_alias_maps.append(window_alias_map)
                    prompts.append(prompt)
        else:
            # Fallback when no segments are available: window the full chunks list.
            all_chunks_text = " ".join(
                (c.get("transcript", "") + " " + c.get("visual_context", "") + " " + c.get("ocr_text", ""))
                for c in chunks_data
            )
            all_entities = _filter_entities_for_segment(entities_data, all_chunks_text)
            if len(all_entities) < 2:
                all_entities = entities_data

            all_entity_list_str, all_alias_map = _build_compact_entity_list(all_entities)

            all_windows, _ = build_extraction_windows(
                chunks_data,
                token_budget=token_budget,
                overlap_chunks=overlap_chunks,
                token_counter=token_counter,
            )

            for w_idx, w in enumerate(all_windows):
                seg_context = w.text
                candidate_prompt = RELATION_EXTRACTION_PROMPT.format(
                    entity_list=all_entity_list_str,
                    context=seg_context,
                )
                prompt_tokens = token_counter(candidate_prompt)

                if prompt_tokens <= max_prompt_budget:
                    window_entity_list_str = all_entity_list_str
                    window_alias_map = all_alias_map
                    prompt = candidate_prompt
                else:
                    neighbor_texts = []
                    if w_idx > 0:
                        neighbor_texts.append(all_windows[w_idx - 1].text)
                    if w_idx < len(all_windows) - 1:
                        neighbor_texts.append(all_windows[w_idx + 1].text)

                    window_entity_list_str, window_alias_map = _prune_entities_for_window(
                        seg_entities=all_entities,
                        current_window_text=w.text,
                        neighbor_window_texts=neighbor_texts,
                        context=seg_context,
                        seg_alias_map=all_alias_map,
                        token_counter=token_counter,
                        max_tokens=max_prompt_budget,
                        seg_id="lecture_all",
                        window_id=w.window_id,
                        stats=stats,
                    )
                    prompt = RELATION_EXTRACTION_PROMPT.format(
                        entity_list=window_entity_list_str,
                        context=seg_context,
                    )

                segment_alias_maps.append(window_alias_map)
                prompts.append(prompt)

        # Batch extract and remap compact aliases → real entity_ids.
        batch_results = _extract_relations_batch(prompts, llm, stats=stats)
        for seg_idx, rel_list in enumerate(batch_results):
            alias_map = segment_alias_maps[seg_idx]
            remapped = _remap_relations(rel_list, alias_map, valid_entity_ids=entity_ids, stats=stats)
            raw_relations.extend(remapped)

    # Validate entity references.
    raw_relations = _validate_entity_references(raw_relations, entity_ids, stats=stats)

    # Deduplicate.
    raw_relations = _deduplicate_relations(raw_relations, stats=stats)

    logger.info(stats.report("A9"))

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
