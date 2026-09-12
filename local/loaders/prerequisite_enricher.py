# local/loaders/prerequisite_enricher.py
# Socratic Prerequisite Back-Tracker: Local Package Enricher.
#
# Enriches an existing lecture knowledge package (data/packages/{lecture_id}/)
# by inferring conceptual dependencies (PREREQUISITE_OF) and persisting them
# into prerequisites.json and the active Neo4j graph database with full provenance.
#
# Leaves cloud processing and Stage A9 completely untouched.

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import LocalSettings, SharedSettings
from cloud.extraction.prerequisite_inference import infer_prerequisites

logger = logging.getLogger(__name__)


def enrich_package(
    package_dir: Path,
    min_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Enrich an existing lecture package directory with inferred prerequisites.

    Reads entities.json, relations.json, multimodal_chunks.json, and segments.json.
    Runs candidate generation, multi-signal scoring, and DAG cycle resolution.
    Writes prerequisites.json into the package directory.

    Args:
        package_dir: Path to the unpacked knowledge package.
        min_confidence: Optional confidence cutoff (defaults to SharedSettings).

    Returns:
        Result dictionary containing 'prerequisites' and 'metrics'.
    """
    package_dir = Path(package_dir)
    entities_path = package_dir / "entities.json"
    relations_path = package_dir / "relations.json"
    chunks_path = package_dir / "multimodal_chunks.json"
    segments_path = package_dir / "segments.json"

    if not entities_path.exists():
        raise FileNotFoundError(f"entities.json not found in {package_dir}")
    if not relations_path.exists():
        raise FileNotFoundError(f"relations.json not found in {package_dir}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"multimodal_chunks.json not found in {package_dir}")
    if not segments_path.exists():
        raise FileNotFoundError(f"segments.json not found in {package_dir}")

    entities = json.loads(entities_path.read_text(encoding="utf-8"))
    relations = json.loads(relations_path.read_text(encoding="utf-8"))
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    segments = json.loads(segments_path.read_text(encoding="utf-8"))

    logger.info(
        "Enriching package %s: %d entities, %d relations, %d chunks, %d segments.",
        package_dir.name,
        len(entities),
        len(relations),
        len(chunks),
        len(segments),
    )

    result = infer_prerequisites(
        entities=entities,
        chunks=chunks,
        segments=segments,
        relations=relations,
        min_confidence=min_confidence,
    )

    # Save prerequisites.json companion artifact
    prereqs_path = package_dir / "prerequisites.json"
    prereqs_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved %d prerequisites to %s", len(result["prerequisites"]), prereqs_path)

    return result


def load_prerequisites_into_neo4j(
    driver: Any,
    prerequisites: List[Dict[str, Any]],
    lecture_id: str,
) -> int:
    """
    Load inferred PREREQUISITE_OF relationships into Neo4j with provenance.

    Each relationship matches on (entity_id, lecture_id) to strictly preserve
    single-lecture isolation.

    Stores provenance:
    - is_inferred: True
    - confidence: float
    - evidence_chunk_id: str
    - evidence_timestamp: float
    - signals: JSON string or map
    - discourse_pattern: str
    - discourse_snippet: str
    """
    count = 0
    with driver.session() as session:
        for prereq in prerequisites:
            query = (
                "MATCH (s {entity_id: $source, lecture_id: $lecture_id}) "
                "MATCH (t {entity_id: $target, lecture_id: $lecture_id}) "
                "MERGE (s)-[r:PREREQUISITE_OF]->(t) "
                "SET r.lecture_id = $lecture_id, "
                "    r.confidence = $confidence, "
                "    r.is_inferred = true, "
                "    r.evidence_chunk_id = $chunk_id, "
                "    r.evidence_timestamp = $timestamp, "
                "    r.signals = $signals, "
                "    r.discourse_pattern = $discourse_pattern, "
                "    r.discourse_snippet = $discourse_snippet, "
                "    r.inferred_relation_id = $relation_id, "
                "    r.relation_id = coalesce(r.relation_id, $relation_id)"
            )
            session.run(
                query,
                source=prereq["source_id"],
                target=prereq["target_id"],
                relation_id=prereq.get("relation_id", f"prereq_{count:06d}"),
                lecture_id=lecture_id,
                confidence=float(prereq.get("confidence", 0.0)),
                chunk_id=prereq.get("evidence_chunk_id", ""),
                timestamp=float(prereq.get("evidence_timestamp", 0.0)),
                signals=json.dumps(prereq.get("signals", {})),
                discourse_pattern=prereq.get("discourse_pattern", ""),
                discourse_snippet=prereq.get("discourse_snippet", ""),
            )
            count += 1

    logger.info("Loaded %d inferred PREREQUISITE_OF edges into Neo4j for lecture '%s'.", count, lecture_id)
    return count


def enrich_and_load(
    package_dir: Path,
    lecture_id: str,
    driver: Optional[Any] = None,
    local_settings: Optional[LocalSettings] = None,
    min_confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Enrich a package with prerequisites and load them into Neo4j.
    """
    settings = local_settings or LocalSettings()
    result = enrich_package(package_dir, min_confidence=min_confidence)

    close_driver = False
    if driver is None:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(settings.NEO4J_URI)
        close_driver = True

    try:
        loaded_count = load_prerequisites_into_neo4j(
            driver=driver,
            prerequisites=result["prerequisites"],
            lecture_id=lecture_id,
        )
        result["loaded_into_neo4j"] = loaded_count
    finally:
        if close_driver:
            driver.close()

    return result
