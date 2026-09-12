# local/loaders/neo4j_loader.py
# D1 — Neo4j Graph Loader.
#
# Loads entities.json as nodes and relations.json as relationships
# into a Neo4j 5.24 Community instance.
#
# Node labels come from entity.type (Concept, Algorithm, Formula, Code, Diagram, LectureSegment).
# Relationships match on entity_id, NEVER on name.
#
# Environment: Local only. Never imports from cloud/.

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import LocalSettings
from schemas.entity import Entity
from schemas.relation import Relation
from schemas.enums import EntityType, RelationType

logger = logging.getLogger(__name__)

ALLOWED_LABELS = {et.value for et in EntityType}
ALLOWED_RELATION_TYPES = {rt.value for rt in RelationType}


class Neo4jLoadError(Exception):
    """Raised when Neo4j loading fails."""
    pass


def _validate_entities(entities: List[Dict]) -> List[Entity]:
    """Validate all entities against the Chunk 1 schema."""
    validated = []
    for i, ent in enumerate(entities):
        try:
            validated.append(Entity(**ent))
        except Exception as exc:
            raise Neo4jLoadError(f"Entity {i} validation failed: {exc}")
    return validated


def _validate_relations(relations: List[Dict], entity_ids: set) -> List[Relation]:
    """Validate all relations and check referential integrity."""
    validated = []
    for i, rel in enumerate(relations):
        try:
            r = Relation(**rel)
        except Exception as exc:
            raise Neo4jLoadError(f"Relation {i} validation failed: {exc}")

        if r.source_entity_id not in entity_ids:
            raise Neo4jLoadError(
                f"Relation '{r.relation_id}' references non-existent source_entity_id: "
                f"{r.source_entity_id}"
            )
        if r.target_entity_id not in entity_ids:
            raise Neo4jLoadError(
                f"Relation '{r.relation_id}' references non-existent target_entity_id: "
                f"{r.target_entity_id}"
            )
        if r.relation.value not in ALLOWED_RELATION_TYPES:
            raise Neo4jLoadError(
                f"Relation '{r.relation_id}' has disallowed type: {r.relation.value}"
            )

        validated.append(r)
    return validated


def _create_nodes(driver: Any, entities: List[Entity], lecture_id: str) -> int:
    """
    Create entity nodes in Neo4j.

    Each node gets: entity_id, name, type, lecture_id.
    Label is derived from entity.type.

    Data isolation: the MERGE key is (entity_id, lecture_id), NOT entity_id
    alone. Cloud pipelines may reuse entity ids (e.g. two packages both
    built from a lecture numbered lec_001), and a package may be
    re-imported under a fresh runtime lecture id. Merging on entity_id only
    would silently re-point the FIRST lecture's nodes at the SECOND import
    and leak graph data across lectures.
    """
    count = 0
    with driver.session() as session:
        for entity in entities:
            label = entity.type.value
            if label not in ALLOWED_LABELS:
                raise Neo4jLoadError(f"Disallowed node label: {label}")

            query = (
                f"MERGE (n:{label} {{entity_id: $entity_id, lecture_id: $lecture_id}}) "
                f"SET n.name = $name, n.type = $type"
            )
            session.run(
                query,
                entity_id=entity.entity_id,
                name=entity.name,
                type=entity.type.value,
                lecture_id=lecture_id,
            )
            count += 1

    logger.info("D1: Created %d nodes in Neo4j.", count)
    return count


def _create_relationships(driver: Any, relations: List[Relation], lecture_id: str) -> int:
    """
    Create relationships in Neo4j.

    Matches on (entity_id, lecture_id), NEVER on name alone.
    Relationship type comes from relation.relation.

    Data isolation: endpoints are matched within the SAME lecture only, so
    a shared entity_id across lectures can never create a cross-lecture edge.
    """
    count = 0
    with driver.session() as session:
        for rel in relations:
            rel_type = rel.relation.value
            query = (
                f"MATCH (s {{entity_id: $source, lecture_id: $lecture_id}}) "
                f"MATCH (t {{entity_id: $target, lecture_id: $lecture_id}}) "
                f"MERGE (s)-[r:{rel_type} {{relation_id: $relation_id}}]->(t) "
                f"SET r.lecture_id = $lecture_id"
            )
            session.run(
                query,
                source=rel.source_entity_id,
                target=rel.target_entity_id,
                relation_id=rel.relation_id,
                lecture_id=lecture_id,
            )
            count += 1

    logger.info("D1: Created %d relationships in Neo4j.", count)
    return count


def load_neo4j(
    package_dir: Path,
    lecture_id: str,
    local_settings: LocalSettings | None = None,
    driver: Optional[Any] = None,
) -> Dict[str, int]:
    """
    Load entities and relations from a knowledge package into Neo4j.

    Args:
        package_dir: Path to the unpacked knowledge package.
        lecture_id: Lecture identifier.
        local_settings: Injected LocalSettings.
        driver: Optional pre-created Neo4j driver (for testing).

    Returns:
        Dict with node_count and relationship_count.
    """
    settings = local_settings or LocalSettings()
    package_dir = Path(package_dir)

    # Load files.
    entities_path = package_dir / "entities.json"
    relations_path = package_dir / "relations.json"

    if not entities_path.exists():
        raise Neo4jLoadError(f"entities.json not found: {entities_path}")
    if not relations_path.exists():
        raise Neo4jLoadError(f"relations.json not found: {relations_path}")

    entities_data = json.loads(entities_path.read_text(encoding="utf-8"))
    relations_data = json.loads(relations_path.read_text(encoding="utf-8"))

    # Validate.
    entities = _validate_entities(entities_data)
    entity_ids = {e.entity_id for e in entities}

    # Check entity_id uniqueness.
    if len(entity_ids) != len(entities):
        raise Neo4jLoadError("Duplicate entity_id found in entities.json.")

    relations = _validate_relations(relations_data, entity_ids)

    # Connect and load.
    close_driver = False
    if driver is None:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(settings.NEO4J_URI)
        close_driver = True

    try:
        node_count = _create_nodes(driver, entities, lecture_id)
        rel_count = _create_relationships(driver, relations, lecture_id)

        # Optional: load enriched prerequisites if prerequisites.json is present
        prereq_count = 0
        prereqs_path = package_dir / "prerequisites.json"
        if prereqs_path.exists():
            try:
                from local.loaders.prerequisite_enricher import load_prerequisites_into_neo4j
                prereqs_data = json.loads(prereqs_path.read_text(encoding="utf-8"))
                prereq_list = prereqs_data.get("prerequisites", []) if isinstance(prereqs_data, dict) else prereqs_data
                prereq_count = load_prerequisites_into_neo4j(driver, prereq_list, lecture_id)
            except Exception as exc:
                logger.warning("Failed to load optional prerequisites.json: %s", exc)

    finally:
        if close_driver:
            driver.close()

    result = {"node_count": node_count, "relationship_count": rel_count}
    if prereq_count > 0:
        result["prerequisite_count"] = prereq_count
    logger.info("D1: Neo4j load complete — %s", result)
    return result
