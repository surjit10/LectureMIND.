# retrieval/graph_retriever/neo4j_retriever.py
# E2a — Neo4j Graph Retriever.
#
# Extracts entity mentions from the query via lightweight regex/keyword matching,
# then runs bounded Cypher traversals (1–3 hops) using PREREQUISITE_OF and INTRODUCED_BEFORE.
#
# Populates state.graph_results only.

import logging
import re
from typing import Any, Dict, List, Optional

from config import LocalSettings

logger = logging.getLogger(__name__)

# Lightweight entity extraction patterns.
# Matches capitalized words, acronyms, and common CS terms.
ENTITY_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]+)*)\b"  # Capitalized multi-word
    r"|\b([A-Z]{2,})\b"  # Acronyms like BFS, DFS
)


def extract_entities(query: str) -> List[str]:
    """
    Extract entity mentions from a query using lightweight regex.

    Not a full NER — just captures capitalized terms and acronyms
    likely to match Neo4j node names.
    """
    matches = ENTITY_PATTERN.findall(query)
    entities = set()
    for groups in matches:
        for g in groups:
            g = g.strip()
            if g and len(g) >= 2:
                entities.add(g)
    return list(entities)


def _build_cypher(entity_names: List[str], max_hops: int = 3, lecture_id: Optional[str] = None) -> tuple:
    """
    Build a bounded Cypher query for graph traversal.

    Uses PREREQUISITE_OF and INTRODUCED_BEFORE relationships only.
    Traversal depth: 1–max_hops.

    Returns:
        (cypher_query, params) tuple.
    """
    where_clause = "WHERE start.name IN $entity_names "
    params = {"entity_names": entity_names}
    if lecture_id:
        where_clause += "AND start.lecture_id = $lecture_id "
        params["lecture_id"] = lecture_id

    cypher = (
        "MATCH (start) "
        + where_clause +
        "OPTIONAL MATCH path = (start)-[:PREREQUISITE_OF|INTRODUCED_BEFORE*1.."
        + str(max_hops)
        + "]->(related) "
        "RETURN start.entity_id AS start_id, start.name AS start_name, "
        "start.type AS start_type, "
        "related.entity_id AS related_id, related.name AS related_name, "
        "related.type AS related_type, "
        "[r IN relationships(path) | type(r)] AS rel_types, "
        "length(path) AS hops"
    )
    return cypher, params


def retrieve_graph(
    query: str,
    driver: Optional[Any] = None,
    local_settings: LocalSettings | None = None,
    max_hops: int = 3,
    lecture_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve graph context for a query.

    1. Extract entity mentions from query.
    2. Run bounded Cypher traversal.
    3. Return list of result dicts.

    Args:
        query: User query string.
        driver: Optional Neo4j driver (for testing).
        local_settings: Injected LocalSettings.
        max_hops: Max traversal depth (1–3).
        lecture_id: Optional lecture ID to filter results.

    Returns:
        List of graph result dicts with entity info and paths.
    """
    settings = local_settings or LocalSettings()

    # Extract entities from query.
    entity_names = extract_entities(query)
    if not entity_names:
        logger.info("E2a: No entities extracted from query.")
        return []

    logger.info("E2a: Extracted entities: %s", entity_names)

    # Connect to Neo4j.
    close_driver = False
    if driver is None:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(settings.NEO4J_URI)
        close_driver = True

    try:
        cypher, params = _build_cypher(entity_names, max_hops, lecture_id)
        results = []

        with driver.session() as session:
            records = session.run(cypher, **params)
            for record in records:
                result = {
                    "start_id": record.get("start_id", ""),
                    "start_name": record.get("start_name", ""),
                    "start_type": record.get("start_type", ""),
                    "related_id": record.get("related_id"),
                    "related_name": record.get("related_name"),
                    "related_type": record.get("related_type"),
                    "rel_types": record.get("rel_types", []),
                    "hops": record.get("hops", 0),
                }
                results.append(result)

        logger.info("E2a: Retrieved %d graph results.", len(results))
        return results

    finally:
        if close_driver:
            driver.close()
