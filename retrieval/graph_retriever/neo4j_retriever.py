# retrieval/graph_retriever/neo4j_retriever.py
# E2a — Neo4j Graph Retriever.
#
# Extracts entity mentions from the query via lightweight regex/keyword matching,
# then runs bounded Cypher traversals (1–3 hops) across all closed relation types.
#
# Populates state.graph_results only.

import difflib
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

# All closed relation types (schemas/enums.py) — the traversal must span
# the full graph, not just prerequisites, or it misses USED_BY / EXPLAINS /
# DERIVED_FROM / VISUALIZED_BY edges and under-answers relational queries.
ALL_RELATION_TYPES = (
    "PREREQUISITE_OF|INTRODUCED_BEFORE|USED_BY|DERIVED_FROM|VISUALIZED_BY|EXPLAINS"
)

# Stopwords never useful as graph entity names. Deliberately NOT filtered:
# domain terms like "relation", "model", "table", "database" — those are
# exactly what a student queries about ("how is a relation connected to a
# table") and CONTAINS matching maps them onto "Relational Model".
_STOPWORDS = {
    "what", "which", "how", "why", "when", "where", "who", "does", "did",
    "the", "and", "are", "was", "were", "with", "from", "that", "this",
    "have", "has", "had", "for", "its", "not", "but", "you", "your",
    "about", "into", "can", "could", "will", "would", "shall", "should",
    "than", "then", "there", "their", "they", "is", "it", "of", "to",
    "in", "on", "at", "by", "be", "or", "as", "an", "a", "lecture",
    "lectures", "explain", "explained", "explaining", "give", "given",
    "tell", "called", "using", "used", "use", "via", "such", "like",
}


def _stem_word(word: str) -> str:
    """Lightweight English suffix normalization for common plurals/inflections."""
    w = word.lower().strip()
    if len(w) > 3:
        if w.endswith("ies"):
            return w[:-3] + "y"
        if w.endswith("ses") or w.endswith("xes") or w.endswith("ches") or w.endswith("shes"):
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
    return w


def _score_fuzzy_entity_match(
    query_tokens: List[str],
    query_stems: set,
    entity_name: str,
) -> float:
    """
    Compute conservative similarity score (0.0 to 1.0) between query tokens and an entity name.

    Combines:
    1. Stem overlap (handles singular/plural inflections like processes -> Process Abstraction)
    2. Token-level fuzzy similarity (handles typos/spelling variations like Dijsktra -> Dijkstra)
    3. Multi-word phrase similarity
    """
    ent_clean = entity_name.strip()
    if not ent_clean:
        return 0.0

    ent_tokens = [
        t.lower()
        for t in re.findall(r"[a-zA-Z0-9]+", ent_clean)
        if t.lower() not in _STOPWORDS and len(t) >= 2
    ]
    if not ent_tokens:
        return 0.0

    ent_stems = {_stem_word(t) for t in ent_tokens}

    # 1. Stem intersection
    common_stems = ent_stems.intersection(query_stems)
    if common_stems:
        stem_recall = len(common_stems) / len(ent_stems)
        if stem_recall == 1.0:
            return 1.0
        if len(ent_stems) >= 2 and len(common_stems) >= 1:
            return 0.70 + (0.30 * stem_recall)

    # 2. Token-level fuzzy similarity (typo resilience)
    tok_sims = []
    for e_tok in ent_tokens:
        best_sim = max(
            (difflib.SequenceMatcher(None, q_tok, e_tok).ratio() for q_tok in query_tokens),
            default=0.0,
        )
        tok_sims.append(best_sim)

    avg_tok_sim = sum(tok_sims) / len(tok_sims) if tok_sims else 0.0
    max_tok_sim = max(tok_sims, default=0.0)

    # Single-word entity: require high token similarity
    if len(ent_tokens) == 1:
        if max_tok_sim >= 0.82:
            return max_tok_sim
        return 0.0

    # Multi-word entity: require high average or phrase similarity
    norm_ent = " ".join(ent_tokens)
    norm_q = " ".join(query_tokens)
    phrase_sim = difflib.SequenceMatcher(None, norm_ent, norm_q).ratio()

    composite = max(avg_tok_sim, phrase_sim)
    return composite if composite >= 0.72 else 0.0


def _find_fuzzy_candidates(
    query: str,
    available_entities: List[str],
    threshold: float = 0.72,
    limit: int = 3,
) -> List[str]:
    """
    Find top candidate entity names from available entities in the active lecture.

    Returns up to `limit` entity names exceeding `threshold`, ordered by score descending.
    """
    q_tokens = [
        t.lower()
        for t in re.findall(r"[a-zA-Z0-9]+", query)
        if t.lower() not in _STOPWORDS and len(t) >= 2
    ]
    if not q_tokens or not available_entities:
        return []

    q_stems = {_stem_word(t) for t in q_tokens}

    scored: List[tuple[float, str]] = []
    for ent_name in set(available_entities):
        score = _score_fuzzy_entity_match(q_tokens, q_stems, ent_name)
        if score >= threshold:
            scored.append((score, ent_name))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in scored[:limit]]


def _fetch_lecture_entity_names(driver: Any, lecture_id: str, limit: int = 500) -> List[str]:
    """Fetch distinct entity names for the active lecture from Neo4j."""
    query = (
        "MATCH (n) WHERE n.lecture_id = $lecture_id AND n.name IS NOT NULL "
        "RETURN DISTINCT n.name AS name LIMIT $limit"
    )
    names = []
    with driver.session() as session:
        records = session.run(query, lecture_id=lecture_id, limit=limit)
        for record in records:
            name = record.get("name")
            if name and isinstance(name, str) and name.strip():
                names.append(name.strip())
    return names


def extract_entities(query: str) -> List[str]:
    """
    Extract entity mentions from a query using lightweight regex.

    Not a full NER — captures capitalized terms, acronyms, AND significant
    lowercase words so natural-language queries like "how is a relation
    connected to a table" still resolve to graph nodes. Matching in Cypher
    is case-insensitive, so lowercase terms find "Relational Model".
    """
    matches = ENTITY_PATTERN.findall(query)
    entities = set()
    for groups in matches:
        for g in groups:
            g = g.strip()
            if g and len(g) >= 2:
                entities.add(g)

    # Lowercase significant words as additional candidates.
    for token in re.findall(r"[a-zA-Z]{3,}", query):
        lower = token.lower()
        if lower not in _STOPWORDS and lower not in entities:
            entities.add(lower)

    return list(entities)


def _build_cypher(
    entity_names: List[str],
    max_hops: int = 3,
    lecture_id: Optional[str] = None,
    partial: bool = False,
) -> tuple:
    """
    Build a bounded Cypher query for graph traversal.

    Uses ALL closed relation types. Traversal depth: 1–max_hops.

    Matching: exact case-insensitive by default. With partial=True, falls
    back to substring matching so a lowercase query word like "relation"
    still resolves to a node named "Relational Model".

    Data isolation: BOTH the start node and every related node are filtered
    on lecture_id (when supplied). Nodes MERGE on (entity_id, lecture_id) in
    the loader, so the same cloud entity name may exist in multiple lectures
    without cross-lecture contamination — the retrieval must mirror that.

    Returns:
        (cypher_query, params) tuple.
    """
    # Case-insensitive matching: nodes store title-case names ("Relational
    # Model") while queries may be all lowercase.
    if partial:
        # Substring match — the term appears anywhere inside the node name.
        where_clause = (
            "WHERE ANY(term IN $entity_names WHERE toLower(start.name) CONTAINS term) "
        )
    else:
        where_clause = "WHERE toLower(start.name) IN $entity_names "
    params = {"entity_names": [n.lower() for n in entity_names]}
    if lecture_id:
        where_clause += "AND start.lecture_id = $lecture_id "
        params["lecture_id"] = lecture_id

    cypher = (
        "MATCH (start) "
        + where_clause +
        "OPTIONAL MATCH path = (start)-[:" + ALL_RELATION_TYPES + "*1.."
        + str(max_hops)
        + "]->(related) "
    )
    if lecture_id:
        # OPTIONAL MATCH leaves related NULL when a start node has no
        # neighbors — keep those rows, but never leak another lecture's nodes.
        cypher += "WHERE related IS NULL OR related.lecture_id = $lecture_id "
    cypher += (
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
    2. Run bounded Cypher traversal (Stage 1: exact, Stage 2: partial, Stage 3: fuzzy fallback).
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

    if not lecture_id:
        # Data isolation: a graph query without a lecture_id must never
        # traverse another lecture's nodes (entities repeat across lectures).
        raise ValueError(
            "E2a: lecture_id is required for graph retrieval — refusing to "
            "traverse across all lectures (cross-lecture data leak)."
        )

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
        # Stage 1: exact case-insensitive match.
        cypher, params = _build_cypher(entity_names, max_hops, lecture_id, partial=False)
        results = _run_traversal(driver, cypher, params)

        # Stage 2: if exact matching found nothing, retry with substring
        # matching so natural-language queries ("relation", "table") still
        # resolve to title-case node names ("Relational Model", "Table").
        if not results and entity_names:
            logger.info("E2a: No exact matches; retrying with partial matching.")
            cypher_partial, params_partial = _build_cypher(
                entity_names, max_hops, lecture_id, partial=True,
            )
            results = _run_traversal(driver, cypher_partial, params_partial)

        # Stage 3: if exact and substring matching both found nothing,
        # run conservative fuzzy/stem candidate resolution against the
        # active lecture's entity nodes to resolve inflections and typos.
        if not results:
            logger.info("E2a: No exact/partial matches; attempting fuzzy fallback resolution.")
            try:
                lecture_entities = _fetch_lecture_entity_names(driver, lecture_id)
                if lecture_entities:
                    candidates = _find_fuzzy_candidates(query, lecture_entities, threshold=0.72, limit=3)
                    if candidates:
                        logger.info("E2a: Fuzzy fallback resolved candidate entities: %s", candidates)
                        cypher_fuzzy, params_fuzzy = _build_cypher(
                            candidates, max_hops, lecture_id, partial=False,
                        )
                        results = _run_traversal(driver, cypher_fuzzy, params_fuzzy)
            except Exception as exc:
                logger.warning("E2a: Fuzzy fallback resolution skipped due to error: %s", exc)

        logger.info("E2a: Retrieved %d graph results.", len(results))
        return results

    finally:
        if close_driver:
            driver.close()


def _run_traversal(driver: Any, cypher: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Execute a traversal Cypher and normalise records to result dicts."""
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
    return results


def get_prerequisites_for_concept(
    concept_name: str,
    lecture_id: str,
    driver: Optional[Any] = None,
    local_settings: Optional[LocalSettings] = None,
    min_confidence: float = 0.65,
    max_depth: int = 3,
) -> List[Dict[str, Any]]:
    """
    Traverse backward across PREREQUISITE_OF edges in Neo4j to find conceptual dependencies.

    Preserves full path topology, depth, and relationship-level provenance (supporting
    DAG branching without lossy flattening).

    Data isolation: strictly filters both nodes and every relationship on lecture_id.

    Args:
        concept_name: Name of target concept to explain / back-track prerequisites for.
        lecture_id: Single lecture scope identifier.
        driver: Optional Neo4j driver instance.
        local_settings: Local settings.
        min_confidence: Minimum edge confidence threshold (default: 0.65).
        max_depth: Maximum backward traversal hops (default: 3).

    Returns:
        List of structured prerequisite items with depth, timestamps, path nodes,
        and intermediate relationship provenance.
    """
    if not lecture_id:
        raise ValueError("lecture_id is required for prerequisite traversal (single-lecture isolation).")

    settings = local_settings or LocalSettings()
    close_driver = False
    if driver is None:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(settings.NEO4J_URI)
        close_driver = True

    cypher = (
        f"MATCH path = (prereq {{lecture_id: $lecture_id}})"
        f"-[r:PREREQUISITE_OF*1..{max_depth}]->"
        f"(target {{lecture_id: $lecture_id}}) "
        f"WHERE toLower(target.name) = toLower($concept_name) "
        f"  AND all(rel IN relationships(path) WHERE rel.lecture_id = $lecture_id AND (rel.confidence IS NULL OR rel.confidence >= $min_confidence)) "
        f"RETURN prereq.entity_id AS entity_id, "
        f"       prereq.name AS concept, "
        f"       prereq.type AS type, "
        f"       target.name AS target_concept, "
        f"       length(path) AS depth, "
        f"       [node IN nodes(path) | node.name] AS path_nodes, "
        f"       [rel IN relationships(path) | {{ "
        f"           source: startNode(rel).name, "
        f"           target: endNode(rel).name, "
        f"           confidence: rel.confidence, "
        f"           chunk_id: rel.evidence_chunk_id, "
        f"           timestamp: rel.evidence_timestamp "
        f"       }}] AS edge_provenance, "
        f"       relationships(path)[0].confidence AS direct_confidence, "
        f"       relationships(path)[0].evidence_chunk_id AS chunk_id, "
        f"       relationships(path)[0].evidence_timestamp AS timestamp "
        f"ORDER BY depth DESC, timestamp ASC"
    )

    params = {
        "concept_name": concept_name.strip(),
        "lecture_id": lecture_id,
        "min_confidence": min_confidence,
    }

    try:
        items = []
        with driver.session() as session:
            records = list(session.run(cypher, **params))

            # Fallback to substring matching if exact match yields no rows
            if not records:
                cypher_partial = (
                    f"MATCH path = (prereq {{lecture_id: $lecture_id}})"
                    f"-[r:PREREQUISITE_OF*1..{max_depth}]->"
                    f"(target {{lecture_id: $lecture_id}}) "
                    f"WHERE toLower(target.name) CONTAINS toLower($concept_name) "
                    f"  AND all(rel IN relationships(path) WHERE rel.lecture_id = $lecture_id AND (rel.confidence IS NULL OR rel.confidence >= $min_confidence)) "
                    f"RETURN prereq.entity_id AS entity_id, "
                    f"       prereq.name AS concept, "
                    f"       prereq.type AS type, "
                    f"       target.name AS target_concept, "
                    f"       length(path) AS depth, "
                    f"       [node IN nodes(path) | node.name] AS path_nodes, "
                    f"       [rel IN relationships(path) | {{ "
                    f"           source: startNode(rel).name, "
                    f"           target: endNode(rel).name, "
                    f"           confidence: rel.confidence, "
                    f"           chunk_id: rel.evidence_chunk_id, "
                    f"           timestamp: rel.evidence_timestamp "
                    f"       }}] AS edge_provenance, "
                    f"       relationships(path)[0].confidence AS direct_confidence, "
                    f"       relationships(path)[0].evidence_chunk_id AS chunk_id, "
                    f"       relationships(path)[0].evidence_timestamp AS timestamp "
                    f"ORDER BY depth DESC, timestamp ASC"
                )
                records = list(session.run(cypher_partial, **params))

            seen_paths = set()
            for rec in records:
                path_nodes = rec.get("path_nodes", [])
                path_key = tuple(path_nodes)
                if path_key in seen_paths:
                    continue
                seen_paths.add(path_key)

                parent_concept = path_nodes[1] if len(path_nodes) > 1 else rec.get("target_concept", "")
                items.append({
                    "concept": rec.get("concept", ""),
                    "entity_id": rec.get("entity_id", ""),
                    "type": rec.get("type", "Concept"),
                    "depth": rec.get("depth", 1),
                    "timestamp": rec.get("timestamp", 0.0),
                    "chunk_id": rec.get("chunk_id", ""),
                    "confidence": rec.get("direct_confidence") or 0.8,
                    "parent_concept": parent_concept,
                    "target_concept": rec.get("target_concept", ""),
                    "path": path_nodes,
                    "edge_provenance": rec.get("edge_provenance", []),
                })

        return items
    finally:
        if close_driver:
            driver.close()


def get_prerequisites_for_query(
    query: str,
    lecture_id: str,
    driver: Optional[Any] = None,
    local_settings: Optional[LocalSettings] = None,
    graph_results: Optional[List[Dict[str, Any]]] = None,
    min_confidence: float = 0.65,
) -> List[Dict[str, Any]]:
    """
    Identify target concepts from a user query or graph retrieval results,
    then trace backward prerequisites for them.
    """
    if not lecture_id:
        return []

    # 1. Identify target candidates
    candidate_targets = []
    # From graph_results first (highest relevance)
    if graph_results:
        for r in graph_results:
            name = r.get("start_name") or r.get("related_name")
            if name and name not in candidate_targets:
                candidate_targets.append(name)

    # From query entity extraction
    extracted = extract_entities(query)
    for name in extracted:
        if name not in candidate_targets:
            candidate_targets.append(name)

    if not candidate_targets:
        return []

    # 2. Retrieve prerequisites for each candidate target
    all_prereqs = []
    seen_concepts = set()

    for target in candidate_targets[:2]:  # focus on top 1-2 concepts
        try:
            items = get_prerequisites_for_concept(
                concept_name=target,
                lecture_id=lecture_id,
                driver=driver,
                local_settings=local_settings,
                min_confidence=min_confidence,
            )
            for item in items:
                cid = (item["concept"], item.get("parent_concept"))
                if cid not in seen_concepts:
                    seen_concepts.add(cid)
                    all_prereqs.append(item)
        except Exception as exc:
            logger.warning("Prerequisite lookup failed for target '%s': %s", target, exc)

    return all_prereqs
