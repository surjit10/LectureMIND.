# evaluation/knowledge_graph/entity_auditor.py
# Read-only Entity Extraction Quality Auditor.
#
# Audits entities extracted by Stage A8 without mutating any files.
# Detects:
#   1. Sentence fragments & incomplete syntax
#   2. Excessively long or clause-like entities
#   3. Generic / non-concept stopwords
#   4. Surface-form duplicate clusters
#   5. Graph orphans (degree == 0)
#
# Computes:
#   - Diagnostic structural rates (fragment_rate, orphan_rate, etc.)
#   - Gold-standard Precision / Recall / F1 when entity_gold.json is provided.

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Sentence fragment starters / connectors
_FRAGMENT_PATTERNS = [
    re.compile(r"^(us\s+a|and\s+then|this\s+means|because\s+the|which\s+means|gives\s+us|so\s+that|therefore\s+we|to\s+fix|now\s+the|we\s+will|that\s+is|due\s+to)\b", re.IGNORECASE),
    re.compile(r"\b(is\s+important\s+because|can\s+only\s+work|will\s+be\s+able\s+to|referring\s+to|known\s+as)\b", re.IGNORECASE),
    re.compile(r"[\.!\?]\s+[A-Z]", re.UNICODE),  # sentence boundary inside entity
]

# Non-concept generic stopwords
_GENERIC_NON_CONCEPTS = {
    "thing", "things", "example", "examples", "important", "process",
    "processes", "system", "systems", "this", "that", "it", "they",
    "video", "videos", "series", "topic", "information", "component",
    "setup", "action", "actions", "function", "functions", "approach",
    "overview", "intro", "conclusion",
}


@dataclass
class EntityIssue:
    entity_id: str
    entity_name: str
    issue_type: str  # "FRAGMENT", "EXCESSIVE_LENGTH", "GENERIC_STOPWORD", "DUPLICATE", "ORPHAN"
    reason: str
    severity: str    # "HIGH", "MEDIUM", "LOW"


@dataclass
class EntityAuditResult:
    total_entities: int = 0
    valid_entities: int = 0
    fragment_count: int = 0
    excessive_length_count: int = 0
    generic_count: int = 0
    orphan_count: int = 0
    duplicate_groups_count: int = 0
    
    # Structural diagnostic rates (NOT called precision)
    fragment_rate: float = 0.0
    generic_rate: float = 0.0
    orphan_rate: float = 0.0
    heuristic_accept_rate: float = 0.0
    
    # Gold-standard metrics (None if gold data not evaluated)
    gold_entity_precision: Optional[float] = None
    gold_entity_recall: Optional[float] = None
    gold_entity_f1: Optional[float] = None
    gold_evaluation_method: Optional[str] = None
    gold_entity_count: int = 0
    matched_gold_entity_count: int = 0
    tp_entity_count: int = 0
    
    issues: List[Dict[str, Any]] = field(default_factory=list)
    duplicate_groups: List[List[Dict[str, str]]] = field(default_factory=list)


def normalize_entity_name(name: str) -> str:
    """Normalize string for fuzzy/cluster duplicate matching."""
    s = name.strip().lower()
    s = re.sub(r"[\-_]+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def audit_entities(
    entities: List[Dict[str, Any]],
    relations: Optional[List[Dict[str, Any]]] = None,
    gold_entities_path: Optional[str | Path] = None,
) -> EntityAuditResult:
    """
    Perform a comprehensive, read-only audit of extracted entities.
    
    Args:
        entities: List of entity dicts (entity_id, name, type, ...)
        relations: Optional list of relation dicts to check graph degree.
        gold_entities_path: Optional path to entity_gold.json.
        
    Returns:
        EntityAuditResult with categorized issues and verified metrics.
    """
    total = len(entities)
    issues: List[EntityIssue] = []
    
    # 1. Compute relationship degrees if relations provided
    degrees: Dict[str, int] = {e["entity_id"]: 0 for e in entities}
    if relations:
        for r in relations:
            src = r.get("source_entity_id") or r.get("source_id")
            tgt = r.get("target_entity_id") or r.get("target_id")
            if src in degrees:
                degrees[src] += 1
            if tgt in degrees:
                degrees[tgt] += 1

    # 2. Check each entity
    fragment_ids = set()
    length_ids = set()
    generic_ids = set()
    orphan_ids = set()

    for e in entities:
        eid = e.get("entity_id", "")
        name = e.get("name", "").strip()

        # A. Fragment Detection
        is_frag = False
        for pat in _FRAGMENT_PATTERNS:
            if pat.search(name):
                issues.append(EntityIssue(
                    entity_id=eid,
                    entity_name=name,
                    issue_type="FRAGMENT",
                    reason=f"Matches sentence fragment / clausal connector pattern: {pat.pattern}",
                    severity="HIGH",
                ))
                fragment_ids.add(eid)
                is_frag = True
                break

        # B. Excessive Length / Clausal structure
        words = name.split()
        if len(words) > 6 or len(name) > 45:
            if not is_frag:
                issues.append(EntityIssue(
                    entity_id=eid,
                    entity_name=name,
                    issue_type="EXCESSIVE_LENGTH",
                    reason=f"Entity name is {len(name)} chars / {len(words)} words, likely a descriptive clause rather than atomic concept.",
                    severity="HIGH",
                ))
            length_ids.add(eid)

        # C. Generic Non-Concept Stopwords
        norm = normalize_entity_name(name)
        if norm in _GENERIC_NON_CONCEPTS or (len(words) == 1 and norm in _GENERIC_NON_CONCEPTS):
            issues.append(EntityIssue(
                entity_id=eid,
                entity_name=name,
                issue_type="GENERIC_STOPWORD",
                reason=f"Matches generic non-concept dictionary: '{name}'",
                severity="MEDIUM",
            ))
            generic_ids.add(eid)

        # D. Orphan Detection
        if relations is not None and degrees.get(eid, 0) == 0:
            issues.append(EntityIssue(
                entity_id=eid,
                entity_name=name,
                issue_type="ORPHAN",
                reason="Entity has degree = 0 (no incident relationships in knowledge graph).",
                severity="LOW",
            ))
            orphan_ids.add(eid)

    # 3. Duplicate Detection (Clustering by normalized name)
    norm_clusters: Dict[str, List[Dict[str, str]]] = {}
    for e in entities:
        eid = e.get("entity_id", "")
        name = e.get("name", "")
        norm = normalize_entity_name(name)
        norm_clusters.setdefault(norm, []).append({"entity_id": eid, "name": name})

    duplicate_groups = [group for group in norm_clusters.values() if len(group) > 1]
    duplicate_ids = set()
    for group in duplicate_groups:
        for item in group:
            duplicate_ids.add(item["entity_id"])
            issues.append(EntityIssue(
                entity_id=item["entity_id"],
                entity_name=item["name"],
                issue_type="DUPLICATE",
                reason=f"Duplicates surface form with {[g['name'] for g in group if g['entity_id'] != item['entity_id']]}",
                severity="MEDIUM",
            ))

    # Calculate valid entities under automated heuristic rules
    flawed_ids = fragment_ids | length_ids | generic_ids
    valid_count = total - len(flawed_ids)
    
    result = EntityAuditResult(
        total_entities=total,
        valid_entities=max(0, valid_count),
        fragment_count=len(fragment_ids),
        excessive_length_count=len(length_ids),
        generic_count=len(generic_ids),
        orphan_count=len(orphan_ids),
        duplicate_groups_count=len(duplicate_groups),
        fragment_rate=round(len(fragment_ids) / total, 4) if total > 0 else 0.0,
        generic_rate=round(len(generic_ids) / total, 4) if total > 0 else 0.0,
        orphan_rate=round(len(orphan_ids) / total, 4) if total > 0 else 0.0,
        heuristic_accept_rate=round(valid_count / total, 4) if total > 0 else 0.0,
        issues=[asdict(i) for i in issues],
        duplicate_groups=duplicate_groups,
    )

    # 4. Gold-standard comparison if gold standard provided
    if gold_entities_path and Path(gold_entities_path).exists():
        with open(gold_entities_path, "r", encoding="utf-8") as fp:
            gold_data = json.load(fp)
        
        gold_list = gold_data.get("entities", [])
        gold_names_map: Dict[str, Set[str]] = {}
        for g in gold_list:
            cname = g["canonical_name"].lower()
            aliases = {a.lower() for a in g.get("aliases", [])}
            aliases.add(cname)
            gold_names_map[cname] = aliases

        # One-to-one greedy matching: each predicted entity can be a TP for at
        # most one gold concept, and each gold concept can be claimed at most
        # once. Duplicate predictions of the same concept yield exactly one TP
        # plus (n-1) FPs — precision is no longer inflated by duplicates.
        matched_gold_keys: Set[str] = set()
        tp = 0
        for e in entities:
            ename = e.get("name", "").strip().lower()
            matched_cname = None
            for cname, alias_set in gold_names_map.items():
                if cname in matched_gold_keys:
                    continue
                if ename in alias_set or any(normalize_entity_name(ename) == normalize_entity_name(a) for a in alias_set):
                    matched_cname = cname
                    break
            if matched_cname is not None:
                tp += 1
                matched_gold_keys.add(matched_cname)

        fp = total - tp
        fn = len(gold_names_map) - len(matched_gold_keys)

        prec = tp / total if total > 0 else 0.0
        rec = tp / len(gold_names_map) if len(gold_names_map) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        result.gold_entity_precision = round(prec, 4)
        result.gold_entity_recall = round(rec, 4)
        result.gold_entity_f1 = round(f1, 4)
        result.gold_evaluation_method = gold_data.get("metadata", {}).get("annotation_method", "LLM-assisted reference labels, pending human verification")
        result.gold_entity_count = len(gold_names_map)
        result.matched_gold_entity_count = len(matched_gold_keys)
        result.tp_entity_count = tp

    return result
