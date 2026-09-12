# evaluation/knowledge_graph/ontology_analyzer.py
# Diagnostic Relation Ontology Analyzer for LectureMIND.
#
# Analyzes the semantic distribution of relation types without mutating the ontology.
# Diagnoses the semantic roles of overloaded relations (particularly USED_BY).

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class OntologyDiagnosticReport:
    total_relations: int
    type_distribution: Dict[str, int]
    used_by_breakdown: Dict[str, Any]
    recommendations: List[str]


def analyze_ontology(
    relations: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
) -> OntologyDiagnosticReport:
    """
    Produce a diagnostic analysis of ontology distribution and semantic overloading.
    """
    total = len(relations)
    entity_map = {e.get("entity_id", ""): e.get("name", "") for e in entities}

    type_counts = Counter(
        (r.get("relation") or r.get("relation_type", "")).strip() for r in relations
    )

    # Analyze USED_BY semantic roles
    used_by_roles = {
        "COMPONENT_OF": [],        # Physical parts/materials
        "FUNCTIONAL_INPUT": [],     # Electrical quantities/operating power
        "SYSTEM_TOPOLOGY": [],      # Wiring/configuration
        "OTHER": [],
    }

    component_keywords = {"core", "coil", "laminated", "sheet", "winding"}
    input_keywords = {"current", "voltage", "alternating current", "ac", "dc", "power"}
    topology_keywords = {"phase", "delta", "star", "wye", "connection", "configuration"}

    for r in relations:
        rel_type = (r.get("relation") or r.get("relation_type", "")).strip()
        if rel_type != "USED_BY":
            continue

        src_id = r.get("source_entity_id") or r.get("source_id", "")
        tgt_id = r.get("target_entity_id") or r.get("target_id", "")
        src = entity_map.get(src_id, src_id).lower()
        tgt = entity_map.get(tgt_id, tgt_id).lower()
        rel_str = f"{entity_map.get(src_id, src_id)} -> USED_BY -> {entity_map.get(tgt_id, tgt_id)}"

        if any(kw in src for kw in component_keywords):
            used_by_roles["COMPONENT_OF"].append(rel_str)
        elif any(kw in src for kw in input_keywords):
            used_by_roles["FUNCTIONAL_INPUT"].append(rel_str)
        elif any(kw in src for kw in topology_keywords):
            used_by_roles["SYSTEM_TOPOLOGY"].append(rel_str)
        else:
            used_by_roles["OTHER"].append(rel_str)

    used_by_total = sum(len(v) for v in used_by_roles.values())
    used_by_summary = {
        "total_used_by": used_by_total,
        "percentage_of_all_relations": round((used_by_total / total) * 100, 1) if total > 0 else 0.0,
        "roles": {
            "COMPONENT_OF": {
                "count": len(used_by_roles["COMPONENT_OF"]),
                "examples": used_by_roles["COMPONENT_OF"][:4],
                "description": "Physical/structural hardware components forming the machine.",
            },
            "FUNCTIONAL_INPUT": {
                "count": len(used_by_roles["FUNCTIONAL_INPUT"]),
                "examples": used_by_roles["FUNCTIONAL_INPUT"][:4],
                "description": "Physical quantities or power sources required for operation.",
            },
            "SYSTEM_TOPOLOGY": {
                "count": len(used_by_roles["SYSTEM_TOPOLOGY"]),
                "examples": used_by_roles["SYSTEM_TOPOLOGY"][:4],
                "description": "Electrical wiring topology or multi-phase organizational layouts.",
            },
            "OTHER": {
                "count": len(used_by_roles["OTHER"]),
                "examples": used_by_roles["OTHER"][:4],
                "description": "Miscellaneous associations mapped to USED_BY.",
            },
        },
    }

    recommendations = [
        "USED_BY currently accounts for over 50% of extracted relations, serving as an ontological catch-all.",
        "Introduce COMPONENT_OF (or HAS_PART) in future schema version to distinguish physical parts (Iron Core, Coil) from operational inputs (Alternating Current, Voltage).",
        "Preserve backwards compatibility by maintaining USED_BY as an allowed alias or sub-property during future migrations.",
    ]

    return OntologyDiagnosticReport(
        total_relations=total,
        type_distribution=dict(type_counts),
        used_by_breakdown=used_by_summary,
        recommendations=recommendations,
    )
