# evaluation/knowledge_graph/canonical_relation_matching.py
"""
Evaluation-only Canonical Relation Normalization.

This module provides deterministic mapping from surface predicates and domain-specific
naming variations into the closed LectureMIND RelationType ontology:
    - PREREQUISITE_OF
    - INTRODUCED_BEFORE
    - USED_BY
    - DERIVED_FROM
    - VISUALIZED_BY
    - EXPLAINS

Rule: This is evaluation-only. It NEVER modifies production knowledge graphs or extraction.
Applied symmetrically to predicted relations and gold annotations.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from evaluation.knowledge_graph.name_matching import normalize_name
from schemas.enums import RelationType

# Canonical ontology mappings: (canonical_relation_name, is_inverted)
# is_inverted = True indicates (A, rel, B) corresponds to (B, canonical_rel, A)
_RELATION_SYNONYM_MAP: Dict[str, Tuple[str, bool]] = {
    # PREREQUISITE_OF
    "prerequisite_of": ("PREREQUISITE_OF", False),
    "is_prerequisite_of": ("PREREQUISITE_OF", False),
    "precedes_conceptually": ("PREREQUISITE_OF", False),
    "depends_on": ("PREREQUISITE_OF", True),
    "requires": ("PREREQUISITE_OF", True),
    "prerequisite": ("PREREQUISITE_OF", True),
    "needs": ("PREREQUISITE_OF", True),
    "relies_on": ("PREREQUISITE_OF", True),

    # USED_BY
    "used_by": ("USED_BY", False),
    "utilized_by": ("USED_BY", False),
    "consumed_by": ("USED_BY", False),
    "operated_on_by": ("USED_BY", False),
    "is_used_by": ("USED_BY", False),
    "uses": ("USED_BY", True),
    "operates_on": ("USED_BY", True),
    "utilizes": ("USED_BY", True),
    "employs": ("USED_BY", True),
    "applies_to": ("USED_BY", True),
    "calls": ("USED_BY", True),
    "invokes": ("USED_BY", True),

    # DERIVED_FROM
    "derived_from": ("DERIVED_FROM", False),
    "calculated_from": ("DERIVED_FROM", False),
    "computed_from": ("DERIVED_FROM", False),
    "built_from": ("DERIVED_FROM", False),
    "sourced_from": ("DERIVED_FROM", False),
    "induced_by": ("DERIVED_FROM", False),
    "derives": ("DERIVED_FROM", True),
    "produces": ("DERIVED_FROM", True),
    "yields": ("DERIVED_FROM", True),
    "forms": ("DERIVED_FROM", True),
    "creates": ("DERIVED_FROM", True),
    "generates": ("DERIVED_FROM", True),
    "induces": ("DERIVED_FROM", True),
    "constitutes": ("DERIVED_FROM", True),

    # INTRODUCED_BEFORE
    "introduced_before": ("INTRODUCED_BEFORE", False),
    "taught_before": ("INTRODUCED_BEFORE", False),
    "presented_before": ("INTRODUCED_BEFORE", False),
    "prior_to": ("INTRODUCED_BEFORE", False),
    "introduced_after": ("INTRODUCED_BEFORE", True),
    "taught_after": ("INTRODUCED_BEFORE", True),
    "follows": ("INTRODUCED_BEFORE", True),

    # EXPLAINS
    "explains": ("EXPLAINS", False),
    "describes": ("EXPLAINS", False),
    "illustrates": ("EXPLAINS", False),
    "defines": ("EXPLAINS", False),
    "clarifies": ("EXPLAINS", False),
    "modeled_by": ("EXPLAINS", False),
    "explained_by": ("EXPLAINS", True),
    "described_by": ("EXPLAINS", True),
    "defined_by": ("EXPLAINS", True),

    # VISUALIZED_BY
    "visualized_by": ("VISUALIZED_BY", False),
    "shown_in": ("VISUALIZED_BY", False),
    "depicted_by": ("VISUALIZED_BY", False),
    "visualizes": ("VISUALIZED_BY", True),
    "depicts": ("VISUALIZED_BY", True),
}


def normalize_predicate(predicate: str) -> Tuple[str, bool]:
    """
    Map a predicate string to its canonical RelationType and direction flag.

    Returns:
        (canonical_relation, is_inverted)
        If predicate is unrecognized, returns (cleaned_uppercase_predicate, False).
    """
    clean = re.sub(r"[^a-zA-Z0-9_]+", "_", predicate.strip().lower()).strip("_")
    if clean in _RELATION_SYNONYM_MAP:
        return _RELATION_SYNONYM_MAP[clean]

    # Check if exact match to closed RelationType enum
    for rt in RelationType:
        if clean == rt.value.lower():
            return (rt.value, False)

    return (predicate.strip().upper(), False)


def canonicalize_triple(source: str, relation: str, target: str) -> Tuple[str, str, str]:
    """
    Normalize endpoints and map relation to canonical direction.

    Returns:
        (canonical_source, canonical_relation, canonical_target)
    """
    s_norm = normalize_name(source)
    t_norm = normalize_name(target)
    canon_rel, inverted = normalize_predicate(relation)

    if inverted:
        return (t_norm, canon_rel, s_norm)
    return (s_norm, canon_rel, t_norm)


def evaluate_relations_dual(
    predicted_triples: List[Tuple[str, str, str]],
    gold_triples: List[Tuple[str, str, str]],
) -> Dict[str, Any]:
    """
    Compute both Exact String F1 and Canonical/Domain-Aware F1.

    Args:
        predicted_triples: List of (source_name, relation, target_name)
        gold_triples: List of (source_name, relation, target_name)

    Returns:
        Dictionary with exact and canonical precision, recall, F1, and breakdown.
    """
    total_pred = len(predicted_triples)
    total_gold = len(gold_triples)

    # 1. Exact string match (case/whitespace normalized)
    gold_exact_set = {
        (normalize_name(s), r.strip().upper(), normalize_name(t))
        for s, r, t in gold_triples
    }

    exact_tp = 0
    claimed_exact: Set[Tuple[str, str, str]] = set()
    for s, r, t in predicted_triples:
        cand = (normalize_name(s), r.strip().upper(), normalize_name(t))
        if cand in gold_exact_set and cand not in claimed_exact:
            exact_tp += 1
            claimed_exact.add(cand)

    exact_p = exact_tp / total_pred if total_pred > 0 else 0.0
    exact_r = exact_tp / total_gold if total_gold > 0 else 0.0
    exact_f1 = (2 * exact_p * exact_r / (exact_p + exact_r)) if (exact_p + exact_r) > 0 else 0.0

    # 2. Canonical ontology match
    gold_canonical_set = {
        canonicalize_triple(s, r, t)
        for s, r, t in gold_triples
    }

    canonical_tp = 0
    claimed_canon: Set[Tuple[str, str, str]] = set()
    matches_breakdown: List[Dict[str, Any]] = []

    for s, r, t in predicted_triples:
        canon_triple = canonicalize_triple(s, r, t)
        if canon_triple in gold_canonical_set and canon_triple not in claimed_canon:
            canonical_tp += 1
            claimed_canon.add(canon_triple)
            matches_breakdown.append({
                "predicted": f"{s} ==[{r}]==> {t}",
                "canonical": f"{canon_triple[0]} ==[{canon_triple[1]}]==> {canon_triple[2]}",
                "status": "CANONICAL_MATCH",
            })

    canon_p = canonical_tp / total_pred if total_pred > 0 else 0.0
    canon_r = canonical_tp / total_gold if total_gold > 0 else 0.0
    canon_f1 = (2 * canon_p * canon_r / (canon_p + canon_r)) if (canon_p + canon_r) > 0 else 0.0

    return {
        "total_predicted": total_pred,
        "total_gold": total_gold,
        "exact_metrics": {
            "true_positives": exact_tp,
            "precision": round(exact_p, 4),
            "recall": round(exact_r, 4),
            "f1": round(exact_f1, 4),
            "description": "Strict exact string-match on predicate and normalized endpoints",
        },
        "canonical_metrics": {
            "true_positives": canonical_tp,
            "precision": round(canon_p, 4),
            "recall": round(canon_r, 4),
            "f1": round(canon_f1, 4),
            "description": "Ontology-aware normalized matching using closed relation taxonomy and directional inverses",
        },
        "canonical_matches": matches_breakdown,
    }
