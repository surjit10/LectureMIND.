# evaluation/tests/test_canonical_relation_matching.py
"""Unit tests for canonical relation normalization and dual evaluation."""

import pytest
from evaluation.knowledge_graph.canonical_relation_matching import (
    canonicalize_triple,
    evaluate_relations_dual,
    normalize_predicate,
)


def test_normalize_predicate_synonyms():
    assert normalize_predicate("prerequisite_of") == ("PREREQUISITE_OF", False)
    assert normalize_predicate("depends_on") == ("PREREQUISITE_OF", True)
    assert normalize_predicate("requires") == ("PREREQUISITE_OF", True)
    assert normalize_predicate("used_by") == ("USED_BY", False)
    assert normalize_predicate("uses") == ("USED_BY", True)
    assert normalize_predicate("operates_on") == ("USED_BY", True)
    assert normalize_predicate("derived_from") == ("DERIVED_FROM", False)
    assert normalize_predicate("forms") == ("DERIVED_FROM", True)
    assert normalize_predicate("creates") == ("DERIVED_FROM", True)
    assert normalize_predicate("introduced_before") == ("INTRODUCED_BEFORE", False)
    assert normalize_predicate("taught_after") == ("INTRODUCED_BEFORE", True)
    assert normalize_predicate("explains") == ("EXPLAINS", False)
    assert normalize_predicate("explained_by") == ("EXPLAINS", True)


def test_canonicalize_triple_inversion():
    # If A uses B -> canonical is B is used by A
    source = "Process"
    relation = "uses"
    target = "Memory"
    canon = canonicalize_triple(source, relation, target)
    assert canon == ("memory", "USED_BY", "process")

    # If A requires B -> B is prerequisite of A
    source = "Transformer"
    relation = "requires"
    target = "Alternating Current"
    canon = canonicalize_triple(source, relation, target)
    assert canon == ("alternating current", "PREREQUISITE_OF", "transformer")


def test_evaluate_relations_dual_bridges_naming_conventions():
    # Gold standard uses passive relation (Alternating Current ==[USED_BY]==> Transformer)
    # Extracted model uses active verb (Transformer ==[uses]==> Alternating Current)
    gold = [
        ("Alternating Current", "USED_BY", "Transformer"),
        ("Magnetic Field", "DERIVED_FROM", "Alternating Current"),
    ]
    predicted = [
        ("Transformer", "uses", "Alternating Current"),      # Semantically correct inverted synonym
        ("Alternating Current", "creates", "Magnetic Field"), # Semantically correct inverted synonym
    ]

    res = evaluate_relations_dual(predicted, gold)

    # Exact string match fails because "uses" != "USED_BY" and endpoints are flipped
    assert res["exact_metrics"]["true_positives"] == 0
    assert res["exact_metrics"]["f1"] == 0.0

    # Canonical evaluation succeeds by resolving inverse direction and synonyms
    assert res["canonical_metrics"]["true_positives"] == 2
    assert res["canonical_metrics"]["precision"] == 1.0
    assert res["canonical_metrics"]["recall"] == 1.0
    assert res["canonical_metrics"]["f1"] == 1.0


def test_evaluate_relations_dual_no_duplicate_claiming():
    gold = [
        ("Alternating Current", "USED_BY", "Transformer"),
    ]
    # Two predicted relations that both resolve to the same canonical gold edge
    predicted = [
        ("Alternating Current", "USED_BY", "Transformer"),
        ("Transformer", "uses", "Alternating Current"),
    ]

    res = evaluate_relations_dual(predicted, gold)
    # Only 1 TP can be claimed from 1 gold edge; second prediction is an FP
    assert res["canonical_metrics"]["true_positives"] == 1
    assert res["canonical_metrics"]["precision"] == 0.5
    assert res["canonical_metrics"]["recall"] == 1.0
