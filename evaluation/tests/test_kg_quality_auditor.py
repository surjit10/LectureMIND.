# evaluation/tests/test_kg_quality_auditor.py
# Comprehensive unit & regression tests for Knowledge Graph Quality Auditor.

import json
from pathlib import Path
import pytest

from evaluation.knowledge_graph.entity_auditor import audit_entities
from evaluation.knowledge_graph.relation_auditor import audit_relations
from evaluation.knowledge_graph.prerequisite_auditor import audit_prerequisites, _detect_cycles
from evaluation.knowledge_graph.ontology_analyzer import analyze_ontology
from evaluation.knowledge_graph.graphrag_evaluator import SimpleBM25, SimpleGraphRetriever, _reciprocal_rank_fusion, evaluate_graphrag


def test_entity_fragment_and_orphan_detection():
    """Verify fragment and orphan detection on synthetic entities."""
    entities = [
        {"entity_id": "e1", "name": "Transformer", "type": "Concept"},
        {"entity_id": "e2", "name": "us a sinusoidal waveform. This is important because", "type": "Concept"},
        {"entity_id": "e3", "name": "Step-Up Transformer", "type": "Concept"},
        {"entity_id": "e4", "name": "Step Up Transformer", "type": "Concept"},
        {"entity_id": "e5", "name": "Isolated Node", "type": "Concept"},
    ]
    relations = [
        {"relation_id": "r1", "source_entity_id": "e1", "relation": "INTRODUCED_BEFORE", "target_entity_id": "e3"},
    ]

    res = audit_entities(entities, relations=relations)

    assert res.total_entities == 5
    assert res.fragment_count == 1
    assert any(i["issue_type"] == "FRAGMENT" for i in res.issues)
    assert any(i["issue_type"] == "ORPHAN" and i["entity_name"] == "Isolated Node" for i in res.issues)
    assert res.duplicate_groups_count == 1  # Step-Up Transformer vs Step Up Transformer


def test_relation_auditor_dangling_and_direction():
    """Verify dangling entity and direction error detection."""
    entities = [
        {"entity_id": "e1", "name": "Magnetic Field", "type": "Concept"},
        {"entity_id": "e2", "name": "Electromotive Force", "type": "Concept"},
    ]
    relations = [
        {
            "relation_id": "r1",
            "source_entity_id": "e1",
            "relation": "DERIVED_FROM",
            "target_entity_id": "e2",
        },
        {
            "relation_id": "r2",
            "source_entity_id": "e1",
            "relation": "USED_BY",
            "target_entity_id": "non_existent_e99",
        },
    ]
    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "The magnetic field constantly disturbs electrons. This movement is known as electromotive force.",
            "start_time": 0.0,
            "end_time": 10.0,
        }
    ]

    res = audit_relations(relations, entities, chunks)

    assert res.total_relations == 2
    assert res.dangling_entity_count == 1
    assert res.suspicious_direction_count == 1
    assert any(i["issue_type"] == "SUSPICIOUS_DIRECTION" for i in res.issues)
    assert any(i["issue_type"] == "DANGLING_ENTITY" for i in res.issues)


def test_evidence_tiering():
    """Verify distinct evidence tiers: DIRECT_EVIDENCE vs CO_OCCURRENCE_ONLY vs NO_EVIDENCE."""
    entities = [
        {"entity_id": "e1", "name": "Transformer", "type": "Concept"},
        {"entity_id": "e2", "name": "Alternating Current", "type": "Concept"},
        {"entity_id": "e3", "name": "Solar Panel", "type": "Concept"},
    ]
    relations = [
        {"relation_id": "r1", "source_entity_id": "e2", "relation": "USED_BY", "target_entity_id": "e1"},
        {"relation_id": "r2", "source_entity_id": "e3", "relation": "USED_BY", "target_entity_id": "e1"},
    ]
    chunks = [
        {
            "chunk_id": "chunk_1",
            "text": "Transformers can only work using alternating current. Here is another sentence about nothing.",
            "start_time": 0.0,
            "end_time": 10.0,
        }
    ]

    res = audit_relations(relations, entities, chunks)
    r1_detail = next(d for d in res.relation_details if d["relation_id"] == "r1")
    r2_detail = next(d for d in res.relation_details if d["relation_id"] == "r2")

    assert r1_detail["evidence_tier"] == "DIRECT_EVIDENCE"
    assert r1_detail["lecture_supported"] is True
    assert r2_detail["evidence_tier"] == "NO_EVIDENCE"
    assert r2_detail["lecture_supported"] is False


def test_prerequisite_cycle_detection():
    """Verify DFS cycle and self-loop detection in prerequisite graphs."""
    edges = [("A", "B"), ("B", "C"), ("C", "A")]
    cycles = _detect_cycles(edges)
    assert len(cycles) >= 1
    assert set(cycles[0]) == {"A", "B", "C"}
    assert cycles[0][0] == cycles[0][-1]  # starts and ends on same node

    prereqs = [
        {"source_name": "A", "target_name": "B", "confidence": 0.9, "signals": {"graph": 1.0, "discourse": 0.0}},
        {"source_name": "B", "target_name": "C", "confidence": 0.9, "signals": {"graph": 1.0, "discourse": 0.0}},
        {"source_name": "C", "target_name": "A", "confidence": 0.9, "signals": {"graph": 1.0, "discourse": 0.0}},
        {"source_name": "D", "target_name": "D", "confidence": 0.9, "signals": {"graph": 1.0, "discourse": 0.0}},
    ]
    entities = [{"entity_id": x, "name": x} for x in ["A", "B", "C", "D"]]
    chunks = [{"chunk_id": "c1", "text": "A B C D", "start_time": 0.0, "end_time": 10.0}]

    res = audit_prerequisites(prereqs, entities, chunks)
    assert res.is_dag is False
    assert res.cycle_count >= 1
    assert res.self_loop_count == 1
    assert any(i["issue_type"] == "SELF_LOOP" for i in res.issues)


def test_ontology_analysis_breakdown():
    """Verify ontology analysis categorizes USED_BY correctly."""
    entities = [
        {"entity_id": "e1", "name": "Iron Core", "type": "Concept"},
        {"entity_id": "e2", "name": "Alternating Current", "type": "Concept"},
        {"entity_id": "e3", "name": "Three Phase Configuration", "type": "Concept"},
        {"entity_id": "e4", "name": "Transformer", "type": "Concept"},
    ]
    relations = [
        {"relation_id": "r1", "source_entity_id": "e1", "relation": "USED_BY", "target_entity_id": "e4"},
        {"relation_id": "r2", "source_entity_id": "e2", "relation": "USED_BY", "target_entity_id": "e4"},
        {"relation_id": "r3", "source_entity_id": "e3", "relation": "USED_BY", "target_entity_id": "e4"},
    ]

    report = analyze_ontology(relations, entities)
    assert report.total_relations == 3
    assert report.used_by_breakdown["total_used_by"] == 3
    assert report.used_by_breakdown["roles"]["COMPONENT_OF"]["count"] == 1
    assert report.used_by_breakdown["roles"]["FUNCTIONAL_INPUT"]["count"] == 1
    assert report.used_by_breakdown["roles"]["SYSTEM_TOPOLOGY"]["count"] == 1


def test_graphrag_downstream_evaluator():
    """Verify Vector, Graph, and Hybrid ranking evaluation logic."""
    chunks = [
        {"chunk_id": "c1", "text": "Transformers operate using alternating current.", "ocr_text": ""},
        {"chunk_id": "c2", "text": "Magnetic fields induce electromotive force in the coil.", "ocr_text": ""},
    ]
    entities = [
        {"entity_id": "e1", "name": "Transformer", "type": "Concept"},
        {"entity_id": "e2", "name": "Alternating Current", "type": "Concept"},
    ]
    relations = [
        {"relation_id": "r1", "source_entity_id": "e2", "relation": "USED_BY", "target_entity_id": "e1"},
    ]
    bench = [
        {
            "query": "Why does a transformer require alternating current?",
            "expected_chunk_ids": ["c1"],
            "question_type": "relational",
        }
    ]

    res = evaluate_graphrag(chunks, entities, relations, bench)
    assert res.total_queries == 1
    assert res.bm25_metrics["Hit@1"] == 1.0
    assert res.vector_rag_metrics["Hit@1"] == 1.0
    assert res.hybrid_rag_metrics["Hit@1"] == 1.0


def test_direction_accounting_breakdown():
    """Verify exact breakdown of correct, reversed, type mismatch, and unmatched relations."""
    entities = [
        {"entity_id": "e1", "name": "Magnetic Field", "type": "Concept"},
        {"entity_id": "e2", "name": "Electromotive Force", "type": "Concept"},
        {"entity_id": "e3", "name": "Transformer", "type": "Concept"},
        {"entity_id": "e4", "name": "Alternating Current", "type": "Concept"},
        {"entity_id": "e5", "name": "Unknown Concept", "type": "Concept"},
    ]
    # Gold reference relations:
    # 1. ("electromotive force", "DERIVED_FROM", "magnetic field")
    # 2. ("alternating current", "USED_BY", "transformer")
    relations = [
        # Correct forward match
        {"relation_id": "r1", "source_entity_id": "e4", "relation": "USED_BY", "target_entity_id": "e3"},
        # Reversed direction
        {"relation_id": "r2", "source_entity_id": "e1", "relation": "DERIVED_FROM", "target_entity_id": "e2"},
        # Type mismatch
        {"relation_id": "r3", "source_entity_id": "e4", "relation": "PREREQUISITE_OF", "target_entity_id": "e3"},
        # Unmatched relation
        {"relation_id": "r4", "source_entity_id": "e1", "relation": "EXPLAINS", "target_entity_id": "e5"},
    ]
    chunks = [
        {"chunk_id": "c1", "transcript": "Transformers use alternating current. Changing magnetic fields induce EMF.", "start_time": 0.0, "end_time": 10.0}
    ]

    # Create temporary gold relations file
    import tempfile
    gold_data = {
        "annotation_method": "LLM-assisted reference labels, pending human verification",
        "relations": [
            {"source_name": "Electromotive Force", "relation": "DERIVED_FROM", "target_name": "Magnetic Field"},
            {"source_name": "Alternating Current", "relation": "USED_BY", "target_name": "Transformer"},
        ]
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(gold_data, tf)
        tf_path = tf.name

    try:
        res = audit_relations(relations, entities, chunks, gold_relations_path=tf_path)
        assert res.total_relations == 4
        assert res.correct_direction_count == 1
        assert res.reversed_direction_count == 1
        assert res.relation_type_mismatch_count == 1
        assert res.unmatched_relation_count == 1
        # Total accounted for: 1 + 1 + 1 + 1 = 4
        assert res.correct_direction_count + res.reversed_direction_count + res.relation_type_mismatch_count + res.unmatched_relation_count == res.total_relations
        # Direction accuracy among gold matched pairs (1 correct out of 2 directional candidates: r1 and r2)
        assert res.gold_matched_direction_accuracy == 0.5
        # Overall acceptance rate: 1 / 4 = 0.25
        assert res.overall_direction_accuracy == 0.25
    finally:
        Path(tf_path).unlink(missing_ok=True)


def test_transcript_key_in_multimodal_chunks():
    """Verify that chunks with 'transcript' instead of 'text' are properly parsed for evidence."""
    entities = [
        {"entity_id": "e1", "name": "Primary Coil", "type": "Concept"},
        {"entity_id": "e2", "name": "Secondary Coil", "type": "Concept"},
    ]
    relations = [
        {"relation_id": "r1", "source_entity_id": "e1", "relation": "CONNECTED_TO", "target_entity_id": "e2"},
    ]
    chunks = [
        {
            "chunk_id": "c1",
            # Note: NO 'text' field, only 'transcript'
            "transcript": "The primary coil is inductively coupled and connected to the secondary coil via the magnetic core.",
            "start_time": 0.0,
            "end_time": 10.0,
        }
    ]

    res = audit_relations(relations, entities, chunks)
    r1 = res.relation_details[0]
    assert r1["evidence_tier"] == "DIRECT_EVIDENCE"
    assert r1["is_cross_chunk"] is False
    assert r1["lecture_supported"] is True


def test_strict_bipartite_prerequisite_matching():
    """Verify strict 1-to-1 exact matching prevents greedy substring theft and duplicate claiming."""
    import tempfile

    entities = [
        {"entity_id": "e1", "name": "Alternating Current"},
        {"entity_id": "e2", "name": "Current"},
        {"entity_id": "e3", "name": "Transformer"},
        {"entity_id": "e4", "name": "Coil"},
        {"entity_id": "e5", "name": "Step Up Transformer"},
    ]
    chunks = [{"chunk_id": "c1", "text": "lecture text", "start_time": 0.0, "end_time": 10.0}]

    # Inferred edges:
    # 1. Exact match for gold edge 1
    # 2. Substring competitor for gold edge 1 (Current -> Transformer)
    # 3. Exact match for gold edge 2
    # 4. Substring competitor for gold edge 2 (Coil -> Transformer)
    inferred = [
        {"source_name": "Alternating Current", "target_name": "Transformer", "confidence": 0.95, "signals": {"graph": 1.0, "discourse": 0.0}},
        {"source_name": "Current", "target_name": "Transformer", "confidence": 0.80, "signals": {"graph": 0.0, "discourse": 0.85}},
        {"source_name": "Coil", "target_name": "Step Up Transformer", "confidence": 0.90, "signals": {"graph": 0.0, "discourse": 0.90}},
        {"source_name": "Coil", "target_name": "Transformer", "confidence": 0.70, "signals": {"graph": 0.0, "discourse": 0.70}},
    ]

    gold_data = {
        "metadata": {"annotation_method": "expert reference"},
        "prerequisites": [
            {"source_name": "Alternating Current", "target_name": "Transformer"},
            {"source_name": "Coil", "target_name": "Step Up Transformer"},
        ]
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
        json.dump(gold_data, tf)
        tf_path = tf.name

    try:
        res = audit_prerequisites(inferred, entities, chunks, gold_prerequisites_path=tf_path)
        # Strict exact metrics:
        # TP = 2 (Alternating Current -> Transformer, Coil -> Step Up Transformer)
        # FP = 2 (Current -> Transformer, Coil -> Transformer)
        # FN = 0
        assert res.gold_prerequisite_precision == 0.50
        assert res.gold_prerequisite_recall == 1.0
        assert res.gold_prerequisite_f1 == 0.6667
        assert res.matched_gold_prerequisite_count == 2
        assert res.tp_prerequisite_count == 2

        # Verify details dictionary structure
        details = res.edge_details[-1]
        assert details["strict_metrics"]["tp"] == 2
        assert details["strict_metrics"]["fp"] == 2
        assert details["strict_metrics"]["fn"] == 0

        # Fuzzy metrics also present
        assert res.fuzzy_prerequisite_precision is not None
        assert res.fuzzy_prerequisite_recall is not None
    finally:
        Path(tf_path).unlink(missing_ok=True)
