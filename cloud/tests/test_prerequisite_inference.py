# cloud/tests/test_prerequisite_inference.py
# Comprehensive test suite for Socratic Prerequisite Inference.

import pytest
from cloud.extraction.prerequisite_inference import (
    DiscourseCueDetector,
    generate_prerequisite_candidates,
    score_prerequisite_candidate,
    enforce_dag_cycles,
    infer_prerequisites,
)
from schemas.entity import Entity
from schemas.chunk import MultimodalChunk
from schemas.segment import LectureSegment
from schemas.relation import Relation
from schemas.enums import EntityType, RelationType


def test_discourse_cue_detection():
    detector = DiscourseCueDetector()

    # 1. "Before we move on to page faults, let's first understand page tables."
    text1 = "Before we move on to page faults, let's first understand page tables."
    score1, pat1, snip1 = detector.detect(text1, "Page Table", "Page Fault")
    assert score1 >= 0.80
    assert pat1 is not None

    # 2. "You need to know X in order to understand Y."
    text2 = "You need to understand virtual memory in order to understand paging."
    score2, pat2, _ = detector.detect(text2, "Virtual Memory", "Paging")
    assert score2 >= 0.85

    # 3. "Y relies on the mechanism we discussed earlier, X."
    text3 = "Thrashing relies on the mechanism we discussed earlier, page fault."
    score3, pat3, _ = detector.detect(text3, "Page Fault", "Thrashing")
    assert score3 >= 0.85

    # 4. "Recall X because we'll use it here."
    text4 = "Recall address space because we will use it here in virtual memory."
    score4, pat4, _ = detector.detect(text4, "Address Space", "Virtual Memory")
    assert score4 >= 0.80

    # Negative case: unrelated text
    text_unrelated = "In this section we discuss operating system design and history."
    score_unrelated, _, _ = detector.detect(text_unrelated, "Page Table", "Thrashing")
    assert score_unrelated == 0.0


def test_candidate_prefiltering_and_metrics_reporting():
    """Verify search space reduction and empirical metric reporting without arbitrary hard-coded claims."""
    # Create 10 entities
    entities = [
        Entity(entity_id=f"ent_{i:03d}", name=f"Concept {i}", type=EntityType.Concept)
        for i in range(10)
    ]
    # Chunks spreading entities out over time
    chunks = [
        MultimodalChunk(
            lecture_id="lec_001",
            chunk_id=f"chunk_{i:03d}",
            timestamp=float(i * 30),
            transcript=f"Concept {i} introduced here.",
            visual_context="",
            ocr_text="",
        )
        for i in range(10)
    ]
    segments = [
        LectureSegment(
            segment_id="seg_001",
            title="Introduction",
            start=0.0,
            end=150.0,
            chunks=[f"chunk_{i:03d}" for i in range(5)],
        ),
        LectureSegment(
            segment_id="seg_002",
            title="Advanced Topics",
            start=150.0,
            end=300.0,
            chunks=[f"chunk_{i:03d}" for i in range(5, 10)],
        ),
    ]
    # Existing relation only between 0 and 1
    relations = [
        Relation(
            relation_id="rel_001",
            source_entity_id="ent_000",
            relation=RelationType.EXPLAINS,
            target_entity_id="ent_001",
        )
    ]

    candidates, metrics = generate_prerequisite_candidates(
        entities=entities,
        chunks=chunks,
        segments=segments,
        relations=relations,
    )

    # 10 entities -> 90 possible directed pairs
    assert metrics["total_possible_pairs"] == 90
    assert metrics["evaluated_pairs"] == 90
    assert "rejections" in metrics
    assert metrics["rejections"]["temporal_order"] > 0
    # Must have significantly reduced search space
    assert metrics["accepted_candidates"] < metrics["total_possible_pairs"]
    assert len(candidates) == metrics["accepted_candidates"]


def test_hard_temporal_gate():
    """Verify t(A) >= t(B) strictly fails (no backward causality)."""
    detector = DiscourseCueDetector()
    candidate = {
        "source_id": "ent_002",
        "source_name": "Thrashing",
        "source_type": "Concept",
        "target_id": "ent_001",
        "target_name": "Page Table",
        "target_type": "Concept",
        "t_source": 50.0,
        "t_target": 10.0,
        "existing_relations": [],
    }
    chunks = [
        MultimodalChunk(
            lecture_id="lec_001",
            chunk_id="chunk_001",
            timestamp=50.0,
            transcript="Page Table is a prerequisite for Thrashing.",
            visual_context="",
            ocr_text="",
        )
    ]
    # In candidate generation, reverse temporal order is rejected
    entities = [
        Entity(entity_id="ent_001", name="Page Table", type=EntityType.Concept),
        Entity(entity_id="ent_002", name="Thrashing", type=EntityType.Concept),
    ]
    segments = [
        LectureSegment(
            segment_id="seg_001",
            title="Overview",
            start=0.0,
            end=100.0,
            chunks=["chunk_001"],
        )
    ]
    candidates, metrics = generate_prerequisite_candidates(entities, chunks, segments, [])
    # Thrashing -> Page Table must be rejected by temporal gate
    thrashing_to_pt = [c for c in candidates if c["source_id"] == "ent_002" and c["target_id"] == "ent_001"]
    assert len(thrashing_to_pt) == 0


def test_hard_evidence_anchor_and_synthetic_scenario():
    """
    CRITICAL USER CORRECTION 2:
    Temporal order alone must NOT make edges pass.
    Scenario:
      t=10: Process
      t=20: Address Space
      t=30: Page Table
      t=40: Page Fault
      t=50: Thrashing
    Verify:
      - Page Table -> Page Fault -> Thrashing passes because of discourse/graph evidence.
      - Process -> Thrashing strictly FAILS because merely appearing earlier is insufficient.
    """
    entities = [
        Entity(entity_id="e_proc", name="Process", type=EntityType.Concept),
        Entity(entity_id="e_addr", name="Address Space", type=EntityType.Concept),
        Entity(entity_id="e_pt", name="Page Table", type=EntityType.Concept),
        Entity(entity_id="e_pf", name="Page Fault", type=EntityType.Concept),
        Entity(entity_id="e_thr", name="Thrashing", type=EntityType.Concept),
    ]

    chunks = [
        MultimodalChunk(
            lecture_id="lec_os",
            chunk_id="c_01",
            timestamp=10.0,
            transcript="Today we begin by discussing the Process abstraction in modern OS.",
            visual_context="Slide: Process Concept",
            ocr_text="Process Management",
        ),
        MultimodalChunk(
            lecture_id="lec_os",
            chunk_id="c_02",
            timestamp=20.0,
            transcript="Every process has its own virtual Address Space.",
            visual_context="",
            ocr_text="",
        ),
        MultimodalChunk(
            lecture_id="lec_os",
            chunk_id="c_03",
            timestamp=30.0,
            transcript="To map virtual addresses, the OS maintains a Page Table.",
            visual_context="Slide: Page Table Architecture",
            ocr_text="Page Table Structures",
        ),
        MultimodalChunk(
            lecture_id="lec_os",
            chunk_id="c_04",
            timestamp=40.0,
            transcript="Before we move on to page faults, let's first understand page tables.",
            visual_context="",
            ocr_text="",
        ),
        MultimodalChunk(
            lecture_id="lec_os",
            chunk_id="c_05",
            timestamp=50.0,
            transcript="Thrashing relies on the mechanism we discussed earlier, page fault.",
            visual_context="Slide: System Thrashing",
            ocr_text="Thrashing & Working Set",
        ),
    ]

    segments = [
        LectureSegment(
            segment_id="s_01",
            title="Processes & Address Spaces",
            start=0.0,
            end=25.0,
            chunks=["c_01", "c_02"],
        ),
        LectureSegment(
            segment_id="s_02",
            title="Paging & Faults",
            start=25.0,
            end=45.0,
            chunks=["c_03", "c_04"],
        ),
        LectureSegment(
            segment_id="s_03",
            title="Performance Issues",
            start=45.0,
            end=60.0,
            chunks=["c_05"],
        ),
    ]

    # Existing relation connecting Page Table -> Page Fault
    relations = [
        Relation(
            relation_id="rel_01",
            source_entity_id="e_pt",
            relation=RelationType.USED_BY,
            target_entity_id="e_pf",
        )
    ]

    result = infer_prerequisites(
        entities=entities,
        chunks=chunks,
        segments=segments,
        relations=relations,
        min_confidence=0.60,
    )

    prereqs = result["prerequisites"]
    pairs = [(p["source_name"], p["target_name"]) for p in prereqs]

    # Page Table -> Page Fault MUST be present
    assert ("Page Table", "Page Fault") in pairs

    # Page Fault -> Thrashing MUST be present
    assert ("Page Fault", "Thrashing") in pairs

    # Process -> Thrashing MUST NOT be present (temporal precedence alone is insufficient!)
    assert ("Process", "Thrashing") not in pairs

    # Check the Process -> Thrashing candidate score explicitly
    cand_proc_thr = {
        "source_id": "e_proc",
        "source_name": "Process",
        "source_type": "Concept",
        "target_id": "e_thr",
        "target_name": "Thrashing",
        "target_type": "Concept",
        "t_source": 10.0,
        "t_target": 50.0,
        "source_in_headings": True,  # even with heading prominence
        "source_in_first_segment": True,  # even in first segment
        "existing_relations": [],  # no relation
    }
    scored_proc_thr = score_prerequisite_candidate(cand_proc_thr, chunks)
    assert scored_proc_thr["confidence"] == 0.0, "Candidate with 0 discourse and 0 graph evidence must have confidence 0.0"


def test_deterministic_dag_cycle_resolution():
    """Verify that directed cycles are broken by deterministically dropping min(confidence) edge."""
    # Cycle: A -> B (0.90), B -> C (0.80), C -> A (0.65)
    cyclic_edges = [
        {"source_id": "A", "target_id": "B", "source_name": "A", "target_name": "B", "confidence": 0.90},
        {"source_id": "B", "target_id": "C", "source_name": "B", "target_name": "C", "confidence": 0.80},
        {"source_id": "C", "target_id": "A", "source_name": "C", "target_name": "A", "confidence": 0.65},
    ]

    dag_edges = enforce_dag_cycles(cyclic_edges)
    assert len(dag_edges) == 2

    edge_pairs = [(e["source_id"], e["target_id"]) for e in dag_edges]
    assert ("A", "B") in edge_pairs
    assert ("B", "C") in edge_pairs
    assert ("C", "A") not in edge_pairs  # 0.65 was lowest, dropped!


def test_confidence_threshold_sweep():
    """Verify configurable threshold filtering behavior."""
    entities = [
        Entity(entity_id="e1", name="Alpha", type=EntityType.Concept),
        Entity(entity_id="e2", name="Beta", type=EntityType.Concept),
    ]
    chunks = [
        MultimodalChunk(
            lecture_id="lec_1",
            chunk_id="c1",
            timestamp=10.0,
            transcript="Alpha is introduced as our core foundation.",
            visual_context="",
            ocr_text="",
        ),
        MultimodalChunk(
            lecture_id="lec_1",
            chunk_id="c2",
            timestamp=20.0,
            transcript="Beta is discussed here. Alpha is needed for Beta.",
            visual_context="",
            ocr_text="",
        ),
    ]
    segments = [
        LectureSegment(
            segment_id="s1",
            title="Intro",
            start=0.0,
            end=30.0,
            chunks=["c1", "c2"],
        )
    ]

    # At lower threshold, passes
    res_low = infer_prerequisites(entities, chunks, segments, [], min_confidence=0.50)
    assert len(res_low["prerequisites"]) == 1

    # At extremely high threshold (> 0.99), filtered out
    res_high = infer_prerequisites(entities, chunks, segments, [], min_confidence=0.99)
    assert len(res_high["prerequisites"]) == 0
