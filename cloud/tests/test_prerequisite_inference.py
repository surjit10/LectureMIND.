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
    """Verify t(A) > t(B) strictly fails (no backward causality)."""
    # In candidate generation, reverse temporal order is strictly rejected:
    # Page Table appears at t=10.0, Thrashing appears at t=50.0.
    # Thrashing (50.0) -> Page Table (10.0) must be rejected.
    chunks = [
        MultimodalChunk(
            lecture_id="lec_001",
            chunk_id="chunk_001",
            timestamp=10.0,
            transcript="First, we define Page Table architecture.",
            visual_context="",
            ocr_text="",
        ),
        MultimodalChunk(
            lecture_id="lec_001",
            chunk_id="chunk_002",
            timestamp=50.0,
            transcript="Later, Thrashing occurs when working sets exceed memory.",
            visual_context="",
            ocr_text="",
        ),
    ]
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
            chunks=["chunk_001", "chunk_002"],
        )
    ]
    candidates, metrics = generate_prerequisite_candidates(entities, chunks, segments, [])
    # Thrashing (t=50) -> Page Table (t=10) must be rejected by temporal gate
    thrashing_to_pt = [c for c in candidates if c["source_id"] == "ent_002" and c["target_id"] == "ent_001"]
    assert len(thrashing_to_pt) == 0
    assert metrics["rejections"]["temporal_order"] > 0

    # Normal forward order: Page Table (t=10) -> Thrashing (t=50) accepted to candidate evaluation
    pt_to_thrashing = [c for c in candidates if c["source_id"] == "ent_001" and c["target_id"] == "ent_002"]
    assert len(pt_to_thrashing) == 1


def test_same_timestamp_candidate_accepted_with_evidence():
    """
    Verify candidates introduced in the same chunk (timestamp(A) == timestamp(B))
    are NOT rejected solely by timestamp equality, allowing discourse/graph evidence
    to determine validity.
    """
    entities = [
        Entity(entity_id="e_ac", name="Alternating Current", type=EntityType.Concept),
        Entity(entity_id="e_tf", name="Transformer", type=EntityType.Concept),
    ]
    # Both first appear in chunk_001 at t=0.0
    chunks = [
        MultimodalChunk(
            lecture_id="lec_001",
            chunk_id="chunk_001",
            timestamp=0.0,
            transcript="Before understanding Transformer, let's first understand Alternating Current.",
            visual_context="",
            ocr_text="",
        )
    ]
    segments = [
        LectureSegment(
            segment_id="seg_001",
            title="Intro",
            start=0.0,
            end=30.0,
            chunks=["chunk_001"],
        )
    ]
    relations = [
        Relation(
            relation_id="rel_001",
            source_entity_id="e_ac",
            relation=RelationType.PREREQUISITE_OF,
            target_entity_id="e_tf",
        )
    ]

    candidates, metrics = generate_prerequisite_candidates(entities, chunks, segments, relations)
    cand_pairs = [(c["source_id"], c["target_id"]) for c in candidates]
    assert ("e_ac", "e_tf") in cand_pairs, "Same-timestamp candidate with evidence must not be rejected"

    # Full inference run: should pass threshold because of discourse and A9 relation
    result = infer_prerequisites(entities, chunks, segments, relations, min_confidence=0.65)
    inferred_pairs = [(p["source_id"], p["target_id"]) for p in result["prerequisites"]]
    assert ("e_ac", "e_tf") in inferred_pairs
    ac_to_tf = [p for p in result["prerequisites"] if p["source_id"] == "e_ac"][0]
    assert ac_to_tf["confidence"] >= 0.65
    assert ac_to_tf["signals"]["temporal"] == 0.5  # Neutral score for equal timestamps


def test_same_timestamp_without_evidence_rejected_by_hard_anchor():
    """
    Verify that equal timestamps alone without discourse or graph evidence
    remain strictly rejected with confidence 0.0 by the Hard Evidence Anchor.
    """
    entities = [
        Entity(entity_id="e_x", name="Concept X", type=EntityType.Concept),
        Entity(entity_id="e_y", name="Concept Y", type=EntityType.Concept),
    ]
    chunks = [
        MultimodalChunk(
            lecture_id="lec_001",
            chunk_id="chunk_001",
            timestamp=0.0,
            transcript="In this section we mention Concept X and Concept Y without relation.",
            visual_context="",
            ocr_text="",
        )
    ]
    segments = [
        LectureSegment(
            segment_id="seg_001",
            title="Intro",
            start=0.0,
            end=30.0,
            chunks=["chunk_001"],
        )
    ]

    result = infer_prerequisites(entities, chunks, segments, [], min_confidence=0.50)
    assert len(result["prerequisites"]) == 0, "No edges must pass without discourse or graph evidence"



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


def test_recalibrated_relation_weights():
    """Verify recalibrated relation weights prevent USED_BY/EXPLAINS from passing alone."""
    from cloud.extraction.prerequisite_inference import (
        RELATION_PREREQUISITE_WEIGHTS,
        score_prerequisite_candidate,
    )
    assert RELATION_PREREQUISITE_WEIGHTS[("PREREQUISITE_OF", True)] == 1.0
    assert RELATION_PREREQUISITE_WEIGHTS[("INTRODUCED_BEFORE", True)] == 0.70
    assert RELATION_PREREQUISITE_WEIGHTS[("DERIVED_FROM", False)] == 0.80
    assert RELATION_PREREQUISITE_WEIGHTS[("USED_BY", True)] == 0.20
    assert RELATION_PREREQUISITE_WEIGHTS[("EXPLAINS", True)] == 0.10

    # A candidate backed ONLY by USED_BY (0.20) and no discourse cue must NOT pass >= 0.65
    candidate = {
        "source_id": "e_mat",
        "source_name": "Iron Core",
        "source_type": "Concept",
        "target_id": "e_dev",
        "target_name": "Transformer",
        "target_type": "Concept",
        "t_source": 10.0,
        "t_target": 20.0,
        "source_chunk_id": "c1",
        "source_in_headings": False,
        "source_in_first_segment": False,
        "existing_forward_relations": ["USED_BY"],
        "existing_reverse_relations": [],
    }
    chunks = [
        MultimodalChunk(
            lecture_id="lec_1",
            chunk_id="c1",
            timestamp=10.0,
            transcript="Iron Core is used by Transformer.",
            visual_context="",
            ocr_text="",
        )
    ]
    scored = score_prerequisite_candidate(candidate, chunks)
    assert scored["confidence"] < 0.65
    assert scored["signals"]["graph"] == 0.20


def test_specialization_subdivision_cue():
    """Verify discourse cue detector captures specialization and subdivision phrasing."""
    from cloud.extraction.prerequisite_inference import DiscourseCueDetector
    detector = DiscourseCueDetector()

    # Specialization: General -> Specialized
    text = "Transformers are manufactured to be step up transformers or step down transformers."
    score, pat, snip = detector.detect(text, "Transformer", "Step Up Transformer")
    assert score >= 0.85
    assert pat == "specialization_subdivision"

    # Type of: Specialized is a type of General
    text2 = "A step down transformer is a type of transformer designed to reduce voltage."
    score2, pat2, snip2 = detector.detect(text2, "Transformer", "Step Down Transformer")
    assert score2 >= 0.85
    assert pat2 == "specialization_type_of"


def test_compound_term_regex_matching():
    """Verify acronym and connector variations in term regex."""
    from cloud.extraction.prerequisite_inference import _build_term_regex
    import re

    re_ac = _build_term_regex("Alternating Current")
    assert re.search(r"\b" + re_ac + r"\b", "works with AC power", re.IGNORECASE)
    assert re.search(r"\b" + re_ac + r"\b", "uses alternating current", re.IGNORECASE)

    re_delta_y = _build_term_regex("Delta Y Connection")
    assert re.search(r"\b" + re_delta_y + r"\b", "known as delta y", re.IGNORECASE)
    assert re.search(r"\b" + re_delta_y + r"\b", "wired in delta-y connection", re.IGNORECASE)


def test_generic_current_negative_lookbehind():
    """Error 2 regression: verify generic 'Current' does not match compound terms."""
    from cloud.extraction.prerequisite_inference import _build_term_regex
    import re

    re_current = _build_term_regex("Current")
    current_pattern = re.compile(r"\b" + re_current + r"\b", re.IGNORECASE)

    # Standalone 'current' must match
    assert current_pattern.search("current flows through the coil")
    assert current_pattern.search("the electric current is measured in amperes")
    assert current_pattern.search("induced currents create opposing fields")

    # Compound terms must NOT match generic Current
    assert not current_pattern.search("alternating current flows through the coil")
    assert not current_pattern.search("alternating-current supply is connected")
    assert not current_pattern.search("AC current flows through the coil")
    assert not current_pattern.search("AC-current flows...")
    assert not current_pattern.search("eddy current losses are significant")
    assert not current_pattern.search("eddy currents are produced in the iron core")
    assert not current_pattern.search("eddy-current heating occurs")
    assert not current_pattern.search("direct current is provided by a battery")
    assert not current_pattern.search("DC current flows...")


def test_a9_relation_prompt_and_schema():
    """Error 1 regression: verify A9 prompt and schema distinguish educational relations."""
    from cloud.extraction.relation_extractor import RELATION_EXTRACTION_PROMPT
    from schemas.enums import RelationType

    # Check that schema contains all 6 relation types
    for rel in ["PREREQUISITE_OF", "USED_BY", "EXPLAINS", "DERIVED_FROM", "INTRODUCED_BEFORE", "VISUALIZED_BY"]:
        assert rel in [t.value for t in RelationType]

    # Check prompt contains strict definition and exclusions for PREREQUISITE_OF
    assert "PREREQUISITE_OF" in RELATION_EXTRACTION_PROMPT
    assert "foundational concept" in RELATION_EXTRACTION_PROMPT
    assert "A -> B" in RELATION_EXTRACTION_PROMPT
    assert "Strict Exclusion" in RELATION_EXTRACTION_PROMPT
    assert "Iron Core" in RELATION_EXTRACTION_PROMPT
    assert "Eddy Currents" in RELATION_EXTRACTION_PROMPT
    assert "Voltage" in RELATION_EXTRACTION_PROMPT
    assert "Current" in RELATION_EXTRACTION_PROMPT
    assert "Sinusoidal Waveform" in RELATION_EXTRACTION_PROMPT


def test_discourse_causal_statements_not_overinferred():
    """Error 3 regression: verify non-prerequisite causal/topological statements are rejected."""
    from cloud.extraction.prerequisite_inference import DiscourseCueDetector
    detector = DiscourseCueDetector()

    # 1. Coil -> EMF: sentence ending in coil followed by force causes electrons to move
    text_coil_emf = "It is connected to the secondary coil. This force causes the electrons to move, creating an electromotive force."
    score_coil_emf, pat_coil_emf, _ = detector.detect(text_coil_emf, "Coil", "Electromotive Force")
    assert score_coil_emf == 0.0, f"Expected 0.0, got {score_coil_emf} via {pat_coil_emf}"

    # 2. Coil -> Delta Y: "connect coils in a configuration known as delta y"
    text_coil_dy = "We connect the coils in a configuration known as delta y connection."
    score_coil_dy, pat_coil_dy, _ = detector.detect(text_coil_dy, "Coil", "Delta Y Connection")
    assert score_coil_dy == 0.0, f"Expected 0.0, got {score_coil_dy} via {pat_coil_dy}"

    # 3. Coil -> Magnetic Field: "first coil then the magnetic field will be stronger"
    text_coil_mf = "We wrap wire around the first coil then the magnetic field will be stronger."
    score_coil_mf, pat_coil_mf, _ = detector.detect(text_coil_mf, "Coil", "Magnetic Field")
    assert score_coil_mf == 0.0, f"Expected 0.0, got {score_coil_mf} via {pat_coil_mf}"

    # Verify genuine pedagogical discourse cues still match
    text_genuine_mf_emf = "Change in the intensity and direction of the magnetic field produces an electromotive force."
    score_genuine, pat_genuine, _ = detector.detect(text_genuine_mf_emf, "Magnetic Field", "Electromotive Force")
    assert score_genuine >= 0.85
    assert pat_genuine in {"inductive_generation", "causal_generation"}

    text_genuine_dy = "A three phase configuration is known as a delta y connection."
    score_dy, pat_dy, _ = detector.detect(text_genuine_dy, "Three Phase Configuration", "Delta Y Connection")
    assert score_dy >= 0.85
    assert pat_dy == "topological_configuration"


def test_pedagogical_inversion_strong_vs_weak():
    """Error 4 regression: verify t(A) > t(B) is allowed with strong evidence but rejected with weak."""
    from cloud.extraction.prerequisite_inference import score_prerequisite_candidate

    chunks = []

    # Case 1: t(A) = 300, t(B) = 60, but explicit PREREQUISITE_OF relation in graph
    candidate_strong = {
        "source_id": "ent_A",
        "source_name": "Foundational Concept",
        "source_type": "Concept",
        "target_id": "ent_B",
        "target_name": "Advanced Topic",
        "target_type": "Concept",
        "t_source": 300.0,
        "t_target": 60.0,
        "source_chunk_id": "chunk_05",
        "source_in_headings": True,
        "source_in_first_segment": False,
        "existing_forward_relations": ["PREREQUISITE_OF"],
        "existing_reverse_relations": [],
    }
    scored_strong = score_prerequisite_candidate(candidate_strong, chunks)
    assert scored_strong["confidence"] >= 0.65, f"Strong inversion should pass threshold, got {scored_strong['confidence']}"

    # Case 2: t(A) = 300, t(B) = 60, with weak relation (e.g. USED_BY weight 0.20) and no discourse
    candidate_weak = {
        "source_id": "ent_A",
        "source_name": "Physical Component",
        "source_type": "Component",
        "target_id": "ent_B",
        "target_name": "Machine",
        "target_type": "Device",
        "t_source": 300.0,
        "t_target": 60.0,
        "source_chunk_id": "chunk_05",
        "source_in_headings": False,
        "source_in_first_segment": False,
        "existing_forward_relations": ["USED_BY"],
        "existing_reverse_relations": [],
    }
    scored_weak = score_prerequisite_candidate(candidate_weak, chunks)
    assert scored_weak["confidence"] < 0.65, f"Weak inversion must not pass threshold, got {scored_weak['confidence']}"


def test_dag_cycle_and_self_loop_elimination():
    """Verify that cycle resolution enforces a strict DAG with no cycles and no self-loops."""
    from cloud.extraction.prerequisite_inference import enforce_dag_cycles
    from evaluation.knowledge_graph.prerequisite_auditor import _detect_cycles

    # Graph with a 3-cycle: A -> B -> C -> A, plus self loop D -> D
    edges = [
        {"source_id": "A", "target_id": "B", "confidence": 0.90},
        {"source_id": "B", "target_id": "C", "confidence": 0.85},
        {"source_id": "C", "target_id": "A", "confidence": 0.70},  # weakest in cycle
        {"source_id": "D", "target_id": "D", "confidence": 0.95},  # self-loop
    ]

    pruned = enforce_dag_cycles(edges)

    # Check no self loops
    for e in pruned:
        assert e["source_id"] != e["target_id"]

    # Check acyclicity using DFS cycle detection
    directed_pairs = [(e["source_id"], e["target_id"]) for e in pruned]
    detected_cycles = _detect_cycles(directed_pairs)
    assert len(detected_cycles) == 0


