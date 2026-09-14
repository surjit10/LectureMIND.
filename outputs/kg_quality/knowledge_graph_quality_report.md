# LectureMIND Knowledge Graph Quality Report

**Lecture ID**: `vidssave_com_How_does_a_Transformer_work_Working_Principle_electrical_engineering_1080P`
**Evaluation Standard**: Read-Only Structural Audit & Reference-Label Evaluation
**KG Quality Diagnostic Score**: **32.1 / 100.0** *(only measurable components counted)*

> [!NOTE]
> **Reference Label Provenance**: Reference concepts, relations, and prerequisite dependencies are **LLM-assisted reference labels, pending human verification**. They serve as an automated evaluation benchmark and have not undergone independent manual double-blind verification by human domain experts.

---

## 1. Dataset & Pipeline Summary

| Metric | Measured Value |
| :--- | :--- |
| **Lecture Duration** | 390.0s (~6.5 min) |
| **Multimodal Chunks** | 9 chunks |
| **Segments** | 1 segments |
| **Extracted Entities** | 9 (accepted: 8, flagged as fragments: 1) |
| **Extracted Relations** | 5 (dangling entity references: 0) |
| **Inferred Prerequisites** | 3 edges (DAG: True, cycles: 0) |
| **QA Benchmark Applied** | Yes — evaluation/datasets/transformer_kg_qa_50.json |
| **Gold Reference Applied** | Yes |

---

## 2. Entity Extraction Quality

* **Reference Label Standard**: `LLM-assisted reference labels, pending human verification`
* **Reference Concepts Available**: 16

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Reference Entity Precision** | **55.6%** | 5 / 9 extracted entities match a reference concept (1-to-1) |
| **Reference Entity Recall** | **31.2%** | 5 / 16 reference concepts captured |
| **Reference Entity F1** | **40.0%** | Balanced entity extraction performance |

| Structural Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Fragment Rate** | **11.1%** | 1 sentence-fragment extraction artifacts detected |
| **Generic Non-Concept Rate** | **0.0%** | 0 generic stopwords detected |
| **Orphan Entity Rate** | **11.1%** | 1 entities with graph degree = 0 |
| **Duplicate Surface Forms** | **0** | groups of entities sharing a normalized surface form |

### Flagged Entity Issues:
- **[ORPHAN]** `blueprint-style grid`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[FRAGMENT]** `us a sinusoidal waveform. This is important because`: Matches sentence fragment / clausal connector pattern: ^(us\s+a|and\s+then|this\s+means|because\s+the|which\s+means|gives\s+us|so\s+that|therefore\s+we|to\s+fix|now\s+the|we\s+will|that\s+is|due\s+to)\b

---

## 3. Relation Extraction, Direction & Evidence Grounding Quality

* **Reference Label Standard**: `LLM-assisted reference labels, pending human verification` (21 reference relations)

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Reference Relation Precision** | **0.0%** | 0 / 5 extracted relations match a reference triple |
| **Reference Relation Recall** | **0.0%** | 0 / 21 reference triples captured |
| **Reference Relation F1** | **0.0%** | Harmonic mean of precision and recall |
| **Gold-Matched Direction Accuracy** | **0.0%** | 0 / 0 gold-matched directional relations have correct orientation |
| **Evidence Coverage Rate** | **60.0%** | 3 / 5 relations supported by direct or indirect transcript evidence |

| Evidence Grounding Tier | Count | Proportion |
| :--- | :---: | :---: |
| **Direct Textual Evidence** | 3 | 60.0% |
| **Indirect Textual Evidence** | 0 | 0.0% |
| **Co-occurrence Only** *(not evidence)* | 1 | 20.0% |
| **No Single-Chunk Evidence** | 1 | 20.0% |

### Flagged Directional Warnings:
- None detected.

---

## 4. Prerequisite Dependency Quality (DAG Validation)

| Prerequisite Metric | Value | Status |
| :--- | :---: | :---: |
| **Total Inferred Prerequisites** | 3 | Multi-signal inferred edges |
| **Graph Topology (Is DAG)** | **True** | Strict DAG — zero cycles |
| **Cycle Count** | **0** | PASS |
| **Self-Loop Count** | **0** | PASS |
| **Temporal Inversion Warnings** | **0** | Diagnostic (LOW severity) |
| **Non-Pedagogical Flags** | **0** | Marker-based heuristic |

| **Reference Prerequisite Precision** | **0.0%** | 0 / 3 inferred prerequisites match reference DAG (strict 1-to-1) |
| **Reference Prerequisite Recall** | **0.0%** | 0 / 10 reference prerequisites captured |
| **Reference Prerequisite F1** | **0.0%** | Prerequisite graph alignment score |

---

## 5. Cross-Chunk Relation Analysis (descriptive — not scored)

The cross-chunk ratio is reported for descriptive purposes only: it reflects how far
relations span the lecture timeline, and correlates with package density — sparse graphs
mechanically show a higher share. It is **not** a quality signal and is excluded from the
composite score, which instead rewards grounding depth and entity participation (see
`graph_coherence` in Section 8).

| Metric | Measured Value |
| :--- | :--- |
| **Same-Chunk Relations** | 4 (80.0%) |
| **Cross-Chunk Relations** | **1 (20.0%)** |
| **Cross-Chunk Ratio** | **0.200** |

---

## 6. Downstream GraphRAG Benchmark Evaluation

Evaluated on the applied QA benchmark (50 queries; lexical BM25 baseline vs graph-only vs hybrid RRF — self-contained evaluators, not the production Qdrant/Neo4j stack):

| Retrieval Route | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Precision@5 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 / Lexical Retrieval** | **0.68** | **0.88** | **0.98** | **0.885** | **0.264** | **0.796** |
| **Graph-Only RAG** | 0.10 | 0.38 | 0.62 | 0.543 | 0.176 | 0.333 |
| **Hybrid RAG (RRF)** | 0.34 | 0.82 | **0.90** | 0.768 | 0.228 | 0.582 |

> [!IMPORTANT]
> **Methodological note:** BM25 is lexical retrieval (term frequency + length saturation), not dense vector search. Dense multimodal-embedding retrieval is a planned future experiment.

---

## 7. Diagnostic Ontology Breakdown

* `USED_BY`: 1 relations (20.0% of all relations)
  - **COMPONENT_OF**: 0 instances — Physical/structural hardware components forming the machine.
  - **FUNCTIONAL_INPUT**: 0 instances — Physical quantities or power sources required for operation.
  - **SYSTEM_TOPOLOGY**: 0 instances — Electrical wiring topology or multi-phase organizational layouts.
  - **OTHER**: 1 instances — Miscellaneous associations mapped to USED_BY.
    Examples: `magnetic field -> USED_BY -> transformers`

---

## 8. Composite Score Breakdown (only measurable components)

| Component | Weight | Contribution |
| :--- | :---: | :---: |
| entity_quality | 20 | 8.0 |
| relation_quality | 20 | 0.0 |
| direction_accuracy | 10 | 0.0 |
| evidence_grounding | 15 | 9.0 |
| graph_coherence | 10 | 7.44 |
| prerequisite_dag | 15 | 0.0 |
| graphrag_usefulness | 10 | 7.68 |
| **Total** | **100** | **32.1** |

---

## 9. Recommendations for Next Iterations

1. **[Medium Priority] Entity Hygiene Pre-Filter**: Filter clausal-fragment entities before saving Stage A8 output.
2. **[Medium Priority] Per-Lecture Gold Labels**: Produce human-verified gold files for every lecture that is claimed to be quality-audited; this audit reports N/A rather than fabricating scores when they are missing.
3. **[Low Priority] Ontology Expansion**: Introduce `COMPONENT_OF` where `USED_BY` is semantically overloaded.
