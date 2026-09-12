# LectureMIND Knowledge Graph Quality Report

**Lecture ID**: `vidssave_com_How_does_a_Transformer_work_Working_Principle_electrical_engineering_1080P`  
**Evaluation Standard**: Read-Only Structural Audit & LLM-Assisted Reference Evaluation  
**KG Quality Diagnostic Score**: **77.0 / 100** *(Heterogeneous Diagnostic Metric — Not Single-Dimension Accuracy)*

> [!NOTE]
> **Reference Label Provenance**: All reference concepts, relations, and prerequisite dependencies are **LLM-assisted reference labels, pending human verification**. They serve as an automated evaluation benchmark and have not undergone independent manual double-blind verification by human domain experts.

---

## 1. Dataset & Pipeline Summary

| Metric | Measured Value | Status / Interpretation |
| :--- | :--- | :---: |
| **Lecture Duration** | 389.1s (~6.5 min) | 100% video timeline covered |
| **Multimodal Chunks** | 10 chunks | Full lecture span ($t=0.0$ to $389.1$s) |
| **Segments** | 1 segment | Valid `LectureSegment` |
| **Extracted Entities** | 15 entities | Extracted: 15 / Accepted: 14 / Noisy: 1 |
| **Extracted Relations** | 26 relations | Zero dangling IDs (`dangling=0`) |
| **Inferred Prerequisites**| 9 edges | Strict DAG, 0 cycles, 0 self-loops |
| **Embedding Dimension**| 1024 (BGE-Large) | Shape: (10, 1024) verified |
| **QA Benchmark** | 50 questions | Verified grounded test queries ([transformer_kg_qa_50.json](file:///home/surjit/Desktop/lecuremid/evaluation/datasets/transformer_kg_qa_50.json)) |

---

## 2. Entity Extraction Quality

* **Reference Label Standard**: `LLM-assisted reference labels, pending human verification`
* **Reference Concepts Available**: 16 domain concepts

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Reference Entity Precision** | **93.3%** | 14 / 15 extracted entities match reference domain concepts |
| **Reference Entity Recall** | **87.5%** | 14 / 16 reference lecture concepts captured |
| **Reference Entity F1** | **90.3%** | Balanced entity extraction performance |
| **Fragment Rate** | **6.7%** | 1 sentence fragment detected (extraction artifact) |
| **Generic Non-Concept Rate** | **0.0%** | 0 generic stopwords detected |
| **Orphan Entity Rate** | **13.3%** | 2 entities with graph degree = 0 |

### Entity Classification Breakdown:
- **Extracted Entities (15)**: All entities extracted by the pipeline.
- **Accepted Entities (14)**: Legitimate domain concepts with semantic utility.
- **Noisy Entity / Artifact (1)**: `us a sinusoidal waveform. This is important because` — flagged as a sentence fragment extraction artifact.
- **Unsupported Entities (0)**: No extracted entities are ungrounded in the lecture material.

### Flagged Entity Issues:
- **[ORPHAN]** `Delta Y Connection`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[FRAGMENT]** `us a sinusoidal waveform. This is important because`: Matches sentence fragment / clausal connector pattern: ^(us\s+a|and\s+then|this\s+means|because\s+the|which\s+means|gives\s+us|so\s+that|therefore\s+we|to\s+fix|now\s+the|we\s+will|that\s+is|due\s+to)\b
- **[ORPHAN]** `us a sinusoidal waveform. This is important because`: Entity has degree = 0 (no incident relationships in knowledge graph).

---

## 3. Relation Extraction, Direction & Evidence Grounding Quality

* **Reference Label Standard**: `LLM-assisted reference labels, pending human verification` (21 reference relations)

| Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Reference Relation Precision** | **73.1%** | 19 / 26 extracted relations match reference semantic pairs |
| **Reference Relation Recall** | **90.5%** | 19 / 21 reference relations captured |
| **Reference Relation F1** | **80.8%** | Harmonic mean of precision and recall |
| **Gold-Matched Direction Accuracy** | **95.0%** | 19 / 20 reference-matched relations have correct source $\to$ target direction |
| **Overall Direction Acceptance Rate**| **73.1%** | 19 / 26 total extracted relations accepted as directional true positives |
| **Evidence Coverage Rate** | **61.5%** | 16 / 26 relations supported by direct or indirect transcript evidence |

### Direction Accounting & Classification (Total: 26 Relations):
| Category | Count | Proportion | Interpretation & Examples |
| :--- | :---: | :---: | :--- |
| **Correct Direction (Gold Matched)** | **19** | 73.08% | 13 exact forward + 6 fuzzy forward matches (`Alternating Current ==[USED_BY]==> Transformer`) |
| **Reversed Direction** | **1** | 3.85% | `Magnetic Field ==[DERIVED_FROM]==> Electromotive Force` (EMF is induced by magnetic field) |
| **Relation Type Mismatch** | **1** | 3.85% | `Sinusoidal Waveform ==[PREREQUISITE_OF]==> Alternating Current` (Reference: `EXPLAINS`) |
| **Unmatched / Granular Relations** | **5** | 19.23% | Valid lecture relations not in reference set (`Sinusoidal Waveform ==[EXPLAINS]==> Transformer`) |

### Evidence Grounding Tiers:
* **Direct Textual Evidence** (13 relations): Explicit relational predicate in the same sentence or clause.
* **Indirect Textual Evidence** (3 relations): Related statements within the same lecture chunk context.
* **Co-occurrence Only** (2 relations): Entities appear in the same chunk without semantic support. *(Treated as ungrounded — co-occurrence is NOT evidence)*.
* **No Single-Chunk Evidence** (8 relations): Cross-chunk relations connecting concepts introduced across different segments.

### Flagged Directional Warnings:
- **[SUSPICIOUS_DIRECTION]** `Magnetic Field -[DERIVED_FROM]-> Electromotive Force`: Faraday's Law of Induction states that changing magnetic flux induces Electromotive Force (EMF). Thus, EMF is DERIVED_FROM Magnetic Field, not vice versa.
- **[SUSPICIOUS_DIRECTION]** `Magnetic Field -[DERIVED_FROM]-> Sinusoidal Waveform`: A magnetic field is physically generated by alternating current, not directly derived from the mathematical sinusoidal waveform.

---

## 4. Prerequisite Dependency Quality (DAG Validation)

| Prerequisite Metric | Value | Status |
| :--- | :---: | :---: |
| **Total Inferred Prerequisites** | 9 | Multi-signal inferred edges |
| **Graph Topology (Is DAG)** | **True** | Strict DAG — Zero cycles detected |
| **Cycle Count** | **0** | PASS |
| **Self-Loop Count** | **0** | PASS |
| **Temporal Inversion Warnings** | **0** | PASS — All dependencies respect chronological/logical flow |
| **Pedagogical Relevance** | **9 / 9** | 100.0% genuine pedagogical prerequisites |
| **Reference Prerequisite Precision** | **77.8%** | 7 / 9 inferred prerequisites match reference DAG |
| **Reference Prerequisite Recall** | **70.0%** | 7 / 10 reference prerequisites captured |
| **Reference Prerequisite F1** | **73.7%** | Prerequisite graph alignment score |

### Pedagogical vs Factual vs Lecture Grounding Distinction:
- **Lecture Grounded**: Both concepts are explicitly introduced in the lecture.
- **Factually Correct**: The scientific relationship is accurate.
- **Pedagogically Justified**: Understanding concept A is actually necessary before learning concept B.
- *Example False Positive*: `Transformer -> Eddy Currents` is factually correct and lecture-grounded (Chunk 8), but non-pedagogical (eddy currents are an electromagnetic core-loss effect explained by transformers, not a prerequisite dependency). The auditor properly flags this.

---

## 5. Cross-Chunk Relation Analysis

| Metric | Measured Value | Importance |
| :--- | :---: | :--- |
| **Same-Chunk Relations** | 18 (69.2%) | Intra-window localized facts |
| **Cross-Chunk Relations** | **8 (30.8%)** | Long-range conceptual bridges |
| **Cross-Chunk Ratio** | **0.308** | Conceptual continuity across video timeline |

### Terminology Definitions:
* **Cross-chunk relation**: Source and target entities have their primary/first mentions in distinct lecture chunks.
* **Cross-window relation**: Spans extraction window boundaries (mitigated by A8/A9 sliding window context).
* **Directly evidenced relation**: Supported by an explicit linguistic clause in source text.
* **Inferred relation**: Derived via multi-signal graph traversal and temporal reasoning rather than direct clause extraction.

*(Note on resolution: Safe chunk extraction inspects transcripts, OCR, and visual context. Earlier code that checked only `text` omitted transcripts, artificially inflating cross-chunk estimates. The verified count is 18 same-chunk, 8 cross-chunk).*

---

## 6. Downstream GraphRAG Benchmark Evaluation

Evaluated across **50 verified grounded questions** ([`transformer_kg_qa_50.json`](file:///home/surjit/Desktop/lecuremid/evaluation/datasets/transformer_kg_qa_50.json)):

| Retrieval Route | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Precision@5 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 / Lexical Retrieval** | **0.78** | **0.96** | **1.00** | **0.940** | **0.280** | **0.879** |
| **Graph-Only RAG** | 0.10 | 0.38 | 0.72 | 0.602 | 0.184 | 0.330 |
| **Hybrid RAG (RRF)** | 0.26 | 0.94 | **1.00** | 0.905 | 0.268 | 0.587 |

> [!IMPORTANT]
> **Methodological Honesty on Retrieval:**
> 1. **BM25 is Lexical Retrieval**: BM25 is based on term frequency and document length saturation, NOT dense vector embeddings. Dense vector retrieval using multimodal embeddings is a planned future experiment.
> 2. **Benchmark Finding**: The current graph retrieval component does not yet outperform the lexical baseline on this benchmark (BM25 MRR 0.879 vs Graph MRR 0.330). This occurs because direct factual queries benefit strongly from exact lexical matching, whereas multi-hop graph expansion through high-degree hub nodes (e.g. `Transformer`) causes precision dilution.
> 3. **Hybrid Complementarity**: Hybrid RRF maintains 100% Hit@5 and 0.905 Recall@5 while enriching candidate pools with graph-linked concepts.

---

## 7. Diagnostic Ontology Breakdown

* `USED_BY`: 16 relations (61.5%)
  - **Component-of**: 4 instances (`Iron Core`, `Coil`)
  - **Functional Input**: 7 instances (`Voltage`, `Current`, `AC`)
  - **System Topology**: 3 instances (`Three Phase`, `Delta Y`)

*Ontology Quality Finding*: `USED_BY` is overloaded across component-of, electrical inputs, and topology. Future iterations should add `COMPONENT_OF` to relieve semantic overloading.

---

## 8. Failure Cases & Limitations

1. **Entity Extraction Artifact**: `us a sinusoidal waveform. This is important because` was extracted as a concept fragment from Chunk 1. It has degree = 0 and causes no downstream harm, but should be filtered by an entity hygiene step.
2. **Reversed Direction on `DERIVED_FROM`**:
   `Magnetic Field ==[DERIVED_FROM]==> Electromotive Force` was extracted with inverted causality. Changing magnetic field induces EMF; therefore, EMF is derived from magnetic field.
3. **Graph-Only Precision Dilution**: Single high-degree entities connect to many chunks, diluting pure graph retrieval precision relative to lexical search.

---

## 9. Recommendations for Next Iterations

1. **[Medium Priority] Entity Hygiene Pre-Filter**: Filter out clausal fragment prefixes before saving Stage A8 entities.
2. **[Medium Priority] Directional Few-Shot Prompts**: Add explicit directional examples for `DERIVED_FROM` in Stage A9 to prevent cause/effect reversal.
3. **[Low Priority] Ontology Expansion**: Introduce `COMPONENT_OF` to relieve semantic overloading on `USED_BY`.
