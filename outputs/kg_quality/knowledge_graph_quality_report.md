# LectureMIND Knowledge Graph Quality Report

**Lecture ID**: `CS162_Lecture_1_What_is_an_Operating_System_720P`
**Evaluation Standard**: Read-Only Structural Audit & Reference-Label Evaluation
**KG Quality Diagnostic Score**: **32.5 / 45.0** *(only measurable components counted)*

> [!NOTE]
> **Reference Label Provenance**: Reference concepts, relations, and prerequisite dependencies are **LLM-assisted reference labels, pending human verification**. They serve as an automated evaluation benchmark and have not undergone independent manual double-blind verification by human domain experts.

---

## 1. Dataset & Pipeline Summary

| Metric | Measured Value |
| :--- | :--- |
| **Lecture Duration** | 4982.1s (~83.0 min) |
| **Multimodal Chunks** | 98 chunks |
| **Segments** | 13 segments |
| **Extracted Entities** | 158 (accepted: 158, flagged as fragments: 0) |
| **Extracted Relations** | 103 (dangling entity references: 0) |
| **Inferred Prerequisites** | 29 edges (DAG: True, cycles: 0) |
| **QA Benchmark Applied** | No — benchmark chunk IDs do not belong to this package (reported N/A) |
| **Gold Reference Applied** | No — gold labels are annotated for a different lecture (reported N/A) |

---

## 2. Entity Extraction Quality

* Gold reference metrics: **N/A** — the packaged gold labels are annotated for a different lecture and were NOT applied.

| Structural Metric | Value | Interpretation |
| :--- | :---: | :--- |
| **Fragment Rate** | **0.0%** | 0 sentence-fragment extraction artifacts detected |
| **Generic Non-Concept Rate** | **1.3%** | 2 generic stopwords detected |
| **Orphan Entity Rate** | **58.9%** | 93 entities with graph degree = 0 |
| **Duplicate Surface Forms** | **0** | groups of entities sharing a normalized surface form |

### Flagged Entity Issues:
- **[ORPHAN]** `Internet`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `ARPANET`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `World Wide Web`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Bell's Law`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Moore's Law`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `L1 cache reference`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Mutex lock/unlock`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `DNS Server`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Multimedia`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Windowing System`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Browser`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Search Query`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Switchboard Operator`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Computer Operator`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[GENERIC_STOPWORD]** `System`: Matches generic non-concept dictionary: 'System'
- **[GENERIC_STOPWORD]** `Process`: Matches generic non-concept dictionary: 'Process'
- **[ORPHAN]** `System Library`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Hypervisor`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Docker`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Register`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Input/Output Controller`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `System Libraries`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Scheduler Queue`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Shared Data`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Idle Process`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `CPU`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Protection Boundary`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Segmentation Fault`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Virtual Machine`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Tiny OS`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `I/O`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Processor protection`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Power management`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Look and feel`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Compiled Program`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `System Libs`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Core OS`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Locking`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Device Driver`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Network File System`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Clustered High-Availability System`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Peer-to-Peer System`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Tessellation`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Ocean Store`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Quantum Computing`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Swarm`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Global Data Plane`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Data Capsule`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Sections TBA`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Neil Kulkarni`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Akshat Gokhale`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Alina Dan`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `William Hsu`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `John Markham`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Taj Shaik`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Early Drop Deadline`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Drop Deadline`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Camera Requirement`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Discussion Sessions`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Design Reviews`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Office Hours`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Exams`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Virtual Class`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Coffee Houses`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `cs162.eecs.berkeley.edu`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Principles and Practices of Operating Systems`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `O'Reilly animal books`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `cloud operating system`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `project zero`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `GitHub account`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Autograder`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Camera`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Design doc`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Slack`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Messenger`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Time-zone survey`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Resource Allocation`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Hyperthreading`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Linux 2.2.0`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Mars Curiosity Rover`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Firefox`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Android`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Linux 3.1`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Windows 7`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Windows Vista`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Facebook`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Mac OS`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Real-time operating system`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[ORPHAN]** `Programming`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[EXCESSIVE_LENGTH]** `t there's a lot of material in this`: Entity name is 35 chars / 8 words, likely a descriptive clause rather than atomic concept.
- **[ORPHAN]** `t there's a lot of material in this`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[EXCESSIVE_LENGTH]** `template for internet history and evolution Title: Evolution`: Entity name is 60 chars / 8 words, likely a descriptive clause rather than atomic concept.
- **[ORPHAN]** `template for internet history and evolution Title: Evolution`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[EXCESSIVE_LENGTH]** `10/26/20 Kubiatowicz CS162 UCB Fall 2020 Lec1.7 what`: Entity name is 52 chars / 8 words, likely a descriptive clause rather than atomic concept.
- **[ORPHAN]** `10/26/20 Kubiatowicz CS162 UCB Fall 2020 Lec1.7 what`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[EXCESSIVE_LENGTH]** `Title Process Abstraction and Isolation in Operating Systems`: Entity name is 60 chars / 8 words, likely a descriptive clause rather than atomic concept.
- **[ORPHAN]** `Title Process Abstraction and Isolation in Operating Systems`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[EXCESSIVE_LENGTH]** `how to write code that runs in a`: Entity name is 32 chars / 8 words, likely a descriptive clause rather than atomic concept.
- **[ORPHAN]** `how to write code that runs in a`: Entity has degree = 0 (no incident relationships in knowledge graph).
- **[EXCESSIVE_LENGTH]** `Common Services and Specialized Operating Systems`: Entity name is 49 chars / 6 words, likely a descriptive clause rather than atomic concept.
- **[ORPHAN]** `Common Services and Specialized Operating Systems`: Entity has degree = 0 (no incident relationships in knowledge graph).

---

## 3. Relation Extraction, Direction & Evidence Grounding Quality

* Gold reference metrics: **N/A** — the packaged gold labels are annotated for a different lecture and were NOT applied.

| Evidence Grounding Tier | Count | Proportion |
| :--- | :---: | :---: |
| **Direct Textual Evidence** | 2 | 1.9% |
| **Indirect Textual Evidence** | 76 | 73.8% |
| **Co-occurrence Only** *(not evidence)* | 17 | 16.5% |
| **No Single-Chunk Evidence** | 8 | 7.8% |

### Flagged Directional Warnings:
- None detected.

---

## 4. Prerequisite Dependency Quality (DAG Validation)

| Prerequisite Metric | Value | Status |
| :--- | :---: | :---: |
| **Total Inferred Prerequisites** | 29 | Multi-signal inferred edges |
| **Graph Topology (Is DAG)** | **True** | Strict DAG — zero cycles |
| **Cycle Count** | **0** | PASS |
| **Self-Loop Count** | **0** | PASS |
| **Temporal Inversion Warnings** | **0** | Diagnostic (LOW severity) |
| **Non-Pedagogical Flags** | **0** | Marker-based heuristic |

* Gold reference metrics: **N/A** — the packaged gold prerequisite labels are annotated for a different lecture and were NOT applied.

---

## 5. Cross-Chunk Relation Analysis (descriptive — not scored)

The cross-chunk ratio is reported for descriptive purposes only: it reflects how far
relations span the lecture timeline, and correlates with package density — sparse graphs
mechanically show a higher share. It is **not** a quality signal and is excluded from the
composite score, which instead rewards grounding depth and entity participation (see
`graph_coherence` in Section 8).

| Metric | Measured Value |
| :--- | :--- |
| **Same-Chunk Relations** | 95 (92.2%) |
| **Cross-Chunk Relations** | **8 (7.8%)** |
| **Cross-Chunk Ratio** | **0.078** |

---

## 6. Downstream GraphRAG Benchmark Evaluation

* **N/A** — the configured QA benchmark's `expected_chunk_ids` belong to a different lecture package. Scoring this package against them would only produce meaningless zeros, so downstream retrieval metrics are not reported. Run the audit with `--benchmark-qa` pointing at a benchmark written for this package.

---

## 7. Diagnostic Ontology Breakdown

* `USED_BY`: 62 relations (60.2% of all relations)
  - **COMPONENT_OF**: 1 instances — Physical/structural hardware components forming the machine.
    Examples: `Multi-core -> USED_BY -> System`
  - **FUNCTIONAL_INPUT**: 0 instances — Physical quantities or power sources required for operation.
  - **SYSTEM_TOPOLOGY**: 0 instances — Electrical wiring topology or multi-phase organizational layouts.
  - **OTHER**: 61 instances — Miscellaneous associations mapped to USED_BY.
    Examples: `Operating System -> USED_BY -> Branch mispredict`, `Operating System -> USED_BY -> L2 cache reference`, `Operating System -> USED_BY -> Main memory reference`, `Operating System -> USED_BY -> Disk seek`

---

## 8. Composite Score Breakdown (only measurable components)

| Component | Weight | Contribution |
| :--- | :---: | :---: |
| entity_quality | 20 | 18.99 |
| evidence_grounding | 15 | 11.36 |
| graph_coherence | 10 | 2.15 |
| **Total** | **45** | **32.5** |

---

## 9. Recommendations for Next Iterations

1. **[Medium Priority] Entity Hygiene Pre-Filter**: Filter clausal-fragment entities before saving Stage A8 output.
2. **[Medium Priority] Per-Lecture Gold Labels**: Produce human-verified gold files for every lecture that is claimed to be quality-audited; this audit reports N/A rather than fabricating scores when they are missing.
3. **[Low Priority] Ontology Expansion**: Introduce `COMPONENT_OF` where `USED_BY` is semantically overloaded.
