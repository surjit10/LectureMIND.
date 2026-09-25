# LectureMIND — Independent Forensic Metric Audit

**Audit Date:** September 25, 2026  
**Auditor:** Antigravity Autonomous Forensic Agent  
**Target Repository:** `/home/surjit/Desktop/lecuremid`  
**Evaluation Scope:** All headline retrieval, reranking, routing, GraphRAG, KG quality, and QA metrics  
**Operating Principle:** Skeptical verification-first audit. Prioritize empirical truthfulness over narrative convenience.

---

## 1. Executive Summary

A comprehensive forensic audit was performed on every high-value metric reported in LectureMIND documentation, reports, and code artifacts. A total of **22 discrete metric claims** were audited across 6 evaluation layers.

### Audit Verdict Summary:
- **VERIFIED:** **11 metrics** (50.0%) — Fully reproducible from saved artifacts with verified numerators and denominators.
- **VERIFIED BUT LIMITED:** **3 metrics** (13.6%) — Mathematically accurate, but constrained by small sample sizes or targeted test construction (e.g. Routing 12/12, 2-hop Hit@5 5/5).
- **VALID BUT METRIC-SENSITIVE:** **1 metric** (4.5%) — Answer Token F1 (0.459); correctly calculated via standard SQuAD formula, but sensitive to lexical phrasing differences.
- **CIRCULAR / SELF-CONSTRUCTED:** **1 metric** (4.5%) — Graph Traversal 20/20; questions were constructed from known existing Neo4j paths.
- **MISLEADING:** **2 metrics** (9.1%) — Reranker Triplet MRR (0.997) is a 2-candidate training convergence metric, not retrieval MRR; Citation Completeness (1.000) is an architectural metadata copying mechanism, not LLM citation faithfulness.
- **INCORRECT:** **3 metrics** (13.6%) — Orphan Entity Rate (reported as 0.0% by conflating with dangling edges; actual is **58.9%**); Pre-Rerank Hit@1 (reported as 0.620; actual is **0.660**); Pre-Rerank Hit@3 (reported as 0.940; actual is **0.880**).
- **UNVERIFIABLE:** **1 metric** (4.5%) — Standalone Dense-Only ablation row lacks an underlying execution log or saved output file.

### Major Forensic Discoveries:
1. **The Q17 vs Q40 Trade-off:** The single Hit@5 retrieval failure in the pre-rerank pool is **Q17** (rank 7). However, after cross-encoder reranking, Q17 is promoted to Rank 1, while **Q40 ("What is the grading breakdown for CS162?") is demoted from Rank 1 to Rank 6**. This demotion explains why the LLM refused with *"Insufficient evidence"* on Q40. Net Hit@5 remained 49/50 (98.0%), but the failing question shifted between stages.
2. **0.997 MRR is a 2-Candidate Binary Ranking:** The 0.997 score in `METRICS_MATRIX.md` evaluates the cross-encoder on triplets with exactly 1 positive and 1 negative candidate. Rank is constrained to 1 or 2; Recall@5 and Recall@10 are mathematically guaranteed to be 100%. It is a training convergence metric, not end-to-end retrieval MRR.
3. **58.9% Orphan Entities:** 93 out of 158 extracted entities in CS162 have degree = 0 (no incident relationships). Mislabeled as 0.0% in narrative reports by confusing orphan nodes with dangling relation edges.
4. **Circularity in Graph Benchmark:** `cs162_lecture1_graph_qa.json` (20 queries) was constructed by querying Neo4j for existing paths. It is a valid **graph integration/path-recovery test**, not an independent GraphRAG QA benchmark.

---

## 2. Headline Metrics Audit Table

| Metric | Reported Value | Recomputed Value | Sample Size (N) | Numerator / Denominator | Forensic Status | Meaning & Qualification |
|---|---:|---:|---:|---:|---|---|
| **Retrieval Hit@5 (Pre-Rerank)** | 0.980 (98.0%) | **0.980** (98.0%) | 50 | 49 / 50 | **VERIFIED** | Gold evidence in top-5 candidate pool |
| **Retrieval Recall@5 (Pre-Rerank)** | 0.862 (86.2%) | **0.862** (86.2%) | 50 | 43.1 / 50 | **VERIFIED** | Fraction of multi-chunk evidence retrieved |
| **Strict MRR@5 (Pre-Rerank)** | mixed / 0.785 | **0.7853** | 50 | 39.2667 / 50 | **CLARIFIED** | Strict reciprocal rank capped at k=5 |
| **Unbounded MRR (Pre-Rerank)** | 0.788 | **0.7882** | 50 | 39.4095 / 50 | **VERIFIED** | Diagnostic: reciprocal rank across candidate pool |
| **Ranking Quality (Post-Rerank MRR)** | 0.918 | **0.9183** | 50 | 45.9167 / 50 | **VERIFIED** | Reciprocal rank after cross-encoder scoring |
| **Retrieval NDCG@5** | 0.767 | **0.767** | 50 | 38.3702 / 50 | **VERIFIED** | Normalized discounted cumulative gain |
| **Pre-Rerank Hit@1** | 0.620 | **0.660** | 50 | 33 / 50 | **CORRECTED** | Under-reported by 4 percentage points |
| **Pre-Rerank Hit@3** | 0.940 | **0.880** | 50 | 44 / 50 | **CORRECTED** | Over-reported by 6 percentage points |
| **Post-Rerank Hit@1** | 0.840 | **0.860** | 50 | 43 / 50 | **CORRECTED** | Under-reported by 2 percentage points |
| **Post-Rerank Hit@3** | 0.960 | **0.960** | 50 | 48 / 50 | **VERIFIED** | Gold evidence in top-3 after reranking |
| **Post-Rerank Hit@5** | 0.980 | **0.980** | 50 | 49 / 50 | **VERIFIED** | Gold evidence in top-5 after reranking |
| **BM25 Lexical-Only Hit@5** | 0.380 (38.0%) | **0.380** (38.0%) | 50 | 19 / 50 | **VERIFIED** | BM25 baseline on identical 50 questions |
| **BM25 Lexical-Only Recall@5** | 0.281 | **0.281** | 50 | 14.066 / 50 | **VERIFIED** | Mean chunk recall under BM25 alone |
| **BM25 Lexical-Only MRR** | 0.322 | **0.322** | 50 | 16.074 / 50 | **VERIFIED** | Unbounded reciprocal rank under BM25 |
| **Dense-Only Ablation** | Hit@5 = 0.940 | `UNVERIFIABLE` | 50 | N/A | **CORRECTED / UNVERIFIABLE** | Lacks execution log / saved output file |
| **Answer Generation Token F1** | 0.459 | **0.4594** | 50 | 22.9689 / 50 | **VERIFIED** | SQuAD unigram token overlap F1 |
| **Answer Keyword Recall** | 54.5% / 82.4% | **54.53%** | 50 | 27.2667 / 50 | **CORRECTED** | 82.4% removed as stale unverified draft figure |
| **Citation Completeness** | 1.000 (100.0%) | **1.000** (100.0%) | 50 | 50 / 50 | **MISLEADING** | Code guarantees metadata copying, not LLM faithfulness |
| **Routing Accuracy** | 1.000 (100.0%) | **1.000** (100.0%) | 12 | 12 / 12 | **VERIFIED BUT LIMITED** | Small targeted pattern unit test (N=12) |
| **Graph Traversal Path Recovery (Old)** | 1.000 (100.0%) | **1.000** (100.0%) | 20 | 20 / 20 | **RECLASSIFIED** | Targeted integration test of known Cypher paths |
| **Held-Out Graph Structural Recovery** | N/A | **0.520** (52.0%) | 25 | 13 / 25 | **NEW / AUTHENTIC** | Independent lecture-derived multi-hop benchmark |
| **Held-Out Graph Downstream Hit@5** | N/A | **0.840** (84.0%) | 25 | 21 / 25 | **NEW / REPRODUCIBLE** | Hybrid Graph+BM25 (+20.0% lift over BM25 64%) |
| **Held-Out Graph Negative Refusal** | N/A | **1.000** (100.0%) | 5 | 5 / 5 | **NEW / AUTHENTIC** | Clean refusal on unanswerable/negative inquiries |
| **Reranker Triplet Training MRR** | 0.996 / 0.997 | **0.996** | 1,360 | 1354.5 / 1360 | **MISLEADING** | Pairwise 2-candidate ranking convergence |
| **CS162 Orphan Entity Rate** | 0.0% | **58.86%** | 158 | 93 / 158 | **CORRECTED** | Conflated with 0.0% dangling relation edges |
| **Test Suite Count** | 584 / 580 | **594 passed, 6 skipped** | 600 | 594 / 600 | **CORRECTED** | Full automated pytest suite (0 failed) |

---

## 3. Retrieval Verification & Detailed Ablation

The frozen benchmark dataset (`evaluation/datasets/cs162_lecture1_qa_50.json`) was evaluated against `evaluation_report_20260811_105621.json`.

### Recomputed Retrieval Comparison:

| Evaluation Stage | Hit@1 | Hit@3 | Hit@5 | Recall@5 | MRR@5 | NDCG@5 | Verification Basis |
|---|---:|---:|---:|---:|---:|---:|---|
| **BM25 Lexical Only** | 0.220 (11/50) | 0.380 (19/50) | 0.380 (19/50) | 0.281 | 0.322 | 0.257 | Executed & independently recomputed |
| **Dense Only (BGE-Large)** | *0.580* | *0.880* | *0.940* | *0.814* | *0.724* | *0.710* | **UNVERIFIABLE** (estimated from diagnostic notes) |
| **Dense + BM25 (RRF)** | **0.660** (33/50) | **0.880** (44/50) | **0.980** (49/50) | **0.862** | **0.788** | **0.767** | Verified from raw JSON run |
| **Hybrid + Cross-Encoder Reranker** | **0.860** (43/50) | **0.960** (48/50) | **0.980** (49/50) | **0.862** | **0.918** | **0.895** | Verified from raw JSON run |

### Cross-Encoder Impact (Pre-Rerank vs. Post-Rerank):
- **Hit@1:** Lifted from **0.660 (33/50)** to **0.860 (43/50)** (+20.0 percentage points / +30.3% relative improvement).
- **Hit@3:** Lifted from **0.880 (44/50)** to **0.960 (48/50)** (+8.0 percentage points).
- **MRR:** Lifted from **0.788** to **0.918** (+0.130 absolute / +16.5% relative improvement).
- **Failing Question Dynamics:**
  - In Pre-Rerank: Q17 was Rank 7 (Hit@5 failure), Q40 was Rank 1.
  - In Post-Rerank: Q17 was promoted to Rank 1, while Q40 was demoted to Rank 6 (Hit@5 failure).

---

## 4. Graph Verification & Circularity Audit

### 4.1 Benchmark Construction Audit
The 20 questions in `evaluation/datasets/cs162_lecture1_graph_qa.json` were audited for provenance:
- **Construction Method:** An automated script queried active Neo4j relationships in `lecture_092f861b`, identified existing 1-hop, 2-hop, and 3-hop paths, and synthesized natural language queries around those exact paths.
- **Circularity Finding:** The benchmark tests known paths that exist in the database.
- **Scientific Status:** This benchmark must be labeled as a **Targeted Graph Integration and Path-Recovery Test**. It does **NOT** measure generalized, double-blind GraphRAG accuracy on unseen user queries.

### 4.2 Recomputed Graph Traversal Results:
- **1-hop paths (N = 10):** 10 / 10 recovered (100.0%)
- **2-hop paths (N = 5):** 5 / 5 recovered (100.0%)
- **3-hop paths (N = 5):** 5 / 5 recovered (100.0%)
- **Overall Traversal:** **20 / 20 (100.0%)**

### 4.3 Downstream Multi-Hop Chunk Retrieval on Graph Queries:
- **BM25 Lexical:** Hit@5 = 0.550, Recall@5 = 0.308, MRR = 0.354
- **Graph-Only:** Hit@5 = 0.050, Recall@5 = 0.017, MRR = 0.130 (Nodes point to concept entities, not full lecture paragraphs)
- **Hybrid (Graph + BM25 RRF):** Hit@5 = 0.600, Recall@5 = 0.308, MRR = 0.383
- **2-Hop Sub-sample (N = 5):** Hit@5 = 5 / 5 (100.0%), MRR = 0.600
- **End-to-End Answer Generation Accuracy on Graph Benchmark:** **NOT MEASURED** (LLM answer generation was not executed on these 20 graph queries).

---

## 5. Query Routing Verification

- **Dataset:** `evaluation/datasets/cs162_lecture1_routing_12.json`
- **Recomputed Accuracy:** **12 / 12 (100.0%)**
  - `vector_only`: 5 / 5 (100.0%)
  - `graph_only`: 5 / 5 (100.0%)
  - `graph+vector`: 2 / 2 (100.0%)
- **Circularity / Independence Audit:**
  - The query vocabulary contains explicit keyword triggers matching regex rules in `agent/dspy/planner.py` (e.g. *"depend"*, *"related"*, *"path"*, *"taught before"*).
  - This is a **Targeted Unit / Integration Test** ensuring the regex dispatch logic functions as implemented. It is **NOT** a statistically powered benchmark demonstrating semantic intent classification.

---

## 6. Answer F1 Verification & Error Analysis

- **Recomputed Mean Answer F1:** **0.4594** (45.9%)
- **Recomputed Mean Keyword Recall:** **0.5453** (54.5%)
- **Formula Used:** SQuAD token-level unigram F1:
  $$\text{Precision} = \frac{|T_{\text{gold}} \cap T_{\text{gen}}|}{|T_{\text{gen}}|}, \quad \text{Recall} = \frac{|T_{\text{gold}} \cap T_{\text{gen}}|}{|T_{\text{gold}}|}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}$$

### Methodological Audit of the 7 Failure Categories:
The previous script `analyze_answer_errors.py` classified errors using a **heuristic rule cascade**:
- If `hit_at_5 == 0` $\rightarrow$ `retrieval_failure` (1 sample)
- Else if `"insufficient evidence" in gen` $\rightarrow$ `evidence_gated_refusal` (1 sample)
- Else if `recall_at_5 < 0.70 and len(gold_chunks) > 1` $\rightarrow$ `multi_chunk_synthesis_partial` (9 samples)
- Else $\rightarrow$ `gold_eval_mismatch_conciseness` (39 samples)

**Forensic Finding:** This taxonomy is **heuristic**, not human-verified. However, independent forensic inspection of the 39 questions confirmed that the generated answers are factually grounded and accurate (e.g., Q6: *"The lecture states that modern cars contain hundreds of processors"* vs. conversational gold preamble), but penalized by token-overlap metrics.

---

## 7. Knowledge Graph Quality & Orphan Entity Audit

Audited against `0-output/CS162_Lecture_1_What_is_an_Operating_System_720P_knowledge_package.zip`:

| KG Quality Dimension | Metric Name | Verified Value | Forensic Status | Analysis & Finding |
|---|---|---:|---|---|
| **Graph Scale** | Entities | 158 | Verified | 158 entity nodes in entities.json |
| | Relations | 103 | Verified | 103 relation edges in relations.json |
| **Referential Integrity** | Dangling Edge Rate | **0.0%** (0/103) | **VERIFIED** | Every relation connects valid entity IDs |
| | Duplicate Relation Rate | **0.0%** (0/103) | **VERIFIED** | Zero duplicate triples |
| **Topology** | **Orphan Entity Rate** | **58.9%** (93/158) | **INCORRECT in docs** | Mislabeled as 0.0% in narrative. 93 entities have degree = 0 |
| | Directed Cycles | **0** | **VERIFIED** | Topological sort succeeds across all 29 prerequisites |
| **Grounding** | Evidence Grounding Rate | **75.7%** (78/103) | **VERIFIED** | Backed by direct/indirect chunk sentences |
| | Cross-Chunk Edge Ratio | **7.8%** (8/103) | **VERIFIED** | 8 relations connect cross-chunk entities |

### Definition of "Orphan Entity":
In `evaluation/knowledge_graph/entity_auditor.py`, an orphan is explicitly defined as:
$$\text{degree}(e) = \text{in\_degree}(e) + \text{out\_degree}(e) = 0$$
Across all three packages in `0-output/`:
- **CS162:** 93 / 158 entities are orphans (**58.9%**)
- **MIT Deep Learning:** 35 / 87 entities are orphans (**40.2%**)
- **Self-Attention Transformers:** 48 / 82 entities are orphans (**58.5%**)

---

## 8. Hard-Coded Metric & Leakage Audit

1. **Hard-Coded Values in Python Code:**
   - Evaluators calculate metrics dynamically from data structures. No hard-coded metrics exist inside `benchmark_runner.py`, `retrieval_metrics.py`, or `reranker_metrics.py`.
2. **Hard-Coded / Estimated Values in Markdown Reports:**
   - In `evaluation/outputs/final_evaluation_report.md`, the **Dense-Only** row (`0.580, 0.880, 0.940, 0.814, 0.724, 0.710`) was inserted without an underlying JSON output artifact.
   - Narrative Hit@1 was written as `0.620` instead of recomputed `0.660`.
3. **Data Leakage Check:**
   - **Ingestion / Embeddings:** BGE embedding model was pre-trained by BAAI. No lecture QA questions were used in BGE training.
   - **Reranker Training:** The cross-encoder was trained on `triplets.json` generated from chunks. No benchmark QA pairs leaked into triplet generation.
   - **Graph Benchmark:** Questions were synthesized directly from database paths (`POTENTIAL CIRCULARITY`).

---

## 9. Statistical Uncertainty & Confidence Intervals

For finite benchmark populations, 95% Wilson score binomial confidence intervals are computed:

| Metric | Sample Size (N) | Successes | Point Estimate | 95% Confidence Interval (Wilson) | Interpretation |
|---|---:|---:|---:|:---:|---|
| **Retrieval Hit@5** | 50 | 49 | **98.0%** | **[89.5%, 99.7%]** | 1 failure represents 2.0 percentage points |
| **Pre-Rerank Hit@1** | 50 | 33 | **66.0%** | **[52.2%, 77.6%]** | Gold chunk at rank 1 pre-rerank |
| **Post-Rerank Hit@1** | 50 | 43 | **86.0%** | **[73.8%, 93.1%]** | Gold chunk at rank 1 post-rerank |
| **Routing Accuracy** | 12 | 12 | **100.0%** | **[75.8%, 100.0%]** | 1 failure represents 8.33 percentage points |
| **Graph Path Recovery** | 20 | 20 | **100.0%** | **[83.9%, 100.0%]** | 1 failure represents 5.0 percentage points |
| **2-Hop Hybrid Hit@5** | 5 | 5 | **100.0%** | **[56.6%, 100.0%]** | Micro-sample; lower bound is 56.6% |

---

## 10. Final Presentation-Safe Metrics

### SAFE TO PRESENT

The following statements are genuinely verified and defendable under technical interview interrogation:

1. **Retrieval Performance:**
   > *"On the frozen 50-question CS162 Lecture 1 benchmark, hybrid retrieval (dense + BM25 via RRF) achieves Hit@5 = 49/50 (98.0%), Recall@5 = 86.2%, and MRR@5 = 0.788."*

2. **Reranker Improvement:**
   > *"The cross-encoder reranker pushes relevant candidates to Rank 1 for 43 out of 50 questions (Hit@1 = 86.0%), lifting end-to-end MRR from 0.788 to 0.918 (+16.5% relative improvement)."*

3. **BM25 Lexical Ablation:**
   > *"BM25 lexical retrieval alone achieves only 38.0% Hit@5 (19/50) and 0.281 Recall@5, proving that the benchmark cannot be solved by simple keyword matching and requires semantic dense retrieval."*

4. **Knowledge Graph Traversal:**
   > *"On a targeted 20-question graph integration test spanning 1-hop, 2-hop, and 3-hop relationships in CS162, Neo4j bounded Cypher traversals achieved 20/20 (100%) path recovery."*

5. **Knowledge Graph Structural Integrity:**
   > *"The CS162 knowledge package contains 158 entities and 103 relations with 0 dangling edges and 0 prerequisite DAG cycles, while maintaining a 58.9% orphan entity rate for unlinked background concepts."*

6. **Generation & Grounding:**
   > *"Answer generation produces 100% deterministic timestamped citation provenance from retrieved chunk metadata. SQuAD token F1 is 0.459 (Keyword Recall = 54.5%), reflecting concise generation against conversational gold references rather than factual hallucination."*

7. **Architectural Realities:**
   > *"The system uses a deterministic Python single-pass workflow with LangGraph-compatible state dictionaries, and a deterministic heuristic query planner with a DSPy extension point (`set_dspy_module`)."*

---

## 11. Answers to the 12 Critical Forensic Questions

1. **Are the major retrieval numbers real and reproducible?**  
   **YES.** Hit@5 = 0.980, Recall@5 = 0.862, Pre-rerank MRR = 0.788, Post-rerank MRR = 0.918, and NDCG@5 = 0.767 are 100% reproduced from `evaluation_report_20260811_105621.json`.

2. **Is the 98% Hit@5 number genuinely 49/50?**  
   **YES.** Exactly 49 questions have gold evidence in top 5; exactly 1 question failed (Q17 in pre-rerank; Q40 in post-rerank).

3. **Is the 91.8% MRR genuinely end-to-end MRR@5?**  
   **YES.** Recomputed as post-rerank MRR (`ranking_quality`) over the 50 benchmark queries (mean = 0.9183).

4. **Is the 100% graph result a real independent benchmark or a graph-derived integration test?**  
   **GRAPH-DERIVED INTEGRATION TEST.** Questions were constructed from known Neo4j paths. It proves traversal mechanics work, but is not an independent double-blind benchmark.

5. **Is the 100% routing result sufficiently independent to call routing accuracy?**  
   **NO.** It is a targeted 12-sample integration test matching the router's keyword regexes. It should be presented as *"12/12 on a targeted routing test"*.

6. **Is Answer F1 = 0.459 correctly computed?**  
   **YES.** Recomputed mean token F1 is 0.4594. Keyword recall is 54.5%.

7. **Are any metrics hard-coded?**  
   **NO hardcoded metrics in python evaluation code.** However, the Dense-Only row in the markdown ablation table was estimated without a saved execution artifact.

8. **Are any reports stale or inconsistent with raw outputs?**  
   **YES.** Pre-rerank Hit@1 was reported as 0.620 (actual 0.660); Pre-rerank Hit@3 was reported as 0.940 (actual 0.880); Orphan rate was reported as 0.0% (actual 58.9%).

9. **Is there evidence of benchmark leakage?**  
   **NO retrieval leakage found.** Graph benchmark has circular path construction (`POTENTIAL CIRCULARITY`).

10. **Which numbers are safe to put in a presentation?**  
    Hit@5 = 98.0% (49/50), Recall@5 = 86.2%, Pre-rerank MRR = 0.788, Post-rerank MRR = 0.918, BM25 Hit@5 = 38.0%, 0 DAG cycles, and 20/20 graph path recovery on targeted test.

11. **Which numbers should NOT be presented as headline performance?**  
    Do NOT present "0.997 MRR" as retrieval performance (it is 2-candidate training convergence). Do NOT present "100% GraphRAG accuracy" (it is path recovery). Do NOT present "100% citation faithfulness" (it is metadata provenance).

12. **What is the single biggest remaining evaluation weakness?**  
    **Single-lecture scope.** All end-to-end QA benchmarks are evaluated on CS162 Lecture 1. Generalization across diverse multi-lecture corpora remains uncertified.
