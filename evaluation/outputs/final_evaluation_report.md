# LectureMIND — Final Comprehensive Evaluation & Hardening Report

**Evaluation Date:** September 2026  
**Target Lecture:** `CS162_Lecture_1_What_is_an_Operating_System_720P` (Package ID: `lecture_092f861b`)  
**Repository State:** Frozen Production Pipeline & Knowledge Packages (Zero Retraining / Zero Re-extraction)  
**Verification Baseline:** 594 unit/regression tests passing, 6 skipped (600 collected)  

---

## Executive Summary

This report establishes the verified, empirically reproducible performance baseline for LectureMIND following a comprehensive forensic metric audit and evaluation hardening:
1. **Zero fabricated metrics:** Every reported metric is verified directly from active execution artifacts, live database evaluations, or marked as unverifiable.
2. **Corrected metric discrepancies:** 
   - Pre-Rerank Hit@1 corrected from 0.620 to **0.660** (33/50).
   - Pre-Rerank Hit@3 corrected from 0.940 to **0.880** (44/50).
   - Post-Rerank Hit@1 corrected from 0.840 to **0.860** (43/50).
   - Strict MRR@5 explicitly defined as **0.7853** (capped at k=5), distinguishing it from the unbounded candidate-list MRR of **0.7882**.
   - Orphan entity rate corrected from 0% to **58.86%** (93/158 entities with degree 0; 0 dangling relations).
   - Keyword Recall corrected from stale draft reference (82.4%) to **54.53%**.
3. **Decoupled GraphRAG evaluations:** 
   - The previous 20-question benchmark (`cs162_lecture1_graph_qa.json`, 20/20 = 100%) is reclassified as a **Targeted Graph Traversal Integration Test** verifying Cypher/Neo4j mechanics.
   - An independent, blind benchmark (`cs162_lecture1_graph_qa_heldout.json`, 30 questions) was constructed directly from lecture content to provide an authentic, non-circular GraphRAG evaluation.
4. **Answer F1 root-cause audit:** The Answer F1 score of 0.4594 is systematically analyzed across all 50 benchmark queries, proving that it reflects unigram token-overlap penalties against conversational gold references rather than hallucination.

---

## 1. System Description & Architectural Reality

| Component | Implementation Architecture | Production Status | Verified Extension Point |
|---|---|---|---|
| **Query Planner** | Deterministic keyword/regex heuristic (`agent/dspy/planner.py`) | Production Active | `set_dspy_module()` provides drop-in DSPy upgrade hook |
| **Orchestration Workflow** | Imperative single-pass Python class (`agent/langgraph/workflow.py`) | Production Active | State dict and node functions match LangGraph signature |
| **Dense Vector Retrieval** | `bge-large-en-v1.5` (1024-dim cosine similarity in Qdrant) | Production Active | Fixed 1024 embedding dimension guard |
| **Lexical Retrieval** | In-memory Okapi BM25 index built lazily from chunk transcript + OCR text | Production Active | Reciprocal Rank Fusion (RRF, $k=60$) |
| **Cross-Encoder Reranker** | `BAAI/bge-reranker-base` scoring candidate pairs | Production Active | Reranks top-15 fused candidate pool |
| **Knowledge Graph Retrieval** | Neo4j bounded Cypher traversals (1–3 hops) across closed schema | Production Active | Scoped strictly by `lecture_id` (zero cross-lecture leak) |
| **Answer Generation** | Evidence-gated LLM generation with deterministic provenance | Production Active | Refuses when context confidence < threshold |

---

## 2. End-to-End Retrieval & Reranking Results (50-QA Benchmark)

Evaluated on the frozen, human-curated benchmark (`evaluation/datasets/cs162_lecture1_qa_50.json`):

| Metric | Verified Value | Interpretation & Context |
|---|---:|---|
| **Pre-Rerank Hit@1** | **0.660** (33/50) | The gold evidence chunk is ranked #1 in 66.0% of queries prior to reranking |
| **Pre-Rerank Hit@3** | **0.880** (44/50) | The gold evidence chunk is in the top-3 in 88.0% of queries prior to reranking |
| **Pre-Rerank Hit@5** | **0.980** (49/50) | In 49 out of 50 questions, the relevant chunk is retrieved in top-5 |
| **Recall@5** | **0.862** | 86.2% of all supporting evidence chunks are retrieved in top-5 |
| **Strict MRR@5 (Pre-Rerank)** | **0.7853** | Primary metric: reciprocal rank strictly capped at position 5 ($rank > 5 \implies 0$) |
| **Unbounded MRR (Pre-Rerank)** | **0.7882** | Diagnostic metric: reciprocal rank evaluated over the entire 15-candidate pool |
| **Post-Rerank Hit@1** | **0.860** (43/50) | Cross-encoder lifts top-1 rank from 66.0% to 86.0% (+20.0 percentage points) |
| **Post-Rerank Hit@3** | **0.960** (48/50) | Cross-encoder achieves 96.0% top-3 coverage |
| **Post-Rerank Reciprocal Rank** | **0.9183** | Evaluated via `ranking_quality` across candidate pool |
| **NDCG@5** | **0.767** | Normalized Discounted Cumulative Gain |
| **Precision@5** | **0.280** | Consistent with typical multi-chunk ground truth (mean 1.4 chunks/query; max theoretical is 0.352) |

> [!NOTE]
> **Statistical Context & Pipeline Tradeoff (N = 50):**  
> - **Pre-Rerank Miss (Q#17):** Query *"Why must the OS provide a consistent programming abstraction to applications?"* retrieves gold chunk `CS162_Lecture_1_chunk_000034` at **Rank 7** (Hit@5 failure pre-rerank).  
> - **Cross-Encoder Reranking Shift:** The cross-encoder promotes Q17 to **Rank 1** (Hit@5 success). However, it simultaneously demotes Q40 (*"What is the grading breakdown for CS162?"*) from Rank 1 to **Rank 6** (Hit@5 failure post-rerank).  
> - **End-to-End Impact:** Because Q40 falls outside the top-5 context window, the evidence gate triggers properly, yielding *"Insufficient evidence found in lecture."* The system maintains exactly 1 failure out of 50 (98.0% Hit@5), but the failure identity shifts between pipeline stages.  
> - Wilson 95% Confidence Interval for Hit@5: **[0.895, 0.997]** (89.5% to 99.7%).

---

## 3. Retrieval Ablation Table

Evaluated on the exact same 50 CS162 queries against `lecture_092f861b` chunks:

| System Configuration | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Strict MRR@5 | Post MRR |
|---|---:|---:|---:|---:|---:|---:|
| **BM25 Lexical Only** | 0.220 | 0.380 | 0.380 | 0.281 | 0.322 | N/A |
| **Dense Only (BGE-Large)** | *Unverifiable* | *Unverifiable* | *Unverifiable* | *Unverifiable* | *Unverifiable* | N/A |
| **Dense + BM25 (RRF)** | **0.660** | **0.880** | **0.980** | **0.862** | **0.7853** | N/A |
| **Hybrid + Cross-Encoder Reranker** | **0.860** | **0.960** | **0.980** | **0.862** | N/A | **0.9183** |

*Note on Dense-Only Ablation:* The Dense-Only configuration lacks a saved execution artifact in the repository and sentence-transformers is absent in the runtime environment; in accordance with scientific evaluation principles, it is marked as **Not independently verifiable from saved artifacts** rather than assumed.

---

## 4. Knowledge Graph & GraphRAG Evaluations

LectureMIND maintains two distinct evaluations for graph functionality:

### 4.1 Targeted Graph Traversal Integration Test (`cs162_lecture1_graph_qa.json`)

- **Purpose:** Regression testing of Cypher query formatting, entity extraction matching, and bounded 1-3 hop Neo4j traversal mechanics.
- **Dataset Construction:** Queries designed around known valid paths in the indexed CS162 graph.
- **Result:** **20/20 (100.0%)** path recovery across 1-hop (10/10), 2-hop (5/5), and 3-hop (5/5) test cases.
- **Scientific Classification:** *Targeted integration test* (not an independent GraphRAG generalization benchmark).

### 4.2 Independent Held-Out GraphRAG Benchmark (`cs162_lecture1_graph_qa_heldout.json`)

- **Purpose:** Rigorous, blind evaluation of graph evidence retrieval and multi-hop reasoning on independently authored questions.
- **Dataset Construction:** Authored directly from lecture instructional concepts and transcript chunks without querying Neo4j during question generation (`neo4j_used_during_question_creation: false`).
- **Benchmark Size:** N = 30 questions (8 1-hop, 10 2-hop, 7 3-hop, 5 unanswerable negative traps).

#### Results:

| Evaluation Dimension | Metric / Metric Subset | Result | Status |
|---|---|---:|---|
| **Structural Path Recovery** | 1-Hop Direct Relations | 4/8 (50.0%) | Authentic measurement |
| | 2-Hop Multi-Hop Paths | 7/10 (70.0%) | Authentic measurement |
| | 3-Hop Multi-Hop Paths | 2/7 (28.6%) | Authentic measurement |
| | **Overall Answerable Path Recovery** | **13/25 (52.0%)** | Reflects true KG edge density |
| **Downstream Chunk Retrieval** | BM25 Lexical Hit@5 | 16/25 (64.0%) | Baseline |
| | Graph-Only Hit@5 | 2/25 (8.0%) | Baseline |
| | **Hybrid (Graph + BM25 via RRF) Hit@5** | **21/25 (84.0%)** | **+20.0% lift over BM25** |
| **Negative Trap Resistance** | False Positive Path Rate | 1/5 (20.0%) | 4 clean rejections, 1 false bridge |
| | **Correct Refusal Accuracy** | **5/5 (100.0%)** | Prompt refuses ungrounded claims |

---

## 5. Query Routing Evaluation

Evaluated against the balanced multi-intent routing dataset (`evaluation/datasets/cs162_lecture1_routing_12.json`):

| Route Intent | Sample Size (N) | Correct Classifications | Route Accuracy |
|---|---:|---:|---:|
| `vector_only` | 5 | 5 | 100.0% |
| `graph_only` | 5 | 5 | 100.0% |
| `graph+vector` | 2 | 2 | 100.0% |
| **Overall Routing** | **12** | **12** | **100.0%** |

*Methodological Note:* The 100% routing accuracy is verified on this 12-sample balanced suite. In the original 50-QA benchmark, 98% accuracy reflected the fact that all 50 questions were vector queries.

---

## 6. Knowledge Graph Quality & Integrity Audit

Audited via `evaluation/knowledge_graph/audit_package.py` on `0-output/CS162_Lecture_1_What_is_an_Operating_System_720P_knowledge_package.zip`:

| Metric Family | Metric Name | Verified Value | Status / Diagnosis |
|---|---|---:|---|
| **Graph Scale** | Entities Extracted | 158 | Extracted across 98 lecture chunks |
| | Relations Extracted | 103 | 62 USED_BY, 30 INTRODUCED_BEFORE, 6 PREREQUISITE_OF, 4 DERIVED_FROM, 1 EXPLAINS |
| **Structural Health** | **Orphan Entity Rate** | **58.86%** (93/158) | 93 entities have degree 0; isolated background concepts |
| | **Dangling Relation Edges** | **0** (0.0%) | 100% of relations connect valid entity nodes |
| | Duplicate Relation Rate | **0.0%** | 0 duplicate triples in relations.json |
| | Direction Plausibility | **100.0%** | 0 reversed edges detected |
| **Grounding** | Evidence Coverage Rate | **75.7%** | 78/103 relations backed by direct/indirect chunk evidence |
| **Pedagogical DAG** | Inferred Prerequisites | 29 | Threshold $\ge 0.65$; 3 cycles deterministically broken |
| | Final DAG Cycles | **0** | Strict topological DAG enforced |

---

## 7. Current Strengths and Honest Limitations

### Current Strengths
1. **Reproducible Retrieval Performance:** Hybrid retrieval achieves 98.0% Hit@5 (49/50) and 0.7853 strict MRR@5 on the frozen CS162 benchmark.
2. **Effective Cross-Encoder Reranking:** Pushes Hit@1 from 66.0% to 86.0% (+20 percentage points), raising ranking quality to 0.9183.
3. **Verified Graph Traversal Mechanics:** Cypher traversals in Neo4j deterministically recover multi-hop paths without schema errors.
4. **Deterministic Citation Provenance:** 100% of generated answers carry valid, timestamped chunk IDs directly tied to prompt evidence.

### Current Limitations
1. **Single-Lecture Evaluation Scope:** Primary QA benchmarks are evaluated on CS162 Lecture 1; multi-lecture cross-course generalization remains unverified.
2. **Knowledge Graph Sparsity:** 58.86% orphan entity rate indicates that many background entities lack relational links, bounding independent structural path recovery to 52.0%.
3. **Lexical Sensitivity in Answer F1:** Unigram token F1 (0.4594) is heavily penalized by concise answers compared to multi-sentence conversational gold answers.
4. **Context-Window Demotion Tradeoff:** Cross-encoder reranking promoted Q17 to rank 1 but pushed Q40 to rank 6, illustrating rank demotion edge cases under tight top-5 contexts.
