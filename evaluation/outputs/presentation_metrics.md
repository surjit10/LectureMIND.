# LectureMIND — Verified Presentation & Interview Reference Sheet

*This summary contains strictly verified metrics and defensible talking points for technical interviews and presentations. Every claim here is backed by executable tests, forensic audits, and reproducible benchmark artifacts.*

---

## Headline Metrics (Verified & Defensible)

### 1. End-to-End Retrieval & Reranking (50-Question CS162 Benchmark)
- **Hit@5:** **98.0%** (49/50 questions) on `evaluation/datasets/cs162_lecture1_qa_50.json`
- **Recall@5:** **86.2%** (recovers 86.2% of all multi-chunk ground truth in top-5)
- **Pre-Rerank Hit@1:** **66.0%** (33/50)
- **Pre-Rerank Hit@3:** **88.0%** (44/50)
- **Pre-Rerank Strict MRR@5:** **0.7853** (reciprocal rank capped at k=5; 0.7882 unbounded over candidate pool)
- **Post-Rerank Hit@1:** **86.0%** (43/50) — lifted from 66.0% by cross-encoder (+20.0 percentage points)
- **Post-Rerank Hit@3:** **96.0%** (48/50)
- **Post-Rerank Reciprocal Rank Mean:** **0.9183** (evaluated via `ranking_quality`)
- **NDCG@5:** **0.767**

### 2. Retrieval Layer Ablation (CS162 Lecture 1)
- **BM25 Lexical-Only:** **38.0% Hit@5**, 0.322 MRR (proves benchmark cannot be solved by simple keyword search)
- **Dense Vector-Only:** *Not independently verifiable from saved artifacts* (omitted from headline claims)
- **Hybrid (Dense + BM25 via RRF):** **98.0% Hit@5**, **0.7853 Strict MRR@5** (recovers exact entities like textbook names and lines of code)
- **Cross-Encoder Reranker:** Pushes top candidates to Rank 1 (Hit@1: 66.0% → 86.0%), lifting ranking quality to **0.9183**

### 3. Knowledge Graph & GraphRAG (Two Decoupled Evaluations)
- **Targeted Graph Traversal Integration Test:** **20/20 (100.0%)** path recovery on `cs162_lecture1_graph_qa.json` (purpose: regression verification of Cypher query and Neo4j traversal mechanics)
- **Independent Held-Out GraphRAG Benchmark:** Evaluated on `cs162_lecture1_graph_qa_heldout.json` (30 questions: 8 1-hop, 10 2-hop, 7 3-hop, 5 negative traps)
  - **Structural Path Recovery (Live Neo4j):** **52.0%** (13/25) — 1-hop: 50.0% (4/8), 2-hop: 70.0% (7/10), 3-hop: 28.6% (2/7)
  - **Downstream Evidence Chunk Hit@5:** **84.0%** (21/25) under Hybrid Graph+BM25 (+20.0% gain over BM25-only 64.0%)
  - **Negative Resistance:** **100% correct refusal** on unanswerable/negative questions (0 false positive claims)
- **Prerequisite DAG Integrity:** **29 prerequisites** in CS162, **0 cycles**, **0 self-loops** (strict topological sort enforced)
- **Orphan Entity Rate:** **58.86%** (93/158 entities have degree = 0; isolated background concepts; 0 dangling relation edges)
- **Graph Routing Accuracy:** **100.0%** (12/12) on balanced multi-route benchmark (5 vector, 5 graph, 2 hybrid)

### 4. Generation & Grounding
- **Answer F1:** **0.4594** (Mean Keyword Recall is **54.53%**; error analysis proves gap is driven by concise LLM answers vs multi-sentence narrative gold answers, not hallucination)
- **Pipeline Rank Tradeoff:** Cross-encoder promotes Q17 to Rank 1 (success) but demotes Q40 to Rank 6, triggering the evidence gate ("Insufficient evidence found in lecture") as the single post-rerank Hit@5 failure
- **Citation Provenance:** **Deterministic citation provenance** attached directly from retrieved chunk metadata (every answer carries chunk IDs and start/end timestamps)
- **Refusal Safety:** Rejects ungrounded queries with *"Insufficient evidence"* rather than hallucinating

### 5. Benchmark Scope
- **Benchmark Size:** N = 50 questions for QA retrieval; N = 20 for Graph Integration; N = 30 for Independent Held-Out Graph; N = 12 for Routing
- **Evaluated Package:** 1 lecture fully evaluated end-to-end (`CS162 Lecture 1`, 98 chunks, 158 entities, 103 relations)

---

## Precise Phrasing for Presentations & Interviews

| Do NOT Claim (Misleading) | SAY THIS INSTEAD (Accurate & Defensible) |
|---|---|
| ❌ *"98% accuracy"* | ✅ *"98% Hit@5 on the 50-question CS162 Lecture 1 retrieval benchmark"* |
| ❌ *"Pre-rerank Hit@1 was 62%"* | ✅ *"Pre-rerank Hit@1 is 66.0% (33/50); post-rerank Hit@1 is 86.0% (43/50)"* |
| ❌ *"0.997 end-to-end retrieval MRR"* | ✅ *"0.997 pairwise cross-encoder training convergence on 2-candidate pairs; 0.7853 strict MRR@5 pre-rerank and 0.9183 post-rerank on live benchmark"* |
| ❌ *"100% GraphRAG accuracy"* | ✅ *"100% (20/20) on targeted traversal regression tests; 52.0% (13/25) structural path recovery and 84.0% downstream chunk Hit@5 on our independent held-out graph benchmark"* |
| ❌ *"0% orphan entity rate"* | ✅ *"58.9% orphan entity rate (93/158 background entities have degree 0, a known KG extraction limitation); 0 dangling relations"* |
| ❌ *"100% LLM / RAGAS citation faithfulness"* | ✅ *"Deterministic citation provenance attached from retrieved source metadata"* |
| ❌ *"DSPy-powered query planning"* | ✅ *"Deterministic heuristic query planner with a DSPy-compatible extension point (`set_dspy_module`)"* |
| ❌ *"LangGraph multi-agent orchestration"* | ✅ *"Deterministic Python single-pass workflow using LangGraph-compatible state structures"* |
| ❌ *"Validated on multi-lecture production dataset"* | ✅ *"Frozen single-lecture CS162 benchmark with knowledge packages packaged for MIT and Transformers"* |
| ❌ *"RAGAS evaluated and load tested to 1000 users"* | ✅ *"Includes integration scaffolds for RAGAS and concurrency load profiling; published metrics use standard IR & QA benchmarks"* |

---

## 30-Second Interview Pitch

> *"LectureMIND is a full-stack, private multimodal GraphRAG system for academic lectures. Instead of naive semantic search, it fuses dense vector embeddings (`bge-large-en-v1.5`) and BM25 lexical search via Reciprocal Rank Fusion, followed by a fine-tuned cross-encoder reranker that achieves **98.0% Hit@5 and lifts Hit@1 from 66.0% to 86.0%** on our 50-question CS162 benchmark. For conceptual dependencies, it navigates a cycle-free Neo4j knowledge graph achieving **84.0% downstream Hit@5 on independently evaluated multi-hop queries**, while guaranteeing deterministic timestamped grounding and clean refusal on unanswerable questions."*
