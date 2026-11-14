# LectureMIND — Metrics Matrix

> **Purpose:** One document that quantifies what LectureMIND is and why it is stronger than generic RAG / video-summarizer baselines.
> **Audience:** Engineers, evaluators, and reviewers who need the measured numbers behind the claims.
> **Last updated:** 2026-11-14 · current QA numbers come from `evaluation/reports/smoke_groq_20261114_newcode/` and `evaluation/reports/retrieval_comparison_20261114/`; older figures are marked historical.
> **Legend:** `[Measured]` verified by live benchmark execution and report.
> **Companion docs:** [EVALUATION.md](./docs/EVALUATION.md) (framework) · [RETRIEVAL_SYSTEM.md](./docs/RETRIEVAL_SYSTEM.md) (retrieval internals)

---

## Table of Contents

- [1. Project at a glance](#1-project-at-a-glance)
- [2. Measured metrics](#2-measured-metrics)
- [3. Comparison matrix — LectureMIND vs baselines](#3-comparison-matrix--lecturemind-vs-baselines)
- [4. Metrics roadmap](#4-metrics-roadmap)
- [5. Credibility notes](#5-credibility-notes)

---

## 1. Project at a glance

| Dimension | Value | Evidence |
|---|---|---|
| Architecture layers | Multimodal ingestion → segmentation → entity/relation extraction → prerequisite DAG inference → hybrid GraphRAG (vector + BM25 + graph) → cross-encoder rerank → evidence-gated generation | `cloud/`, `retrieval/`, `agent/pipeline/` |
| Modalities captured | Audio (100% chunk coverage) + slide visuals + OCR at slide keyframes | `multimodal_fusion.py`, live scan |
| Serving model | Local Ollama `qwen2.5:3b` (offline, fully private) **or** cloud APIs — OpenAI, Gemini, Groq, OpenRouter, Anthropic — hot-swappable via `ProviderRegistry` | `local/llm/provider_registry.py` |
| Embedding | `bge-large-en-v1.5` (1024-dim) | startup log |
| Reranker | Global cross-encoder `BAAI/bge-reranker-base` (XLM-R), CPU, process singleton; dynamic int8 quantization (`RERANKER_QUANTIZE`) with automatic FP32 fallback | `rerank_service.py`, `reranker_loader.py`, `app.py` |
| Hybrid retrieval | Dense (`bge-large-en-v1.5`) + BM25 lexical candidates fused via Reciprocal Rank Fusion (RRF) before cross-encoder reranking; enabled by default (`ENABLE_HYBRID_RETRIEVAL`) | `retrieval/hybrid/bm25_retriever.py`, `config.py` |
| Knowledge graph | Production run: CS162 (158 entities / 103 relations; 93 orphan entities = 58.9% isolated background concepts, 0 dangling relations), MIT (87 entities / 94 relations), Self-Attention (82 entities / 65 relations); **262 total relations**, 0 dangling — all re-verified by audit against the current packages in `0-output/` | Kaggle package audit (`audit_package.py`) |
| Prerequisite DAG | Strict DAG enforced via deterministic DFS cycle resolution; 29 prerequisites (CS162), 15 (MIT), 14 (Self-Attention), **58 total prerequisites**, 0 cycles / 0 self-loops across all packages (re-verified). Reference-label F1 is only defined for the Transformer lecture (the sole annotated one) — see §2.6 | `evaluation/knowledge_graph/` |
| Content scale | Ingested long-lecture packages (48–98 chunks / 44–83 min); package size: **208–437 KB** (replacing 1+ GB video) | `0-output/` |
| Automated tests | **649 passing** (2026-11-14, `.venv_clean`) — planner, retrieval, hybrid retrieval, reranker, context builder, prerequisite inference, KG auditor, loader, isolation, pipeline, extraction, evaluation metrics | `pytest` |

---

## 2. Measured metrics

### 2.1 Retrieval & reranking quality (cross-encoder eval on lecture triplets)

Source: `data/packages/<lecture>/training_metrics.json` — ranking eval on generated triplets (hard-negative sampling).

| Lecture | Triplets | MRR | Recall@5 | Recall@10 | NDCG@10 |
|---|---|---|---|---|---|
| lecture_bb2ee3fa | 1,086 | **0.9949** | 1.000 | 1.000 | **0.9963** |
| lecture_6a80d31a | 263 | **0.9981** | 1.000 | 1.000 | **0.9986** |
| lecture_53b21041 | 11 | **1.000** | 1.000 | 1.000 | **1.000** |
| **Aggregate** | **1,360** | **0.996** | **1.000** | **1.000** | **0.997** |

> **Interpretation caveat (important):** this table evaluates the reranker **on its own training objective** — each query ranks exactly 2 candidates (1 positive / 1 negative) drawn from the same package used for training, with no held-out split and no hard guarantee of truly negative hard examples. Near-perfect scores here are expected and demonstrate training convergence only; they are **not** a generalization measure. The externally meaningful retrieval numbers are the live smoke benchmark in §2.4.

### 2.2 Grounding, citations & hallucination resistance (live-verified)

| Metric | Value | How verified |
|---|---|---|
| Out-of-scope refusal (no hallucination) | Refuses with "Insufficient evidence…" instead of inventing | Live trap question (capital of France + fabricated topic) |
| Source-grounded answers | 100% of answers carry `chunk_id` + `timestamp` sources built only from retrieved chunks | `_build_sources`, live query |
| Lecture-wide synthesis | Summary context spans **87% of the lecture timeline on average** (worst prompt 86%; 7.7 of 10 deciles) across 9 summary prompts on 3 lectures | `evaluation/reports/retrieval_comparison_20261114/` |
| Visual grounding | Answered from slide content when planner routes `diagram/slide` intent | Live visual query |
| Graph grounding | Traversal answers (`Relational Model → Relational Algebra → Database Systems → SQL`) | Live lowercase graph query |
| Language consistency | Answers pinned to question language (English-only verified; 0 non-English chars) | Live test |
| Cross-lecture data leakage | **0 foreign chunks** in leak test; hard guards raise without `lecture_id`; Neo4j scoped by `(entity_id, lecture_id)` | Live test + Cypher inspection |

### 2.3 Latency & throughput (live, real LLM)

Measured 2026-11-14 with **Groq `openai/gpt-oss-20b`** (client paced at 6,000 tokens/min); embeddings and the cross-encoder run on local CPU. Source: `evaluation/reports/smoke_groq_20261114_reviewed/` (45 questions, 15 candidates, 6,000-char context).

| Stage | Mean | p95 |
|---|---|---|
| Retrieval (dense + graph, live Qdrant/Neo4j) | 0.15 s | 0.26 s |
| Reranker + context (local CPU, 15 candidates) | 4.54 s | 6.51 s |
| Generation (Groq, rate-limiter paced) | 22.82 s | 27.53 s |
| **Total end-to-end** | **27.51 s** | **31.20 s** |

> Generation time is dominated by the 6,000 tokens/min client pacing and Groq network time; the larger 6,000-char context raised it from 16.0 s (4,000-char context, same day). The CPU reranker varied by ~0.4 s between same-day runs of identical code. Load: the recorded load test (`evaluation/reports/load_test_report.csv`) shows 0 errors at 1–2 concurrent users and **2/4 errors at 4 concurrent users (p95 47.7 s)** — no high-concurrency capacity is claimed.

### 2.4 End-to-end QA accuracy (45-question three-lecture smoke set, reviewed labels)

Source: `evaluation/reports/smoke_groq_20261114_reviewed/` — **45/45 questions, 0 failures**, Groq `openai/gpt-oss-20b`, real `QueryWorkflow`. Dataset: `evaluation/datasets/multilecture_smoke_45_reviewed.json` (15 per lecture; 17 factual / 12 conceptual / 10 visual / 1 definition / 3 graph / 2 unanswerable). 40 questions were LLM-drafted from one chunk each and then given an **AI-assisted label review** (3 reference answers corrected, 25 missing supporting chunks added, 3 graph-fact questions flagged); **not human-reviewed**.

| Metric | Mean | 95% CI | n |
|---|---:|---:|---:|
| **Post-rerank Hit@5** | **0.907** | 0.814 – 0.977 | 43 |
| **Post-rerank MRR@5** | **0.801** | 0.691 – 0.903 | 43 |
| Post-rerank Recall@5 / nDCG@5 | 0.880 / 0.813 | — | 43 |
| **Context recall** | **0.843** | 0.736 – 0.936 | 43 |
| Pre-rerank Hit@5 / MRR@5 (RRF order) | 0.884 / 0.804 | — | 43 |
| Routing accuracy / visual routing accuracy | 0.933 / 0.911 | — | 45 |
| Answer token F1 (lexical) | 0.256 | 0.201 – 0.313 | 45 |
| Keyword recall | 0.786 | 0.682 – 0.878 | 45 |
| Correct refusals (unanswerable) | 2/2 | — | 2 |
| Answerable questions refused | 4/43 | — | 43 |
| Mean / p95 total latency | 27.5 s / 31.2 s | — | 45 |

**By question type (post-rerank):** factual (17) Hit@5 0.94 / MRR@5 0.83 / context recall 0.90 · conceptual (12) 0.92 / 0.86 / 0.92 · visual (10) 0.90 / 0.85 / 0.84 · definition (1) 1.00 / 1.00 / 1.00 · graph (3) 0.67 / 0.18 / 0.17.

**Component ablation** (`evaluation/reports/ablation_20261114_reviewed/`, no LLM):

| Configuration | Hit@1 | Hit@5 | MRR@5 | Context recall |
|---|---:|---:|---:|---:|
| Dense only | 0.674 | 0.814 | 0.734 | 0.793 |
| + BM25 (RRF) | 0.744 | 0.884 | 0.804 | 0.824 |
| + cross-encoder | 0.744 | 0.907 | 0.801 | 0.874 |
| + graph context (production) | 0.744 | 0.907 | 0.801 | 0.843 |
| Production, previous 4,000-char / 50%-graph budget | 0.744 | 0.907 | 0.801 | 0.783 |

**Error analysis** (43 answerable): 36 answered with the evidence in the prompt · 2 refused with the evidence in the prompt (visual questions not flagged `need_visual`) · 4 evidence retrieved but not placed in the prompt · 1 retrieval miss.

**What the numbers say:**
- BM25 is the largest single gain; the cross-encoder adds +0.023 Hit@5 and +0.05 context recall, with no MRR gain on this set.
- The graph layer does not change chunk ranking and costs prompt space (graph-routed context recall 0.750 → 0.528); its benefit for relational answers is not yet demonstrated.
- The 6,000-char / 25%-graph budget was chosen on this same set (no dev split), so its gain is optimistic; it raised Groq-paced latency from 21.8 s to 27.5 s.
- n = 43 gives ±0.1 confidence intervals; graph (3) and unanswerable (2) slices are too small for claims. No judge-based faithfulness metric has been run.

**Effect of the 2026-11 correctness fixes** (`evaluation/reports/retrieval_comparison_20261114/`, no LLM calls, same data and models; old-code run reproduces the 2026-10-10 report exactly):

| | Before | After |
|---|---:|---:|
| Post-rerank Hit@5 / MRR@5 | 0.884 / 0.759 | 0.884 / 0.759 |
| Routing accuracy | 0.889 | 0.933 |
| Summary prompts — mean timeline span in context | 0.807 | 0.872 |
| Summary prompts — worst timeline span | 0.464 | 0.863 |
| Summary prompts — lecture deciles covered (of 10) | 5.3 | 7.7 |
| Incoming graph relations of "Process Abstraction" shown correctly | 1/16 | 16/16 |

**Historical (not reproducible):** the 2026-08-11 run on `cs162_lecture1_qa_50.json` (`evaluation/outputs/evaluation_report_20260811_105621.*`; Hit@5 0.980, MRR@5 0.785, routing 0.980, answer F1 0.459) cannot be rerun on the current packages — its `lecture_id` is no longer loaded and its gold chunk IDs now point to different text (keywords match for 21/50 and 17/50 questions in the two CS162 packages). Several CS162 tuning decisions were made on that set, so it was never a held-out measurement.

### 2.5 Storage & compression

LectureMIND replaces raw video with structured knowledge. An **85-minute, 720p lecture (~1.2 GB at typical encoding)** compresses into a **~437 KB knowledge package** (~2,800× smaller):

| Artifact | Size | Notes |
|---|---|---|
| Source video (85 min, 720p) | ≈ 1.2 GB | Not stored or shipped |
| CS162 Knowledge package (`.zip`) | **437 KB** | 98 chunks, 158 entities, 103 relations, 29 prerequisites (~2,800× smaller than video) |
| MIT 6.S191 Knowledge package (`.zip`) | **312 KB** | 70 chunks, 87 entities, 94 relations, 15 prerequisites |
| Self-Attention Knowledge package (`.zip`)| **208 KB** | 48 chunks, 82 entities, 65 relations, 14 prerequisites |
| Per-query LLM context | ≈ 3.5 KB | the only text the model reads per question |
| Code snapshot (`dist/lecturemind-code-kaggle.zip`) | **131.89 KB** | 64 files, 335 KB of source → 2.5× zip ratio |
| Whole corpus (725 packages) | ≈ 85 MB | replaces an estimated 100+ GB of source video |

This is the storage story the architecture is built around: the expensive, bulky artifact (video) is processed **once** in the cloud, and everything a student needs — searchable chunks, embeddings, graph, captions, prerequisites — ships as a small portable ZIP that runs fully offline on a laptop.

### 2.6 Knowledge Graph Quality & Prerequisite DAG Metrics (Live Audited)

Source: `evaluation/knowledge_graph/audit_package.py` — audits regenerated 2026-09-15 from the **latest packages in `0-output/`** (Transformer lecture + CS162/MIT/Self-Attention long-lecture packages).

> **Applicability warning:** the reference labels in `evaluation/knowledge_graph/*_gold.json` are **LLM-assisted labels pending human verification**, and they were annotated for the **6.5-minute Transformer lecture only**. Gold-reference metrics are therefore computed **only** where the gold lecture matches the audited package (`gold_applies` flag in `outputs/*/metrics.json`); for all other lectures they are reported as `null` — never as 0.0 and never replaced by optimistic fallbacks.

#### Gold-reference scores (Transformer lecture only — the sole annotated lecture)

> These were measured while the extraction prompts and prerequisite rules contained this lecture's entities (removed 2026-11-14), so they are not an independent measurement.

| Metric | Precision | Recall | F1 | Basis |
|---|---|---|---|---|
| Entity extraction | 55.6% (5/9) | 31.3% (5/16) | **40.0%** | one-to-one matching against 16 reference concepts |
| Relation extraction | 0.0% (0/5) | 0.0% (0/21) | **0.0%** | exact + fuzzy matching against 21 reference relations |
| Prerequisite DAG | 0.0% (0/3) | 0.0% (0/10) | **0.0%** strict (fuzzy diagnostic F1: 15.4%) | against 10 reference prerequisites |

#### Structural metrics (measured, gold-independent — all audited packages, latest `0-output/` versions)

| Metric | CS162 (98 chunks) | MIT (70 chunks) | Self-Attention (48 chunks) | Transformer (9 chunks) |
|---|---|---|---|---|
| Strict DAG guarantee | True | True | True | True |
| Cycle count | 0 | 0 | 0 | 0 |
| Self-loop count | 0 | 0 | 0 | 0 |
| Relations (0 dangling) | 103 | 94 | 65 | 5 |
| Prerequisite edges (DAG) | 29 | 15 | 14 | 3 |
| Pedagogically supported prerequisites | 29/29 (100%) | n/a | n/a | n/a |
| Composite diagnostic (only measurable components counted) | **37.4 / 55** | **25.0 / 45** | **27.2 / 45** | 32.1 / 100 |
| — of which graph coherence (direct-evidence rate + entity participation, 10 pts) | 2.15 | 3.31 | 2.15 | 7.4 |
| — of which downstream GraphRAG usefulness (Recall@5 contribution, 10 pts) | 4.86 | n/a | n/a | n/a |

All four audits were computed from the packages in `0-output/`. The CS162 figures (98 chunks, 158 entities, 103 relations) describe package `lecture_05086e79`; the QA and held-out benchmarks use `lecture_092f861b` (93 chunks, 138 entities, 189 relations). Strict DAG, 0 cycles and 0 dangling relations are guaranteed by the DFS cycle breaker and loader validation — they confirm the constraints run, not extraction quality.

Provenance note (updated 2026-09-15): every number in this file traces to either (a) a stored benchmark report generated by the code in this repo (`evaluation/outputs/`), (b) a regenerated audit artifact (`outputs/kg_quality_latest_0output/`), or (c) a live execution recorded at the time of measurement.

#### Knowledge Graph Evaluations (Decoupled Benchmark Suite)

LectureMIND maintains two distinct evaluations for graph functionality:

##### A. Targeted Graph Traversal Integration Test (`cs162_lecture1_graph_qa.json`)
*Regression verification of Neo4j Cypher traversal and schema mechanics:*
- **Sample size:** N = 20 (10 1-hop, 5 2-hop, 5 3-hop)
- **Path Recovery Success:** **20/20 (100.0%)** on targeted paths
- **Status:** Reclassified as an internal integration test verifying Cypher query formation rather than an independent GraphRAG accuracy metric.

##### B. Held-Out GraphRAG Benchmark (`cs162_lecture1_graph_qa_heldout.json`)
*Questions written without querying Neo4j (`neo4j_used_during_question_creation: false`); Neo4j was used during gold-label verification (`graph_heldout_creation_metadata.json`). Run 2026-09-25 on the pre-fix code; report now generated from the JSON with no hard-coded values. The evaluator uses `SimpleBM25` + `SimpleGraphRetriever` — not the production dense + cross-encoder pipeline.*
- **Sample size:** N = 30 (8 1-hop, 10 2-hop, 7 3-hop, 5 negative)
- **Structural path recovery (direction ignored):** **52.0%** (13/25) — 1-hop 4/8, 2-hop 7/10, 3-hop 2/7
- **Hybrid (graph + BM25 RRF) chunk Hit@5:** **84.0%** (21/25) — 1-hop 5/8, 2-hop 10/10, 3-hop 6/7; vs. BM25 alone: Hit@5 +20.0 pts, **Hit@1 −24.0 pts, MRR −9.8 pts**
- **Negative questions:** 5/5 refused; 1/5 had a false-positive graph path
- **Lenient answer match** (keyword recall ≥ 0.5 or token F1 ≥ 0.25): 25/30 — 9 of the 20 answerable matches passed on keyword recall alone; not a correctness judgment

#### Downstream GraphRAG Retrieval Benchmark (CS162 — 12 Gold Routing Queries)

Evaluated via `evaluation/knowledge_graph/graphrag_evaluator.py` against `0-output/CS162_...zip`:

| Retrieval Route | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Precision@5 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 / Lexical Retrieval** | 0.2500 | 0.4167 | 0.5000 | 0.4028 | 0.1167 | 0.3885 |
| **Graph-Only RAG** | 0.0833 | 0.4167 | 0.4167 | 0.2361 | 0.1000 | 0.2795 |
| **Hybrid RAG (RRF)** | **0.2500** | **0.4167** | **0.6667** | **0.4861** | **0.1667** | **0.4204** |

#### Multi-Lecture Extraction Yield (Full Cloud Execution)

Empirical extraction yield across three complete production lecture packages in `0-output/` using sliding-window chunking, compact entity alias remapping (`E1, E2...`), and 8192-token retry budgets:

| Lecture Package | Duration / Chunks | Extracted Entities | Extracted Relations | Inferred Prerequisites | DAG Status | Package Size |
|---|---|---|---|---|---|---|
| **CS162 Operating Systems** | ~83 min (98 chunks) | **158** | **103** | **29** | **Strict DAG (0 cycles)** | **437 KB** |
| **MIT 6.S191 Deep Learning** | ~56 min (70 chunks) | **87** | **94** | **15** | **Strict DAG (0 cycles)** | **312 KB** |
| **Self-Attention in Transformers**| ~44 min (48 chunks) | **82** | **65** | **14** | **Strict DAG (0 cycles)** | **208 KB** |
| **Total Across Corpus** | **216 chunks** | **327 entities** | **262 relations** | **58 prerequisites** | **100% Acyclic** | **957 KB total** |

### 2.7 Scale & content coverage

| Metric | Value |
|---|---|
| Videos ingested | 725 knowledge packages on disk (Kaggle-produced) |
| Avg chunks / lecture | Long lectures: **48–98 multimodal chunks / 6–13 segments** |
| Transcript coverage | 100% of chunks carry transcript |
| Visual + OCR coverage | Fused at slide keyframes; 10 of 45 smoke-benchmark questions target slide content |
| Graph density | 327 entities / 262 relations across 3 recent full lectures; 0.0% dangling edges |

---

## 3. Comparison matrix — LectureMIND vs baselines

| Capability | **LectureMIND** | Naive vector RAG | Generic video summarizer | Closed-box LLM (ChatGPT etc.) |
|---|---|---|---|---|
| Multimodal capture (audio + slide + OCR) | Yes — audio 100% + keyframe slides/OCR | No — text-only | Partial — transcript only | No — cannot ingest |
| Timestamped, chunk-level grounding | Yes — every chunk carries timestamp | No | No | No |
| Knowledge-graph reasoning (entities/relations/traversal) | Yes — 6 relation types; CS162 benchmark package 138 entities / 189 relations; 1–3-hop traversal in both edge directions (rendered as `[Graph]` lines, not reranked) | No | No | No |
| Hybrid retrieval (dense + BM25/RRF + graph + cross-encoder rerank) | Yes — BM25 +0.07 Hit@5 over dense; cross-encoder +0.023 Hit@5 (ablation, §2.4) | Partial — top-k only, no rerank | No | No |
| Lecture-wide synthesis (summary/notes/quiz) | Yes — 8-bucket temporal sampling; summary context spans 87% of the lecture on average | No — single-pass top-k | Partial — chapter list only | No |
| Hallucination control (evidence-gated, source-only) | Yes — live-verified refusal | No — invents freely | No | No |
| Cross-lecture data isolation | Yes — hard guards, tested | Partial | n/a | n/a |
| Domain-adaptive reranker training | Yes — dedicated training pipeline and automated triplet dataset generator | No | No | No |
| Private / offline (no API dependence) | Yes — local Ollama + Neo4j + Qdrant | Yes | Yes | No — requires API |
| Benchmark infra (QA harness, KG audit suite, no-LLM code comparison; scaffolded RAGAS/load-test) | Yes — live 45-question smoke run (post-rerank Hit@5 0.907, MRR@5 0.801, context recall 0.843, routing 0.933; 95% CIs) + component ablation | No | No | No |
| Measured eval numbers to show | Yes — retrieval/rerank + live QA + latency + graph/visual types | Partial | No | No |
| Source-video footprint | 83-min 720p lecture → **436 KB knowledge package** (~2,800× smaller); corpus ≈ 85 MB for 725 lectures | Stores raw video | Stores raw video | n/a |

**The one-line pitch this matrix supports:** *LectureMIND is a full-stack, private, multimodal lecture-QA system — hybrid dense + BM25 retrieval with cross-encoder reranking, a knowledge-graph layer, timestamped grounding and lecture-wide synthesis — with a reproducible evaluation harness (post-rerank Hit@5 0.907, 95% CI 0.81–0.98, on a 45-question smoke set with AI-reviewed labels). Human-reviewed labels, judge-based faithfulness and a GraphRAG ablation are still open.*

---

## 4. Evaluation Suite & Verification

The evaluation suite executes the real pipeline with verified datasets:
- **Comprehensive Benchmark (`evaluation/benchmark_runner.py`):** Real single-pass execution across Qdrant, Neo4j, CrossEncoder reranker, and LLM backends with full metric breakdowns.
- **RAGAS Integration (`evaluation/ragas/eval_ragas.py`):** Structured evaluation harness for Faithfulness, Answer Relevancy, and Context Precision. Runnable via `python -m evaluation.ragas.eval_ragas --report <benchmark_report.json>`; requires the optional `ragas` package and an LLM judge — exits with a clear message if they are missing (no RAGAS results have been produced yet).
- **Load Testing (`evaluation/load_testing/load_test.py`):** High-concurrency throughput and latency profiling (100 / 500 / 1000 concurrent virtual users). Runnable via `python -m evaluation.load_testing.load_test --url <server> --users 100`; the only recorded run is a small smoke test (1/2/4 concurrent users; 2/4 errors at 4 users) in `evaluation/reports/load_test_report.csv`.
- **Interactive Visual Dashboard (`evaluation/dashboard/`):** Standalone dashboard rendering performance analytics and score breakdowns.

> **Provenance note:** every number in this file traces to either (a) a stored benchmark report generated by the code in this repo (`evaluation/outputs/`), (b) a regenerated audit artifact (`outputs/kg_quality*/`), or (c) a live execution recorded at the time of measurement. The MIT/Self-Attention audits were regenerated from the latest packages in `0-output/` and confirm the published extraction counts (92/114 relations, 11/3 prerequisites).

---
*Maintained by the evaluation stack in `evaluation/` — regenerate reports with `python -m evaluation.benchmark_runner`, `python -m evaluation.ragas.eval_ragas`, `python -m evaluation.load_testing.load_test`.*
