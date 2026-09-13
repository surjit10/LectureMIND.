# LectureMIND — Metrics Matrix

> **Purpose:** One document that quantifies what LectureMIND is and why it is stronger than generic RAG / video-summarizer baselines.
> **Audience:** Engineers, evaluators, and reviewers who need the measured numbers behind the claims.
> **Last updated:** 2026-08-15 · every number below traces to a file or a live test in this repo.
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
| Architecture layers | Multimodal ingestion → segmentation → entity/relation extraction → prerequisite DAG inference → hybrid GraphRAG (vector + BM25 + graph) → cross-encoder rerank → evidence-gated generation | `cloud/`, `retrieval/`, `agent/langgraph/` |
| Modalities captured | Audio (100% chunk coverage) + slide visuals + OCR at slide keyframes | `multimodal_fusion.py`, live scan |
| Serving model | Local Ollama `qwen2.5:3b` (offline, fully private) **or** cloud APIs — OpenAI, Gemini, Groq, OpenRouter, Anthropic — hot-swappable via `ProviderRegistry` | `local/llm/provider_registry.py` |
| Embedding | `bge-large-en-v1.5` (1024-dim) | startup log |
| Reranker | Global cross-encoder `BAAI/bge-reranker-base` (XLM-R), CPU, process singleton; dynamic int8 quantization (`RERANKER_QUANTIZE`) with automatic FP32 fallback | `rerank_service.py`, `reranker_loader.py`, `app.py` |
| Hybrid retrieval | Dense (`bge-large-en-v1.5`) + BM25 lexical candidates fused via Reciprocal Rank Fusion (RRF) before cross-encoder reranking; enabled by default (`ENABLE_HYBRID_RETRIEVAL`) | `retrieval/hybrid/bm25_retriever.py`, `config.py` |
| Knowledge graph | Production run: CS162 (138 entities / 189 relations), MIT (78 entities / 92 relations), Self-Attention (60 entities / 114 relations); **395 total relations**, 0.0% dangling rate | Kaggle package audit (`audit_package.py`) |
| Prerequisite DAG | Strict DAG enforced via deterministic DFS cycle resolution; 27 prerequisites (CS162), 11 (MIT), 3 (Self-Attention); benchmark F1: **73.7%** (77.8% precision / 70.0% recall, 0 cycles) | `evaluation/knowledge_graph/` |
| Content scale | Ingested long-lecture packages (46–93 chunks / 40–85 min); package size: **202–428 KB** (replacing 1+ GB video) | `0-output/` |
| Automated tests | **569 passing** (planner, retrieval, hybrid retrieval, reranker, prerequisite inference, KG auditor, loader, isolation, pipeline, extraction) | `pytest` |

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

### 2.2 Grounding, citations & hallucination resistance (live-verified)

| Metric | Value | How verified |
|---|---|---|
| Out-of-scope refusal (no hallucination) | Refuses with "Insufficient evidence…" instead of inventing | Live trap question (capital of France + fabricated topic) |
| Source-grounded answers | 100% of answers carry `chunk_id` + `timestamp` sources built only from retrieved chunks | `_build_sources`, live query |
| Lecture-wide synthesis | Summary answer grounded on **56 timestamped sources** across the timeline | Live summary query |
| Visual grounding | Answered from slide content when planner routes `diagram/slide` intent | Live visual query |
| Graph grounding | Traversal answers (`Relational Model → Relational Algebra → Database Systems → SQL`) | Live lowercase graph query |
| Language consistency | Answers pinned to question language (English-only verified; 0 non-English chars) | Live test |
| Cross-lecture data leakage | **0 foreign chunks** in leak test; hard guards raise without `lecture_id`; Neo4j scoped by `(entity_id, lecture_id)` | Live test + Cypher inspection |

### 2.3 Latency & throughput (live, real LLM)

Measured on the live local stack with **cloud inference active (Groq `openai/gpt-oss-120b`)**. Reranker runs on local CPU; generation runs on the cloud API. Source: `evaluation/outputs/evaluation_report_20260811_105621.*` (curated 50-question run, `top_k = 15`).

| Stage | Mean | p95 |
|---|---|---|
| Planner | 0.0004 s | 0.0008 s |
| Retrieval (dense + BM25, live Qdrant) | 0.13 s | 0.17 s |
| Reranker (local CPU, k=15) | 3.82 s | 5.33 s |
| Generation (cloud LLM, rate-limiter paced) | 16.56 s | 18.97 s |
| **Total end-to-end** | **20.51 s** | **22.82 s** |

> Generation latency includes per-provider rate-limiter pacing (`local/llm/rate_limiter.py`) under the provider's token budget — the limiter keeps full benchmark runs on a single backend with zero mid-run quota failures. The reranker is the deliberate precision stage: it scales linearly with candidate count, so `top_k` is the primary latency/recall lever. Load-test harness (`evaluation/load_testing/load_test.py`) is ready for a 100/500/1000-user run.

### 2.4 End-to-end QA accuracy (curated 50-question benchmark)

Source: `evaluation/outputs/evaluation_report_20260811_105621.*` — **50/50 questions, 0 errors, single backend (Groq `openai/gpt-oss-120b`)**, run through the real `QueryWorkflow` (Qdrant → Neo4j → reranker → LLM). Dataset: `evaluation/datasets/cs162_lecture1_qa_50.json` — every expected chunk ID is keyword-verified against chunk content and confirmed present in Qdrant (23 factual / 17 conceptual / 5 visual / 4 definition / 1 summary).

| Metric | Mean | p95 |
|---|---|---|
| Routing accuracy | **0.980** (49/50) | 1.000 |
| Visual routing accuracy (`need_visual`) | **1.000** | 1.000 |
| Hit@5 *(Primary Sufficiency)* | **0.980** | 1.000 |
| MRR@5 *(Primary Rank-1)* | **0.788** | 1.000 |
| Recall@5 *(Primary Coverage)* | **0.862** | 1.000 |
| NDCG@5 *(Primary Ranking Order)* | **0.767** | 1.000 |
| Precision@5 *(Secondary IR)* | 0.280 | 0.400 |
| Ranking quality (rerank MRR) | **0.918** | 1.000 |
| Answer F1 | **0.459** | 0.714 |
| Keyword recall | **0.545** | 1.000 |
| Citation completeness | **1.000** | 1.000 |
| Citation coverage | **0.927** | 1.000 |
| Chunk coverage | **1.000** | 1.000 |
| Mean end-to-end latency | 20.51 s | 22.82 s |

> **Note on Precision@5 (0.280) vs. Primary Metrics**: In single-lecture QA, ground-truth evidence is localized. In `cs162_lecture1_qa_50.json`, 27 questions (54%) have only 1 relevant chunk and 12 questions (24%) have only 2. The absolute mathematical upper bound for Precision@5 across this dataset is **0.3520 (35.20%)**. A score of 0.280 represents **79.5% of the theoretical maximum achievable by any system**. For this reason, the primary retrieval evaluation metrics for LectureMIND are **Hit@5 (0.980)**, **MRR@5 (0.788)**, **Recall@5 (0.862)**, and **NDCG@5 (0.767)**.

**By question type (measured):**

| Type | n | MRR@5 | Hit@5 | Answer F1 |
|---|---|---|---|---|
| factual | 23 | 0.844 | 1.000 | 0.422 |
| conceptual | 17 | 0.805 | 0.941 | 0.406 |
| visual (`need_visual`) | 5 | 0.567 | 1.000 | 0.631 |
| definition | 4 | 0.750 | 1.000 | 0.632 |
| summary (lecture-wide) | 1 | 0.500 | 1.000 | 0.682 |

**System features behind these numbers:**

1. **Semantic chunk merging.** Whisper segments are merged at ingestion into 97 self-contained semantic chunks; answer chunks carry complete passages, which raises Answer F1 and keyword recall directly.
2. **Hybrid BM25 lexical retrieval (RRF).** Exact entity names and numeric facts that dense embeddings miss are recovered before reranking — e.g. the "Linux lines of code" evidence sits at BM25 rank 0 while absent from the dense top-15 (verified against the live index).
3. **Score-ordered context selection.** `ContextBuilder` selects evidence by rerank score and renders it chronologically, so a top-ranked chunk late in the timeline is never dropped by the context budget.
4. **OCR noise sanitization.** URLs and social handles are stripped from otherwise-educational slides instead of discarding the whole slide.
5. **Rate limiter + retry.** Per-provider token/request budgets with exponential backoff (`local/llm/rate_limiter.py`) keep runs on a single backend with 0 errors.

Citation completeness of 1.0 means every cited source was actually present in the prompt context supplied to the LLM: zero hallucinated or ungrounded citations.

### 2.5 Storage & compression

LectureMIND replaces raw video with structured knowledge. An **85-minute, 720p lecture (~1.2 GB at typical encoding)** compresses into a **~428 KB knowledge package** (~2,800× smaller):

| Artifact | Size | Notes |
|---|---|---|
| Source video (85 min, 720p) | ≈ 1.2 GB | Not stored or shipped |
| CS162 Knowledge package (`.zip`) | **428 KB** | 93 chunks, 138 entities, 189 relations, 27 prerequisites (~2,800× smaller than video) |
| MIT 6.S191 Knowledge package (`.zip`) | **308 KB** | 69 chunks, 78 entities, 92 relations, 11 prerequisites |
| Self-Attention Knowledge package (`.zip`)| **202 KB** | 46 chunks, 60 entities, 114 relations, 3 prerequisites |
| Per-query LLM context | ≈ 3.5 KB | the only text the model reads per question |
| Code snapshot (`dist/lecturemind-code-kaggle.zip`) | **131.89 KB** | 64 files, 335 KB of source → 2.5× zip ratio |
| Whole corpus (725 packages) | ≈ 85 MB | replaces an estimated 100+ GB of source video |

This is the storage story the architecture is built around: the expensive, bulky artifact (video) is processed **once** in the cloud, and everything a student needs — searchable chunks, embeddings, graph, captions, prerequisites — ships as a small portable ZIP that runs fully offline on a laptop.

### 2.6 Knowledge Graph Quality & Prerequisite DAG Metrics (Live Audited)

Source: `evaluation/knowledge_graph/audit_package.py` — audited across the benchmark package and newly ingested long-lecture packages:

| Metric | Measured Value | Benchmark Baseline | Delta / Health Status |
|---|---|---|---|
| **Prerequisite Strict Precision** | **77.8%** (7/9) | 53.8% (7/13) | **+24.0% absolute gain** |
| **Prerequisite Strict Recall** | **70.0%** (7/10) | 70.0% (7/10) | **100% preserved (zero regression)** |
| **Prerequisite Strict F1** | **73.7%** | 60.9% | **+12.8% absolute gain** |
| **Strict DAG Guarantee** | **True** | True | 100% acyclic across all packages |
| **Cycle Count** | **0** | 0 | Mutual cycles deterministically pruned |
| **Self-Loop Count** | **0** | 0 | Zero self-dependencies ($A \to A$) |
| **Pedagogical Relevance** | **100%** | 93.3% | Zero physical parts/losses mislabeled as prerequisites |
| **Dangling Relation Rate** | **0.0%** (0/395) | 0.0% | 100% referential integrity to `entities.json` |
| **Relation Inverse Consistency** | **100%** | 100% | Symmetric and reverse mappings verified |

#### Multi-Lecture Extraction Yield (Full Cloud Execution)

Comparison between legacy run (truncated by token caps) and the current sliding-window + compact alias pipeline:

| Lecture Package | Duration | Entities (Old $\to$ New) | Relations (Old $\to$ New) | Inferred Prerequisites | Extraction Gain |
|---|---|---|---|---|---|
| **CS162 Operating Systems** | ~85 min (93 chunks) | $128 \to \mathbf{138}$ | $20 \to \mathbf{189}$ | **27** | **+845% (+169 relations)** |
| **MIT 6.S191 Deep Learning** | ~60 min (69 chunks) | $75 \to \mathbf{78}$ | $20 \to \mathbf{92}$ | **11** | **+360% (+72 relations)** |
| **Self-Attention in Transformers**| ~40 min (46 chunks) | $43 \to \mathbf{60}$ | $10 \to \mathbf{114}$ | **3** | **+1040% (+104 relations)** |
| **Total Across Corpus** | **208 chunks** | **276 entities** | **395 relations** | **41 prerequisites** | **+690% (+345 relations)** |

### 2.7 Scale & content coverage

| Metric | Value |
|---|---|
| Videos ingested | 725 knowledge packages on disk (Kaggle-produced) |
| Avg chunks / lecture | Long lectures: **46–93 multimodal chunks / 4–14 segments** |
| Transcript coverage | 100% of chunks carry transcript |
| Visual + OCR coverage | Fused at slide keyframes; 5 of 50 benchmark questions target slide content |
| Graph density | 276 entities / 395 relations across 3 recent full lectures; 0.0% dangling edges |

---

## 3. Comparison matrix — LectureMIND vs baselines

| Capability | **LectureMIND** | Naive vector RAG | Generic video summarizer | Closed-box LLM (ChatGPT etc.) |
|---|---|---|---|---|
| Multimodal capture (audio + slide + OCR) | Yes — audio 100% + keyframe slides/OCR | No — text-only | Partial — transcript only | No — cannot ingest |
| Timestamped, chunk-level grounding | Yes — every chunk carries timestamp | No | No | No |
| Knowledge-graph reasoning (entities/relations/traversal) | Yes — 128 entities / 20 relations (demo), 930 / 127 collection, 6 relation types | No | No | No |
| Hybrid retrieval (dense + BM25/RRF + graph + cross-encoder rerank) | Yes — 0.997 MRR on lecture triplets; 50-set MRR@5 0.788 | Partial — top-k only, no rerank | No | No |
| Lecture-wide synthesis (summary/notes/quiz) | Yes — 8-bucket temporal sampling, 56-source summary | No — single-pass top-k | Partial — chapter list only | No |
| Hallucination control (evidence-gated, source-only) | Yes — live-verified refusal | No — invents freely | No | No |
| Cross-lecture data isolation | Yes — hard guards, tested | Partial | n/a | n/a |
| Domain-adaptive reranker training | Yes — dedicated training pipeline and automated triplet dataset generator | No | No | No |
| Private / offline (no API dependence) | Yes — local Ollama + Neo4j + Qdrant | Yes | Yes | No — requires API |
| Benchmark infra (RAGAS, QA harness, load test) | Yes — harness + live run: **50-QA curated (MRR@5 0.788, Hit@5 0.980, routing 0.980)** | No | No | No |
| Measured eval numbers to show | Yes — retrieval/rerank + live QA + latency + graph/visual types | Partial | No | No |
| Source-video footprint | 83-min 720p lecture → **436 KB knowledge package** (~2,800× smaller); corpus ≈ 85 MB for 725 lectures | Stores raw video | Stores raw video | n/a |

**The one-line pitch this matrix supports:** *LectureMIND is a full-stack, private, multimodal GraphRAG system — with timestamped grounding, graph reasoning, lecture-wide synthesis, and hallucination controls — that has an evaluation harness and measured ranking metrics (0.997 MRR) out of the box.*

---

## 4. Evaluation Suite & Verification

The evaluation suite executes the real pipeline with verified datasets:
- **Comprehensive Benchmark (`evaluation/benchmark_runner.py`):** Real single-pass execution across Qdrant, Neo4j, CrossEncoder reranker, and LLM backends with full metric breakdowns.
- **RAGAS Integration (`evaluation/ragas/eval_ragas.py`):** Structured evaluation harness for Faithfulness, Answer Relevancy, and Context Precision.
- **Load Testing (`evaluation/load_testing/load_test.py`):** High-concurrency throughput and latency profiling (100 / 500 / 1000 concurrent virtual users).
- **Interactive Visual Dashboard (`evaluation/dashboard/`):** Standalone dashboard rendering performance analytics and score breakdowns.

---
*Maintained by the evaluation stack in `evaluation/` — regenerate reports with `evaluation/benchmark_runner.py`, `evaluation/ragas/eval_ragas.py`, `evaluation/load_testing/load_test.py`.*
