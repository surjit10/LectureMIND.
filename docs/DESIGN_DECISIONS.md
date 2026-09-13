# DESIGN_DECISIONS.md

> **Purpose:** Every significant architectural choice — the decision, the rationale, and the production trade-offs.
> **Audience:** Engineers and evaluators reviewing system design.
> **Last updated:** 2026-08-15 · consistent with the current production codebase.
> **Companion docs:** [ARCHITECTURE.md](./ARCHITECTURE.md) (what) · [PIPELINE.md](./PIPELINE.md) (how) · [RETRIEVAL_SYSTEM.md](./RETRIEVAL_SYSTEM.md) (retrieval internals)

---

## Table of Contents

- [1. Cloud / Local Split Architecture](#1-cloud--local-split-architecture)
- [2. GraphRAG over Vanilla RAG](#2-graphrag-over-vanilla-rag)
- [3. Two-Stage Retrieval](#3-two-stage-retrieval-bi-encoder--cross-encoder)
- [4. LangGraph-Style Orchestration](#4-langgraph-style-orchestration-custom-queryworkflow)
- [5. Strategy + Factory for LLM Backends](#5-strategy--factory-pattern-for-llm-backends)
- [6. Lazy LLM vs Eager Reranker Loading](#6-lazy-llm-initialization-vs-eager-reranker-loading)
- [7. Decoupled Startup](#7-decoupled-startup-zero-llm-initialization-at-boot)
- [8. pydantic-settings Isolation](#8-pydantic-settings-environment-isolation)
- [9. GraphML for Graph Portability (Superseded)](#9-graphml-for-knowledge-graph-portability)
- [10. SSE Streaming (reversed)](#10-sse-streaming-for-query-responses)
- [11. Knowledge Package Interface](#11-knowledge-package-as-the-cloudlocal-interface)
- [12. Evaluation HTTP Bypass](#12-evaluation-framework-http-bypass)
- [13. Operational Hardening & Production Reliability](#13-operational-hardening--production-reliability)
- [14. Potential Future Improvements](#14-potential-future-improvements)
- [15. Architectural Strengths](#15-architectural-strengths)
- [16. Multi-Signal Prerequisite DAG Inference](#16-multi-signal-prerequisite-dag-inference)
- [17. Compact Entity Alias Remapping for Relation Extraction](#17-compact-entity-alias-remapping-for-relation-extraction)

---

## 1. Cloud / Local Split Architecture

### Decision
Process all computationally heavy tasks (transcription, OCR, embedding, graph extraction) once on GPU infrastructure. Deliver results to students as a portable **Knowledge Package** that a lightweight local server consumes.

### Rationale
- Students should not need GPU hardware. A 5-year-old laptop should suffice for the learning interface.
- Heavy models (Faster-Whisper, Qwen2-VL, PaddleOCR, bge-large-en-v1.5) require gigabytes of VRAM and minutes of processing per lecture. Running these per student per lecture is neither economical nor practical.
- Kaggle provides free GPU compute hours, making cloud processing accessible without infrastructure cost.

### Trade-offs
| Accepted | Rejected |
|---|---|
| Students must import a pre-processed package | Students cannot upload raw video directly from the local UI |
| Processing happens offline and ahead of time | Real-time processing is not supported |
| Package files can be large (50–500 MB for embeddings) | Instant availability of new lectures |

### Scalability Considerations
- The split naturally scales: one cloud run processes one lecture, and the resulting package can be imported by unlimited students without additional compute.
- The local server is stateless regarding lecture processing — it only manages database connections and LLM calls.

---

## 2. GraphRAG over Vanilla RAG

### Decision
Combine semantic vector search (Qdrant) with knowledge graph traversal (Neo4j) rather than using vector similarity alone.

### Rationale
Standard RAG (vector-only) fails on relational queries. A student asking "How does the light reaction relate to the Calvin cycle and where does their product go?" needs a system that understands the *relationship graph*, not just semantic proximity. Vector search might retrieve relevant individual chunks but cannot reason about the path between concepts.

GraphRAG adds:
- **Structured reasoning**: Neo4j graph edges explicitly encode `PRODUCES`, `REQUIRES`, `ENABLES` relationships.
- **Multi-hop traversal**: The system can follow a chain (`A → B → C`) without needing the full chain to appear in a single text chunk.
- **Complementarity**: Vector search excels at definitions and specific passages; graph search excels at relational questions. Together they cover the full query space.

### Trade-offs
| Accepted | Rejected |
|---|---|
| Increased cloud processing complexity (entity and relation extraction) | Simpler, faster cloud pipeline |
| Two databases to maintain locally (Qdrant + Neo4j) | Single database simplicity |
| Higher package size (Graph JSON adds ~10–100 KB) | Smaller packages |
| Graph quality depends on extraction model accuracy | Graph-free approach |

### Scalability Considerations
- Neo4j handles millions of nodes and edges efficiently with index-backed Cypher queries.
- Graph quality is bounded by the entity/relation extractor's accuracy. Errors in extraction propagate permanently into the package.

---

## 3. Two-Stage Retrieval: Bi-Encoder + Cross-Encoder

### Decision
Use a fast **bi-encoder** (bge-large-en-v1.5) for broad candidate retrieval, followed by a slow but accurate **cross-encoder** (bge-reranker-base) for precision reranking.

### Rationale
- A pure bi-encoder applied to millions of chunks is fast but imprecise — it encodes query and document independently, missing fine-grained relevance signals.
- A pure cross-encoder applied to all chunks is accurate but O(N) in compute — infeasible at scale.
- The two-stage design gets the best of both: retrieve a candidate pool (top-15, `NORMAL_TOP_K`) cheaply with a bi-encoder, then score it precisely with a cross-encoder.
- This is the standard industry pattern (as used in production systems at Microsoft, Google, and most modern RAG deployments).

### Reranker Design Details
- The reranker is loaded **eagerly at startup** as a global singleton (`rerank_service`), preventing per-query model loading overhead (~560 MB model).
- The reranker is **lecture-agnostic** — it can score any (query, chunk) pair without needing to know which lecture is active.
- Score threshold of `0.3` filters out low-quality candidates before they reach the LLM context window.

### Trade-offs
| Accepted | Rejected |
|---|---|
| Reranker adds ~50–200ms latency per query (cross-encoder inference) | Pure bi-encoder speed (no reranker latency) |
| 560 MB model always in RAM | Zero reranker memory footprint |
| Higher precision for LLM context | Noisier context (more irrelevant chunks reach the LLM) |

---

## 4. LangGraph-Style Orchestration (Custom QueryWorkflow)

### Decision
Structure the query pipeline as a **typed, node-based orchestrator** rather than a monolithic function. `agent/langgraph/workflow.py` exposes a `QueryWorkflow` class whose node functions (`planner_node`, `vector_retriever_node`, `graph_retriever_node`, `reranker_node`, `answer_generator_node`) operate on a shared `QueryState` TypedDict — but the orchestrator is a **plain Python single-pass class**, with no `langgraph` dependency installed.

### Rationale
- A **typed, inspectable state machine**: each node reads from and writes to a shared `QueryState` TypedDict, making data flow explicit and debuggable.
- Nodes are independently testable — the `reranker_node` can be unit-tested without running the full pipeline.
- The node functions are **drop-in LangGraph-compatible**: if the `langgraph` dependency is ever added, the same nodes can be wired into a `StateGraph` without rewriting logic.
- The evaluation framework invokes the same `QueryWorkflow` object directly (bypassing HTTP) and inspects intermediate state fields — impossible with a monolithic function.

### Trade-offs
| Accepted | Rejected |
|---|---|
| A thin hand-rolled orchestrator (no extra dependency) | Monolithic function chain |
| Explicit per-node state updates | Framework-provided state accumulation |
| Standard Python debugging | LangGraph-specific debugging tools |

---

## 5. Strategy + Factory Pattern for LLM Backends

### Decision
Abstract all LLM interactions behind `LLMBackend` (ABC), with `OllamaBackend` and `OnlineBackend` as concrete implementations, selected by `ProviderRegistry` (factory).

### Rationale
- LangGraph nodes (`answer_generator_node`, `learning_service`) depend on `LLMBackend` — they are completely unaware of whether generation happens locally or in the cloud.
- Switching providers (Ollama → Gemini → Claude) requires only updating `data/llm_config.json`. No code change is needed anywhere in the pipeline.
- `ProviderRegistry` produces **ephemeral** (non-cached) instances on each call. This ensures that provider switches take effect on the very next request without any state invalidation.

### Trade-offs
| Accepted | Rejected |
|---|---|
| New backend creation overhead per query | Cached backend with stale provider config risk |
| `OnlineBackend` must handle multiple provider SDKs | One class per provider (more files, less unified) |
| `ProviderRegistry` reads JSON file on every backend request | In-memory config cache (would require cache invalidation) |

### Scalability Considerations
- JSON file reads are cheap (microseconds). The ephemeral instantiation model scales well for single-user local deployment.
- For multi-user deployment, a proper config cache with invalidation events would be needed.

---

## 6. Lazy LLM Initialization vs. Eager Reranker Loading

### Decision
- **LLM backends**: Instantiated lazily (on demand, per query).
- **Reranker model**: Loaded eagerly (at startup, once).

### Rationale
**Reranker is eager** because:
- The 560 MB cross-encoder model takes several seconds to load from disk.
- It is used on every single query without exception.
- It is lecture-agnostic — no need to reload it per lecture switch.
- Loading it at startup ensures zero latency on the first query.

**LLM backends are lazy** because:
- The user may configure a provider but not immediately query.
- Provider configuration can change at runtime — eager initialization would capture stale config.
- Cloud providers (OpenAI, Gemini) have no local "model" to load — the client is a thin SDK wrapper that initializes in microseconds.
- Ollama's model is managed by the daemon process — the `OllamaBackend` only needs a client handle, not the model weights.

---

## 7. Decoupled Startup (Zero LLM Initialization at Boot)

### Decision
The FastAPI server startup sequence validates only infrastructure (databases, reranker) and performs **zero LLM provider validation** at boot time.

### Rationale
**Before this decision:** Startup unconditionally checked Ollama for `qwen2.5:3b` even when the user intended to use cloud APIs. This caused startup failures for users with no local Ollama installation.

**After:** Provider status is deferred entirely to user-initiated `/settings/status` calls. The server boots in all environments:
- No API keys configured → boots cleanly.
- No Ollama installed → boots cleanly.
- Both configured → boots cleanly.
- Neither configured → boots cleanly, reports `NOT_CONFIGURED` when queried.

### Trade-offs
| Accepted | Rejected |
|---|---|
| Provider errors surface only at query time, not at boot | Immediate boot-time failure for misconfigured providers |
| Users must explicitly check provider status | Automatic status reporting at startup |
| `ENABLE_AUTO_OLLAMA_PULL` env var is now effectively dead code | Removing the dead code (backward compat preserved) |

---

## 8. `pydantic-settings` Environment Isolation

### Decision
Three distinct settings classes (`SharedSettings`, `CloudSettings`, `LocalSettings`) with `extra="ignore"` on each.

### Rationale
- The `.env` file contains variables for both cloud and local environments. Without `extra="ignore"`, Pydantic rejects variables that belong to a different settings class, causing startup failures.
- Strict class separation prevents cloud-only heavy imports (PaddleOCR, CUDA libraries) from being loaded by the local server.
- `SharedSettings.EMBEDDING_DIMENSION` with a `field_validator` enforces dimension consistency across the entire system — a mismatched dimension (e.g., using `bge-base` at 768 dims) would produce silently wrong Qdrant search results without this guard.

---

## 9. GraphML for Knowledge Graph Portability

> **[SUPERSEDED]** — this decision was evaluated and reversed. The current implementation (`cloud/packaging/exporter.py`, `serving/` importer) serializes the knowledge graph as **`entities.json` + `relations.json`** (plus `embeddings.npy` for entity vectors) inside the knowledge package, and loads them into Neo4j at import time. GraphML is not used anywhere in the pipeline.

### Decision (original)
Serialize the knowledge graph as GraphML (XML-based) rather than a Neo4j-specific format.

### Rationale (original)
- GraphML is database-agnostic and parseable by any graph library (NetworkX, JanusGraph, etc.).
- Future migrations away from Neo4j would not require reformatting existing packages.
- NetworkX's `nx.read_graphml()` is the standard Python import path, with no proprietary dependency.

### Trade-offs (original)
- GraphML is verbose XML — smaller packages would result from binary formats (e.g., GraphSON, Apache TinkerPop).
- GraphML does not natively support all Neo4j property types — some metadata may be coerced to strings.

### Why it was reversed
Plain JSON keeps the package human-inspectable, trivially diffable, and directly loadable into Neo4j's import tooling; the portability argument did not justify the extra serialization layer.

---

## 10. SSE Streaming for Query Responses

The current API returns **plain JSON responses** (see `serving/fastapi/routes/`); there is no SSE streaming endpoint. Latency is handled by routing generation to fast cloud LLM backends (Groq measured at ~1.3–2.5 s per answer).

### Decision (original)
Stream LLM-generated tokens to the frontend via Server-Sent Events (SSE) rather than returning a complete response after generation finishes.

### Rationale (original)
- LLM generation for a 500-word answer takes 5–15 seconds on a local model.
- Without streaming, the user sees a blank screen for the entire duration.
- SSE enables the first token to appear in under 1 second regardless of total answer length.
- SSE is a standard browser-supported protocol (unlike WebSockets, it requires no special handshake and works over standard HTTP).

### Why it was reversed
With cloud-backed generation the full round-trip is already interactive; streaming added SSE plumbing, frontend event handling, and partial-answer caching complexity without a meaningful UX gain at current latencies.

---

## 11. Knowledge Package as the Cloud/Local Interface

### Decision
Use a ZIP archive of structured files (embeddings, graph, chunks, metadata) as the sole transfer mechanism between the cloud pipeline and local server.

### Rationale
- No network dependency during local usage — packages work offline after import.
- Packages are versionable and shareable — a professor can email a package to students.
- The format is inspectable — all files are standard formats (NumPy, JSON, ZIP).
- Import is a one-time operation; subsequent queries never access the package directory.

### Trade-offs
| Accepted | Rejected |
|---|---|
| Large package sizes (50–500 MB) | Streaming knowledge transfer (complex) |
| Manual import step required | Automatic cloud sync |
| No real-time updates to knowledge base | Live lecture ingestion |

---

## 12. Evaluation Framework HTTP Bypass

### Decision
The `BenchmarkRunner` instantiates `QueryWorkflow` directly as a Python object rather than sending HTTP requests to the running server.

### Rationale
- HTTP adds serialization overhead and hides intermediate `QueryState` fields.
- The evaluator needs access to `vector_results`, `reranked_results`, `telemetry` — none of which are exposed via the streaming API.
- Direct instantiation makes evaluation deterministic (no HTTP timeouts, no connection management).
- The evaluation framework can run without a running server instance.

---

## 13. Operational Hardening & Production Reliability

| Area | Implementation & Strategy | Architecture Impact |
|---|---|---|
| **Rate Limiter Pacing** | Token and request budget enforcement with jittered exponential backoff (`rate_limiter.py`) | Prevents mid-stream quota exhaustion on cloud LLM APIs; ensures 100% benchmark completion rates |
| **Multi-Modal Hybrid Fusion** | Reciprocal Rank Fusion (RRF) between dense BGE embeddings and BM25 lexical retriever | Ensures exact term/numerical recall while retaining deep semantic matching |
| **Dynamic Quantization** | Dynamic `int8` quantization for CrossEncoder reranker with automated FP32 fallback | Reduces memory consumption and accelerates CPU inference on student hardware |
| **Lecture Isolation Guarantees** | Strict `(entity_id, lecture_id)` scoping in Neo4j and Qdrant queries | Guarantees zero cross-contamination between courses and lectures |
| **Evidence Gating** | Strict verification against retrieved chunks before generating answers | Prevents hallucinations; ensures 100% citation completeness |
| **Prerequisite DAG Guarantees** | Deterministic DFS cycle resolution (`cloud/extraction/prerequisite_extractor.py`) | Enforces strictly acyclic prerequisite graphs (0 cycles, 0 self-loops) for valid curriculum sequencing |
| **Referential Integrity** | Pre-export validation against extracted entity indices (`cloud/packaging/validator.py`) | Enforces 0.0% dangling relations across all processed lectures (395/395 verified relations) |

---

## 14. Potential Future Improvements

| Improvement | Addresses |
|---|---|
| Offload Ollama download to a background thread/task | Event loop blocking (HIGH risk) |
| Add `asyncio.Lock` or file-level locking to `ProviderRegistry` | Concurrent JSON write corruption |
| Replace Jaccard `answer_similarity` with a semantic embedding similarity | Metric accuracy |
| Extend `expected_keywords` coverage in the QA dataset | Broader `keyword_recall` coverage (the field is populated and measured — 0.545 on the QA set) |
| Add API token authentication (e.g., bearer token in `.env`) | Security for network-exposed deployments |
| Implement conversation history via multi-turn `QueryState` accumulation | Multi-turn chat context |
| Add `DOWNLOADING` state to `ProviderManager` for Ollama download progress | User experience during model download |
| Implement binary embedding storage (e.g., quantized int8) to reduce package size | Package portability |
| Add Kafka-based streaming ingestion for live lecture processing | Real-time lecture availability |
| Add a web-based package upload UI | Ease of import |
| Implement async evaluation runner for large benchmark datasets | Evaluation speed |

---

## 15. Architectural Strengths

| Strength | Description |
|---|---|
| **Clean separation of concerns** | Cloud and local runtimes are completely decoupled via the Knowledge Package interface |
| **Dependency inversion throughout** | LangGraph nodes, `LearningService`, and evaluation all depend on the `LLMBackend` abstraction — never on concrete implementations |
| **Zero-restart provider switching** | LLM provider can switch from local to cloud mid-session via a single API call |
| **Offline-first capability** | The entire local server runs without any cloud connectivity once packages are imported |
| **Inspectable pipeline state** | `QueryState` makes every intermediate result available — critical for evaluation and debugging |
| **Portable knowledge format** | Knowledge Packages use only open, standard formats (NumPy, JSON, ZIP) — **202–428 KB** replacing 1+ GB video |
| **Dual retrieval modes** | GraphRAG handles both semantic and relational queries — covering the full student query space |
| **Strict Graph Topology** | Prerequisite graphs are strictly acyclic DAGs with verified 77.8% precision / 70.0% recall |
| **Modular evaluation framework** | Independent benchmark and audit suites (`benchmark_runner.py` and `audit_package.py`) |

---

## 16. Multi-Signal Prerequisite DAG Inference

### Decision
Extract prerequisite dependencies between entities using a composite scoring function combining temporal precedence ($W=0.30$), lexical co-occurrence with negative lookbehinds ($W=0.30$), segment containment ($W=0.25$), and pedagogical inversion, followed by deterministic DFS cycle breaking.

### Rationale
- Pure LLM prompting for prerequisites is notoriously prone to hallucinated cycles ($A \to B \to A$), temporal reversal, and inclusion of superficial components (e.g., slide titles).
- Pure chronological order fails when a foundational concept is briefly referenced late in a lecture.
- Multi-signal scoring grounds the dependency in transcript evidence while enforcing that prerequisite candidates must temporally precede or co-occur in introductory contexts.
- Deterministic DFS cycle pruning guarantees mathematical acyclicity (strict DAG = True), which is required for topological sorting and pedagogical sequencing in the Socratic Back-Tracker.

---

## 17. Compact Entity Alias Remapping for Relation Extraction

### Decision
Remap extracted entities into short token aliases (`E1, E2, ... En`) within sliding chunk windows before passing to the relation extraction prompt, then resolve them back to canonical names.

### Rationale
- Full 85-minute lectures (e.g., CS162 with 138 entities and 93 chunks) exceed standard prompt token limits if all entity names and descriptions are repeatedly serialized.
- Legacy extraction without alias remapping suffered token truncation, collapsing extracted relations from 189 down to only 20.
- Compact alias remapping reduces prompt token usage by ~65%, enabling dense relation extraction (395 relations across 3 lectures) with zero dangling edges.

---

*Phase 3 documentation. Cross-references: ARCHITECTURE.md §4, PIPELINE.md §1, RETRIEVAL_SYSTEM.md §4, EVALUATION.md §7.*
