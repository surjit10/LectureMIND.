# FOLDER_STRUCTURE.md

> **Purpose:** Module-by-module map of the repository — what each folder contains and why it exists.
> **Audience:** Engineers onboarding to the codebase; anyone checking the repo layout.
> **Last updated:** 2026-08-11 · verified from source code analysis of the current tree.
> **Companion docs:** [ARCHITECTURE.md](./ARCHITECTURE.md) (design) · [TECH_STACK.md](./TECH_STACK.md) (technologies)

---

## Table of Contents

- [Repository Root](#repository-root)
- [Folder-by-Folder Explanation](#folder-by-folder-explanation)

---

## Repository Root

```
lecturemind/
├── cloud/                     # Cloud ingestion pipeline (heavy GPU processing)
├── local/                     # Local inference business logic
│   ├── llm/                   # LLM abstraction layer
│   ├── loaders/               # Knowledge Package importer, prerequisite enricher, Ollama loader
│   └── docker/
│       └── local_runtime/     # Docker volume mounts (data persistence)
│           ├── models/        # Cached model weights
│           ├── neo4j_data/    # Neo4j graph database persistent storage
│           └── qdrant_data/   # Qdrant vector database persistent storage
├── agent/                     # Query orchestration engine (LangGraph-compatible)
│   └── langgraph/
│       ├── nodes/             # Individual DAG node implementations
│       └── workflow.py        # Graph definition and QueryWorkflow class
├── retrieval/                 # Vector, graph, hybrid BM25, and course retrieval clients
├── serving/
│   └── fastapi/
│       ├── app.py             # FastAPI application, lifespan, startup
│       ├── routes/            # HTTP route handlers (query, prerequisites, lectures, settings)
│       ├── learning_service.py# Flashcard / notes / quiz generation
│       └── rerank_service.py  # Global reranker singleton wrapper
├── frontend/                  # Web UI (student-facing interface)
├── schemas/                   # Shared Pydantic models (cross-module DTOs)
├── evaluation/                # Offline benchmark & auditing framework
│   ├── benchmark_runner.py    # Main evaluation orchestrator
│   ├── dataset_loader.py      # BenchmarkSample loading + validation
│   ├── datasets/              # QA sets (cs162_lecture1_qa_50.json; sample_dataset.json)
│   ├── knowledge_graph/       # Knowledge Graph Quality & Prerequisite Audit Suite
│   ├── dashboard/             # HTML dashboard generator (renders outputs only)
│   ├── load_testing/          # 100/500/1000-user load test
│   ├── ragas/                 # RAGAS answer-quality scoring
│   ├── metrics/               # Domain-specific metric modules
│   ├── reports/               # Report generation (CSV, JSON, Markdown)
│   └── outputs/               # Generated benchmark reports
├── data/                      # Runtime data storage
│   ├── llm_config.json        # LLM provider configuration (active source of truth)
│   ├── lecture_registry.json  # Imported lecture metadata
│   ├── courses.json           # Course index (metadata only)
│   └── packages/              # Knowledge Package directories
│       └── lecture_{id}/      # Per-lecture package (embeddings + graph + prereqs + chunks)
├── config.py                  # Pydantic-settings environment configuration
├── requirements.txt           # Cloud + full dependency set
└── local_requirements.txt     # Minimal local-only dependencies
```

---

## Folder-by-Folder Explanation

---

### `cloud/`

**Purpose:** Orchestrates the full cloud-side ingestion pipeline. This is the "expensive" half of the system, designed to run on GPU-equipped infrastructure like Kaggle notebooks.

**What happens here:**
1. Accepts raw lecture video files as input (A1 metadata via FFprobe).
2. Extracts frames and audio tracks concurrently (A2/A3).
3. Dispatches to transcription (Faster-Whisper), visual captioning (Qwen2-VL, A4), and OCR (PaddleOCR, A5).
4. Fuses transcript + visual + OCR into multimodal chunks (A6), segments (A7), extracts entities/relations (A8/A9), embeds (B0), generates triplets (B1).
5. Validates (C1) and exports the final Knowledge Package ZIP (C2).

**Key concepts:**
- PaddleOCR for raw slide text; Qwen2-VL for slide descriptions; Qwen2.5-7B-Instruct for segmentation / entities / relations / triplets.
- Faster-Whisper for speech-to-text transcription.
- `BAAI/bge-large-en-v1.5` for embedding text chunks.

**Dependency on:** `schemas/`, `config.py` (`CloudSettings`).

---

### `local/`

**Purpose:** All local-runtime business logic. The counterpart to `cloud/`. This is code that runs on the student's machine.

**Sub-folders:**

#### `local/llm/`
The LLM abstraction layer. Contains:
- `base.py`: `LLMBackend` Abstract Base Class — defines `generate()`, `stream()`, `test_connection()`.
- `ollama_backend.py`: `OllamaBackend` — connects to local Ollama daemon at `localhost:11434`. Uses the `ollama` Python SDK. Initialized lazily (client acquired on first call).
- `online_backend.py`: `OnlineBackend` — unified wrapper for remote AI providers. Routes to OpenAI, Google Gemini, Anthropic (Claude), etc. based on `active_provider_id` in `llm_config.json`. Every call flows through the rate limiter and a bounded retry loop (429s honour `Retry-After`; otherwise exponential backoff with jitter).
- `rate_limiter.py`: `RateLimiter` — per-provider+model token/request budgets shared across all callers, with token estimation and retry-aware pacing.
- `provider_registry.py`: `ProviderRegistry` — reads `data/llm_config.json`. Implements `get_active_backend()` factory. Is the single source of truth for what LLM is currently active. Returns ephemeral (non-cached) `LLMBackend` instances.
- `provider_manager.py`: `ProviderManager` — operational health controller. Tests SDK connectivity. Maps errors to `ProviderStatus` enum values: `AVAILABLE`, `NOT_INSTALLED`, `MODEL_MISSING`, `MISSING_API_KEY`, `INVALID_API_KEY`, `NETWORK_ERROR`, `NOT_CONFIGURED`.

#### `local/loaders/`
Package and model management:
- `ollama_loader.py`: `check_model_ready()` and `pull_model_if_needed()` — queries the Ollama daemon for model availability and initiates downloads. Accepts `model_name` as a parameter.
- `package_validator.py`: verifies required package files (`manifest.json`, `entities.json`, `relations.json`, `embeddings.npy`, `multimodal_chunks.json`).
- `neo4j_loader.py`: loads entities, relations, and `PREREQUISITE_OF` directed edges into Neo4j with lecture-scoped property indexing.
- `qdrant_loader.py`: creates lecture-scoped collections and upserts 1024-dim dense vectors from `embeddings.npy` and chunk text payloads.
- `prerequisite_enricher.py`: infers prerequisite dependencies and writes companion `prerequisites.json` if missing.

#### `local/docker/local_runtime/`
Docker bind-mount targets for persistent data:
- `models/`: Cached model weights (e.g., reranker model files).
- `neo4j_data/`: Neo4j database persistent storage (graph data survives container restarts).
- `qdrant_data/`: Qdrant vector database persistent storage (embeddings survive container restarts).

**Key configuration link:** `config.py` → `LocalSettings` (Neo4j URI, Qdrant URL, cache directories).

---

### `agent/`

**Purpose:** The reasoning engine. Defines and executes the query processing pipeline using a single-pass orchestrator with LangGraph-compatible state structure.

**Sub-structure:**

#### `agent/langgraph/workflow.py`
- Defines `QueryWorkflow` class.
- Executes the single-pass pipeline connecting all nodes in sequence.
- Manages `QueryState` (TypedDict) that accumulates data across nodes.
- Records telemetry (per-node latency timestamps) into `state["telemetry"]`.

#### `agent/langgraph/nodes/`
Each file is a self-contained stage in the single-pass flow (structured to be drop-in
LangGraph-compatible; the orchestrator itself is a plain Python class — no `langgraph` dependency):

| File | Function | Inputs from State | Outputs to State |
|---|---|---|---|
| `vector_retriever.py` | `vector_retriever_node` | `query`, `lecture_id` | `vector_results` |
| `graph_retriever.py` | `graph_retriever_node` | `query`, `lecture_id` | `graph_results` |
| `reranker.py` | `reranker_node` | `query`, `graph_results`, `vector_results` | `reranked_results`, `final_context` |
| `answer_generator.py` | `answer_generator_node` | `query`, `final_context` | `answer`, `sources`, `graph_path` |

Routing is done by `QueryPlanner` (`agent/dspy/planner.py`), invoked directly by `workflow.run()`.

**Design note:** All nodes are completely independent. `answer_generator_node` depends on `LLMBackend` through `ProviderRegistry`, not on any concrete backend. This enables seamless switching between Ollama and cloud APIs.

---

### `retrieval/`

**Purpose:** Provides retrieval client implementations for both data stores.

**What lives here:**
- `vector_retriever/qdrant_retriever.py`: Wraps the `qdrant_client` SDK. Performs approximate nearest-neighbor search using 1024-dim query embeddings. Returns ranked `(chunk_id, score, payload)` tuples.
- `graph_retriever/neo4j_retriever.py`: Wraps the `neo4j` Python driver. Executes Cypher queries to traverse entity–relationship paths and prerequisite dependencies.
- `hybrid/bm25_retriever.py`: Lazy per-lecture BM25 index built from Qdrant payloads; fuses dense + lexical results with Reciprocal Rank Fusion (RRF) before reranking (default-on via `ENABLE_HYBRID_RETRIEVAL`).
- `reranker/rerank_service.py`: Global cross-encoder singleton (optional int8 quantization) + candidate deduplication + graph-path rendering into context.
- `context_builder.py`: Dedupe → chronological sort → adjacent merge → OCR-noise filter → budget enforcement; selects evidence by rerank score and renders it chronologically.

**Embedding model used at query time:** The same `BAAI/bge-large-en-v1.5` model used during cloud ingestion (enforced by `SharedSettings.EMBEDDING_DIMENSION = 1024`).

---

### `serving/`

**Purpose:** The FastAPI application and all HTTP-layer code.

#### `serving/fastapi/app.py`
- Defines the FastAPI `app` object.
- Implements the `lifespan` context manager (startup / shutdown hooks).
- Startup sequence: `_run_startup_audit()` → `_run_model_recovery()` → `_try_restore_active_lecture()`.
- Registers all routers.

#### `serving/fastapi/routes/`
| File | Routes | Purpose |
|---|---|---|
| `query.py` | `POST /query` | Runs the single-pass workflow; returns JSON `{answer, sources[], graph_path[], debug{}}` |
| `prerequisites.py` | `GET /lectures/{id}/prerequisites/{concept}` | Socratic Back-Tracker: reverse prerequisite dependency traversal, anchor chunks, and topological sequence |
| `lectures.py` | `POST /upload`, `GET /lectures`, `POST /lectures/{id}/load`, learning endpoints | Package import, listing, activation, active-learning content |
| `courses.py` | `POST /courses`, `POST /courses/query` | Course index + fan-out query (Feature 1) |
| `reranker.py` | `POST /api/reranker/upload`, status/reload | Global reranker upload → atomic replace → hot reload |
| `settings.py` | `GET /settings/status`, provider add/test/patch/delete, inference-mode | LLM provider management |
| `debug.py` | Debug endpoints | Internal state inspection |

#### `serving/fastapi/learning_service.py`
- Handles flashcard, note, and quiz generation.
- Uses `ProviderRegistry.get_active_backend()` — same path as the query pipeline.
- Operates over vector embeddings of the active lecture context.

#### `serving/fastapi/rerank_service.py`
- Singleton wrapper for the global reranker model (`BAAI/bge-reranker-base`).
- Loaded eagerly at startup, persisted in application state.
- Prevents reloading the cross-encoder model on every lecture switch or query.

---

### `frontend/`

**Purpose:** Student-facing web application. Communicates with the FastAPI backend via HTTP REST (JSON).

**Features exposed:**
- Chat interface (grounded answers with timestamped sources + Developer-Mode pipeline trace).
- Socratic learning interface (backward prerequisite traversal and curriculum sequencing).
- Active learning tools (flashcards, structured notes, quizzes).
- Provider settings panel (switch between Ollama / cloud APIs, manage API keys, trigger model downloads).
- Lecture management (import and activate Knowledge Packages).

---

### `schemas/`

**Purpose:** Shared Pydantic data models used across both the cloud and local sides of the system. Prevents model duplication and ensures consistent validation.

---

### `evaluation/`

**Purpose:** Offline RAG quality benchmark and knowledge graph quality audit framework.

#### `evaluation/benchmark_runner.py`
- `BenchmarkRunner(dataset_path, output_dir)` — main orchestrator.
- Iterates over `BenchmarkSample` objects from a JSON dataset.
- For each sample: invokes `QueryWorkflow.run()`, extracts `QueryState`, computes metrics.

#### `evaluation/knowledge_graph/`
- `audit_package.py`: Knowledge Graph Quality & Prerequisite Audit Suite CLI — validates schema integrity, orphan rate, referential consistency (0.0% dangling edges), strict DAG topology (0 cycles), and bipartite prerequisite matching against gold labels.
- `prerequisite_evaluator.py`: Bipartite matching algorithm and metric computer.
- `prerequisite_gold.json`: Ground-truth benchmark prerequisite dependencies.

#### `evaluation/metrics/`
| Module | Metrics Computed |
|---|---|
| `planner_metrics.py` | `routing_accuracy` |
| `retrieval_metrics.py` | `precision@5`, `recall@5`, `hit@5`, `mrr`, `ndcg_at_5` |
| `reranker_metrics.py` | `ranking_quality`, `avg_cross_encoder_score` |
| `answer_metrics.py` | `answer_similarity`, `keyword_recall`, `context_length`, `answer_length` |
| `citation_metrics.py` | `citation_coverage`, `citation_completeness`, `citation_count`, `chunk_coverage` |
| `latency_metrics.py` | `planner_latency`, `retrieval_latency`, `reranker_latency`, `generation_latency`, `total_latency` |

#### `evaluation/reports/`
- `report_generator.py`: `ReportGenerator` — aggregates per-sample metrics, writes CSV / JSON / Markdown.

---

### `data/`

**Purpose:** Runtime data directory. Not a source code folder.

| Path | Contents |
|---|---|
| `data/llm_config.json` | Active LLM provider configuration — authoritative source of truth |
| `data/packages/lecture_{id}/` | Individual Knowledge Package directories |
| `data/packages/lecture_{id}/metadata.json` | Package title, date, duration, pipeline config |
| `data/packages/lecture_{id}/manifest.json` | Validated manifest (files, checksums, statistics) |
| `data/packages/lecture_{id}/multimodal_chunks.json` | Fused transcript + visual + OCR chunks (timestamped) |
| `data/packages/lecture_{id}/segments.json` | Topic segments + `chunk_segment_map.json` |
| `data/packages/lecture_{id}/entities.json` | Extracted entities (typed) |
| `data/packages/lecture_{id}/relations.json` | Extracted typed relations (entity → relation → entity) |
| `data/packages/lecture_{id}/prerequisites.json` | Strictly acyclic prerequisite DAG (prerequisite → concept) |
| `data/packages/lecture_{id}/embeddings.npy` + `embedding_ids.json` | 1024-dim chunk embeddings + ids (loads into Qdrant) |
| `data/packages/lecture_{id}/triplets.json` | Relation triplets for offline reranker training |

---

### `config.py`

**Purpose:** Settings isolation using `pydantic-settings`. Three distinct settings classes prevent accidental cross-environment imports.

| Class | Used By | Key Settings |
|---|---|---|
| `CloudSettings` | `cloud/` | Kaggle output paths, OCR toggle, model names |
| `LocalSettings` | `serving/`, `local/` | Neo4j URI, Qdrant URL, cache directories, `OLLAMA_MODEL` fallback |
| `SharedSettings` | Both | `EMBEDDING_DIMENSION = 1024` (prevents model mismatch) |

---

### `requirements.txt`

**Purpose:** Full dependency list for the cloud pipeline and local server combined. Includes heavy GPU libraries (PaddleOCR, CUDA-accelerated packages).

### `local_requirements.txt`

**Purpose:** Minimal dependency list for running only the local inference server. Excludes cloud-only heavy dependencies (PaddleOCR, GPU tools) to enable lightweight local development.

---

*Phase 1 documentation. Further structural detail will be added in subsequent phases.*
