# TECH_STACK.md

> **Purpose:** The complete technology inventory — languages, frameworks, models, databases, and shared constants.
> **Audience:** Engineers, evaluators, and interviewers who need the exact stack with versions and roles.
> **Last updated:** 2026-08-10 · every technology is evidenced from source files, imports, and configuration artifacts.

---

## Table of Contents

- [1. Languages](#1-languages)
- [2. Frameworks](#2-frameworks)
- [3. AI / ML Libraries](#3-ai--ml-libraries)
- [4. Databases](#4-databases)
- [5. Infrastructure](#5-infrastructure)
- [6. Python Ecosystem Dependencies](#6-python-ecosystem-dependencies)
- [7. Configuration and Serialization](#7-configuration-and-serialization)
- [8. Shared Constants](#8-shared-constants)

---

## 1. Languages

| Language | Version | Usage |
|---|---|---|
| **Python** | 3.x (inferred) | Entire backend (cloud pipeline, local server, agent, evaluation) |
| **HTML / CSS / JavaScript** | — | Frontend web application (`frontend/`) |
| **Cypher** | — | Neo4j graph database query language (used in `graph_retriever.py`) |

---

## 2. Frameworks

| Framework | Purpose |
|---|---|
| **FastAPI** | Local inference server — HTTP REST API (JSON responses) |
| **Single-pass orchestrator** | `agent/langgraph/workflow.py` — imperative single-pass pipeline (planner → retrievers → reranker → generator); node functions are structured to be drop-in LangGraph-compatible, with no external `langgraph` dependency |
| **Pydantic** | Data validation and serialization (DTOs, `QueryState`, schemas) |
| **pydantic-settings** | Environment-variable-driven configuration (`CloudSettings`, `LocalSettings`, `SharedSettings`) |
| **uvicorn** | ASGI server — serves the FastAPI application |

---

## 3. AI / ML Libraries

### 3.1 Inference & Embeddings

| Library / Model | Role |
|---|---|
| **sentence-transformers** | Local cross-encoder reranker loading and scoring (`BAAI/bge-reranker-base`) |
| **transformers** (HuggingFace) | Model loading infrastructure used by `sentence_transformers` and vision models |
| **BAAI/bge-large-en-v1.5** | Primary embedding model — 1024-dimensional dense vectors for all lecture chunks |
| **BAAI/bge-reranker-base** | Cross-encoder reranker — re-scores retrieved chunks for precision before LLM generation; optional dynamic int8 quantization (`RERANKER_QUANTIZE`) with FP32 fallback |
| **BM25 (pure Python, stdlib)** | `retrieval/hybrid/bm25_retriever.py` — lexical retrieval fused with dense candidates via Reciprocal Rank Fusion (no external dependency) |

### 3.2 Cloud Ingestion Models & Heuristic Engines

| Model / Engine | Role |
|---|---|
| **Faster-Whisper** | ASR (Automatic Speech Recognition) — transcribes lecture audio to timestamped text |
| **Qwen2-VL** | Vision-Language Model — generates structured slide *descriptions* (visual captioning, stage A4) |
| **PaddleOCR** | OCR engine — extracts raw on-screen text from slide frames (stage A5, dedicated subprocess to avoid CUDA-context contamination) |
| **Qwen2.5-7B-Instruct** | Cloud-side LLM — segmentation (A7), entity extraction (A8), sliding-window relation extraction with compact aliases (A9), triplet generation (B1) |
| **Multi-Signal Prerequisite Fuser & DFS DAG Breaker** | Heuristic prerequisite inference (`cloud/extraction/prerequisite_extractor.py`, `local/loaders/prerequisite_enricher.py`) — temporal precedence + lexical mentions (negative lookbehinds) + segment containment + pedagogical inversion + DFS cycle resolution |

### 3.3 LLM Backends

| Backend | Type | Provider |
|---|---|---|
| **Ollama** (via `ollama` SDK) | Local, offline | Runs any GGUF-compatible model (default: `qwen2.5:3b`); GPU-accelerated via `docker-compose.gpu.yml` override + `nvidia-container-toolkit` |
| **OpenAI** (via `openai` SDK) | Cloud | GPT-4o, GPT-4, etc. |
| **Groq** (via `openai` SDK, OpenAI-compatible) | Cloud | Llama 3.3 70B, etc. — very fast inference |
| **OpenRouter** (via `openai` SDK, OpenAI-compatible) | Cloud | Many models behind one API |
| **Google Gemini** (via `google-generativeai` SDK) | Cloud | Gemini 1.5 Pro, Gemini Flash, etc. |
| **Anthropic** (via `anthropic` SDK) | Cloud | Claude Sonnet, Claude Haiku, etc. |

> **Note:** The `OnlineBackend` is a unified wrapper — all cloud providers share the same Python class and are distinguished only by `active_provider_id` in `data/llm_config.json`.

---

## 4. Databases

| Database | Type | Role |
|---|---|---|
| **Qdrant** | Vector database | Stores 1024-dim chunk embeddings; serves semantic nearest-neighbor queries at inference time |
| **Neo4j** | Graph database | Stores entity-relationship knowledge graph and strictly acyclic `PREREQUISITE_OF` dependencies; serves Cypher traversal and Socratic back-tracking queries |

**Client libraries:**
- `qdrant-client` (Python SDK) — used in `retrieval/vector_retriever/qdrant_retriever.py`
- `neo4j` (official Python driver) — used in `retrieval/graph_retriever/neo4j_retriever.py` and `serving/fastapi/routes/prerequisites.py`

**Persistence:**
- Both databases run in Docker containers with bind mounts to `local/docker/local_runtime/qdrant_data/` and `local/docker/local_runtime/neo4j_data/`.
- Data persists across container restarts and server shutdowns.
- Qdrant knowledge is loaded from `embeddings.npy` + `embedding_ids.json` contained in Knowledge Packages.
- Neo4j knowledge is loaded from `entities.json`, `relations.json`, and `prerequisites.json` contained in Knowledge Packages.

---

## 5. Infrastructure

| Component | Purpose |
|---|---|
| **Docker** | Containerizes Qdrant and Neo4j services for local deployment |
| **Ollama daemon** | Serves local GGUF language models at `localhost:11434`; managed via `ollama` CLI and SDK |
| **Kaggle** | Cloud GPU execution environment for the heavy ingestion pipeline (entry: `cloud/orchestration/run_ingestion_pipeline.py`) |

---

## 6. Python Ecosystem Dependencies

### Core Backend
| Package | Function |
|---|---|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `pydantic` / `pydantic-settings` | Data models and env config |
### AI / ML
| Package | Function |
|---|---|
| `sentence-transformers` | Reranker model loading and scoring |
| `transformers` | HuggingFace model hub loader |
| `faster-whisper` | GPU-accelerated ASR transcription |
| `torch` | PyTorch runtime (required by transformers, sentence-transformers) |
| `Pillow` / `opencv-python` | Image processing for frame extraction and OCR pre-processing |

### Databases & Clients
| Package | Function |
|---|---|
| `qdrant-client` | Qdrant vector DB client |
| `neo4j` | Neo4j graph DB Python driver |
| `ollama` | Ollama local daemon SDK |

### Cloud Provider SDKs
| Package | Provider |
|---|---|
| `openai` | OpenAI (GPT series), Groq, OpenRouter, custom OpenAI-compatible endpoints |
| `google-generativeai` | Google Gemini |
| `anthropic` | Anthropic Claude |

### Cloud Pipeline Only (excluded from `local_requirements.txt`)
| Package | Function |
|---|---|
| `paddleocr` | PaddleOCR visual text extraction |
| `paddlepaddle` | PaddlePaddle runtime (CPU wheels by default; GPU wheels only when the Kaggle image provides them) |

---

## 7. Configuration and Serialization

| Technology | Usage |
|---|---|
| **JSON** | `data/llm_config.json` — LLM provider config; `transcript.json` — transcription output |
| **JSON** | `entities.json` / `relations.json` / `prerequisites.json` — serialized knowledge graph (nodes + typed edges + prerequisite DAG) inside Knowledge Packages |
| **Pydantic JSON Schema** | Validation of Knowledge Package metadata and API request/response bodies |
| **`.env` files** | Environment variable injection for `pydantic-settings` (`LocalSettings`, `CloudSettings`) |

---

## 8. Shared Constants

| Constant | Value | Importance |
|---|---|---|
| `EMBEDDING_DIMENSION` | `1024` | Enforced in `SharedSettings`. Prevents vector dimension mismatch between cloud-generated embeddings and local Qdrant queries. |
| Default local LLM | `qwen2.5:3b` | Defined in `config.py` as `OLLAMA_MODEL`. Used as a fallback if `llm_config.json` is absent. |
| Ollama default port | `localhost:11434` | Hardcoded in `ollama_backend.py` (Ollama SDK default). |
| Reranker model | `BAAI/bge-reranker-base` | Loaded eagerly at startup in `rerank_service.py`. |
| `NORMAL_TOP_K` / `LECTURE_WIDE_TOP_K` | `15` | `agent/dspy/planner.py` — retrieval candidate pool. The reranker sees 15 candidates; the LLM still receives a budget-capped context. |
| `MIN_TRANSCRIPT_CHARS` / `RETRIEVAL_FETCH_SLACK` | `15` / `15` | `retrieval/vector_retriever/qdrant_retriever.py` — transcript-only filler filter + fetch slack so filtering never shrinks the pool. |
| Context budget | `4,000` chars normal / `6,000` lecture-wide | `agent/dspy/planner.py`; `rerank_service.MAX_CONTEXT_CHARS = 4000` fallback. |
| Prerequisite threshold | `0.65` | `cloud/extraction/prerequisite_extractor.py` — minimum multi-signal score for prerequisite edge admission. |

---

*Phase 1 documentation. More dependency details will be verified in later phases when requirements files are read directly.*
