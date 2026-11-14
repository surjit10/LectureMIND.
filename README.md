# LectureMIND

> **AI-powered GraphRAG learning platform** that transforms lecture videos into interactive, queryable knowledge bases.

---

## Contents

- [Overview](#overview) · [Motivation](#motivation) · [Features](#features)
- [High-Level Architecture](#high-level-architecture) · [System Components](#system-components)
- [Processing Pipeline](#processing-pipeline) · [Retrieval Pipeline](#retrieval-pipeline) · [Folder Structure](#folder-structure)
- [Technologies Used](#technologies-used) · [Installation](#installation) · [Configuration](#configuration) · [Running the Project](#running-the-project)
- [Evaluation](#evaluation) · [Design Decisions](#design-decisions) · [Production Architecture](#production-architecture)
- [Future Improvements](#future-improvements) · [Documentation Index](#documentation-index)

---

## Overview

LectureMIND converts lecture videos into structured, AI-queryable knowledge bases that students can interact with through natural language. During processing, the system automatically extracts speech, visual slide content, and on-screen text to build a multimodal representation of the lecture. Once a lecture is processed, students can ask questions, generate flashcards, produce structured notes, and receive grounded, citation-backed answers—all running locally on a standard laptop.

The system is built around **GraphRAG (Graph-augmented Retrieval-Augmented Generation)**, combining dense semantic vector search with knowledge graph traversal to answer both factual and relational questions grounded in the lecture content.

---

## Motivation

Traditional lecture recordings are difficult to revisit effectively. Students must scrub through hours of video to find specific information, with no way to ask follow-up questions or verify what they remember. LectureMIND addresses this by:

- **Making lectures searchable** at a semantic level — ask a question, get an answer from the lecture content.
- **Compressing gigabytes of video into kilobytes of structured data** — an 83-minute 720p lecture (~1.2 GB) becomes a ~436 KB knowledge package (~2,800× smaller), because the video is replaced by transcripts, slide text, embeddings, and a knowledge graph.
- **Separating expensive computation from the learning interface** — heavy AI processing happens once on cloud GPU infrastructure; students interact with a lightweight local server.
- **Supporting multiple learning modes** — conversational Q&A, flashcard generation, structured notes — all grounded in the actual lecture content.
- **Preserving privacy** — students can run the entire inference pipeline locally using Ollama, with no data leaving their machine after the initial package import.

---

## Features

- **Multimodal Ingestion** — Audio transcription (Faster-Whisper), slide visual captioning (Qwen2-VL), and OCR (PaddleOCR), fused into timestamped multimodal chunks at slide keyframes; semantic chunk merging combines whisper segments into self-contained passages
- **GraphRAG Query Pipeline** — Hybrid retrieval: dense vectors + BM25 lexical scores fused via Reciprocal Rank Fusion (RRF) + knowledge-graph traversal, cross-encoder reranking, and evidence-gated generation
- **Socratic Prerequisite Back-Tracker** — Deterministic DAG prerequisite inference engine; traces foundational knowledge gaps backwards when students struggle with advanced concepts, surfacing specific timestamps and chunks
- **Knowledge Graph Quality Audit Suite** — Automated evaluation layer auditing entity fragment/orphan rates, relation referential integrity (0.0% dangling relations in audited packages), strict DAG acyclicity (0 cycles), gold-reference matching where applicable (only the Transformer lecture is annotated: Entity F1 0.40, Relation F1 0.00 — LLM-assisted labels pending human verification), and GraphRAG downstream retrieval
- **Conversational Q&A** — Grounded JSON answers with timestamped source citations; refuses to answer when the lecture lacks evidence
- **Active Learning** — Auto-generated notes, flashcards, quizzes, and learning paths from lecture context
- **Course Index** — Lightweight course layer (pure metadata) that queries across lectures via per-lecture fan-out, without ever merging knowledge graphs
- **Retrieval Explainability** — Per-query pipeline trace (route, planner intent, per-stage scores, timings, selected evidence) surfaced in Developer Mode
- **QA Evaluation** — Grounded benchmark sets with keyword-verified ground-truth chunks across factual, conceptual, definition, summary, and visual (`need_visual`) query types
- **Cross-Lecture Isolation** — Hard per-`lecture_id` guards in Qdrant and Neo4j; nodes/edges scoped by `(entity_id, lecture_id)`; no data leaks between lectures or across re-imports
- **Fine-tunable Global Reranker** — `cloud/reranker_training/` builds training data from `triplets.json` shipped inside knowledge packages and fine-tunes the cross-encoder; hot-reloadable via API
- **Flexible LLM Backends** — Seamlessly switch between local Ollama models and cloud APIs (OpenAI, Gemini, Groq, OpenRouter, Anthropic) without server restart; per-provider rate limiting with exponential backoff
- **Knowledge Packages** — Portable ZIP archives encapsulating processed lectures; long 90-minute lectures (~1.2 GB video) compress into **~428 KB packages** (~2,800× smaller than source video)
- **Offline-First** — Full query and prerequisite navigation capability with no internet connection once packages are imported

---

## High-Level Architecture

LectureMIND operates on a strict two-phase architecture. Expensive processing happens once in the cloud; all subsequent student interactions happen locally.

```mermaid
graph TD
    subgraph Cloud[" Cloud Runtime (Kaggle / GPU)"]
        Video[" Lecture Video"]
        Whisper["Faster-Whisper\n(A2: Audio Transcription)"]
        VLM["Qwen2-VL\n(A4: Visual Slide Captioning)"]
        OCR["PaddleOCR\n(A5: Slide Text Extraction)"]
        Chunker["Multimodal Fusion & Segmentation\n(A6+A7: Timestamped Chunks)"]
        EntityExt["Entity Extraction\n(A8: Qwen2.5-7B)"]
        RelExt["Relation Extraction\n(A9: Sliding-Window + Compact Aliases)"]
        PrereqInf["Prerequisite Inference Engine\n(A10: Multi-Signal Scoring + DAG DFS)"]
        Embedder["Embedding Model\n(B0: bge-large-en-v1.5, 1024-dim)"]
        TripletGen["Triplet Generator\n(B1: Contrastive Hard Negatives)"]
        Exporter["Package Exporter\n(C1 validate → C2 zip)"]

        Video --> Whisper & VLM & OCR
        Whisper & VLM & OCR --> Chunker
        Chunker --> EntityExt --> RelExt --> PrereqInf
        Chunker --> Embedder & TripletGen
        Embedder & RelExt & PrereqInf & TripletGen --> Exporter
    end

    Exporter -->|"Knowledge Package (.zip)"| LocalServer

    subgraph LocalServer[" Local Runtime (FastAPI + QueryWorkflow)"]
        Import["Package Import\n(Qdrant Vectors + Neo4j Graph + Prereqs)"]
        Course["Course Index\n(course_registry + isolation fan-out)"]
        QueryWorkflow["QueryWorkflow\n(Single-pass orchestrator)"]
        Planner["QueryPlanner\n(regex heuristics)"]
        VecRet["VectorRetriever + BM25\n(Qdrant ANN + RRF fusion)"]
        GraphRet["GraphRetriever\n(Neo4j 1-3 hop traversal)"]
        PrereqTracker["Socratic Prerequisite Back-Tracker\n(/lecture/{id}/prerequisites/{concept})"]
        Rerank["reranker_node\n(Global Cross-Encoder BGE)"]
        Ctx["ContextBuilder\n(dedupe/merge/budget/priority)"]
        Gen["answer_generator_node\n(evidence-gated generation)"]
        ProvReg["ProviderRegistry\n(data/llm_config.json)"]
        Ollama["OllamaBackend\n(Local Private LLM)"]
        Online["OnlineBackend\n(Cloud LLM APIs)"]
        Learning["LearningService\n(Notes/Quiz/Flashcards)"]
        KGAudit["KG Quality Audit Suite\n(entity/relation/prereq DAG/RAG)"]

        Import --> QueryWorkflow & Course & PrereqTracker
        Course --> QueryWorkflow
        QueryWorkflow --> Planner --> VecRet & GraphRet
        VecRet & GraphRet --> Rerank --> Ctx --> Gen
        Gen --> ProvReg --> Ollama & Online
        Learning --> ProvReg
        Import --> KGAudit
    end

    subgraph Frontend[" Next.js Frontend"]
        UI["Web UI\n(Chat / Prerequisite Navigator / Quiz / Graph / Dev Mode)"]
    end

    Frontend -->|"HTTP REST (JSON)"| LocalServer
```

---

## System Components

| Component | Location | Role |
|---|---|---|
| **Cloud Ingestion Pipeline** | `cloud/` | Multi-stage lecture processing: transcription, OCR, fusion, segmentation, entity/relation extraction, prerequisite inference, embeddings, triplets |
| **Prerequisite Inference Engine (A10)** | `cloud/extraction/prerequisite_inference.py` | Multi-signal semantic scoring ($S_{\text{discourse}} + S_{\text{graph}} + S_{\text{prominence}} + S_{\text{temporal}}$), temporal causality + pedagogical inversion gates, deterministic DFS DAG cycle resolution |
| **Socratic Prerequisite Back-Tracker** | `serving/fastapi/routes/prerequisites.py` | Bounded Neo4j backward prerequisite dependency traversal, timestamped concept anchoring |
| **KG Quality Audit Suite** | `evaluation/knowledge_graph/` | Full audit suite: entity fragment/orphan rate, relation referential integrity (0.0% dangling), gold-reference matching gated by lecture applicability (only Transformer lecture annotated — see `outputs/kg_quality*/`), GraphRAG downstream retrieval |
| **Relation Extraction (A9)** | `cloud/extraction/relation_extractor.py` | Sliding-window token management with compact entity alias remapping (`E1, E2...`), 8192-token retry cap, strict pedagogical exclusion rules |
| **Global Training Orchestration** | `scripts/train_global_reranker.py` | Local: merge package triplets → fine-tune → install into `GLOBAL_RERANKER_DIR` → hot-reload |
| **Reranker Training Pipeline (B)** | `cloud/reranker_training/` | Independent pipeline: discover packages → extract `triplets.json` → merge/dedupe → fine-tune → evaluate → version → export `global_reranker_v{N}.zip`. Never touches videos |
| **Knowledge Package** | `data/packages/lecture_{id}/` | Self-contained processed lecture archive (embeddings, graph, prerequisites, chunks, triplets) |
| **FastAPI Server** | `serving/fastapi/app.py` | Local HTTP server; lifespan, model recovery, router registration |
| **QueryWorkflow** | `agent/pipeline/workflow.py` | Single-pass query orchestration: planner → conditional retrieval → rerank → evidence-gated answer |
| **QueryPlanner** | `agent/planner/planner.py` | Heuristic route selection (`graph_only`/`vector_only`/`graph_and_vector`) + 10-intent `plan_full()` |
| **Vector Retriever** | `retrieval/vector_retriever/qdrant_retriever.py` | Qdrant ANN search (1024-dim) + lecture-wide timeline sampling; **raises without `lecture_id`** |
| **Graph Retriever** | `retrieval/graph_retriever/neo4j_retriever.py` | Neo4j bounded 1–3-hop traversal, all 6 relation types, exact→partial match; **raises without `lecture_id`** |
| **Course Retriever** | `retrieval/course_retriever.py` | Course fan-out: queries only member lectures through the lecture-scoped retrievers, dedupes on `(lecture_id, chunk_id)` |
| **Context Builder** | `retrieval/context_builder.py` | Dedupe, chronological order, adjacent-merge, OCR-noise filter, transcript>OCR>visual priority, budget enforcement |
| **Global Reranker** | `retrieval/reranker/rerank_service.py` + `local/services/reranker_service.py` | BAAI/bge-reranker-base cross-encoder singleton; loaded once at startup; hot-reloadable |
| **Lecture Registry** | `local/storage/lecture_registry.py` | JSON-backed knowledge-package metadata + active-lecture lifecycle + startup audit |
| **Course Registry** | `local/storage/course_registry.py` | JSON-backed course metadata (name + lecture_ids); never writes to Qdrant/Neo4j |
| **Provider Registry** | `local/llm/provider_registry.py` | JSON-backed factory for LLM backend selection |
| **Provider Manager** | `local/llm/provider_manager.py` | Runtime health checker for configured providers |
| **Learning Service** | `serving/fastapi/learning_service.py` | Notes, flashcards, quiz, and learning-path generation reusing the retrieval pipeline |
| **Evaluation Framework** | `evaluation/benchmark_runner.py` | Offline RAG quality benchmark over a QA dataset, bypassing HTTP |
| **RAGAS Scaffold** | `evaluation/ragas/eval_ragas.py` | Scaffolded evaluation interface for optional Faithfulness, Answer Relevancy, Context Precision |
| **Load Test Scaffold** | `evaluation/load_testing/load_test.py` | Configurable concurrency profiling; reports avg/p50/p95/p99 latency, errors, and RPS |
| **Frontend** | `frontend/` | Next.js 14 + React 18 + TypeScript + custom CSS design system; Student/Developer modes |

---

## Processing Pipeline

The cloud pipeline processes a raw lecture video through sequential stages (orchestrated by `cloud/orchestration/run_ingestion_pipeline.py`):

| Stage | Name | Input | Output | Model Used |
|---|---|---|---|---|
| A1 | Metadata Extraction | Video file | `metadata.json` | FFprobe |
| A2+A3 | Transcription + Frame Extraction (concurrent) | Audio + Video | `transcript.json`, `frames/*.jpg` | Faster-Whisper, OpenCV / FFmpeg |
| A4 | Visual Captioning | Keyframes | `vlm_output.jsonl` | Qwen2-VL |
| A5 | OCR Extraction | Keyframes | `ocr_output.jsonl` | PaddleOCR (dedicated subprocess) |
| A6 | Multimodal Fusion | Transcript + VLM + OCR | `multimodal_chunks.json` | Rule-based fusion (±2 s window) |
| A7 | Topic Segmentation | Chunks | `segments.json`, `chunk_segment_map.json` | Qwen2.5-7B-Instruct |
| A8 | Entity Extraction | Segments + Chunks | `entities.json` | Qwen2.5-7B-Instruct |
| A9 | Relation Extraction | Entities + Chunks | `relations.json` | Qwen2.5-7B (Sliding-window + compact aliases) |
| A10 | Prerequisite Inference | Entities + Relations + Chunks | `prerequisites.json` | Multi-Signal Fuser + DFS Cycle Breaker |
| B0 | Embeddings | Chunks | `embeddings.npy`, `embedding_ids.json` | BAAI/bge-large-en-v1.5 (1024-dim) |
| B1 | Triplet Generation | Segments + Chunks | `triplets.json` | Qwen2.5-7B-Instruct |
| B2 | Reranker Fine-tune *(optional)* | `triplets.json` | `reranker_model/`, `training_metrics.json` | BAAI/bge-reranker-base |
| C1 | Validation | All outputs | validation report | Comprehensive Schema Validator |
| C2 | Package Export | All outputs | `lecture_{id}_knowledge_package.zip` | ZIP Deflate level 9 |

> Stages A2 and A3 run **concurrently**. Stage A5 (PaddleOCR) runs inside the cloud pipeline's dedicated subprocess, so its CUDA context never contaminates the parent process.
>
> B2 is **off by default** (`ENABLE_PIPELINE_RERANKER_TRAINING=False`). Reranker fine-tuning is intended to run offline via `scripts/train_global_reranker.py`; inference only ever loads from `local_runtime/models/global_reranker/`.

---

## Retrieval Pipeline

Every query passes through a single-pass pipeline (`agent/pipeline/workflow.py`):

```mermaid
flowchart TD
    Q["User Question"] --> P["QueryPlanner<br/>(agent/planner/planner.py)<br/>plan → route, plan_full → intent"]

    P --> R{"Retrieval route"}

    R -->|"graph_only · graph_and_vector"| G["GraphRetriever<br/>Neo4j · bounded 1-3 hop traversal<br/>all 6 relation types · exact → partial match<br/>scoped by (entity_id, lecture_id)"]
    R -->|"vector_only · graph_only · graph_and_vector"| V["VectorRetriever + BM25<br/>Qdrant ANN · bge-large-en-v1.5<br/>BM25 lexical index · RRF fusion<br/>top_k=15 · lecture_id filter<br/>filler-chunk filter (min 15 chars)"]

    G --> M["reranker_node<br/>merge + dedupe by chunk_id"]
    V --> M

    M -->|"cross-encoder<br/>bge-reranker-base"| X["Reranked candidates<br/>+ graph paths rendered as<br/>Graph: EntityA → EntityB (RELATION)"]

    X --> C["ContextBuilder<br/>dedupe → chronological sort → adjacent merge<br/>OCR-noise filter · Transcript → OCR → Visual<br/>budget enforcement (4,000 / 6,000 chars)"]

    C --> AG{"Evidence gated?"}

    AG -->|"context present"| L["answer_generator_node<br/>active LLM backend<br/>Ollama (local) or OnlineBackend (cloud)"]
    L --> R2["JSON Response<br/>(answer, sources, graph_path, debug)"]

    AG -->|"no evidence"| I["Insufficient evidence refusal<br/>(refuses rather than hallucinates)"]
    I --> R2
```

**Pipeline stages:**

1. **Query Planning** — `QueryPlanner.plan` → retrieval route (`graph_only` / `vector_only` / `graph_and_vector`); `plan_full` → intent, `is_lecture_wide`, `top_k`, `context_budget`, `need_visual`.
2. **Retrieval** — Conditional: `graph_retriever` (Neo4j bounded 1–3-hop traversal, all 6 relation types, exact→partial match) and `vector_retriever` (Qdrant ANN). With hybrid retrieval enabled (default), BM25 lexical results are fused with the dense candidates via Reciprocal Rank Fusion (RRF) before reranking, recovering exact entity names and numeric facts that dense similarity misses. `graph_only` routes **always** also retrieve vectors — the graph supplies structure, vectors supply the text the answer must be grounded in. Every call is lecture-scoped.
3. **Fusion & Rerank** — `reranker_node` merges + dedupes by `chunk_id`, then the global CrossEncoder (`bge-reranker-base`, optionally int8-quantized) scores pairs.
4. **Context Assembly** — `ContextBuilder`: dedupe → chronological sort → merge adjacent chunks → OCR-noise filter → priority Transcript > OCR > Visual → budget enforcement; lecture-wide intents sample 8 timeline buckets.
5. **LLM Generation** — `answer_generator_node` via the active LLM backend (Ollama or online): evidence-gated, language-pinned, falls back to `"Insufficient evidence found in lecture."` when context is empty; sources built only from retrieved chunks.

**Why this design?** The bi-encoder retrieves broadly (fast, high recall); the cross-encoder reranks precisely (slow, high precision). Neither alone achieves both goals. `debug` is an additive pipeline trace (route, planner intent, per-stage scores, timings) consumed only by the frontend's Developer Mode — normal responses are unchanged.

**Course queries** follow the same flow with one extra step: `POST /courses/query` fans out to each member lecture through the lecture-scoped retrievers (isolation guards apply per call), merges with `(lecture_id, chunk_id)` dedup, then reuses the global reranker and answer generator.

See [RETRIEVAL_SYSTEM.md](./docs/RETRIEVAL_SYSTEM.md) for the complete algorithmic detail.

---

## Folder Structure

```
lecturemind/
│
├── cloud/                          # Cloud ingestion pipeline (Kaggle)
│   ├── orchestration/
│   │   └── run_ingestion_pipeline.py   # Main cloud orchestrator (A1–C2)
│   ├── training/
│   │   ├── triplet_generator.py    # B1 — reranker training triplets
│   │   └── reranker_trainer.py     # B2 — CrossEncoder fine-tuning (not run by default)
│   ├── reranker_training/          # Pipeline B — INDEPENDENT reranker training
│   │   ├── package_discovery.py    # discover packages, extract ONLY triplets.json
│   │   ├── dataset_builder.py      # merge / dedupe / filter / train-dev split
│   │   ├── versioning.py           # v1/ v2/ … best/ latest/ + index.json
│   │   ├── export.py               # global_reranker_v{N}.zip
│   │   └── pipeline.py             # orchestrator + CLI (python -m …)
│   └── packaging/                  # validator / exporter / manifest_builder
│
├── agent/                          # Query reasoning pipeline
│   ├── planner/
│   │   ├── planner.py              # QueryPlanner (route + 10-intent plan_full)
│   │   └── query_plan.py           # QueryPlan dataclass
│   └── pipeline/
│       ├── workflow.py             # QueryWorkflow (single-pass orchestrator)
│       ├── state.py                # GraphState (frozen QueryPipelineState)
│       └── nodes/
│           ├── vector_retriever.py # Qdrant retrieval node
│           ├── graph_retriever.py  # Neo4j retrieval node
│           ├── reranker.py         # Cross-encoder rerank + ContextBuilder node
│           └── answer_generator.py # Evidence-gated LLM node
│
├── retrieval/                      # Retrieval layer (packages)
│   ├── vector_retriever/qdrant_retriever.py   # Qdrant ANN + lecture-wide sampling
│   ├── graph_retriever/neo4j_retriever.py     # Neo4j bounded traversal (lecture-scoped)
│   ├── reranker/rerank_service.py             # Global CrossEncoder singleton + rerank()
│   ├── course_retriever.py                    # Course fan-out (Feature 1)
│   └── context_builder.py                     # Context assembly (dedupe/merge/budget)
│
├── local/                          # Local-side business logic
│   ├── llm/                         # LLMBackend, Ollama/online backends, provider registry
│   ├── loaders/                     # package_validator, qdrant_loader, neo4j_loader,
│   │                                #   reranker_loader, manifest_validator
│   ├── storage/                     # lecture_registry.py, course_registry.py,
│   │                                #   registry_provider.py
│   ├── services/                    # reranker_service.py, llm_service.py
│   └── docker/                      # docker-compose (Qdrant + Neo4j)
│
├── serving/fastapi/                 # FastAPI HTTP layer
│   ├── app.py                       # App object, lifespan, model recovery, routers
│   ├── learning_service.py          # Notes / flashcards / quiz / learning path
│   └── routes/
│       ├── lectures.py              # POST /upload, lecture CRUD, learning endpoints
│       ├── query.py                 # POST /query (JSON + debug trace)
│       ├── courses.py               # Course CRUD + POST /courses/query (Feature 1)
│       ├── reranker.py              # Global reranker upload/status/reload/settings
│       ├── settings.py              # Provider config and status routes
│       └── debug.py                 # Debug endpoints
│
├── evaluation/                     # Offline evaluation framework
│   ├── benchmark_runner.py          # QA benchmark over the real workflow
│   ├── ragas/eval_ragas.py          # Scaffold for Faithfulness / Answer Relevancy / Context Precision
│   ├── load_testing/load_test.py    # Scaffold for 100/500/1000 concurrent users
│   ├── dashboard/dashboard_generator.py  # HTML dashboard from existing outputs
│   ├── metrics/                     # 6 metric families (planner/retrieval/reranker/…)
│   ├── reports/report_generator.py  # JSON/CSV/Markdown reports
│   └── outputs/                     # Generated benchmark reports
│
├── scripts/                         # Operational scripts
│   └── train_global_reranker.py     # Merge triplets → fine-tune → install → reload
│
├── frontend/                        # Next.js web UI (chat, learning, settings, dev mode)
├── schemas/                         # Frozen Pydantic models + closed enums
├── data/
│   ├── llm_config.json              # Active LLM provider configuration
│   ├── lecture_registry.json        # Imported lecture metadata
│   ├── courses.json                 # Course index (metadata only)
│   └── packages/                    # Imported Knowledge Package directories
├── config.py                        # SharedSettings / CloudSettings / LocalSettings
├── METRICS_MATRIX.md                # Measured metrics + comparison matrix
├── requirements.txt                 # Cloud (Kaggle) dependencies
└── local_requirements.txt           # Local-only dependencies
```

---

## Technologies Used

### Backend
| Technology | Version | Role |
|---|---|---|
| Python | 3.12 | Primary language |
| FastAPI | Latest | Local HTTP server |
| uvicorn | Latest | ASGI server |
| Pydantic / pydantic-settings | v2 | Data models, env config |

### AI / ML
| Technology | Role |
|---|---|
| BAAI/bge-large-en-v1.5 | Chunk embedding (1024-dim dense vectors) |
| BAAI/bge-reranker-base | Cross-encoder reranking (post-retrieval precision) |
| Faster-Whisper | GPU ASR — lecture audio transcription |
| Qwen2-VL | Vision-Language Model — slide visual captioning |
| Qwen2.5-7B-Instruct | Cloud-side LLM — segmentation, entity/relation extraction, triplet synthesis |
| PaddleOCR | OCR — raw text extraction from slide frames |
| sentence-transformers | Reranker model loader |
| Ollama (`qwen2.5:3b` default) | Local offline LLM inference (optionally GPU-accelerated via `docker-compose.gpu.yml`) |
| OpenAI / Gemini / Groq / OpenRouter / Anthropic | Cloud LLM inference (via unified OnlineBackend) |

### Databases & Infrastructure
| Technology | Role |
|---|---|
| Qdrant | Vector database — semantic ANN search |
| Neo4j | Graph database — knowledge graph traversal |
| Docker | Containerizes Qdrant and Neo4j for local deployment |
| Kaggle | Cloud GPU environment for lecture ingestion |

### Frontend
| Technology | Role |
|---|---|
| Next.js 14 | React framework with SSR |
| React 18 | UI component library |
| TypeScript | Type-safe frontend code |
| Custom CSS design system | `globals.css` CSS-variable theming + CSS modules |
| @tanstack/react-query | Server state synchronization |
| reactflow | Knowledge graph visualization |
| react-markdown | Markdown answer rendering |

---

## Installation

### Prerequisites
- Docker and Docker Compose
- Python 3.12 with `pip`
- Node.js 18+ and npm
- Ollama (for offline inference) — [install guide](https://ollama.ai)

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/lecturemind.git
cd lecturemind
```

### 2. Start Databases (Docker)
```bash
docker compose -f local/docker/docker-compose.yml up -d
```
This starts Qdrant (port 6333) and Neo4j (port 7474/7687) with bind mounts to `local/docker/local_runtime/`.

### 3. Create Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r local_requirements.txt
```
> **Note:** Use `requirements.txt` only when running the **cloud ingestion pipeline** on GPU infrastructure. `local_requirements.txt` strips out PaddleOCR, PaddlePaddle-GPU, and other cloud-only heavy dependencies.

### 4. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### 5. Configure Environment
```bash
cp .env.example .env
# Edit .env — see Configuration section below
```

### 6. Pull Ollama Model (for offline inference)
```bash
ollama pull qwen2.5:3b
```

> **Optional — GPU acceleration for Ollama (recommended if an NVIDIA GPU is available).**
> By default Ollama runs on CPU inside Docker, which makes answer generation slow
> (tens of seconds per query). To run the model on an NVIDIA GPU instead:
>
> 1. Install the NVIDIA container toolkit (host-level, one-time, ~8 MB download):
>    ```bash
>    sudo apt-get install -y nvidia-container-toolkit
>    sudo nvidia-ctk runtime configure --runtime=docker
>    sudo systemctl restart docker
>    ```
> 2. Recreate only the ollama container with the GPU override (base compose file stays CPU-safe):
>    ```bash
>    docker compose -f local/docker/docker-compose.yml -f local/docker/docker-compose.gpu.yml up -d ollama
>    ```
> 3. Verify: `docker exec lecturemind-ollama ollama ps` should show `100% GPU` (or a CPU/GPU split) instead of `100% CPU`.
>
> **Revert anytime** (no data loss): `docker compose -f local/docker/docker-compose.yml up -d ollama`.
> Smaller models (e.g. `qwen2.5:1.5b`) fit entirely in VRAM and are faster on small GPUs.
> See `local/docker/docker-compose.gpu.yml` for full instructions.

---

## Configuration

Edit `.env` in the project root:

```env
# ── Infrastructure ─────────────────────────────────────────────────
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

QDRANT_URL=http://localhost:6333

# ── Embedding (do not change — must match cloud pipeline) ───────────
EMBEDDING_DIMENSION=1024

# ── Ollama (offline inference) ──────────────────────────────────────
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_HOST=http://localhost:11434

# ── Cloud API Keys (online inference) ──────────────────────────────
# Set via the /settings UI at runtime — do not commit to version control
OPENAI_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
```

> **Important:** `EMBEDDING_DIMENSION` is hard-enforced at `1024` by a Pydantic `field_validator`. This must match the embedding model used during cloud ingestion (`BAAI/bge-large-en-v1.5`). Changing this value causes a startup error.

LLM provider selection is managed at runtime via `data/llm_config.json` (written by `ProviderRegistry`). Do not edit this file manually; use the `/settings` API or UI instead.

### Cloud vs. local inference

#### Local ML runtime setup

The local API needs a CPU/GPU embedding runtime for semantic retrieval. On a
CPU-only machine, install PyTorch from the CPU index before the pinned local
requirements so pip does not select a large CUDA distribution:

```bash
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.3.1
.venv/bin/python -m pip install -r local_requirements.txt
```

The reranker model is recovered at API startup. Answer generation additionally
needs either the configured Ollama model or an online provider in `/settings`.

Answer generation runs through a pluggable `LLMBackend` selected by `ProviderRegistry` at runtime:

- **Offline mode (default):** local Ollama model (e.g. `qwen2.5:3b`) — fully private, no internet needed.
- **Online mode:** a cloud API keyed provider (OpenAI, Google Gemini, Groq, OpenRouter, Anthropic, or any OpenAI-compatible `base_url`).

Add a provider and switch modes via the settings UI or directly:

```bash
# Add a provider (key is stored locally in data/llm_config.json, masked in API responses)
curl -X POST http://localhost:8000/settings/providers \
  -H 'Content-Type: application/json' \
  -d '{"provider":"groq","model":"llama-3.3-70b-versatile","api_key":"gsk_...","display_name":"Groq"}'

# Switch to online inference and activate the provider ID returned above
curl -X PATCH http://localhost:8000/settings \
  -H 'Content-Type: application/json' \
  -d '{"inference_mode":"online","active_provider_id":"<provider-id>"}'

# Validate a key without saving it
curl -X POST http://localhost:8000/settings/providers/test \
  -H 'Content-Type: application/json' \
  -d '{"provider":"groq","api_key":"gsk_..."}'
```

Cloud inference is dramatically faster than local CPU and higher quality than a small local
model — measured: a lecture-wide summary query dropped from ~180 s (local CPU) to ~8 s total
(2.5 s generation) with Groq `llama-3.3-70b-versatile`. If a provider fails at runtime, the
registry automatically falls back to the local Ollama backend.

> **Security:** API keys live only in `data/llm_config.json`, which is `.gitignore`d and never
> returned through the API (always masked as `***`). Never commit this file.

---

## Running the Project

### Start the Backend
```bash
source .venv/bin/activate
uvicorn serving.fastapi.app:app --reload
```
Server starts at `http://localhost:8000`. The startup sequence:
1. Validates the Knowledge Package registry (filesystem audit).
2. Loads the Global Reranker (`BAAI/bge-reranker-base`) into memory.
3. Attempts to restore the previously active lecture (reconnects to Qdrant and Neo4j).

### Start the Frontend
```bash
cd frontend
npm run dev
```
Frontend available at `http://localhost:3000`.

### Import a Knowledge Package
Use the web UI to upload a `.zip` package, or call the API directly:
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@lecture_abc12345.zip" \
  -F "display_name=Introduction to Photosynthesis"
```

### Query the Active Lecture
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the role of chlorophyll in photosynthesis?", "lecture_id": "lecture_abc12345"}'
```
Returns JSON: `{answer, sources[], graph_path[], debug{}}`. The `debug` field is populated only for Developer-Mode clients.

### Course Queries (Feature 1)
```bash
# Create a course, add lectures, query across them
curl -X POST http://localhost:8000/courses -H "Content-Type: application/json" -d '{"name": "Databases"}'
curl -X POST http://localhost:8000/courses/{course_id}/lectures -H "Content-Type: application/json" -d '{"lecture_id": "lecture_aaa"}'
curl -X POST http://localhost:8000/courses/query -H "Content-Type: application/json" \
  -d '{"course_id": "{course_id}", "query": "Explain indexing across all lectures"}'
```

### Check Provider Status
```bash
curl http://localhost:8000/settings/status
```

### Train the Global Reranker (Pipeline B)

The reranker training pipeline is **completely independent** from lecture
processing. Its only input is the `triplets.json` shipped inside every
knowledge package — never raw videos.

```bash
# Merge extracted package triplets → fine-tune → install → hot-reload:
python scripts/train_global_reranker.py
```

The full pipeline (discover packages → merge/dedupe → fine-tune → evaluate →
version → export `global_reranker_v{N}.zip`) lives in `cloud/reranker_training/`;
upload the exported ZIP via `POST /api/reranker/upload` for an atomic hot-reload
without restarting the server.

---

## Evaluation

The evaluation framework measures RAG pipeline quality offline — no running server required.

### Run Benchmark
```bash
./.venv_clean/bin/python scripts/run_mlops_smoke.py
```

The default command validates the 45-question, three-lecture smoke benchmark and refreshes
the dashboard without calling an LLM. To run the real workflow (Qdrant → Neo4j → cross-encoder
reranker → active LLM backend), use `scripts/run_mlops_smoke.py --run`; this may consume provider
quota. Reports are written to the selected run directory in CSV, JSON, and Markdown formats,
with run-manifest and package-hash provenance.

### Measured Results

#### Current results — 45-question smoke set, reviewed labels (2026-11-14)

Report: `evaluation/reports/smoke_groq_20261114_reviewed/` (45/45 questions, 0 failures).
Pipeline: local `bge-large-en-v1.5`, BM25 + RRF, `bge-reranker-base`, Qdrant, Neo4j; answers from
Groq `openai/gpt-oss-20b`. Dataset: `evaluation/datasets/multilecture_smoke_45_reviewed.json` —
15 questions per lecture (CS162, MIT 6.S191, Self-Attention); 40 were LLM-drafted from one chunk
each, then **AI-assisted review** (`review_status: llm_assisted_review`): 3 wrong reference
answers corrected, 25 missing supporting chunks added, 3 graph-fact questions flagged. **Not
human-reviewed** — the validator's `release` profile rejects it until it is.

| Metric | Mean | 95% CI (bootstrap) | n |
|---|---:|---:|---:|
| Post-rerank Hit@5 | **0.907** | 0.814 – 0.977 | 43 |
| Post-rerank MRR@5 | **0.801** | 0.691 – 0.903 | 43 |
| Context recall (gold chunks placed in the LLM prompt) | **0.843** | 0.736 – 0.936 | 43 |
| Routing accuracy | 0.933 | 0.844 – 1.000 | 45 |
| Answer token F1 (lexical, vs. short reference answers) | 0.256 | 0.201 – 0.313 | 45 |
| Keyword recall | 0.786 | 0.681 – 0.878 | 45 |
| Correct refusals on unanswerable questions | 2/2 | — | 2 |
| Answerable questions refused | 4/43 | — | 43 |
| Mean total latency (Groq paced at 6,000 tokens/min) | 27.5 s | 25.7 – 29.0 s | 45 |

**Component ablation** (`evaluation/reports/ablation_20261114_reviewed/`, same questions, no LLM):

| Configuration | Hit@1 | Hit@5 | MRR@5 | Context recall |
|---|---:|---:|---:|---:|
| Dense only | 0.674 | 0.814 | 0.734 | 0.793 |
| + BM25 (RRF) | 0.744 | 0.884 | 0.804 | 0.824 |
| + cross-encoder reranker | 0.744 | 0.907 | 0.801 | 0.874 |
| + knowledge-graph context (production) | 0.744 | 0.907 | 0.801 | 0.843 |
| Production with the previous 4,000-char / 50%-graph budget | 0.744 | 0.907 | 0.801 | 0.783 |

**Error analysis** (`analysis.md` in the report folder; 43 answerable questions): 36 evidence reached
the prompt and the model answered · 2 refused although the evidence was in the prompt · 4 evidence
retrieved but not placed in the prompt · 1 retrieval miss.

What this shows:
- BM25 is the largest single gain (+0.07 Hit@5 / MRR@5 over dense only); the cross-encoder adds
  +0.023 Hit@5 and +0.05 context recall but no MRR gain on this set.
- Graph context does not change chunk ranking (graph results are relation lines, not chunks) and
  costs prompt space: context recall on graph-routed questions is 0.750 without graph lines and
  0.528 with them. Its value for relational answers has not been shown with an answer-level test.
- Raising the point-query context budget to 6,000 characters with a 25% graph share lifted context
  recall from 0.783 to 0.843; it was selected on this same question set (no separate dev set), so
  the gain is optimistic, and larger prompts raised Groq-paced latency (21.8 s → 27.5 s).
- Two refusals happened with the evidence present: visual questions the planner did not flag
  `need_visual`, so slide descriptions were left out of the context.
- With n = 43, confidence intervals are about ±0.1: differences smaller than that are not
  established. Answer F1 is lexical and penalises long answers; no judge-based faithfulness or
  correctness metric has been run.

#### Effect of the 2026-11 correctness fixes (no-LLM retrieval comparison)

`evaluation/reports/retrieval_comparison_20261114/` runs commit `891648c` and the fixed code through
the same retrieval → rerank → context pipeline on the same data (original labels):

| | Before fixes | After fixes |
|---|---:|---:|
| Post-rerank Hit@5 / MRR@5 (43 answerable) | 0.884 / 0.759 | 0.884 / 0.759 |
| Routing accuracy (45) | 0.889 | 0.933 |
| Lecture-wide summaries: mean timeline span in context (9 prompts) | 0.807 | **0.872** |
| Lecture-wide summaries: worst prompt's timeline span | 0.464 | **0.863** |
| Lecture-wide summaries: lecture deciles covered (of 10) | 5.3 | **7.7** |
| Incoming graph relations of "Process Abstraction" shown correctly | 1/16 | **16/16** |

Earlier runs on the unreviewed labels (`smoke_groq_20261010_fixed/`, `smoke_groq_20261114_newcode/`)
are kept for traceability.

#### Historical single-lecture benchmark (comparison only)

The archived report `evaluation/outputs/evaluation_report_20260811_105621.*` contains a
50-question, single-lecture run from an earlier configuration. It is retained for audit
traceability only and is **not the current headline evaluation**. Do not compare its
metrics directly with the current three-lecture smoke run: the datasets, scope, and
run configuration differ. It also **cannot be reproduced on the current packages**: its
`lecture_id` (`lecture_cs162_v17`) is no longer loaded, and its gold chunk IDs point to
different text in both current CS162 packages (keywords match the gold chunks for only
21/50 and 17/50 questions). See `reports/lecturemind_runtime_evaluation_forensic_audit.md`
for the historical discrepancy analysis.

### Knowledge Graph Quality & Prerequisite Audit

Audited via `evaluation/knowledge_graph/audit_package.py` on the short Transformer lecture against reference labels (`evaluation/knowledge_graph/prerequisite_gold.json`) — **LLM-assisted reference labels, pending human verification**, not an independent human gold standard. These scores apply to that lecture only: for other packages the auditor reports gold metrics as N/A (null) rather than misleading zeros, and the composite score counts only measurable components.

| Metric | Measured Score | Diagnostic Context |
|---|---|---|
| **Graph Topology (Strict DAG)** | **True** | Deterministic DFS cycle resolution guarantees acyclicity |
| **Cycle Count** | **0** | Zero feedback loops in prerequisite graph |
| **Self-Loop Count** | **0** | Zero self-dependencies ($A \to A$) |
| **Orphan Entity Rate** | **58.86%** | 93/158 background entities in CS162 have degree 0 (known KG limitation) |
| **Dangling Relation Rate** | **0.0%** | 100% referential integrity across all entities (0 dangling edges) |
| Entity / Relation / Prereq reference F1 | See audit output | Not independently human-verified; see provenance note above |

#### Multi-Lecture Extraction Yield (Full Cloud Execution)

Results across three full-length lectures processed with sliding-window chunking, compact entity alias remapping (`E1, E2...`), and 8192-token retry budgets:

| Lecture Package | Duration / Chunks | Extracted Entities | Extracted Relations | Inferred Prerequisites | DAG Status |
|---|---|---|---|---|---|
| **CS162 Operating Systems** | ~83 min (98 chunks) | **158** | **103** | **29** | **Strict DAG (0 cycles)** |
| **MIT 6.S191 Deep Learning** | ~56 min (70 chunks) | **87** | **94** | **15** | **Strict DAG (0 cycles)** |
| **Self-Attention in Transformers**| ~44 min (48 chunks) | **82** | **65** | **14** | **Strict DAG (0 cycles)** |
| **Total Across Corpus** | **216 chunks** | **327 entities** | **262 relations** | **58 prerequisites** | **100% Acyclic** |

#### Downstream GraphRAG Benchmark (CS162 Operating Systems)

Evaluated via `evaluation/knowledge_graph/graphrag_evaluator.py`:

| Retrieval Route | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Precision@5 | MRR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BM25 / Lexical Retrieval** | 0.25 | 0.42 | 0.50 | 0.403 | 0.117 | 0.389 |
| **Graph-Only RAG** | 0.08 | 0.42 | 0.42 | 0.236 | 0.100 | 0.280 |
| **Hybrid RAG (RRF)** | **0.25** | **0.42** | **0.67** | **0.486** | **0.167** | **0.420** |

Run the auditor on any knowledge package:
```bash
./.venv/bin/python evaluation/knowledge_graph/audit_package.py \
  --package 0-output/CS162_Lecture_1_What_is_an_Operating_System_720P_knowledge_package.zip \
  --output-dir outputs/kg_quality_cs162/
```

### RAGAS Answer Quality (requires a live server + LLM backend)
```bash
pip install ragas datasets   # optional dependency group
python -m evaluation.ragas.eval_ragas  # writes evaluation/reports/ragas_report.json
```
> Requires `ragas` + `datasets` to be installed; without them the command refuses to run (it will not write placeholder scores). No RAGAS report exists yet in this repository.

### Load Testing (requires a live server)
```bash
python -m evaluation.load_testing.load_test  # writes evaluation/reports/load_test_report.csv
```
> Requires the FastAPI server to be running. The current small report is
> `evaluation/reports/load_test_report.csv`; it is a smoke test, not a production capacity claim.

### Benchmark Dashboard (renders existing outputs — never re-runs evals)
```bash
./.venv_clean/bin/python scripts/run_mlops_smoke.py
# → evaluation/dashboard/index.html (self-contained, inline SVG charts)
```

### Metrics Computed

| Metric | Type | Description |
|---|---|---|
| `routing_accuracy` | Planner | Fraction of queries routed to the correct retrieval strategy |
| `visual_routing_accuracy` | Planner | Fraction of queries where `need_visual` matched the ground truth |
| `post_rerank_hit_at_5` / `post_rerank_mrr_at_5` / `post_rerank_recall_at_5` / `post_rerank_ndcg_at_5` | Retrieval | Hit / MRR (0 beyond rank 5) / Recall / nDCG at 5 on the **cross-encoder order** — the ranking the generator actually receives (primary) |
| `context_recall` | Retrieval | Fraction of expected chunks placed in the LLM prompt after reranking, merging and the context budget (primary end-to-end retrieval metric) |
| `hit@5` | Retrieval | Hit@5 on the **pre-rerank** candidate order (dense + BM25 via RRF); kept for report continuity |
| `mrr_at_5` | Retrieval | Pre-rerank reciprocal rank of the first relevant chunk within top-5, 0 beyond rank 5 |
| `mrr` | Retrieval | Pre-rerank unbounded reciprocal rank (diagnostic) |
| `recall@5` | Retrieval | Pre-rerank fraction of relevant chunks in the top-5 |
| `ndcg_at_5` | Retrieval | Pre-rerank nDCG at rank 5 |
| `precision@5` | Retrieval | Pre-rerank fraction of top-5 chunks that are relevant (low by construction: most questions have one relevant chunk) |
| `r_precision` | Retrieval | Pre-rerank precision at rank $R = \|\text{expected}\|$ |
| `ranking_quality` | Reranker | Unbounded MRR of the reranked list |
| `avg_cross_encoder_score` | Reranker | Mean cross-encoder relevance score across all 15 candidate chunks |
| `top1_cross_encoder_score`| Reranker | Relevance score of the #1 ranked candidate |
| `answer_similarity` | Answer | Lexical Jaccard token overlap vs. ground truth answer (supplementary) |
| `answer_f1` | Answer | SQuAD-style token F1 vs. ground truth answer |
| `keyword_recall` | Answer | Fraction of ground-truth keywords present in the answer (0.0 on empty keywords) |
| `context_length` | Answer | Character count of assembled context |
| `answer_length` | Answer | Character count of generated answer |
| `citation_coverage` | Citation | Fraction of expected chunks among the returned sources. Sources are the chunks placed in the prompt (the model does not select citations), so this equals `context_recall` |
| `citation_provenance_precision` | Citation | Fraction of sources that were in the prompt context — 1.0 by construction; a sanity check, not a quality metric |
| `citation_count` | Citation | Number of unique source chunks returned |
| `chunk_coverage` | Citation | Fraction of reranked candidates that reached the prompt |
| `*_latency` | Latency | Per-node and total pipeline latency (seconds) |

> **Benchmark Dataset Format:** Each sample requires `query`, `expected_route`, `expected_chunk_ids`, and `ground_truth_answer`; optional fields are `keywords` (checked by `keyword_recall`) and `need_visual` (checked by `visual_routing_accuracy`). See `evaluation/datasets/cs162_lecture1_qa_50.json` for a real example, and `evaluation/datasets/sample_dataset.json` for a minimal format sample.

---

## Known Limitations

Measured or verified in this repository; see `docs/EVALUATION.md` for the evidence.

- **Evaluation labels are not human-reviewed.** The 45-question benchmark was LLM-drafted and
  then AI-reviewed; with n = 43 answerable questions, confidence intervals are about ±0.1.
- **No judge-based faithfulness or correctness metric** has been run (RAGAS is scaffolded only);
  answer F1 is lexical.
- **The knowledge-graph layer is experimental.** Graph results are rendered as relation lines,
  not linked to chunks, so they do not improve chunk ranking and they use prompt space; their
  benefit for relational answers has not been shown. On the only gold-labelled lecture, relation
  and prerequisite F1 are 0.0 (and were measured while the extraction prompts still contained
  that lecture's entities — removed in 2026-11).
- **The prerequisite Back-Tracker is experimental.** Prerequisites are inferred by rules at import
  time; a manual read of the 27 CS162 edges judged roughly a quarter to a third pedagogically
  sound (others are course-logistics entities or inverted part-of edges). Not human-validated;
  inferred-only edges are therefore kept out of GraphRAG answer context.
- **Routing is regex-based** (93% on the smoke set); missed `need_visual` flags cause refusals
  when the answer is only in a slide description.
- **Citations are context provenance**, not model-selected citations: the LLM does not see chunk
  IDs, and "sources" are the chunks placed in the prompt.
- **Concurrency:** the recorded load test fails 2 of 4 requests at 4 concurrent users.
- **No long-context baseline** (whole transcript in the prompt) has been measured yet.

## Design Decisions

The following table summarizes the key architectural decisions. See [DESIGN_DECISIONS.md](./DESIGN_DECISIONS.md) for the full rationale and trade-off analysis.

| Decision | Rationale |
|---|---|
| Cloud/Local split | GPU hardware should not be required for student interaction |
| GraphRAG over vanilla RAG | Relational queries require graph traversal, not just semantic similarity |
| Two-stage retrieval (bi-encoder + cross-encoder) | Balances retrieval recall (fast ANN) with generation precision (slow cross-encoder) |
| Single-pass orchestrator (`agent/pipeline/`) | Imperative planner → retrievers → reranker → generator flow; independently testable node functions, structured to be drop-in LangGraph-compatible with zero external dependency; evaluation-friendly |
| Strategy + Factory for LLM backends | Zero-restart provider switching; clean abstraction for Ollama  cloud APIs |
| Lazy LLM / eager reranker | Reranker is lecture-agnostic and query-critical — load once as a global singleton; LLM config changes at runtime |
| Decoupled startup | Server must boot cleanly with zero providers configured |
| Knowledge Package format | Offline-first, portable, open standards (NumPy, JSON, ZIP) |
| **Lecture isolation** | Both retrievers **raise** without a `lecture_id`; Neo4j nodes/edges scoped by `(entity_id, lecture_id)` — re-imports and shared cloud entity ids can never leak data between lectures |
| **Course = metadata only** | Courses are a JSON registry; course queries fan out to lecture-scoped retrievers and merge in the app layer — lecture graphs are never merged |
| **Global reranker singleton** | Loaded once at startup from `local_runtime/models/global_reranker/`; fine-tunes install there offline and hot-reload — inference code is never touched |
| **Additive debug schema** | `QueryResponse.debug` is an optional field defaulting to `{}` — backward compatible; only Developer Mode renders it |
| **Dashboard reads outputs** | The dashboard is a pure renderer over existing benchmark/RAGAS/load-test files; it never recomputes or re-runs evaluations |

---

## Production Architecture & Operational Characteristics

For the production Compose stack, provide a non-default Neo4j password:

```bash
export NEO4J_PASSWORD='replace-with-a-strong-secret'
docker compose -f docker/docker-compose.prod.yml up -d
```

| Dimension | Implementation | Production Characteristic |
|---|---|---|
| **Inference Scalability** | Per-provider token & rate limiter (`local/llm/rate_limiter.py`) | Smoothly manages API quotas with exponential backoff; ensures zero mid-stream interruptions across large-scale workloads |
| **Data Isolation** | Strict `(entity_id, lecture_id)` scoping plus lecture-scoped Qdrant point IDs | Prevents cross-lecture collisions in the tested retrieval and import paths; this is not a substitute for user authentication/authorization |
| **Memory Efficiency** | Dynamic quantization (`int8`) with automated FP32 fallback for CrossEncoder | Optimized for lightweight laptop deployment and minimal memory footprint |
| **Knowledge Portability** | Compressed Knowledge Package archives (.zip) | Portable JSON/NumPy artifacts with upload, expanded-size, file-count, schema, and content-hash validation |
| **Grounding Guardrails** | Evidence gating and source-ID provenance binding | Detects whether citations came from supplied context; claim-level faithfulness still requires a real RAGAS/judge evaluation and is not guaranteed |

---

## Future Roadmap

- [ ] Multi-user authentication & enterprise SSO integration
- [ ] Distributed live lecture streaming via Kafka ingestion pipeline
- [ ] Multi-turn conversation memory extension
- [ ] Quantized embedding storage (int8) for ultra-compact packages
- [ ] Course-level automated curriculum synthesis

---

## Documentation Index

| File | Contents |
|---|---|
| [ARCHITECTURE.md](./docs/ARCHITECTURE.md) | System overview, Cloud/Local split, Mermaid diagram, design patterns |
| [FOLDER_STRUCTURE.md](./docs/FOLDER_STRUCTURE.md) | Module-by-module directory analysis |
| [TECH_STACK.md](./docs/TECH_STACK.md) | Full technology inventory with versions and roles |
| [PIPELINE.md](./docs/PIPELINE.md) | Step-by-step execution flows for all major pipelines |
| [RETRIEVAL_SYSTEM.md](./docs/RETRIEVAL_SYSTEM.md) | Complete retrieval algorithm documentation |
| [EVALUATION.md](./docs/EVALUATION.md) | Evaluation framework, all metric implementations, known issues |
| [DESIGN_DECISIONS.md](./docs/DESIGN_DECISIONS.md) | Architectural decisions, trade-offs, limitations, future improvements |
| [METRICS_MATRIX.md](./METRICS_MATRIX.md) | Measured metrics, comparison vs baselines, and the metrics roadmap |

---

## References

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Neo4j Cypher Manual](https://neo4j.com/docs/cypher-manual/current/)
- [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) — Embedding model
- [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) — Cross-encoder reranker
- [Faster-Whisper](https://github.com/guillaumekln/faster-whisper) — ASR engine
- [Qwen2-VL](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct) — Vision-Language Model
- [Ollama](https://ollama.ai) — Local LLM inference daemon
- [FastAPI](https://fastapi.tiangolo.com) — Python HTTP framework
- [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) — Environment configuration

---

<div align="center">
<sub>LectureMIND — private, offline-first GraphRAG for lecture video.</sub>
</div>
