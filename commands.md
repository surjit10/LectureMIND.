# LectureMIND — Execution & Commands Cheatsheet

A quick, copy-paste reference for running, testing, evaluating, and managing the **LectureMIND** platform.

---

## Table of Contents

1. [Environment Setup & Installation](#1-environment-setup--installation)
2. [Infrastructure & Databases (Docker)](#2-infrastructure--databases-docker)
3. [Local LLM Engine (Ollama)](#3-local-llm-engine-ollama)
4. [Running the Application](#4-running-the-application)
5. [API & Query Execution](#5-api--query-execution)
6. [Course-Level Retrieval (Multi-Lecture)](#6-course-level-retrieval-multi-lecture)
7. [Testing Suite](#7-testing-suite)
8. [Evaluation & Benchmarking](#8-evaluation--benchmarking)
9. [Global Reranker Fine-Tuning (Pipeline B)](#9-global-reranker-fine-tuning-pipeline-b)
10. [Troubleshooting & Maintenance](#10-troubleshooting--maintenance)

---

## 1. Environment Setup & Installation

### Create & Activate Virtual Environment
```bash
# Navigate to project root
cd /home/surjit/Desktop/lecturemind/lecturemind

# Create virtual environment (Python 3.12 recommended)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install local dependencies
pip install --upgrade pip
pip install -r local_requirements.txt
```

### Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

### Configure Environment Variables
```bash
# Copy example configuration if .env does not exist
cp .env.example .env

# Verify EMBEDDING_DIMENSION=1024 (do not change, matches BGE-large-en-v1.5)
```

---

## 2. Infrastructure & Databases (Docker)

### Start Databases (Qdrant & Neo4j)
```bash
# Start Qdrant (port 6333) and Neo4j (ports 7474, 7687)
docker compose -f local/docker/docker-compose.yml up -d
```

### Verify Container Health
```bash
# Check running containers
docker ps --filter "name=lecturemind"

# Inspect database logs if needed
docker logs -f lecturemind-neo4j
docker logs -f lecturemind-qdrant
```

### Stop Databases
```bash
docker compose -f local/docker/docker-compose.yml down
```

---

## 3. Local LLM Engine (Ollama)

### Pull the Default Model
```bash
# Pull default local inference model
ollama pull qwen2.5:3b
```

### Optional: Enable GPU Acceleration for Ollama
```bash
# 1. Ensure NVIDIA container toolkit is installed
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 2. Start Ollama with GPU override
docker compose -f local/docker/docker-compose.yml -f local/docker/docker-compose.gpu.yml up -d ollama

# 3. Verify GPU allocation
docker exec lecturemind-ollama ollama ps
```

---

## 4. Running the Application

### Option A: Start the FastAPI Backend
```bash
# Terminal 1: Backend
source .venv/bin/activate
uvicorn serving.fastapi.app:app --host 0.0.0.0 --port 8000 --reload
```
* Backend API documentation available at: `http://localhost:8000/docs`

### Option B: Start the Next.js Frontend
```bash
# Terminal 2: Frontend
cd frontend
npm run dev
```
* Web UI available at: `http://localhost:3000`

---

## 5. API & Query Execution

### 1. Import a Knowledge Package (.zip)
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data/packages/lecture_sample.zip" \
  -F "display_name=CS162 Operating Systems - Lecture 1"
```

### 2. Check Active Lecture Status & List Lectures
```bash
# List all imported lectures
curl -s http://localhost:8000/lectures | jq

# Check status of specific lecture
curl -s http://localhost:8000/status/lecture_bb2ee3fa | jq
```

### 3. Ask a Question (Single-Pass Hybrid GraphRAG)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the process abstraction in operating systems?",
    "lecture_id": "lecture_bb2ee3fa"
  }' | jq
```

### 4. Switch to Cloud LLM Backend (e.g., Groq / OpenAI / Gemini)
```bash
# Register Groq Provider
curl -X POST http://localhost:8000/settings/providers \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "groq",
    "model": "llama-3.3-70b-versatile",
    "api_key": "YOUR_GROQ_API_KEY",
    "display_name": "Groq LLaMA 3.3"
  }'

# Switch active mode to online
curl -X POST http://localhost:8000/settings/inference-mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "online"}'

# Revert to local offline mode (Ollama)
curl -X POST http://localhost:8000/settings/inference-mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "local"}'
```

### 5. Generate Active Learning Content
```bash
# Generate Structured Notes
curl -X POST http://localhost:8000/notes \
  -H "Content-Type: application/json" \
  -d '{"lecture_id": "lecture_bb2ee3fa"}' | jq

# Generate Flashcards
curl -X POST http://localhost:8000/flashcards \
  -H "Content-Type: application/json" \
  -d '{"lecture_id": "lecture_bb2ee3fa", "count": 5}' | jq

# Generate Interactive Quiz
curl -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{"lecture_id": "lecture_bb2ee3fa", "count": 5}' | jq
```

---

## 6. Course-Level Retrieval (Multi-Lecture)

```bash
# 1. Create a Course
curl -X POST http://localhost:8000/courses \
  -H "Content-Type: application/json" \
  -d '{"name": "Operating Systems & Networking"}' | jq

# 2. Add Lectures to Course (replace {course_id} and {lecture_id})
curl -X POST http://localhost:8000/courses/{course_id}/lectures \
  -H "Content-Type: application/json" \
  -d '{"lecture_id": "lecture_bb2ee3fa"}' | jq

# 3. Query Across All Lectures in Course
curl -X POST http://localhost:8000/courses/query \
  -H "Content-Type: application/json" \
  -d '{
    "course_id": "{course_id}",
    "query": "How is virtual memory managed across different OS architectures?"
  }' | jq
```

---

## 7. Testing Suite

### Run Entire Test Suite (498 Tests)
```bash
source .venv/bin/activate
pytest
```

### Run Specific Test Modules
```bash
# Graph retrieval & fuzzy entity matching tests
pytest retrieval/tests/test_e2a_graph_retriever.py -v

# Hybrid BM25 & vector retrieval tests
pytest retrieval/tests/test_hybrid_retrieval.py -v

# Cross-encoder reranker tests
pytest retrieval/tests/test_e3_reranker.py -v

# LangGraph single-pass workflow tests
pytest agent/tests/test_e2_workflow.py -v

# FastAPI route & isolation tests
pytest serving/tests/test_e5_api.py -v
```

---

## 8. Evaluation & Benchmarking

### 1. Run Offline QA Benchmark (50 Grounded Questions)
```bash
source .venv/bin/activate
python -c "
from evaluation.benchmark_runner import BenchmarkRunner
runner = BenchmarkRunner(
    dataset_path='evaluation/datasets/cs162_lecture1_qa_50.json',
    output_dir='evaluation/outputs'
)
runner.setup()
runner.run()
"
```
* Reports generated in `evaluation/outputs/` (`.json`, `.csv`, `.md`).

### 2. Generate Interactive Benchmark Dashboard
```bash
source .venv/bin/activate
python -m evaluation.dashboard.dashboard_generator
```
* Output saved to `evaluation/dashboard/index.html` (self-contained with inline charts).
* Open in browser: `xdg-open evaluation/dashboard/index.html`

### 3. Run RAGAS Answer Quality Evaluation
```bash
source .venv/bin/activate
pip install ragas datasets   # optional dependency group
python -m evaluation.ragas.eval_ragas --report evaluation/outputs/<benchmark_report>.json
```
* Requires the optional `ragas` package and an LLM judge; exits with a clear message if missing.
* No RAGAS report exists yet in this repository.

### 4. Run Load Testing
```bash
source .venv/bin/activate
python -m evaluation.load_testing.load_test --url http://localhost:8000 --users 100
```
* Requires the FastAPI server to be running.
* No load-test report exists yet in this repository.

---

## 9. Global Reranker Fine-Tuning (Pipeline B)

```bash
# Merge package triplets -> fine-tune cross-encoder -> install to local_runtime/models/ -> hot-reload
source .venv/bin/activate
python scripts/train_global_reranker.py
```

### Inspect Query Pipeline Trace (CLI)
```bash
source .venv/bin/activate
python scripts/trace_query.py "Explain process synchronization primitives" --lecture-id "lecture_bb2ee3fa"
```

---

## 10. Troubleshooting & Maintenance

### Check SDK & Hardware Compatibility
```bash
source .venv/bin/activate
python scripts/check_sdks.py
```

### Reset / Clear Database Volumes (Clean State)
```bash
# Stop containers and remove volumes
docker compose -f local/docker/docker-compose.yml down -v

# Restart fresh containers
docker compose -f local/docker/docker-compose.yml up -d
```
